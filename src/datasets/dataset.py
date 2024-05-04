import torch 
import os
import numpy as np
import numpy.random as random
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.datasets import Planetoid, SNAPDataset
from torch_geometric.utils import subgraph, to_networkx, coalesce
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.loader import LinkNeighborLoader

from datasets.fair_rw import FairRW
from datasets.custom_datasets import NBADataset, CollegiateSocNet
from datasets.abstract_dataset import AbstractDatasetInfos, AbstractDataModule


class SampledDataset(LightningDataset):
    def __init__(self, cfg, sampler, n_samples):
        self.data = []
        self.path = "../data"

        if cfg.dataset.name == "Cora":
            self.graph = Planetoid(self.path,"Cora").get(0)
            self.path += "/Cora"
            self.sensitive_attribute = self.graph.y.detach().clone()
        elif cfg.dataset.name == "NBA":
            self.graph = NBADataset(self.path).get(0)
            self.path += "/nba"
            #Country index 36
            self.sensitive_attribute = self.graph.x[:,36].detach().clone()
        elif cfg.dataset.name == "Facebook":
            #1045 nodes 
            self.graph = SNAPDataset(self.path, 'ego-facebook').get(1)
 
            self.path += "/ego-facebook"
            #Male-Female indexes 666,667
            ids = np.array(list(range(self.graph.x.shape[0])))
            keep_rows = torch.logical_or(self.graph.x[:,666] == 1, self.graph.x[:,667] == 1)

            self.graph.x = self.graph.x[keep_rows]

            id_map = dict(zip(ids[keep_rows], list(range(keep_rows.sum()))))
            
            existing_edges = (np.sum(np.isin(self.graph.edge_index.T, list(id_map.keys())), axis=1) == 2)
            
            edge_index = np.vectorize(id_map.get)(self.graph.edge_index.T[existing_edges])
            edge_index = torch.tensor(edge_index.T)
            self.graph.edge_index = coalesce(edge_index)
            self.sensitive_attribute = self.graph.x[:,666].detach().clone()

        elif cfg.dataset.name == "Oklahoma97":
            self.graph = CollegiateSocNet(self.path, "oklahoma97").get(0)
            self.path += "/oklahoma97"
            self.sensitive_attribute = self.graph.x[:,1].detach().clone()
        elif cfg.dataset.name == "UNC28":
            self.graph = CollegiateSocNet(self.path, "unc28").get(0)
            self.path += "/unc28"
            self.sensitive_attribute = self.graph.x[:,1].detach().clone()

        ##Embeddings phase
        if os.path.isfile(self.path + "/processed/embeddings.pt"):
            self.node_embeddings = torch.load(self.path + "/processed/embeddings.pt")
        else:
            print("Embeddings not found. Starting to compute")

            model = GraphSAGE(in_channels=self.graph.x.shape[1], hidden_channels=512, num_layers=2, out_channels=128)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            loader = LinkNeighborLoader(self.graph, num_neighbors=[10,10], neg_sampling_ratio=0.5, batch_size=128, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            
            model.train()
            for i in range(10):
                total_loss = 0
                for batch in loader:
                    batch.to(device)
                    optimizer.zero_grad()
                    h = model(batch.x, batch.edge_index)
                    h_src = h[batch.edge_label_index[0]]
                    h_dst = h[batch.edge_label_index[1]]
                    pred = (h_src * h_dst).sum(dim=-1)
                    loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() / pred.size(0)
                print(f"Epoch {i+1}/10 Loss: {total_loss}")

            model.eval()
            self.graph.to(device)
            with torch.no_grad():
                self.node_embeddings = model(self.graph.x, self.graph.edge_index).to('cpu')
            
            torch.save(self.node_embeddings, self.path + "/processed/embeddings.pt") 
        
        self.graph.to('cpu')
        self.G = to_networkx(self.graph, to_undirected=True)
        if n_samples:
            sampled_graphs = [list(set(sampler.sample(self.G, 20))) for i in range(n_samples)]
        else:
            sampled_graphs = [list(set(sampler.sample(self.G, 20, starting_node=i))) for i in 4 * list(range(self.graph.x.shape[0]))]

        sampled_graphs_dict = [dict(zip(sample,range(len(sample)))) for sample in sampled_graphs]
        sampled_edge_index = [subgraph(sample, self.graph.edge_index)[0].apply_(lambda x : sampled_graphs_dict[idx][x]) for idx, sample in enumerate(sampled_graphs)]
        sampled_edge_attr = [torch.stack([torch.zeros(len(edge_index[0])), torch.ones(len(edge_index[0]))]).T for edge_index in sampled_edge_index]
        sampled_x = [self.node_embeddings[sample] for sample in sampled_graphs]
        self.data = [Data(x = sampled_x[idx], edge_index = sampled_edge_index[idx], edge_attr=sampled_edge_attr[idx], y=torch.zeros(1,0)) for idx, sample in enumerate(sampled_graphs)] #this can be used for fairness of links does this impact fairness?

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    

class SampledDataModule(AbstractDataModule):
    def __init__(self, cfg):
        datasets = {'train': SampledDataset(cfg, FairRW(),  None),
                    'val': SampledDataset(cfg, FairRW(), 1),
                    'test': SampledDataset(cfg, FairRW(), 1)}
        self.datasets = datasets
        super().__init__(cfg, datasets)


class SampledDatasetInfo(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)