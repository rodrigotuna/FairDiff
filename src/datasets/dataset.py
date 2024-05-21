import torch 
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.datasets import Planetoid, SNAPDataset
from torch_geometric.utils import subgraph, to_networkx, coalesce
from datasets.abstract_dataset import AbstractDatasetInfos, AbstractDataModule
from datasets.fair_rw import FairRW
from datasets.custom_datasets import NBADataset, CollegiateSocNet
import numpy as np
import numpy.random as random

class SampledDataset(LightningDataset):
    def __init__(self, cfg, sampler, n_samples):
        self.data = []
        #qm9 = QM9("../data")

        if cfg.dataset.name == "Cora":
            self.graph = Planetoid("../data","Cora").get(0)
            self.sensitive_attribute = self.graph.y.detach().clone()
        elif cfg.dataset.name == "NBA":
            self.graph = NBADataset("../data").get(0)
            #Country index 36
            self.sensitive_attribute = self.graph.x[:,36].detach().clone()
        elif cfg.dataset.name == "Facebook":
            #1045 nodes 
            self.graph = SNAPDataset("../data", 'ego-facebook').get(1)
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
            self.graph = CollegiateSocNet("../data", "oklahoma97").get(0)
            self.sensitive_attribute = self.graph.x[:,1].detach().clone()
        elif cfg.dataset.name == "UNC28":
            self.graph = CollegiateSocNet("../data", "unc28").get(0)
            self.sensitive_attribute = self.graph.x[:,1].detach().clone()

        self.G = to_networkx(self.graph, to_undirected=True)
        
        if n_samples == None:
            num_samples = (2,2) if cfg.dataset.fair else (4,0)
            sampled_graphs = []
            for i in num_samples[0] * list(range(self.graph.x.shape[0])):
                sampled_graphs.append(list(set(sampler.sample(self.G, 20, starting_node=i, sensitive_attribute=self.sensitive_attribute if cfg.dataset.fair else None))))

            ##Sample from G_k
            for i in num_samples[1] * list(range(self.graph.x.shape[0])):
                sampled_graphs.append(list(set(sampler.sample(self.G, 20, k=True, starting_node=i, sensitive_attribute=self.sensitive_attribute if cfg.dataset.fair else None))))

        else:
            sampled_graphs = [list(set(sampler.sample(self.G, 20, 
                                                    sensitive_attribute=self.sensitive_attribute if cfg.dataset.fair else None))) for i in range(n_samples)]

        sampled_graphs_dict = [dict(zip(sample,range(len(sample)))) for sample in sampled_graphs]
        sampled_edge_index = [subgraph(sample, self.graph.edge_index)[0].apply_(lambda x : sampled_graphs_dict[idx][x]) for idx, sample in enumerate(sampled_graphs)]
        sampled_edge_attr = [torch.stack([torch.zeros(len(edge_index[0])), torch.ones(len(edge_index[0]))]).T for edge_index in sampled_edge_index]
        sampled_x = [F.one_hot(torch.tensor(sample), num_classes = self.G.number_of_nodes()).float() for sample in sampled_graphs]
        self.data = [Data(x = sampled_x[idx], edge_index = sampled_edge_index[idx], edge_attr=sampled_edge_attr[idx], y=torch.zeros(1,0)) for idx, sample in enumerate(sampled_graphs)] #this can be used for fairness of links does this impact fairness?

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    

class SampledDataModule(AbstractDataModule):
    def __init__(self, cfg):
        datasets = {'train': SampledDataset(cfg, FairRW(), None),
                    'val': SampledDataset(cfg, FairRW(), 100),
                    'test': SampledDataset(cfg, FairRW(), 100)}
        self.datasets = datasets
        super().__init__(cfg, datasets)


class SampledDatasetInfo(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.num_edges = datamodule.datasets['train'].G.number_of_edges()
        print(self.num_edges)
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)