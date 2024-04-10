import torch 
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.datasets import Planetoid, SNAPDataset
from torch_geometric.utils import subgraph, to_networkx
from datasets.abstract_dataset import AbstractDatasetInfos, AbstractDataModule
from datasets.fair_rw import FairRW
from datasets.custom_datasets import NBADataset

class SampledDataset(LightningDataset):
    def __init__(self, dataset, sampler, n_samples):
        self.dataset = dataset
        self.data = []
        #qm9 = QM9("../data")

        if self.dataset == "Cora":
            self.graph = Planetoid("../data","Cora").get(0)
            self.sensitive_attribute = self.graph.y
        elif self.dataset == "NBA":
            self.graph = NBADataset("../data").get(0)
            #Country index 36
            self.sensitive_attribute = self.graph.x[:,36]
        elif self.dataset == "Facebook":
            #1045 nodes 
            self.graph = SNAPDataset("../data", 'ego-facebook').get(1)
            #Male-Female indexes 666,667
            self.sensitive_attribute = self.graph.x[:,666].detach().clone()
            self.sensitive_attribute[torch.logical_and(self.graph.x[:,666] == 0, self.graph.x[:,667] == 0)] = 2

        elif self.dataset == "Oklahoma97":
            pass
        elif self.dataset == "UNC28":
            pass

        self.G = to_networkx(self.graph)
        self.G = self.G.to_undirected()
        sampled_graphs = [list(set(sampler.sample(self.G, 50))) for i in range(n_samples)]

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
        datasets = {'train': SampledDataset(cfg.dataset.name, FairRW(), 500),
                    'val': SampledDataset(cfg.dataset.name, FairRW(), 100),
                    'test': SampledDataset(cfg.dataset.name, FairRW(), 100)}
        self.datasets = datasets
        super().__init__(cfg, datasets)


class SampledDatasetInfo(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)