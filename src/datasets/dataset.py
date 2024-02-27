import torch 
from torch_geometric.data import Data
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.datasets import Planetoid, QM9
from torch_geometric.utils import subgraph
import networkx as nx

class SampledDataset(LightningDataset):
    def __init__(self, dataset, sampler, n_samples):
        self.dataset = dataset
        self.data = []

        if self.dataset == "Cora":
            self.sensitive_attribute = "gender"
            self.graph = Planetoid("../data","Cora")

        edges = [(int(edge[0]), int(edge[1])) for edge in self.graph.edge_index.T]
        self.G = nx.from_edgelist(edges, create_using=nx.DiGraph)
        print(self.G)
        sampled_graphs = [sampler.sample(self.G, 10) for i in range(n_samples)]

        sampled_graphs_dict = [dict(zip(sample, range(len(sample)))) for sample in sampled_graphs]
        self.data = [Data(x = self.graph.x[sample], edge_index = subgraph(sample, self.graph.edge_index)[0].apply_(lambda x : sampled_graphs_dict[idx][x])) for idx, sample in enumerate(sampled_graphs)] 


    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
