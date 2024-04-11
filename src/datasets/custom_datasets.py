from typing import Callable, Optional, Any
import torch
import os
import os.path as osp
from torch_geometric.data import makedirs, download_url, InMemoryDataset, Data
from torch_geometric.utils import coalesce
import pandas as pd
import scipy.io as sio
import numpy as np

class NBADataset(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/LavinWong/Graph-Fairness-Data/main/NBA'
    files = ['nba.csv', 'nba_relationship.txt']

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.name = 'nba'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    def _download(self):
        if osp.isdir(self.raw_dir) and len(os.listdir(self.raw_dir)) > 0:
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        for name in self.files:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self):
        raw_dir = self.raw_dir
        raw_files = [osp.join(raw_dir, f) for f in self.files]

        data_list = []

        x = pd.read_csv(raw_files[0])
        id_map = x.user_id.to_dict()
        id_map = dict(zip(id_map.values(), id_map.keys()))
        x = torch.from_numpy(x.values)[:, 1:]
        

        row = pd.read_csv(raw_files[1], sep='\t', header=None,
                              usecols=[0]).squeeze()
        col = pd.read_csv(raw_files[1], sep='\t', header=None,
                              usecols=[1]).squeeze()

        row = torch.tensor([id_map[i] for i in row])
        col = torch.tensor([id_map[i] for i in col])

        edge_index = torch.stack([row, col], dim=0)
        edge_index = coalesce(edge_index)

        data_list.append(Data(x=x, edge_index=edge_index))

        if len(data_list) > 1 and self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
        


class CollegiateSocNet(InMemoryDataset):
    '''

    '''
    url = "https://raw.githubusercontent.com/rodrigotuna/Graph-Datasets/main/"
    datasets = ["unc28", "oklahoma97"]
    files = {
        "unc28" : ["UNC28.edgelist", "UNC28.mat"],
        "oklahoma97":  ["Oklahoma97.edgelist", "Oklahoma97.mat"],
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.name = name.lower()
        assert self.name in self.datasets
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    def _download(self):
        if osp.isdir(self.raw_dir) and len(os.listdir(self.raw_dir)) > 0:
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        for file in self.files[self.name]:
            download_url(f'{self.url}/{self.name}/{file}', self.raw_dir)

    def process(self):
        raw_dir = self.raw_dir
        raw_files = [osp.join(raw_dir, f) for f in self.files[self.name]]
        data_list = []

        mat = sio.loadmat(raw_files[1])
        x = mat['local_info']

        total_id_map = dict(zip(x[:,0], list(range(x.shape[0]))))
        non_missinng_val_rows = (np.sum(x == 0, axis=1) == 0)
        x = x[non_missinng_val_rows]

        id_map = dict(zip(np.vectorize(total_id_map.get)(x[:,0]), list(range(x.shape[0]))))
        x = torch.from_numpy(x)[:, 1:]

        edges = np.loadtxt(raw_files[0])
        existing_edges = (np.sum(np.isin(edges, list(id_map.keys())), axis=1) == 2)
        edge_index = np.vectorize(id_map.get)(edges[existing_edges])
        edge_index = torch.tensor(edge_index.T)
        edge_index = coalesce(edge_index)


        data_list.append(Data(x=x, edge_index=edge_index))


        if len(data_list) > 1 and self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])