import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
import os
from torch_geometric.data import Data
import pickle
import numpy as np

from dBandDiff.data_utils import (preprocess, preprocess_tensors, add_scaled_lattice_prop)

import pytorch_lightning as pl
from typing import Optional, Sequence
from pathlib import Path
import random
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
from dBandDiff.data_utils import preprocess, preprocess_tensors, add_scaled_lattice_prop, get_scaler_from_data_list



class CrystDataset(Dataset):
    def __init__(self, name, path, prop, niggli, primitive, graph_method, preprocess_workers, lattice_scale_method, save_path, tolerance, use_space_group, use_pos_index, **kwargs):
        super().__init__()
        self.path = path
        self.folder = None
        self.name = name
        #self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance
        #self.max_num_atoms = max_num_atoms
        self.preprocess(save_path, preprocess_workers, prop)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def preprocess(self, save_path, preprocess_workers, prop):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess(
            self.path,
            self.folder,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop],
            use_space_group=self.use_space_group,
            tol=self.tolerance)
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        #prop = self.scaler.transform(data_dict[self.prop])
        (frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms) = data_dict['graph_arrays']
        mp_id = data_dict['mp_id']
        d_band_center = data_dict['d_band_center']
        

        
        data = Data(
            mp_id = mp_id,
            d_band_center = torch.Tensor([d_band_center]).view(1, -1),
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(edge_indices.T).contiguous(),
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
        )

        if self.use_space_group:
            data.spg_number = torch.Tensor([data_dict['spacegroup']]).view(1, -1)
            data.ops = torch.Tensor(data_dict['wyckoff_ops'])
            data.anchor_index = torch.LongTensor(data_dict['anchors'])
            data.ops_inv = torch.linalg.pinv(data.ops[:,:3,:3])
        
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class CrystDataModule(pl.LightningDataModule):
    def __init__(self, datasets, num_workers, batch_size, scaler_path=None):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.scaler_path = scaler_path

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Sequence[Dataset]] = None
        self.test_dataset: Optional[Sequence[Dataset]] = None

        self.get_scaler(scaler_path)

    def prepare_data(self) -> None:
        pass

    def get_scaler(self, scaler_path):
        #if scaler_path is None:
            #train_dataset = self.datasets['train']
            #self.lattice_scaler = get_scaler_from_data_list(
                #train_dataset.cached_data,
                #key='scaled_lattice')
            #self.scaler = get_scaler_from_data_list(
                #train_dataset.cached_data,
                #key=train_dataset.prop)
        #else:
            #self.lattice_scaler = torch.load(
                #Path(scaler_path) / 'lattice_scaler.pt')
            #self.scaler = torch.load(Path(scaler_path) / 'prop_scaler.pt')
        pass
        
    def setup(self, stage: Optional[str] = None):
        if stage is None or stage == "fit":
            self.val_dataset = self.datasets['val']
            self.train_dataset = self.datasets['train']
            #self.train_dataset.lattice_scaler = self.lattice_scaler
            #self.train_dataset.scaler = self.scaler


    def train_dataloader(self, shuffle=True):

        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size['train'],
            num_workers=self.num_workers['train']
        )
        
    def val_dataloader(self, shuffle=False):
        return DataLoader(
            self.val_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size['val'],
            num_workers=self.num_workers['val'],
        )





def main():
    # Example configuration in place of Hydra's config system
    dataset_config = {   
        'train': CrystDataset(
            name="train",
            path="dataset/mpdos_TM_30_train.csv",
            prop=None,
            niggli=False,
            primitive=False,
            graph_method='crystalnn',
            preprocess_workers=1,
            lattice_scale_method="scale_length",
            save_path="dataset/train_data.pt",
            tolerance=0.1,
            use_space_group=True,
            use_pos_index=False
        ),
        'val': CrystDataset(
            name="val",
            path="dataset/mpdos_TM_30_val.csv",
            prop=None,
            niggli=False,
            primitive=False,
            graph_method='crystalnn',
            preprocess_workers=1,
            lattice_scale_method="scale_length",
            save_path="dataset/val_data.pt",
            tolerance=0.1,
            use_space_group=True,
            use_pos_index=False
        )
          
    }

    num_workers = {'train': 16, 'val': 16, 'test': 16}
    batch_size = {'train': 64, 'val': 32, 'test': 32}

    # Initialize data module and setup
    datamodule = CrystDataModule(
        datasets=dataset_config,
        num_workers=num_workers,
        batch_size=batch_size
    )
    datamodule.setup('fit')
    return datamodule



if __name__ == "__main__":
    datamodule = main()

