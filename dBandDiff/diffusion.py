import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Any, Dict, Optional, Sequence
from torch_geometric.data import DataLoader
from torch_scatter import scatter
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm
import numpy as np
from dBandDiff.diff_utils import d_log_p_wrapped_normal
import hydra
import omegaconf
from torch_scatter.composite import scatter_softmax
from dBandDiff.cspnet import CSPNet 
from dBandDiff.data_utils import (EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, 
                        lattice_params_to_matrix_torch, frac_to_cart_coords, min_distance_sqr_pbc)
from dBandDiff.crystal_family import CrystalFamily
from dBandDiff.diff_utils import d_log_p_wrapped_normal
from dBandDiff.diff_utils import BetaScheduler, SigmaScheduler
from copy import deepcopy as dc
import json

MAX_ATOMIC_NUM = 118


class BaseModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        #self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.decoder = self._instantiate(cfg['decoder'])
        self.beta_scheduler = self._instantiate(cfg['beta_scheduler'])
        self.sigma_scheduler = self._instantiate(cfg['sigma_scheduler'])
        self.time_dim = self._instantiate(cfg['time_dim'])
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        #self.spacegroup_embedding = nn.Embedding(num_embeddings=230, embedding_dim=self._instantiate(cfg['spg_number_dim']))
        self.ops_dim = self._instantiate(cfg['ops_dim'])
        self.ops_embedding = nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, self.ops_dim)
        )        
        self.d_band_center_dim = self._instantiate(cfg['d_band_center_dim'])
        self.d_band_center_embedding = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, self.d_band_center_dim)
        )

        self.crystal_family = CrystalFamily()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 初始化device
        self.to(self.device)  # 将模型移到设备上 


    '''
    def _instantiate(self, class_path, **kwargs):
        # 如果class_path是字符串类型，才用eval
        if isinstance(class_path, str):
            class_obj = eval(class_path)
            return class_obj(**kwargs)
        # 如果class_path不是字符串，直接返回它的值
        return class_path
    '''
    def _instantiate(self, module_cfg):
        if isinstance(module_cfg, (int, float, bool, str)):
            return module_cfg  # 直接返回基本类型
        elif isinstance(module_cfg, dict):
            cls = globals()[module_cfg['type']]
            params = module_cfg.get('params', {})
            return cls(**params)
        else:
            raise ValueError("Unsupported module config format.")
    
        
    def forward(self, batch, batch_idx = None):


        batch_size = batch.num_graphs
        #spg_number_emb = self.spacegroup_embedding((batch.spg_number - 1).long())
        #spg_number = spg_number_emb.squeeze(1)
        wyckoff_ops = batch.ops
        wyckoff_ops = wyckoff_ops.view(wyckoff_ops.size(0), -1)
        wyckoff_ops = self.ops_embedding(wyckoff_ops)
        
        d_band_center = batch.d_band_center
        d_band_center = self.d_band_center_embedding(d_band_center)
        
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        #print(f't shape: {time_emb.shape}, num_atoms shape: {batch.num_atoms.shape}')
        #print(f't: {time_emb}, num_atoms: {batch.num_atoms}')
        
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        lattices = self.crystal_family.de_so3(lattices)
        frac_coords = batch.frac_coords

        rand_x = torch.randn_like(frac_coords)

        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]

        
        rand_x_anchor = rand_x[batch.anchor_index]
        rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        spacegroup = batch.spg_number.squeeze(-1)
        spacegroup = spacegroup.to(torch.long)
        ori_crys_fam = self.crystal_family.m2v(lattices)
        ori_crys_fam = self.crystal_family.proj_k_to_spacegroup(ori_crys_fam, spacegroup)
        rand_crys_fam = torch.randn_like(ori_crys_fam)
        rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, spacegroup)
        input_crys_fam = c0[:, None] * ori_crys_fam + c1[:, None] * rand_crys_fam
        input_crys_fam = self.crystal_family.proj_k_to_spacegroup(input_crys_fam, spacegroup)

        gt_atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
        rand_t = torch.randn_like(gt_atom_types_onehot)[batch.anchor_index]
        atom_type_probs = (c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t)
        atom_type_probs = atom_type_probs[batch.anchor_index]

        pred_crys_fam, pred_x, pred_t = self.decoder(time_emb, wyckoff_ops, d_band_center, atom_type_probs, input_frac_coords, input_crys_fam, batch.num_atoms, batch.batch)
        pred_crys_fam = self.crystal_family.proj_k_to_spacegroup(pred_crys_fam, spacegroup)

        pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)

        tar_x_anchor = d_log_p_wrapped_normal(sigmas_per_atom * rand_x_anchor, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_crys_fam, rand_crys_fam)
        loss_coord = F.mse_loss(pred_x_proj, tar_x_anchor)
        loss_type = F.mse_loss(pred_t, rand_t)


        loss = (
            cfg['cost_lattice'] * loss_lattice +
            cfg['cost_coord'] * loss_coord + 
            cfg['cost_type'] * loss_type)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
            'loss_type' : loss_type
        }

    @torch.no_grad()
    def sample(self, batch, diff_ratio = 1.0, step_lr = 1e-5):


        batch_size = batch.num_graphs
        #spg_number_emb = self.spacegroup_embedding((batch.spg_number - 1).long())
        #spg_number = spg_number_emb.squeeze(1)
        wyckoff_ops = batch.ops
        wyckoff_ops = wyckoff_ops.view(wyckoff_ops.size(0), -1)
        wyckoff_ops = self.ops_embedding(wyckoff_ops)
        
        spacegroup = batch.spg_number.squeeze(-1)
        spacegroup = spacegroup.to(torch.long)
        
        d_band_center = batch.d_band_center
        d_band_center = self.d_band_center_embedding(d_band_center)
        
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        crys_fam_T = torch.randn([batch_size, 6]).to(self.device)
        crys_fam_T = self.crystal_family.proj_k_to_spacegroup(crys_fam_T, spacegroup)
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)
        #print(t_T)

        time_start = self.beta_scheduler.timesteps - 1

        l_T = self.crystal_family.v2m(crys_fam_T)

        x_T_all = torch.cat([x_T[batch.anchor_index], torch.ones(batch.ops.size(0),1).to(x_T.device)], dim=-1).unsqueeze(-1) # N * 4 * 1

        x_T = (batch.ops @ x_T_all).squeeze(-1)[:,:3] % 1. # N * 3

        t_T = t_T[batch.anchor_index]

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : t_T,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T,
            'crys_fam': crys_fam_T
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)

            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]


            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)


            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            crys_fam_t = traj[t]['crys_fam']
            t_t = traj[t]['atom_types']

            # Corrector

            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            
            step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            rand_x_anchor = rand_x[batch.anchor_index]
            rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
            rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

            pred_crys_fam, pred_x, pred_t = self.decoder(time_emb, wyckoff_ops, d_band_center, t_t, x_t, crys_fam_t, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)
            pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce = 'mean')[batch.anchor_index]

            pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1) 

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x

            crys_fam_t_minus_05 = crys_fam_t

            frac_coords_all = torch.cat([x_t_minus_05[batch.anchor_index], torch.ones(batch.ops.size(0),1).to(x_t_minus_05.device)], dim=-1).unsqueeze(-1) # N * 4 * 1

            x_t_minus_05 = (batch.ops @ frac_coords_all).squeeze(-1)[:,:3] % 1. # N * 3

            t_t_minus_05 = t_t

            # Predictor

            rand_crys_fam = torch.randn_like(crys_fam_T)
            rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, spacegroup)
            ori_crys_fam = crys_fam_t_minus_05
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            rand_x_anchor = rand_x[batch.anchor_index]
            rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
            rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

            rand_t = rand_t[batch.anchor_index]

            pred_crys_fam, pred_x, pred_t = self.decoder(time_emb, wyckoff_ops, d_band_center, t_t_minus_05, x_t_minus_05, crys_fam_t_minus_05, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)
            
            crys_fam_t_minus_1 = c0 * (ori_crys_fam - c1 * pred_crys_fam) + sigmas * rand_crys_fam
            crys_fam_t_minus_1 = self.crystal_family.proj_k_to_spacegroup(crys_fam_t_minus_1, spacegroup)

            pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)
            pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce = 'mean')[batch.anchor_index]
            pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1) 

            pred_t = scatter(pred_t, batch.anchor_index, dim=0, reduce = 'mean')[batch.anchor_index]


            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x

            l_t_minus_1 = self.crystal_family.v2m(crys_fam_t_minus_1)


            frac_coords_all = torch.cat([x_t_minus_1[batch.anchor_index], torch.ones(batch.ops.size(0),1).to(x_t_minus_1.device)], dim=-1).unsqueeze(-1) # N * 4 * 1

            x_t_minus_1 = (batch.ops @ frac_coords_all).squeeze(-1)[:,:3] % 1. # N * 3

            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

            t_t_minus_1 = t_t_minus_1[batch.anchor_index]


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1,
                'crys_fam': crys_fam_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        res = dc(traj[0])
        res['atom_types'] = res['atom_types'].argmax(dim=-1) + 1

        return res, traj_stack




# Example configuration (replacing Hydra with direct config dictionary)
with open('dBandDiff/conf.json', "r") as f:
    cfg = json.load(f)



# Assuming the train_loader is passed from elsewhere (e.g., from your datamodule)
# Create and pass the train_loader to your model
model = CSPDiffusion(cfg=cfg)