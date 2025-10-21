import numpy as np
import pandas as pd
import networkx as nx
import torch
import copy
import itertools
import os
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from networkx.algorithms.components import is_connected
from sklearn.metrics import accuracy_score, recall_score, precision_score
from torch_scatter import scatter
from torch_scatter import segment_coo, segment_csr
from p_tqdm import p_umap
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group
from pyxtal import pyxtal
from pathos.pools import ProcessPool as Pool
# from multiprocessing import Pool
from tqdm import tqdm 
from functools import partial 

import faulthandler
faulthandler.enable()

def get_symmetry_info(cif_path, tol=0.01):
    structure = Structure.from_file(cif_path)
    spga = SpacegroupAnalyzer(structure, symprec=tol)
    crystal = spga.get_refined_structure()
    c = pyxtal()
    try:
        c.from_seed(crystal, tol=0.01)
    except:
        c.from_seed(crystal, tol=0.0001)
    space_group = c.group.number
    species = []
    anchors = []
    matrices = []
    coords = []
    for site in c.atom_sites:
        specie = site.specie
        anchor = len(matrices)
        coord = site.position
        for syms in site.wp:
            species.append(specie)
            matrices.append(syms.affine_matrix)
            coords.append(syms.operate(coord))
            anchors.append(anchor)
    anchors = np.array(anchors)
    matrices = np.array(matrices)
    coords = np.array(coords) % 1.
    sym_info = {
        'anchors':anchors,
        'wyckoff_ops':matrices,
        'spacegroup':space_group
    }
    crystal = Structure(
        lattice=Lattice.from_parameters(*np.array(c.lattice.get_para(degree=True))),
        species=species,
        coords=coords,
        coords_are_cartesian=False,
    )
    return crystal, c, sym_info