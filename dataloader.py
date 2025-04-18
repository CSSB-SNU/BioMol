import os
from typing import List, Tuple, Dict, Any, overload
import torch
from constant.chemical import *
from utils.parser import parse_cif
from biomol import BioMol
from utils.MSA import MSA, ComplexMSA
import pickle
import lmdb
from torch.utils.data import Dataset, DataLoader
import random
import torch.distributed as dist

def load_biomol(path : str) -> BioMol:
    """Helper function to load a pickle file."""
    with open(path, "rb") as f:
        return path, pickle.load(f)
    

# For now, I only load protein data.
# For now, no distillation set like FB data.
    
class BioMolDataset(Dataset):
    def __init__(self,
                 meta_data_dict: Dict[Any],
                 params: Dict[str, Any],
                 ):
        self.meta_data_dict = meta_data_dict
        self.params = params
        self._set_seed(params['seed'])
        
    def _set_seed(self, seed = 0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def __getitem__(self, cluster_idx):
        biomol_path = ""
        biomol = load_biomol(biomol_path)
        bioassembly_id, model_id, label_alt_id = None, None, None
        biomol.choose(bioassembly_id, model_id, label_alt_id)        
        biomol.crop_and_load_msa(self.params)
        
        return biomol
    
class DistributedWeightedSampler(torch.utils.data.Sampler):
    def __init__(self, 
                 dataset, 
                 pdb_weights, 
                 FB_weights,
                 use_FB = True, 
                 num_example_per_epoch=25600,
                 fraction_FB = 0.3, 
                 num_replicas=None, 
                 rank=None, 
                 replacement=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        assert num_example_per_epoch % num_replicas == 0 or num_example_per_epoch == -1

        self.dataset = dataset
        self.use_FB = use_FB
        self.num_replicas = num_replicas

        if self.use_FB:
            if num_example_per_epoch == -1:
                self.num_pdb_per_epoch = self.dataset.num_PDB_cluster
                self.num_FB_per_epoch = self.dataset.num_FB
                self.num_example_per_epoch = self.num_pdb_per_epoch + self.num_FB_per_epoch
            else : 
                self.num_FB_per_epoch = int(round(num_example_per_epoch*fraction_FB))
                self.num_pdb_per_epoch = num_example_per_epoch - self.num_FB_per_epoch
        else :
            self.num_FB_per_epoch = 0
            if num_example_per_epoch == -1:
                num_example_per_epoch = self.dataset.num_PDB_cluster
                self.num_pdb_per_epoch = self.dataset.num_PDB_cluster
            else : 
                self.num_pdb_per_epoch = min(num_example_per_epoch, self.dataset.num_PDB_cluster)

        self.total_size = self.num_pdb_per_epoch + self.num_FB_per_epoch
        if self.total_size % self.num_replicas != 0 :
            self.num_pdb_per_epoch += self.num_replicas - self.total_size % self.num_replicas
            self.total_size = self.num_pdb_per_epoch + self.num_FB_per_epoch
        self.num_samples = self.total_size // self.num_replicas
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement
        self.pdb_weights = pdb_weights
        self.FB_weights = FB_weights

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        np.random.seed(self.epoch)
        
        # get indices (fb + pdb models)
        indices = torch.arange(len(self.dataset))

        # weighted subsampling
        # 1. subsample fb and pdb based on length
        sel_indices = torch.tensor((),dtype=int)
        
        if (self.num_pdb_per_epoch>0):
            if self.pdb_weights is None :
                self.pdb_weights = torch.ones(self.dataset.num_PDB_cluster)
            pdb_sampled = torch.multinomial(self.pdb_weights, self.num_pdb_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[pdb_sampled]))

        if (self.num_FB_per_epoch>0):
            if self.FB_weights is None :
                self.FB_weights = torch.ones(self.dataset.num_FB)
            FB_sampled = torch.multinomial(self.FB_weights, self.num_FB_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[FB_sampled + self.dataset.num_PDB_cluster]))

        # shuffle indices
        indices = sel_indices[torch.randperm(len(sel_indices), generator=g)]

        # per each gpu
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples, f"{len(indices)} != {self.num_samples}"

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch