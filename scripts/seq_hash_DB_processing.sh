#!/bin/bash
#SBATCH -J seq_hash_DB
#SBATCH --ntasks-per-node=1         
#SBATCH -c 40                        
#SBATCH --mem=480g
#SBATCH -p cpu-farm
#SBATCH -o ./log/seq_hash_DB_atom_lmdb_%a.out
#SBATCH -e ./log/seq_hash_DB_atom_lmdb__%a.err
#SBATCH --array=0-14                  

srun python -u ./preprocessing/seq_hash_DB_multi_node.py
