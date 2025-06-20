#!/bin/bash
#SBATCH -J cif_DB
#SBATCH --ntasks-per-node=1         
#SBATCH -c 100                        
#SBATCH --mem=480g
#SBATCH -p cpu-farm
#SBATCH -o ./log/cif_lmdb_%a.out
#SBATCH -e ./log/cif_lmdb__%a.err
#SBATCH --array=0-4                  

srun python -u ./BioMol/lmdb/cif_lmdb_multi_node.py
