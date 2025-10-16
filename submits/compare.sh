#!/bin/bash
#SBATCH -J compare_cif_DB
#SBATCH --ntasks-per-node=1         
#SBATCH -c 60                        
#SBATCH --mem=360g
#SBATCH -p cpu
#SBATCH -w node02
#SBATCH -o ./log/compare_cif_lmdb.out
#SBATCH -e ./log/compare_cif_lmdb.err

srun python -u scripts/compare_to_old.py