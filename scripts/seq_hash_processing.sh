#!/bin/bash
#SBATCH -J seq_hash
#SBATCH --ntasks-per-node=1         
#SBATCH -c 100                        
#SBATCH --mem=480g
#SBATCH -p cpu-farm
#SBATCH -o ./log/seq_hash_.out
#SBATCH -e ./log/seq_hash__.err

srun python -u ./preprocessing/sequence_hash_all_molecules.py
