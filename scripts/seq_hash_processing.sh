#!/bin/bash
#SBATCH -J seq_hash
#SBATCH --ntasks-per-node=1         
#SBATCH -c 100                        
#SBATCH --mem=480g
#SBATCH -p cpu-farm
#SBATCH -o ./log/seq_cluster_.out
#SBATCH -e ./log/seq_cluster_.err

# srun python -u ./preprocessing/sequence_hash_all_molecules.py
srun python -u ./preprocessing/sequence_clustering_all_molecules.py
