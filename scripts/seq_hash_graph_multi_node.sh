#!/bin/bash
#SBATCH -J seq_hash_DB
#SBATCH --ntasks-per-node=1         
#SBATCH -c 40                        
#SBATCH --mem=480g
#SBATCH -p cpu-farm
#SBATCH -o ./log/seq_hash_graph_.out
#SBATCH -e ./log/seq_hash_graph__.err

srun python -u ./preprocessing/seq_hash_graph.py
