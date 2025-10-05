#!/bin/bash
#SBATCH -J biomol_old_np
#SBATCH --ntasks-per-node=1         
#SBATCH -c 100                        
#SBATCH --mem=480g
#SBATCH -p cpu-farm
#SBATCH -o ./log/biomol_old_np_%a.out
#SBATCH -e ./log/biomol_old_np_%a.err
#SBATCH --array=0-4                 

torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$RANK save_np.py
