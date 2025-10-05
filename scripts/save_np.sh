#!/bin/bash
#SBATCH -J biomol_old
#SBATCH --ntasks-per-node=1         
#SBATCH -c 80                        
#SBATCH --mem=480g
#SBATCH -p cpu
#SBATCH -w node02
#SBATCH -o ./log/biomol_old_np_%a.out
#SBATCH -e ./log/biomol_old_np_%a.err
#SBATCH --array=0                

srun python -u save_np.py
