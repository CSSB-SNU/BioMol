#!/bin/bash
#SBATCH -J test
#SBATCH --ntasks-per-node=1         
#SBATCH -c 2                      
#SBATCH --mem=4g
#SBATCH -p cpu
#SBATCH -w node01
#SBATCH -o ./log/test.out
#SBATCH -e ./log/test.err

srun --cpu-bind=none python test.py