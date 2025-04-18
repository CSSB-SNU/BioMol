#!/bin/sh 
#SBATCH -J msa_proc
#SBATCH -p cpu-farm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=104
#SBATCH -o ./log/msa_signalp.out
#SBATCH -e ./log/msa_signalp.err

# python -u ./preprocessing/check_MSA.py
python -u ./preprocessing/MSA_remapping.py
