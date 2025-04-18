#!/bin/sh 
#SBATCH -J msa_proc
#SBATCH -p cpu-farm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=104
#SBATCH -o ./log/multistate_analysis.out
#SBATCH -e ./log/multistate_analysis.err

# python -u ./preprocessing/check_MSA.py
export PYTHONPATH="~/biomol:$PYTHONPATH"
python -u ./statistics/multi_state_protiens.py
