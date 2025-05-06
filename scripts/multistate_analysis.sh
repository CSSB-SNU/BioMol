#!/bin/sh 
#SBATCH -J seq_hash_DB
#SBATCH --mem=480g
#SBATCH -p cpu
#SBATCH -c 80
#SBATCH -o ./log/multistate_analysis.out
#SBATCH -e ./log/multistate_analysis.err

# python -u ./preprocessing/check_MSA.py
python -u ./statistics/multi_state_protiens2.py
