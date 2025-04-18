#!/bin/sh 
#SBATCH -J cif_lmdb
#SBATCH -p cpu-farm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=100
#SBATCH -o ./log/cif_lmdb.out
#SBATCH -e ./log/cif_lmdb.err

# python -u ./preprocessing/check_MSA.py
export PYTHONPATH="~/biomol:$PYTHONPATH"
python -u ./preprocessing/cif_lmdb.py
