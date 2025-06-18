#!/bin/sh
#SBATCH -J MSADB
#SBATCH --ntasks-per-node=1         
#SBATCH -c 100                   
#SBATCH --mem=480g
#SBATCH -p cpu-farm
#SBATCH -o ./log/MSA_lmdb.out
#SBATCH -e ./log/MSA_lmdb.err

# python -u ./preprocessing/check_MSA.py
python -u ./BioMol/lmdb/MSA_lmdb.py
