#!/bin/sh 
#SBATCH -J MSA_lmdb
#SBATCH --mem=500g
#SBATCH -p gpu
#SBATCH -w gpu05
#SBATCH -c 60
#SBATCH -o ./log/MSA_lmdb.out
#SBATCH -e ./log/MSA_lmdb.err

# python -u ./preprocessing/check_MSA.py
python -u ./BioMol/lmdb/MSA_lmdb.py
