#!/bin/sh 
#SBATCH -J seq_hash_DB
#SBATCH --mem=500g
#SBATCH -w gpu05
#SBATCH -c 60
#SBATCH -o ./log/seq_hash_DB_lmdb.out
#SBATCH -e ./log/seq_hash_DB_lmdb.err

# python -u ./preprocessing/check_MSA.py
python -u ./preprocessing/seq_hash_DB.py
