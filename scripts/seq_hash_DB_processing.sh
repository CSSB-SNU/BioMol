#!/bin/sh 
#SBATCH -J seq_hash_DB
#SBATCH --mem=480g
#SBATCH -p cpu
#SBATCH -c 80
#SBATCH -o ./log/seq_hash_DB_lmdb_atom.out
#SBATCH -e ./log/seq_hash_DB_lmdb_atom.err

# python -u ./preprocessing/check_MSA.py
python -u ./preprocessing/seq_hash_DB.py
