#!/bin/sh 
#SBATCH -J cif_lmdb
#SBATCH --mem=480g
#SBATCH -p gpu
#SBATCH -w gpu05
#SBATCH -c 80
#SBATCH -o ./log/cif_lmdb.out
#SBATCH -e ./log/cif_lmdb.err

# python -u ./preprocessing/check_MSA.py
python -u ./BioMol/lmdb/cif_lmdb.py
