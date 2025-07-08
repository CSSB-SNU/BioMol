#!/bin/sh
#SBATCH -J MSADB
#SBATCH --ntasks-per-node=1         
#SBATCH -c 100                   
#SBATCH --mem=480g
#SBATCH -p cpu-farm
#SBATCH -o ./log/MSA_lmdb_%a.out
#SBATCH -e ./log/MSA_lmdb__%a.err
#SBATCH --array=0-9               

# python -u ./preprocessing/check_MSA.py
python -u ./BioMol/lmdb/MSA_lmdb_multi_node.py
