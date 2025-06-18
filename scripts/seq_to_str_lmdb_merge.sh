#!/bin/bash
#SBATCH -J cif_DB
#SBATCH --ntasks-per-node=1         
#SBATCH -c 16                    
#SBATCH --mem=480g
#SBATCH -p cpu-farm
#SBATCH -o ./log/seq_to_str_lmdb_merge.out
#SBATCH -e ./log/seq_to_str_lmdb_merge.err

srun python -u preprocessing/seq_hash_DB_merge.py /data/psk6950/BioMolDB_2024Oct21/seq_to_str/ /data/psk6950/BioMolDB_2024Oct21/seq_to_str/atom.lmdb --map-size 2000000000000
