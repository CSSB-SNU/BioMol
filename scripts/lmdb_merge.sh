#!/bin/bash
#SBATCH -J cif_DB
#SBATCH --ntasks-per-node=1         
#SBATCH -c 100                        
#SBATCH --mem=480g
#SBATCH -p cpu-farm
#SBATCH -o ./log/lmdb_merge.out
#SBATCH -e ./log/lmdb_merge.err

# srun python -u ./BioMol/lmdb/cif_lmdb_merge.py /data/psk6950/BioMolDB_2024Oct21/cif_all_molecules.lmdb /data/psk6950/BioMolDB_2024Oct21/cif_all_molecules.lmdb/merged.lmdb --map-size 2000000000000
srun python -u ./BioMol/lmdb/lmdb_merge.py /data/psk6950/BioMolDB_2024Oct21/MSA.lmdb /data/psk6950/BioMolDB_2024Oct21/MSA.lmdb/merged.lmdb --map-size 2000000000000
