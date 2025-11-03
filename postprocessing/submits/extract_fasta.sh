#!/bin/bash
#SBATCH -J extract_fasta
#SBATCH --ntasks-per-node=1         
#SBATCH -c 80                        
#SBATCH --mem=488g
#SBATCH -p cpu
#SBATCH -w node02
#SBATCH -o ./log/extract_fasta.out
#SBATCH -e ./log/extract_fasta.err

set -euo pipefail

# ========= 사용자 설정 =========
CIF_DB_PATH="/public_data/BioMolDBv2_2024Oct21/cif.lmdb"
OUT_PATH="/public_data/BioMolDBv2_2024Oct21/fasta/"
N_JOBS=80
# =================================

mkdir -p ./log

echo "CIF_DB_PATH=${CIF_DB_PATH}"
echo "OUT_PATH=${OUT_PATH}"

srun python -m postprocessing.run extract_fasta \
  "${CIF_DB_PATH}" \
  "${OUT_PATH}" \
  --njobs "${N_JOBS}" \
