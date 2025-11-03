#!/bin/bash
#SBATCH -J compare_fasta
#SBATCH --ntasks-per-node=1         
#SBATCH -c 80                        
#SBATCH --mem=488g
#SBATCH -p cpu
#SBATCH -w node02
#SBATCH -o ./log/compare_fasta.out
#SBATCH -e ./log/compare_fasta.err

set -euo pipefail

# ========= 사용자 설정 =========
V1_PATH="/public_data/BioMolDB_2024Oct21/fasta/"
V2_PATH="/public_data/BioMolDBv2_2024Oct21/fasta/"
N_JOBS=80
# =================================

mkdir -p ./log

echo "V1_PATH=${V1_PATH}"
echo "V2_PATH=${V2_PATH}"

srun python -m postprocessing.compare compare_fasta \
  "${V1_PATH}" \
  "${V2_PATH}" \
  --njobs "${N_JOBS}" \
