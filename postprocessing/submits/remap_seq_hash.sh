#!/bin/bash
#SBATCH -J remap_seq_hash
#SBATCH --ntasks-per-node=1         
#SBATCH -c 90                        
#SBATCH --mem=160g
#SBATCH -p cpu
#SBATCH -w node01
#SBATCH -o ./log/remap_seq_hash.out
#SBATCH -e ./log/remap_seq_hash.err

set -euo pipefail

# ========= 사용자 설정 =========
V1_PATH="/public_data/BioMolDB_2024Oct21/metadata/metadata_psk_new.csv"
V2_PATH="/public_data/BioMolDBv2_2024Oct21/metadata/seq_hash_map.tsv"
N_JOBS=80
# =================================

mkdir -p ./log

echo "V1_PATH=${V1_PATH}"
echo "V2_PATH=${V2_PATH}"

srun python -m postprocessing.compare remap_seq_hash \
  "${V1_PATH}" \
  "${V2_PATH}" \
  --njobs "${N_JOBS}" \
