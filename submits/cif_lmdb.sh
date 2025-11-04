#!/bin/bash
#SBATCH -J cif_DB
#SBATCH --ntasks-per-node=1         
#SBATCH -c 40                        
#SBATCH --mem=800g
#SBATCH -p gpu
#SBATCH -w gpu05
#SBATCH -o ./log/cif_lmdb_%a.out
#SBATCH -e ./log/cif_lmdb_%a.err
#SBATCH --array=0

set -euo pipefail

# ========= 사용자 설정 =========
CIF_DIR="/public_data/BioMolDB_2024Oct21/cif/cif_raw/"
CCD_DB_PATH="/public_data/CCD/biomol_CCD.lmdb"
ENV_PATH="/public_data/BioMolDBv2_2024Oct21/cif.lmdb"
PARSER="biomol.io.parsers.cif_parser:parse"
RECIPE_PATH="./plans/cif_recipe_book.py"
MAP_SIZE="1e12"
N_SHARDS=1
# =================================

mkdir -p ./log

echo "[$(date)] Starting shard $SLURM_ARRAY_TASK_ID / $N_SHARDS on $(hostname)"
echo "CIF_DIR=${CIF_DIR}"
echo "ENV_PATH=${ENV_PATH}"
echo "PARSER=${PARSER}"
echo "RECIPE_PATH=${RECIPE_PATH}"

python -u scripts/cif_lmdb.py build \
  "${CIF_DIR}" \
  "${ENV_PATH}" \
  "${PARSER}" \
  "${RECIPE_PATH}" \
  --map-size "${MAP_SIZE}" \
  --shard-idx "${SLURM_ARRAY_TASK_ID}" \
  --n-shards "${N_SHARDS}" \
  --ccd-db-path "${CCD_DB_PATH}"
