#!/bin/bash
#SBATCH -J ccd_DB
#SBATCH --ntasks-per-node=1         
#SBATCH -c 40                       
#SBATCH --mem=60g
#SBATCH -p cpu
#SBATCH -w node01
#SBATCH -o ./log/ccd_lmdb_%a.out
#SBATCH -e ./log/ccd_lmdb_%a.err
#SBATCH --array=0

set -euo pipefail

# ========= 사용자 설정 =========
CIF_DIR="/public_data/CCD/components_tmp"
ENV_PATH="/public_data/CCD/biomol_CCD.lmdb"
PARSER="biomol.io.parsers.cif_parser:parse"
RECIPE_PATH="./plans/ccd_recipe_book.py"
MAP_SIZE="1e10"
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
