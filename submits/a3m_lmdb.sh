#!/bin/bash
#SBATCH -J a3m_DB
#SBATCH --ntasks-per-node=1         
#SBATCH -c 20                        
#SBATCH --mem=200g
#SBATCH -p gpu
#SBATCH -w gpu04
#SBATCH -o ./log/a3m_lmdb_%a.out
#SBATCH -e ./log/a3m_lmdb_%a.err
#SBATCH --array=0

set -euo pipefail

# ========= 사용자 설정 =========
A3M_DIR="/public_data/BioMolDBv2_2024Oct21/a3m/"
ENV_PATH="/public_data/BioMolDBv2_2024Oct21/a3m.lmdb"
PARSER="biomol.io.parsers.a3m_parser:parse"
RECIPE_PATH="./plans/a3m_recipe_book.py"
MAP_SIZE="1e12" # 1TB
N_SHARDS=1
# =================================

mkdir -p ./log

echo "[$(date)] Starting shard $SLURM_ARRAY_TASK_ID / $N_SHARDS on $(hostname)"
echo "A3M_DIR=${A3M_DIR}"
echo "ENV_PATH=${ENV_PATH}"
echo "PARSER=${PARSER}"
echo "RECIPE_PATH=${RECIPE_PATH}"

srun python -u scripts/a3m_lmdb.py build \
  "${A3M_DIR}" \
  "${ENV_PATH}" \
  "${PARSER}" \
  "${RECIPE_PATH}" \
  --map-size "${MAP_SIZE}" \
  --shard-idx "${SLURM_ARRAY_TASK_ID}" \
  --n-shards "${N_SHARDS}" \
