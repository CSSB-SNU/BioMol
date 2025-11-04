#!/bin/bash
#SBATCH -J graph_lmdb
#SBATCH --ntasks-per-node=1         
#SBATCH -c 70                        
#SBATCH --mem=450g
#SBATCH -p cpu
#SBATCH -w node02
#SBATCH -o ./log/graph_lmdb.out
#SBATCH -e ./log/graph_lmdb.err

set -euo pipefail

# ========= 사용자 설정 =========
CIF_DB_PATH="/public_data/BioMolDBv2_2024Oct21/cif.lmdb/"
SEQ_HASH_MAP="/public_data/BioMolDBv2_2024Oct21/metadata/seq_hash_map.tsv"
SEQ_CLUSTER="/public_data/BioMolDBv2_2024Oct21/seq_cluster/seq_clusters.tsv"
GRAPH_LMDB="/public_data/BioMolDBv2_2024Oct21/graph.lmdb"
# =================================

mkdir -p ./log

echo "CIF_DB_PATH=${CIF_DB_PATH}"
echo "SEQ_HASH_MAP=${SEQ_HASH_MAP}"
echo "SEQ_CLUSTER=${SEQ_CLUSTER}"
echo "GRAPH_LMDB=${GRAPH_LMDB}"


srun python -u -m postprocessing.run graph_lmdb \
  "${CIF_DB_PATH}" \
  "${SEQ_HASH_MAP}" \
  "${SEQ_CLUSTER}" \
  "${GRAPH_LMDB}"
