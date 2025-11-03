#!/bin/bash
#SBATCH -J graph_cluster
#SBATCH --ntasks-per-node=1         
#SBATCH -c 40                        
#SBATCH --mem=600g
#SBATCH -p gpu
#SBATCH -w gpu05
#SBATCH -o ./log/graph_cluster.out
#SBATCH -e ./log/graph_cluster.err

set -euo pipefail

# ========= 사용자 설정 =========
CIF_DB_PATH="/public_data/BioMolDBv2_2024Oct21/cif.lmdb/"
SEQ_HASH_MAP="/public_data/BioMolDBv2_2024Oct21/metadata/seq_hash_map.tsv"
SEQ_CLUSTER="/public_data/BioMolDBv2_2024Oct21/seq_cluster/seq_clusters.tsv"
GRAPH_CLUSTER="/public_data/BioMolDBv2_2024Oct21/cluster/graph_cluster/"
# =================================

mkdir -p ./log

echo "CIF_DB_PATH=${CIF_DB_PATH}"
echo "SEQ_HASH_MAP=${SEQ_HASH_MAP}"
echo "SEQ_CLUSTER=${SEQ_CLUSTER}"
echo "GRAPH_CLUSTER=${GRAPH_CLUSTER}"


srun python -u -m postprocessing.run graph_cluster \
  "${CIF_DB_PATH}" \
  "${SEQ_HASH_MAP}" \
  "${SEQ_CLUSTER}" \
  "${GRAPH_CLUSTER}" \
  --n_digits 6
