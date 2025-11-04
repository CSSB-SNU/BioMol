#!/bin/bash
#SBATCH -J graph_cluster
#SBATCH --ntasks-per-node=1         
#SBATCH -c 75                        
#SBATCH --mem=450g
#SBATCH -p cpu
#SBATCH -w node02
#SBATCH -o ./log/graph_cluster.out
#SBATCH -e ./log/graph_cluster.err

set -euo pipefail

# ========= 사용자 설정 =========
GRAPH_LMDB="/public_data/BioMolDBv2_2024Oct21/graph.lmdb"
GRAPH_CLUSTER="/public_data/BioMolDBv2_2024Oct21/cluster/graph_cluster/"
UNIQUE_GRAPH_LMDB="/public_data/BioMolDBv2_2024Oct21/unique_graph.lmdb"
# =================================

mkdir -p ./log

echo "GRAPH_CLUSTER=${GRAPH_CLUSTER}"
echo "GRAPH_LMDB=${GRAPH_LMDB}"
echo "UNIQUE_GRAPH_LMDB=${UNIQUE_GRAPH_LMDB}"


srun python -u -m postprocessing.run graph_cluster \
  "${GRAPH_LMDB}" \
  "${GRAPH_CLUSTER}" \
  "${UNIQUE_GRAPH_LMDB}"
