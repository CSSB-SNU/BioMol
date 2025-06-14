#!/bin/sh 
#SBATCH -J graph_clustering
#SBATCH -p cpu-farm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=104
#SBATCH -o ./log/graph_hash.out
#SBATCH -e ./log/graph_hash.err

# python -u ./preprocessing/graph_cluster.py
python -u ./preprocessing/graph_hash_v3.py
