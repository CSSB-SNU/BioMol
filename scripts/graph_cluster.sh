#!/bin/sh 
#SBATCH -J graph_clustering
#SBATCH -p cpu-farm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=104
#SBATCH -o ./log/graph_clustering.out
#SBATCH -e ./log/graph_clustering.err

python -u ./preprocessing/graph_cluster.py
