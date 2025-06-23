#!/bin/sh 
#SBATCH -J graph_clustering
#SBATCH -p cpu-farm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=480g
#SBATCH -o ./log/train_valid_cluster.out
#SBATCH -e ./log/train_valid_cluster.err

python -u ./preprocessing/graph_cluster.py
