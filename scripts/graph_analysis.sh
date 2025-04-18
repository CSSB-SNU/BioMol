#!/bin/sh 
#SBATCH -J graph_analysis
#SBATCH -p cpu-farm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=104
#SBATCH -o ./log/graph_analysis2.out
#SBATCH -e ./log/graph_analysis2.err

python -u ./statistics/graph_analysis.py
