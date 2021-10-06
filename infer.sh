#!/bin/bash 
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 100G
#SBATCH --time 50:00:00
#SBATCH --gres gpu:volta:2

module load gcc/8.4.0-cuda  python/3.7.7
source ../../../../argo-env/bin/activate
python -V
python inference.py 