#!/bin/bash

#SBATCH -J danQ_preprocessing

#**********************
# SLURM JOB INFORMATION
#**********************
#SBATCH --time=72:00:00
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100G
#SBATCH -o preprocessing_%J.log

module load miniconda3/23.11.0s

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh 

conda activate danq_env

python3 ./preprocessing/preprocessing.py
