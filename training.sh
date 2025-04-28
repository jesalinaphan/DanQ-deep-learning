#!/bin/bash

#SBATCH -J danq_training
#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o training_%J.log
#SBATCH --gres=gpu:1

# SLURM JOB INFORMATION
#***********************
#***********************

module load miniconda3/23.11.0s

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate danq_env

# run the training script
python3 ./main.py
