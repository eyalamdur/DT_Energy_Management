#!/bin/bash
#SBATCH --job-name=models_training
#SBATCH --output=logs/models/train_RL_models.out
#SBATCH --error=logs/models/train_RL_models.err
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00          # hh:mm:ss, 48 hours

# --- Email notifications ---
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eyal.amdur@campus.technion.ac.il

# Activate your conda/env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anm

# Run your Python script
python ./code/models/train_models.py