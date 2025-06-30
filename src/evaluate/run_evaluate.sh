#!/bin/bash
#SBATCH --job-name=evaluate_dt
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00          # hh:mm:ss, 72 hours

# --- Email notifications ---
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eyal.amdur@campus.technion.ac.il

# Activate conda/env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anm

# Run the evaluation script
python ./src/evaluate/evaluate.py
