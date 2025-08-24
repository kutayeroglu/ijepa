#!/bin/bash

#SBATCH --job-name=eval_lin_probe 
#SBATCH --output=eval_lin_probe-%j.out
#SBATCH --error=eval_lin_probe-%j.err

#SBATCH --container-image ghcr.io\#kutayeroglu/ijepa
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Job: $SLURM_JOB_ID ---"
echo "Running on host: $(hostname)"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo ""

echo "--- Checking GPU with nvidia-smi ---"
nvidia-smi
echo ""

echo "--- Executing main script ---"
python3 "$HOME/projects/ijepa/main_linprobe.py"

echo "--- Job Finished Successfully ---"

