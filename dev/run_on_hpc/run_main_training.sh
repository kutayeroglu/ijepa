#!/bin/bash

#SBATCH --job-name=ijepa_main_training 
#SBATCH --output=logs/ijepa_main_training-%j.out
#SBATCH --error=logs/ijepa_main_training-%j.err

#SBATCH --container-image ghcr.io\#kutayeroglu/ijepa
#SBATCH --container-mounts /stratch/dataset:/datasets
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=05:00:00

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Job: $SLURM_JOB_ID ---"
echo "Running on host: $(hostname)"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo ""

echo "--- Checking GPU with nvidia-smi ---"
nvidia-smi
echo ""

cd "$HOME/projects/ijepa"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU

echo "--- Executing main script ---"

python3 main.py \
    --fname configs/HPC_in1k_vith14_ep300.yaml \
    --devices cuda:0 \
    ${EXTRA_ARGS}

echo "--- Job Finished Successfully ---"

