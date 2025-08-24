#!/bin/bash

#SBATCH --job-name=lib_gpu_check      # Give the job a descriptive name
#SBATCH --output=lib_gpu_check-%j.out # Standard output log file (%j is the job ID)
#SBATCH --error=lib_gpu_check-%j.err  # Standard error log file

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

echo "--- Checking Core Libraries and GPU with Python/Torch ---"
python3 -c '
import sys
import torch

print(f"Python Version: {sys.version}")
print(f"Torch Version: {torch.__version__}")

print("-" * 20)

gpu_available = torch.cuda.is_available()
print(f"Is GPU available via Torch? -> {gpu_available}")

if gpu_available:
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU detected by PyTorch.")
'

echo "--- Job Finished Successfully ---"

