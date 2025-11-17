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

echo "--- GPU Details ---"
echo "GPU Name: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "GPU Memory Free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
echo ""

cd "$HOME/projects/ijepa"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU
# Help reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "--- Executing main script ---"

# Build command with optional extra args
CMD_ARGS=(
    --fname configs/HPC_in1k_vith14_ep300.yaml
    --devices cuda:0
)

# Add EXTRA_ARGS if provided (split by spaces to handle multiple arguments)
if [ -n "${EXTRA_ARGS:-}" ]; then
    # Split EXTRA_ARGS by spaces and add to array
    read -ra EXTRA_ARRAY <<< "${EXTRA_ARGS}"
    CMD_ARGS+=("${EXTRA_ARRAY[@]}")
fi

python3 main.py "${CMD_ARGS[@]}"

echo "--- Job Finished Successfully ---"

