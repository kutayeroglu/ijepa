#!/bin/bash

#SBATCH --job-name=lin_prob_ijepa 
#SBATCH --output=logs/lin_prob_ijepa-%j.out
#SBATCH --error=logs/lin_prob_ijepa-%j.err

#SBATCH --container-image ghcr.io\#kutayeroglu/ijepa
#SBATCH --container-mounts /stratch/dataset:/datasets
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=5-05:00:00

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Job: $SLURM_JOB_ID ---"
echo "Running on host: $(hostname)"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo ""

echo "--- Executing main script ---"

# Set the base dataset directory (inside container, data is mounted at /datasets)
DATA_DIR="/datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
VAL_DATA_DIR="/users/kutay.eroglu/datasets/imagenet/val"
MODEL_PATH="/users/kutay.eroglu/logs/ijepa/pretraining/jepa-ep300.pth.tar"
# The code expects DATASET_DIR/in1k structure, so pass the parent directory
python3 "$HOME/projects/ijepa/main_linprobe.py" \
    --dataset_dir "$DATA_DIR" \
    --val_dir "$VAL_DATA_DIR" \
    --model_path "$MODEL_PATH" \
    --batch_size 128 \
    --learning_rate 0.001 \
    ${EXTRA_ARGS}

echo "--- Job Finished Successfully ---"

