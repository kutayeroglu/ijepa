#!/bin/bash

#SBATCH --job-name=eval_lin_probe 
#SBATCH --output=logs/eval_lin_probe-%j.out
#SBATCH --error=logs/eval_lin_probe-%j.err

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

echo "--- Executing main script ---"

# Set the base dataset directory (inside container, data is mounted at /datasets)
DATA_DIR="/datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
VAL_LABELS_FILE="/users/kutay.eroglu/artifacts/ILSVRC2012_validation_ground_truth.txt"
VAL_DATA_DIR="/users/kutay.eroglu/datasets/imagenet/val"
# The code expects DATASET_DIR/in1k structure, so pass the parent directory
python3 "$HOME/projects/ijepa/main_linprobe.py" \
    --dataset_dir "$DATA_DIR" \
    --val_dir "$VAL_DATA_DIR" \
    --val_labels_file "$VAL_LABELS_FILE" \
    ${EXTRA_ARGS}

echo "--- Job Finished Successfully ---"

