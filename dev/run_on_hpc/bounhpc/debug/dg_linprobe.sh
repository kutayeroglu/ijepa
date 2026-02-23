#!/bin/bash

#SBATCH --job-name=dgf_linprob 
#SBATCH --output=logs/dgf_linprob-%j.out
#SBATCH --error=logs/dgf_linprob-%j.err

#SBATCH --container-image ghcr.io\#kutayeroglu/ijepa
#SBATCH --container-mounts /stratch/dataset:/datasets
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=16G
#SBATCH --time=00:30:00

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
MODEL_PATH="/users/kutay.eroglu/logs/ijepa/pretraining/pretraining/uskumru-multnoise-latest.pth.tar"

# The code expects DATASET_DIR/in1k structure, so pass the parent directory
python3 "$HOME/projects/ijepa/main_linprobe.py" \
    --dataset_dir "$DATA_DIR" \
    --val_dir "$VAL_DATA_DIR" \
    --model_path "$MODEL_PATH" \
    --batch_size 16 \
    --learning_rate 4.88e-5 \
    --train_frac 0.1 \
    --val_frac 1.0 \
    --num_epochs 15 \
    ${EXTRA_ARGS}

echo "--- Job Finished Successfully ---"

