#!/bin/bash
#SBATCH --job-name=aw_linprobe
#SBATCH --qos=acc_debug
#SBATCH --account=etur91
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=aw_linprobe_%j.out
#SBATCH --error=aw_linprobe_%j.err
#SBATCH --chdir=.

set -e

# --- Environment Setup ---
cd "$HOME/ijepa"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRATCH_DIR="/gpfs/scratch/etur91"
REAL_DATA_PATH="/gpfs/projects/etur91/boga222803/datasets/imagenet"
REAL_LOG_PATH="$SCRATCH_DIR/logs"
LOCAL_DATA_DIR="$TMPDIR/imagenet"

mkdir -p "$REAL_LOG_PATH/ijepa/linprobe"
mkdir -p "$LOCAL_DATA_DIR/train"
mkdir -p "$LOCAL_DATA_DIR/val"

# --- Data Staging (The bottleneck) ---
echo "--- Staging and Extracting Data to Local SSD ($TMPDIR) ---"

# Extract Main Train Tar
echo "Extracting main train tar from GPFS..."
tar -xf "$REAL_DATA_PATH/ILSVRC2012_img_train.tar" -C "$LOCAL_DATA_DIR/train"

# Extract class sub-tars in parallel
echo "Extracting class sub-tars in parallel..."
cd "$LOCAL_DATA_DIR/train"
find . -name "*.tar" -print0 | xargs -0 -P 8 -I {} sh -c '
    dir="${1%.tar}"
    mkdir -p "$dir"
    tar -xf "$1" -C "$dir"
    rm "$1"
' -- {}
cd -

# Extract Validation Tar
echo "Extracting validation tar from GPFS..."
tar -xf "$REAL_DATA_PATH/ILSVRC2012_img_val.tar" -C "$LOCAL_DATA_DIR/val"

echo "--- Data extraction complete ---"

# --- Container Execution ---
BIND_ARGS="$LOCAL_DATA_DIR:/mnt/data/imagenet,$REAL_LOG_PATH:/mnt/logs"
SIF_IMAGE="/gpfs/projects/etur91/boga222803/ijepa-env.sif"

# TODO: replace with the actual checkpoint path under /gpfs/scratch/etur91/logs/
MODEL_PATH="/mnt/logs/ijepa/pretraining/jepa-latest.pth.tar"

module purge
module load singularity/4.1.5

echo "--- Executing Linear Probe ---"   
singularity exec --nv \
    --bind "$BIND_ARGS" \
    "$SIF_IMAGE" \
    python main_linprobe.py \
        --dataset_dir /mnt/data/imagenet \
        --val_dir /mnt/data/imagenet/val \
        --model_path "$MODEL_PATH" \
        --batch_size 128 \
        --learning_rate 0.001 \
        ${EXTRA_ARGS}

echo "--- Job Finished Successfully ---"
