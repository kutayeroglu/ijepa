#!/bin/bash
#SBATCH --job-name=siumb
#SBATCH --qos=acc_ehpc
#SBATCH --account=etur91
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=%j_siu1prc_mb.out
#SBATCH --error=%j_siu1prc_mb.err
#SBATCH --chdir=.

set -e

# --- Environment Setup ---
cd "$HOME/ijepa"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRATCH_DIR="/gpfs/scratch/etur91"
REAL_DATA_PATH="/gpfs/projects/etur91/boga222803/datasets/imagenet"
REAL_LOG_PATH="$SCRATCH_DIR/logs"
LOCAL_DATA_DIR="$TMPDIR/imagenet"
SCRIPT_PATH="dev/run_on_hpc/mn5/linear_probe/lp1_mb.sh"
RUN_ID="${SLURM_JOB_ID:-manual}_lp1_mb"
SOURCE_CHECKPOINT_TAG="balon_mblock_vitb"
RUN_OUTPUT_DIR="$REAL_LOG_PATH/ijepa/linprobe/$SOURCE_CHECKPOINT_TAG/runs/$RUN_ID"
MODEL_PATH="/mnt/logs/ijepa/pretraining/balon_mblock_vitb/balon_mblock_vitb-latest.pth.tar"
export IJEPA_LAUNCHER_SCRIPT="$SCRIPT_PATH"

source "$HOME/ijepa/dev/run_on_hpc/mn5/common.sh"

mkdir -p "$REAL_LOG_PATH/ijepa/linprobe"
mkdir -p "$REAL_LOG_PATH/ijepa/linprobe/$SOURCE_CHECKPOINT_TAG"
mkdir -p "$RUN_OUTPUT_DIR"
mkdir -p "$LOCAL_DATA_DIR/train"
mkdir -p "$LOCAL_DATA_DIR/val"

# --- Data Staging (The bottleneck) ---
echo "--- Staging and Extracting Data to Local SSD ($TMPDIR) ---"

# Extract Main Train Tar
echo "Extracting main train tar from GPFS..."
tar -xf "$REAL_DATA_PATH/ILSVRC2012_img_train.tar" --strip-components=1 -C "$LOCAL_DATA_DIR/train"

# Extract class sub-tars in parallel
echo "Extracting class sub-tars in parallel..."
cd "$LOCAL_DATA_DIR/train"
find . -name "*.tar" -print0 | xargs -0 -P 20 -I {} sh -c '
    dir="${1%.tar}"
    mkdir -p "$dir"
    tar -xf "$1" -C "$dir"
    rm "$1"
' -- {}
cd -

# Extract Validation Tar
echo "Extracting validation tar from GPFS..."
tar -xf "$REAL_DATA_PATH/ILSVRC2012_img_val.tar" --strip-components=1 -C "$LOCAL_DATA_DIR/val"

echo "--- Data extraction complete ---"

# --- Sanity Check ---
echo "--- Sanity check: verifying dataset layout ---"
echo "Train top-level dirs (first 5):"
ls "$LOCAL_DATA_DIR/train" | head -5
echo "Train top-level dir count:"
ls -d "$LOCAL_DATA_DIR/train"/*/ | wc -l
echo "Val top-level dirs (first 5):"
ls "$LOCAL_DATA_DIR/val" | head -5
echo "Val top-level dir count:"
ls -d "$LOCAL_DATA_DIR/val"/*/ | wc -l

# --- Container Execution ---
BIND_ARGS="$LOCAL_DATA_DIR:/mnt/data/imagenet,$REAL_LOG_PATH:/mnt/logs"
SIF_IMAGE="/gpfs/projects/etur91/boga222803/ijepa-env.sif"

module purge
module load singularity/4.1.5

print_run_header
echo "--- Executing Linear Probe ---"
singularity exec --nv \
    --bind "$BIND_ARGS" \
    "$SIF_IMAGE" \
    python main_linprobe.py \
        --dataset_dir /mnt/data/imagenet \
        --val_dir /mnt/data/imagenet/val \
        --model_path "$MODEL_PATH" \
        --model_name vit_base \
        --patch_size 16 \
        --batch_size 1024 \
        --learning_rate 0.00625 \
        --weight_decay 0.0005 \
        --num_workers 10 \
        --train_frac 0.01 \
        --num_epochs 100 \
        --output_root /mnt/logs/ijepa/linprobe \
        --run_id "$RUN_ID" \
        ${EXTRA_ARGS}

echo "--- Job Finished Successfully ---"
