#!/bin/bash
#SBATCH --job-name=balnoise
#SBATCH --qos=acc_debug
#SBATCH --account=etur91 
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --output=balon_noise_%j.out
#SBATCH --error=balon_noise_%j.err
#SBATCH --chdir=.

set -e

# --- Environment Setup ---
cd "$HOME/ijepa"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Remove trailing slashes for consistency
SCRATCH_DIR="/gpfs/scratch/etur91"
REAL_DATA_PATH="/gpfs/projects/etur91/boga222803/datasets/imagenet"
REAL_LOG_PATH="$SCRATCH_DIR/logs"
LOCAL_DATA_DIR="$TMPDIR/imagenet"
GREEN_NOISE_HOST_PATH="/gpfs/projects/etur91/boga222803/datasets/green_noise_data_3072.npz"

mkdir -p "$REAL_LOG_PATH/green_mask/pretraining"
mkdir -p "$LOCAL_DATA_DIR/train"

# --- Data Staging (The bottleneck) ---
echo "--- Staging and Extracting Data to Local SSD ($TMPDIR) ---"

# Extract Main Train Tar
echo "Extracting main train tar from GPFS..."
tar -xf "$REAL_DATA_PATH/ILSVRC2012_img_train.tar" -C "$LOCAL_DATA_DIR/train"

# Extract sub-tars (This takes a long time!)
echo "Extracting class sub‑tars in parallel..."
cd "$LOCAL_DATA_DIR/train"
# find . -name "*.tar" | xargs -I {} sh -c "mkdir -p \${1%.tar} && tar -xf \$1 -C \${1%.tar} && rm \$1" -- {}
find . -name "*.tar" -print0 | xargs -0 -P 8 -I {} sh -c '
    dir="${1%.tar}"
    mkdir -p "$dir"
    tar -xf "$1" -C "$dir"
    rm "$1"
' -- {}
cd -

echo "--- Data extraction complete ---"

# --- Container Execution ---
BIND_ARGS="$LOCAL_DATA_DIR:/mnt/data/imagenet,$REAL_LOG_PATH:/mnt/logs,$GREEN_NOISE_HOST_PATH:/mnt/green_noise_data_3072.npz"
SIF_IMAGE="/gpfs/projects/etur91/boga222803/ijepa-env.sif"

CMD_ARGS=(
    --fname configs/balon_mnoise.yaml
    --devices cuda:0 cuda:1 cuda:2 cuda:3
)

module purge
module load singularity/4.1.5

echo "--- Executing I-JEPA (noise mask) ---"
singularity exec --nv \
    --bind "$BIND_ARGS" \
    "$SIF_IMAGE" \
    python main.py "${CMD_ARGS[@]}"

echo "--- Job Finished Successfully ---"
