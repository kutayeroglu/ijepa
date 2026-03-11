#!/bin/bash
#SBATCH --job-name=dw_mnoise_2n
#SBATCH --qos=acc_debug
#SBATCH --account=etur91 
#SBATCH --time=01:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --output=aw_mnoise_2n_%j.out
#SBATCH --error=aw_mnoise_2n_%j.err
#SBATCH --chdir=.

set -e

# --- Environment Setup ---
cd "$HOME/ijepa"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRATCH_DIR="/gpfs/scratch/etur91"
REAL_DATA_PATH="/gpfs/projects/etur91/boga222803/datasets/imagenet"
REAL_LOG_PATH="$SCRATCH_DIR/logs"
LOCAL_DATA_DIR="$TMPDIR/imagenet"
GREEN_NOISE_HOST_PATH="/gpfs/projects/etur91/boga222803/datasets/green_noise_data_3072.npz"

mkdir -p "$REAL_LOG_PATH/green_mask/pretraining"

# --- Data Staging (runs once per node, both nodes in parallel) ---
echo "--- Staging and Extracting Data to Local SSD ($TMPDIR) on all nodes ---"

srun --ntasks-per-node=1 --ntasks="$SLURM_NNODES" bash -c '
    LOCAL_DATA_DIR="$TMPDIR/imagenet"
    mkdir -p "$LOCAL_DATA_DIR/train"

    echo "[$(hostname)] Extracting main train tar from GPFS..."
    tar -xf "'"$REAL_DATA_PATH"'/ILSVRC2012_img_train.tar" -C "$LOCAL_DATA_DIR/train"

    echo "[$(hostname)] Extracting class sub-tars in parallel..."
    cd "$LOCAL_DATA_DIR/train"
    find . -name "*.tar" -print0 | xargs -0 -P 8 -I {} sh -c '\''
        dir="${1%.tar}"
        mkdir -p "$dir"
        tar -xf "$1" -C "$dir"
        rm "$1"
    '\'' -- {}
    cd -

    echo "[$(hostname)] Data extraction complete"
'

echo "--- Data staging finished on all nodes ---"

# --- Container Execution ---
BIND_ARGS="$LOCAL_DATA_DIR:/mnt/data/imagenet,$REAL_LOG_PATH:/mnt/logs,$GREEN_NOISE_HOST_PATH:/mnt/green_noise_data_3072.npz"
SIF_IMAGE="/gpfs/projects/etur91/boga222803/ijepa-env.sif"

module purge
module load singularity/4.1.5

echo "--- Executing I-JEPA (noise mask, 2 nodes x 4 GPUs) ---"
srun --ntasks=8 --ntasks-per-node=4 \
    singularity exec --nv \
    --bind "$BIND_ARGS" \
    "$SIF_IMAGE" \
    python -u -c "
import yaml
from src.train import main as app_main
with open('configs/lufer_mnoise.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
app_main(args=params)
"

echo "--- Job Finished Successfully ---"
