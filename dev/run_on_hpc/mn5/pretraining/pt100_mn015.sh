#!/bin/bash
#SBATCH --job-name=cmr015
#SBATCH --qos=acc_ehpc
#SBATCH --account=etur91
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --output=cmr015_%j.out
#SBATCH --error=cmr015_%j.err
#SBATCH --chdir=.

set -e

cd "$HOME/ijepa"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRATCH_DIR="/gpfs/scratch/etur91"
REAL_DATA_PATH="/gpfs/projects/etur91/boga222803/datasets/imagenet"
REAL_LOG_PATH="$SCRATCH_DIR/logs"
LOCAL_DATA_DIR="$TMPDIR/imagenet"
GREEN_NOISE_HOST_PATH="/gpfs/projects/etur91/boga222803/datasets/green_noise_data_3072.npz"

mkdir -p "$REAL_LOG_PATH/ijepa/pretraining"
mkdir -p "$LOCAL_DATA_DIR/train"

echo "--- Staging and Extracting Data to Local SSD ($TMPDIR) ---"
tar -xf "$REAL_DATA_PATH/ILSVRC2012_img_train.tar" -C "$LOCAL_DATA_DIR/train"

cd "$LOCAL_DATA_DIR/train"
find . -name "*.tar" -print0 | xargs -0 -P 8 -I {} sh -c '
    dir="${1%.tar}"
    mkdir -p "$dir"
    tar -xf "$1" -C "$dir"
    rm "$1"
' -- {}
cd -

echo "--- Data extraction complete ---"

BIND_ARGS="$LOCAL_DATA_DIR:/mnt/data/imagenet,$REAL_LOG_PATH:/mnt/logs,$GREEN_NOISE_HOST_PATH:/mnt/green_noise_data_3072.npz"
SIF_IMAGE="/gpfs/projects/etur91/boga222803/ijepa-env.sif"

CMD_ARGS=(
    --fname configs/balon_mnoise_cmr015.yaml
    --devices cuda:0 cuda:1 cuda:2 cuda:3
)

module purge
module load singularity/4.1.5

echo "--- Executing I-JEPA (color_mask_ratio=0.15) ---"
singularity exec --nv \
    --bind "$BIND_ARGS" \
    "$SIF_IMAGE" \
    python main.py "${CMD_ARGS[@]}"

echo "--- Job Finished Successfully ---"
