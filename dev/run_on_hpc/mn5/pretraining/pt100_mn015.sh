#!/bin/bash
#SBATCH --job-name=cmr015
#SBATCH --qos=acc_ehpc
#SBATCH --account=etur91
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --output=%j_cmr015.out
#SBATCH --error=%j_cmr015.err
#SBATCH --chdir=.

set -e

cd "$HOME/ijepa"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRATCH_DIR="/gpfs/scratch/etur91"
REAL_DATA_PATH="/gpfs/projects/etur91/boga222803/datasets/imagenet"
REAL_LOG_PATH="$SCRATCH_DIR/logs"
LOCAL_DATA_DIR="$TMPDIR/imagenet"
GREEN_NOISE_HOST_PATH="/gpfs/projects/etur91/boga222803/datasets/green_noise_data_3072.npz"
SCRIPT_PATH="dev/run_on_hpc/mn5/pretraining/pt100_mn015.sh"
CONFIG_PATH="configs/balon_mnoise_cmr015.yaml"
RUN_TAG="balon_mnoise_cmr015_vitb"
RUN_ID="${SLURM_JOB_ID:-manual}_${RUN_TAG}"
RUN_OUTPUT_DIR="$REAL_LOG_PATH/ijepa/pretraining/$RUN_TAG/runs/$RUN_ID"
export IJEPA_LAUNCHER_SCRIPT="$SCRIPT_PATH"
export IJEPA_RUN_ID="$RUN_ID"

source "$HOME/ijepa/dev/run_on_hpc/mn5/common.sh"

mkdir -p "$REAL_LOG_PATH/ijepa/pretraining"
mkdir -p "$REAL_LOG_PATH/ijepa/pretraining/$RUN_TAG"
mkdir -p "$RUN_OUTPUT_DIR"
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
    --fname "$CONFIG_PATH"
    --devices cuda:0 cuda:1 cuda:2 cuda:3
)

module purge
module load singularity/4.1.5

print_run_header
echo "--- Executing I-JEPA (color_mask_ratio=0.15) ---"
singularity exec --nv \
    --bind "$BIND_ARGS" \
    "$SIF_IMAGE" \
    python main.py "${CMD_ARGS[@]}"

echo "--- Job Finished Successfully ---"
