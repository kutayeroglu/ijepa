#!/bin/bash
#SBATCH --job-name=pt-mn50
#SBATCH --qos=acc_ehpc
#SBATCH --account=etur91
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --output=%j_pt_mn50_vitb.out
#SBATCH --error=%j_pt_mn50_vitb.err
#SBATCH --chdir=.

set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$HOME/ijepa"

# -- Script --
PROJECT_ROOT="$HOME/ijepa"
PROJECTS_BASE="/gpfs/projects/etur91/boga222803"
SCRIPT_PATH="$(realpath --relative-to="$PROJECT_ROOT" "$0")"
source "$PROJECT_ROOT/dev/run_on_hpc/mn5/common.sh"
export IJEPA_LAUNCHER_SCRIPT="$SCRIPT_PATH"

# -- Config --
CONFIG_NAME="bal_mn50_vitb" # TODO
CONFIG_PATH="configs/${CONFIG_NAME}.yaml"

# -- Logs -- 
SCRATCH_DIR="/gpfs/scratch/etur91"
REAL_LOG_PATH="$SCRATCH_DIR/logs"

# -- Data -- 
NOISE_FILENAME="green_noise_data_3072.npz"
NOISE_HOST_PATH="$PROJECTS_BASE/datasets/$NOISE_FILENAME"
REAL_DATA_PATH="$PROJECTS_BASE/datasets/imagenet"
LOCAL_DATA_DIR="$TMPDIR/imagenet"

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

# -- Container execution --
NOISE_CONTAINER_PATH="/mnt/$NOISE_FILENAME"
BIND_ARGS="$LOCAL_DATA_DIR:/mnt/data/imagenet,$REAL_LOG_PATH:/mnt/logs,$NOISE_HOST_PATH:$NOISE_CONTAINER_PATH"
SIF_IMAGE="$PROJECTS_BASE/ijepa-env.sif"

CMD_ARGS=(
    --fname "$CONFIG_PATH"
    --devices cuda:0 cuda:1 cuda:2 cuda:3
)

module purge
module load singularity/4.1.5

print_run_header
echo "--- Executing I-JEPA ---"
singularity exec --nv \
    --bind "$BIND_ARGS" \
    "$SIF_IMAGE" \
    python main.py "${CMD_ARGS[@]}"

echo "--- Job Finished Successfully ---"
