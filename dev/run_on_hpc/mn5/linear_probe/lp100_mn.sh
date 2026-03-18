#!/bin/bash
#SBATCH --job-name=pHmn50
#SBATCH --qos=acc_ehpc
#SBATCH --account=etur91
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=%j_lp100_mn50.out
#SBATCH --error=%j_lp100_mn50.err
#SBATCH --chdir=.

set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# -- Script --
PROJECT_ROOT="$HOME/ijepa"
cd "$PROJECT_ROOT"
PROJECTS_BASE="/gpfs/projects/etur91/boga222803"
SCRIPT_PATH="$(realpath --relative-to="$PROJECT_ROOT" "$0")"
source "$PROJECT_ROOT/dev/run_on_hpc/mn5/common.sh"
export IJEPA_LAUNCHER_SCRIPT="$SCRIPT_PATH"

# -- Config --
# TODO: Set CONFIG_TAG to match pretraining config write_tag (e.g. balon_mnoise_cmr060_vitb)
CONFIG_TAG="bal_mn50_vitb"
# TODO: Set PRETRAINING_RUN_ID to the pretraining run to probe, or export before sbatch
PRETRAINING_RUN_ID="${PRETRAINING_RUN_ID:-REPLACE_WITH_PRETRAINING_RUN_ID}"
RUN_ID="${SLURM_JOB_ID:-manual}_lp100_mn"

# -- Logs --
SCRATCH_DIR="/gpfs/scratch/etur91"
REAL_LOG_PATH="$SCRATCH_DIR/logs"

# -- Data --
REAL_DATA_PATH="$PROJECTS_BASE/datasets/imagenet"
LOCAL_DATA_DIR="$TMPDIR/imagenet"

MODEL_PATH="$REAL_LOG_PATH/ijepa/pretraining/$CONFIG_TAG/runs/$PRETRAINING_RUN_ID/$CONFIG_TAG-latest.pth.tar"
CONTAINER_MODEL_PATH="/mnt/logs/ijepa/pretraining/$CONFIG_TAG/runs/$PRETRAINING_RUN_ID/$CONFIG_TAG-latest.pth.tar"
PRETRAINING_RUN_DIR="$(dirname "$MODEL_PATH")"
OUTPUTS_DIR="$PRETRAINING_RUN_DIR/linprobe/$RUN_ID"
CONTAINER_OUTPUTS_DIR="/mnt/logs/ijepa/pretraining/$CONFIG_TAG/runs/$PRETRAINING_RUN_ID/linprobe/$RUN_ID"
RUN_OUTPUT_DIR="$OUTPUTS_DIR"

mkdir -p "$OUTPUTS_DIR"
mkdir -p "$LOCAL_DATA_DIR/train"
mkdir -p "$LOCAL_DATA_DIR/val"

echo "--- Staging and Extracting Data to Local SSD ($TMPDIR) ---"
tar -xf "$REAL_DATA_PATH/ILSVRC2012_img_train.tar" --strip-components=1 -C "$LOCAL_DATA_DIR/train"

cd "$LOCAL_DATA_DIR/train"
find . -name "*.tar" -print0 | xargs -0 -P 20 -I {} sh -c '
    dir="${1%.tar}"
    mkdir -p "$dir"
    tar -xf "$1" -C "$dir"
    rm "$1"
' -- {}
cd -

tar -xf "$REAL_DATA_PATH/ILSVRC2012_img_val.tar" --strip-components=1 -C "$LOCAL_DATA_DIR/val"

echo "--- Data extraction complete ---"

# -- Container execution --
BIND_ARGS="$LOCAL_DATA_DIR:/mnt/data/imagenet,$REAL_LOG_PATH:/mnt/logs"
SIF_IMAGE="$PROJECTS_BASE/ijepa-env.sif"

CMD_ARGS=(
    --dataset_dir /mnt/data/imagenet
    --val_dir /mnt/data/imagenet/val
    --model_path "$CONTAINER_MODEL_PATH"
    --model_name vit_base
    --patch_size 16
    --batch_size 1024
    --learning_rate 0.00625
    --weight_decay 0.0005
    --num_workers 10
    --num_epochs 30
    --outputs_dir "$CONTAINER_OUTPUTS_DIR"
    --run_id "$RUN_ID"
    ${EXTRA_ARGS}
)

module purge
module load singularity/4.1.5

print_run_header
echo "--- Executing Linear Probe ---"
singularity exec --nv \
    --bind "$BIND_ARGS" \
    "$SIF_IMAGE" \
    python main_linprobe.py "${CMD_ARGS[@]}"

echo "--- Job Finished Successfully ---"
