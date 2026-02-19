#!/bin/bash
#SBATCH --job-name=ijepa_frac
#SBATCH --qos=acc_debug 
#SBATCH --account=etur91 
#SBATCH --time=01:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=ijepa_%j.out
#SBATCH --error=ijepa_%j.err
#SBATCH --chdir=.

set -e

echo "--- Starting Job: $SLURM_JOB_ID ---"
echo "Running on host: $(hostname)"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo ""

echo "--- Checking GPU with nvidia-smi ---"
nvidia-smi
echo ""

echo "--- GPU Details ---"
echo "GPU Name: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "GPU Memory Free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
echo ""



cd "$HOME/ijepa"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Bind scratch so the container can see your data
SCRATCH_DIR="/gpfs/scratch/etur91/"
REAL_DATA_PATH="/gpfs/projects/etur91/boga222803/datasets/imagenet/"
REAL_LOG_PATH="$SCRATCH_DIR/logs"

# Ensure log directory exists on host
mkdir -p "$REAL_LOG_PATH/ijepa/pretraining"

# Put binds in a variable
BIND_ARGS="$REAL_DATA_PATH:/mnt/data/imagenet,$REAL_LOG_PATH:/mnt/logs"

# Debug: Check host directory
echo "Debug: HOST REAL_LOG_PATH=$REAL_LOG_PATH" >&2
ls -ld "$REAL_LOG_PATH/ijepa/pretraining" >&2
chmod 755 "$REAL_LOG_PATH/ijepa/pretraining"

# Define args
CMD_ARGS=(
    --fname configs/train_frac.yaml
    --devices cuda:0
)

if [ -n "${EXTRA_ARGS:-}" ]; then
    read -ra EXTRA_ARRAY <<< "${EXTRA_ARGS}"
    CMD_ARGS+=("${EXTRA_ARRAY[@]}")
fi

echo "--- Executing main script with explicit bind ---" >&2
SIF_IMAGE="/gpfs/projects/etur91/boga222803/ijepa-env.sif"

module purge
module load singularity/4.1.5
which singularity

# 3. Execute with Singularity
singularity exec --nv \
    --bind "$BIND_ARGS" \
    "$SIF_IMAGE" \
    python main.py "${CMD_ARGS[@]}"

echo "--- Job Finished Successfully ---" >&2

