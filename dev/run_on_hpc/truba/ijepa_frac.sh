#!/bin/bash
#SBATCH -p barbun-cuda           
#SBATCH -A keroglu               
#SBATCH -J ijepa_train_frac      
#SBATCH -o %j.out          
#SBATCH --gres=gpu:1
#SBATCH -N 1                     # Node
#SBATCH -n 1                     # Task
#SBATCH --cpus-per-task 20                    
#SBATCH --time=02:00:00          
#SBATCH --error=%j.err           

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
SCRATCH_DIR="/arf/scratch/keroglu"
REAL_DATA_PATH="/arf/repo/ImageNet"
REAL_LOG_PATH="$SCRATCH_DIR/logs"

# Ensure log directory exists on host
mkdir -p "$REAL_LOG_PATH/ijepa/pretraining"

# Put binds in a variable
BIND_ARGS="$REAL_DATA_PATH:/mnt/data/imagenet,$REAL_LOG_PATH:/mnt/logs"

# Debug: Check host directory
echo "Debug: HOST REAL_LOG_PATH=$REAL_LOG_PATH"
ls -ld "$REAL_LOG_PATH/ijepa/pretraining"

# Define args
CMD_ARGS=(
    --fname configs/train_frac.yaml
    --devices cuda:0
)

if [ -n "${EXTRA_ARGS:-}" ]; then
    read -ra EXTRA_ARRAY <<< "${EXTRA_ARGS}"
    CMD_ARGS+=("${EXTRA_ARRAY[@]}")
fi

echo "--- Executing main script with explicit bind ---"
# Execute script with explicit bind to ensure it works
SIF_IMAGE="$HOME/ijepa.sif"

# Debug: Check container view
apptainer exec --bind "$BIND_ARGS" "$SIF_IMAGE" ls -R /mnt/logs

# Run Python
apptainer exec --nv --bind "$BIND_ARGS" "$SIF_IMAGE" python3 main.py "${CMD_ARGS[@]}"
echo "--- Job Finished Successfully ---"