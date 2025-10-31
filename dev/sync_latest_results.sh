#!/bin/bash

# Script to sync the latest training results from remote server
# Syncs training_loss_plot.png and training_log to local outputs directory
#
# Usage:
#   ./sync_latest_results.sh                    # Syncs the latest run
#   ./sync_latest_results.sh 2025-10-31_14-50-36 # Syncs a specific run by timestamp

set -e  # Exit on error

# Remote server details
REMOTE_USER="kutay.eroglu"
REMOTE_HOST="79.123.177.160"
REMOTE_BASE_PATH="/users/kutay.eroglu/outputs/ijepa"

# Local base path
LOCAL_BASE_PATH="/home/kergolu/outputs"

# Check if a specific folder argument was provided
if [ -n "$1" ]; then
    # Use the provided argument
    FOLDER_ARG="$1"
    
    # If the argument doesn't start with "linprobe_", prepend it
    if [[ "$FOLDER_ARG" != linprobe_* ]]; then
        LATEST_FOLDER="linprobe_${FOLDER_ARG}"
    else
        LATEST_FOLDER="$FOLDER_ARG"
    fi
    
    echo "Using specified folder: $LATEST_FOLDER"
    
    # Verify the folder exists on remote
    if ! ssh "${REMOTE_USER}@${REMOTE_HOST}" "test -d ${REMOTE_BASE_PATH}/${LATEST_FOLDER}" 2>/dev/null; then
        echo "Error: Folder '${LATEST_FOLDER}' not found on remote server."
        exit 1
    fi
else
    # Find the most recent folder matching linprobe_* pattern
    # Uses SSH to list directories, filters for linprobe_* pattern, sorts by modification time (most recent first)
    echo "Connecting to remote server to find latest training folder..."
    
    LATEST_FOLDER_PATH=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" \
        "ls -td ${REMOTE_BASE_PATH}/linprobe_* 2>/dev/null | head -n1")
    
    if [ -z "$LATEST_FOLDER_PATH" ]; then
        echo "Error: No folders matching pattern 'linprobe_*' found on remote server."
        exit 1
    fi
    
    # Extract folder name from full path
    LATEST_FOLDER=$(basename "$LATEST_FOLDER_PATH")
    
    echo "Found latest folder: $LATEST_FOLDER"
fi

# Create local directory if it doesn't exist
LOCAL_DIR="${LOCAL_BASE_PATH}/${LATEST_FOLDER}"
mkdir -p "$LOCAL_DIR"
echo "Created/verified local directory: $LOCAL_DIR"

# Remote paths
REMOTE_DIR="${REMOTE_BASE_PATH}/${LATEST_FOLDER}"
REMOTE_PLOT="${REMOTE_DIR}/training_loss_plot.png"
REMOTE_LOG="${REMOTE_DIR}/training_log"

# Sync training_loss_plot.png
echo "Syncing training_loss_plot.png..."
rsync -avhP "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PLOT}" "$LOCAL_DIR/" || {
    echo "Warning: Failed to sync training_loss_plot.png"
}

# Sync training_log
echo "Syncing training_log..."
rsync -avhP "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_LOG}" "$LOCAL_DIR/" || {
    echo "Warning: Failed to sync training_log"
}

echo "Sync complete! Files saved to: $LOCAL_DIR"

