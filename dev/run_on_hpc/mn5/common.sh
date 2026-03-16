#!/bin/bash

print_run_header() {
    echo "--- Run Metadata ---"
    echo "Timestamp: $(date --iso-8601=seconds)"
    echo "Hostname: $(hostname)"
    echo "SLURM Job ID: ${SLURM_JOB_ID:-N/A}"
    echo "SLURM Job Name: ${SLURM_JOB_NAME:-N/A}"

    if [ -n "${CONFIG_PATH:-}" ]; then
        echo "Config Path: $CONFIG_PATH"
    fi
    if [ -n "${RUN_TAG:-}" ]; then
        echo "Run Tag: $RUN_TAG"
    fi
    if [ -n "${MODEL_PATH:-}" ]; then
        echo "Model Path: $MODEL_PATH"
    fi
    if [ -n "${RUN_ID:-}" ]; then
        echo "Run ID: $RUN_ID"
    fi
    if [ -n "${RUN_OUTPUT_DIR:-}" ]; then
        echo "Output Dir: $RUN_OUTPUT_DIR"
    fi
    if [ -n "${LOCAL_DATA_DIR:-}" ]; then
        echo "Local Data Dir: $LOCAL_DATA_DIR"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU Name(s):"
        nvidia-smi --query-gpu=name --format=csv,noheader || true
    fi

    echo "--------------------"
}
