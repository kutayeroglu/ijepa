#!/bin/bash
#
# Submit a pretraining job followed by an automatic linear probe job.
# The LP job waits in the Slurm queue and starts only after pretraining succeeds.
#
# Usage:
#   ./submit_pipeline.sh <pretraining_script> <linear_probe_script> [options]
#
# Options:
#   --config-tag TAG              Override the config write_tag (auto-extracted from YAML by default)
#   --checkpoint-suffix SUFFIX    Checkpoint to probe (default: latest)
#
# Examples:
#   ./dev/run_on_hpc/mn5/submit_pipeline.sh \
#       dev/run_on_hpc/mn5/pretraining/dg-pt100_mn50_vitb.sh \
#       dev/run_on_hpc/mn5/linear_probe/lp100_mn.sh
#
#   ./dev/run_on_hpc/mn5/submit_pipeline.sh \
#       dev/run_on_hpc/mn5/pretraining/pt100_mb_vitb.sh \
#       dev/run_on_hpc/mn5/linear_probe/lp100_mb.sh \
#       --config-tag bal_mb_vitb --checkpoint-suffix ep300

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }

# --- Parse arguments ---
PT_SCRIPT=""
LP_SCRIPT=""
CONFIG_TAG_OVERRIDE=""
CHECKPOINT_SUFFIX="latest"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-tag)
            CONFIG_TAG_OVERRIDE="$2"; shift 2 ;;
        --checkpoint-suffix)
            CHECKPOINT_SUFFIX="$2"; shift 2 ;;
        -h|--help)
            head -24 "$0" | tail -22; exit 0 ;;
        -*)
            die "Unknown option: $1" ;;
        *)
            if [[ -z "$PT_SCRIPT" ]]; then
                PT_SCRIPT="$1"
            elif [[ -z "$LP_SCRIPT" ]]; then
                LP_SCRIPT="$1"
            else
                die "Unexpected argument: $1"
            fi
            shift ;;
    esac
done

[[ -n "$PT_SCRIPT" ]] || die "Missing pretraining script (first positional argument)"
[[ -n "$LP_SCRIPT" ]] || die "Missing linear probe script (second positional argument)"
[[ -f "$PT_SCRIPT" ]] || die "Pretraining script not found: $PT_SCRIPT"
[[ -f "$LP_SCRIPT" ]] || die "Linear probe script not found: $LP_SCRIPT"

# --- Resolve config tag ---
if [[ -n "$CONFIG_TAG_OVERRIDE" ]]; then
    WRITE_TAG="$CONFIG_TAG_OVERRIDE"
else
    CONFIG_NAME=$(grep -oP 'CONFIG_NAME="\K[^"]+' "$PT_SCRIPT" | head -1) \
        || die "Could not extract CONFIG_NAME from $PT_SCRIPT"
    CONFIG_PATH_TEMPLATE=$(grep -oP 'CONFIG_PATH="\K[^"]+' "$PT_SCRIPT" | head -1) \
        || die "Could not extract CONFIG_PATH from $PT_SCRIPT"
    CONFIG_PATH_REL="${CONFIG_PATH_TEMPLATE//\$\{CONFIG_NAME\}/$CONFIG_NAME}"
    CONFIG_PATH="$PROJECT_ROOT/$CONFIG_PATH_REL"

    [[ -f "$CONFIG_PATH" ]] || die "Config YAML not found: $CONFIG_PATH"

    WRITE_TAG=$(grep -oP '^\s*write_tag:\s*\K\S+' "$CONFIG_PATH") \
        || die "Could not extract write_tag from $CONFIG_PATH"

    echo "Auto-detected write_tag: $WRITE_TAG  (from $CONFIG_PATH)"
fi

# --- Submit pretraining job ---
PT_JOB_ID=$(sbatch --parsable "$PT_SCRIPT") \
    || die "Failed to submit pretraining job"

PRETRAINING_RUN_ID="${PT_JOB_ID}_${WRITE_TAG}"

echo "--- Pipeline Summary ---"
echo "Pretraining job:  $PT_JOB_ID  ($PT_SCRIPT)"
echo "Run ID will be:   $PRETRAINING_RUN_ID"

# --- Submit linear probe job with dependency ---
LP_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:"$PT_JOB_ID" \
    --export=ALL,PRETRAINING_RUN_ID="$PRETRAINING_RUN_ID",CONFIG_TAG="$WRITE_TAG",CHECKPOINT_SUFFIX="$CHECKPOINT_SUFFIX" \
    "$LP_SCRIPT") \
    || die "Failed to submit linear probe job"

echo "Linear probe job: $LP_JOB_ID  ($LP_SCRIPT)  [depends on $PT_JOB_ID]"
echo "Checkpoint:       $WRITE_TAG-$CHECKPOINT_SUFFIX.pth.tar"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Cancel both:   scancel $PT_JOB_ID $LP_JOB_ID"
