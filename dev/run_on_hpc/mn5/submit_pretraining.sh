#!/bin/bash
#
# Submit a pre-training Slurm job with optional environment overrides.
#
# Usage:
#   ./submit_pretraining.sh <pretraining_script> [KEY=VALUE ...]
#
# Environment overrides are passed to the job via sbatch --export=ALL,KEY=VALUE,...
# Values must not contain commas (Slurm --export uses comma as separator).
#
# Examples:
#   ./dev/run_on_hpc/mn5/submit_pretraining.sh \
#       dev/run_on_hpc/mn5/pretraining/pt100_mn50_vitb.sh
#
#   ./dev/run_on_hpc/mn5/submit_pretraining.sh \
#       dev/run_on_hpc/mn5/pretraining/pt100_mn50_vitb.sh \
#       PYTHONUNBUFFERED=1

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

show_help() {
    cat <<'EOF'
Submit a pre-training Slurm job with optional environment overrides.

Usage:
  ./submit_pretraining.sh <pretraining_script> [KEY=VALUE ...]

Environment overrides are passed to the job via sbatch --export=ALL,KEY=VALUE,...
Values must not contain commas (Slurm --export uses comma as separator).

Examples:
  ./dev/run_on_hpc/mn5/submit_pretraining.sh \
      dev/run_on_hpc/mn5/pretraining/pt100_mn50_vitb.sh

  ./dev/run_on_hpc/mn5/submit_pretraining.sh \
      dev/run_on_hpc/mn5/pretraining/pt100_mn50_vitb.sh \
      PYTHONUNBUFFERED=1
EOF
}

validate_override() {
    local pair="$1"
    local key="${pair%%=*}"
    local value="${pair#*=}"

    [[ "$pair" == *"="* ]] || die "Override must be KEY=VALUE: $pair"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "Invalid env key in: $pair"
    [[ "$value" != *","* ]] || die "VALUE must not contain commas (Slurm limitation): $pair"
}

# --- Parse arguments ---
PT_SCRIPT=""
declare -a OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            die "Unknown option: $1"
            ;;
        *=*)
            [[ -n "$PT_SCRIPT" ]] || die "Pre-training script must be the first argument (before KEY=VALUE overrides)"
            validate_override "$1"
            OVERRIDES+=("$1")
            ;;
        *)
            [[ -z "$PT_SCRIPT" ]] || die "Unexpected argument: $1 (overrides must look like KEY=VALUE)"
            PT_SCRIPT="$1"
            ;;
    esac
    shift
done

[[ -n "$PT_SCRIPT" ]] || die "Missing pre-training script (first argument)"
[[ -f "$PT_SCRIPT" ]] || die "Pre-training script not found: $PT_SCRIPT"
[[ -r "$PT_SCRIPT" ]] || die "Pre-training script not readable: $PT_SCRIPT"

# --- Build --export list (preserve caller environment in job + overrides) ---
EXPORT="ALL"
for o in "${OVERRIDES[@]}"; do
    EXPORT+=",$o"
done

# --- Submit ---
JOB_ID=$(sbatch --parsable --export="$EXPORT" "$PT_SCRIPT") \
    || die "sbatch failed"

echo "--- Pre-training submitted ---"
echo "Job ID:   $JOB_ID"
echo "Script:   $PT_SCRIPT"
echo "Export:   $EXPORT"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Cancel:   scancel $JOB_ID"
