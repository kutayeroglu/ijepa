#!/bin/bash
#
# Submit a linear probe Slurm job with optional environment overrides.
#
# Usage:
#   ./submit_linear_probe.sh <linear_probe_script> [KEY=VALUE ...]
#
# Environment overrides are passed to the job via sbatch --export=ALL,KEY=VALUE,...
# Values must not contain commas (Slurm --export uses comma as separator).
#
# Examples:
#   ./dev/run_on_hpc/mn5/submit_linear_probe.sh \
#       dev/run_on_hpc/mn5/linear_probe/lp100_qn50_4gpu.sh
#
#   ./dev/run_on_hpc/mn5/submit_linear_probe.sh \
#       dev/run_on_hpc/mn5/linear_probe/lp100_qn50_4gpu.sh \
#       PRETRAINING_RUN_ID=12345678_bal_qn50_vitb
#
#   ./dev/run_on_hpc/mn5/submit_linear_probe.sh \
#       dev/run_on_hpc/mn5/linear_probe/lp100_mb_4gpu.sh \
#       PRETRAINING_RUN_ID=37972861_bal_mb_vitb \
#       CONFIG_TAG=bal_mb_vitb \
#       CHECKPOINT_SUFFIX=latest

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

show_help() {
    cat <<'EOF'
Submit a linear probe Slurm job with optional environment overrides.

Usage:
  ./submit_linear_probe.sh <linear_probe_script> [KEY=VALUE ...]

Environment overrides are passed to the job via sbatch --export=ALL,KEY=VALUE,...
Values must not contain commas (Slurm --export uses comma as separator).

Examples:
  ./dev/run_on_hpc/mn5/submit_linear_probe.sh \
      dev/run_on_hpc/mn5/linear_probe/lp100_qn50_4gpu.sh

  ./dev/run_on_hpc/mn5/submit_linear_probe.sh \
      dev/run_on_hpc/mn5/linear_probe/lp100_qn50_4gpu.sh \
      PRETRAINING_RUN_ID=12345678_bal_qn50_vitb

  ./dev/run_on_hpc/mn5/submit_linear_probe.sh \
      dev/run_on_hpc/mn5/linear_probe/lp100_mb_4gpu.sh \
      PRETRAINING_RUN_ID=37972861_bal_mb_vitb \
      CONFIG_TAG=bal_mb_vitb \
      CHECKPOINT_SUFFIX=latest
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
LP_SCRIPT=""
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
            [[ -n "$LP_SCRIPT" ]] || die "Linear probe script must be the first argument (before KEY=VALUE overrides)"
            validate_override "$1"
            OVERRIDES+=("$1")
            ;;
        *)
            [[ -z "$LP_SCRIPT" ]] || die "Unexpected argument: $1 (overrides must look like KEY=VALUE)"
            LP_SCRIPT="$1"
            ;;
    esac
    shift
done

[[ -n "$LP_SCRIPT" ]] || die "Missing linear probe script (first argument)"
[[ -f "$LP_SCRIPT" ]] || die "Linear probe script not found: $LP_SCRIPT"
[[ -r "$LP_SCRIPT" ]] || die "Linear probe script not readable: $LP_SCRIPT"

# --- Build --export list (preserve caller environment in job + overrides) ---
EXPORT="ALL"
for o in "${OVERRIDES[@]}"; do
    EXPORT+=",$o"
done

# --- Submit ---
JOB_ID=$(sbatch --parsable --export="$EXPORT" "$LP_SCRIPT") \
    || die "sbatch failed"

echo "--- Linear probe submitted ---"
echo "Job ID:   $JOB_ID"
echo "Script:   $LP_SCRIPT"
echo "Export:   $EXPORT"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Cancel:   scancel $JOB_ID"
