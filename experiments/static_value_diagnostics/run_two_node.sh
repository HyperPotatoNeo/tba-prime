#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH:-/pscratch/sd/s/siddart2}/static_value_diagnostics}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-rgmix-qwen4b-classifier-staged-2node}"
export STATIC_VALUE_CONFIG="${STATIC_VALUE_CONFIG:-$SCRIPT_DIR/../../examples/static_value_rg_mix/static_value.toml}"

sbatch "$SCRIPT_DIR/launch_staged_two_node.sbatch"
