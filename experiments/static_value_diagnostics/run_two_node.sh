#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH:-/pscratch/sd/s/siddart2}/static_value_diagnostics}"
export WANDB_PROJECT="${WANDB_PROJECT:-prime-rl-static-value-diagnostics}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-rgmix-qwen4b-classifier-staged-2node}"
export VALUE_STEPS="${VALUE_STEPS:-100}"
export VALUE_WARMUP_STEPS="${VALUE_WARMUP_STEPS:-50}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export VALUE_UPDATES_PER_BATCH="${VALUE_UPDATES_PER_BATCH:-1}"
export VALUE_DISABLE_COMPILE="${VALUE_DISABLE_COMPILE:-1}"
export VALUE_DISABLE_OPTIM_CPU_OFFLOAD="${VALUE_DISABLE_OPTIM_CPU_OFFLOAD:-1}"
export VALUE_RELOAD_RECORDS_INTERVAL="${VALUE_RELOAD_RECORDS_INTERVAL:-1}"
export SEQ_LEN="${SEQ_LEN:-8192}"
export MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-$((SEQ_LEN - 512))}"
export GROUP_SIZE="${GROUP_SIZE:-8}"
export TRAIN_EPISODES="${TRAIN_EPISODES:-10000}"
export EVAL_EPISODES="${EVAL_EPISODES:-1024}"

sbatch "$SCRIPT_DIR/launch_staged_two_node.sbatch"
