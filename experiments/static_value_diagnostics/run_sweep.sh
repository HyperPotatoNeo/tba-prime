#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH:-/pscratch/sd/s/siddart2}/static_value_diagnostics}"
export WANDB_PROJECT="${WANDB_PROJECT:-prime-rl-static-value-diagnostics}"
export VALUE_STEPS="${VALUE_STEPS:-50}"
export VALUE_WARMUP_STEPS="${VALUE_WARMUP_STEPS:-50}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export VALUE_MICRO_BATCH_SIZE="${VALUE_MICRO_BATCH_SIZE:-4}"
export VALUE_DISABLE_COMPILE="${VALUE_DISABLE_COMPILE:-1}"
export SEQ_LEN="${SEQ_LEN:-8192}"
export MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-$((SEQ_LEN - 512))}"
export GROUP_SIZE="${GROUP_SIZE:-16}"
export NUM_TRAIN_PROMPTS="${NUM_TRAIN_PROMPTS:-64}"
export NUM_VAL_PROMPTS="${NUM_VAL_PROMPTS:-16}"
export NUM_TEST_PROMPTS="${NUM_TEST_PROMPTS:-16}"

submit_one() {
  local name="$1"
  local infer_gpus="$2"
  local value_gpus="$3"
  sbatch --export=ALL,EXPERIMENT_NAME="$name",INFER_GPUS="$infer_gpus",VALUE_GPUS="$value_gpus" \
    "$SCRIPT_DIR/launch_one_node.sbatch"
}

# One full premium GPU node. Collection and value training run concurrently
# against the same static policy. The 2-inference/2-value split is available
# as an opt-in throughput experiment; it has been less stable under NCCL on
# Perlmutter than the default 3-inference/1-value split.
submit_one "rgmix-qwen4b-classifier-infer3-value1" 3 1
if [[ "${INCLUDE_2GPU_VALUE_SPLIT:-0}" == "1" ]]; then
  submit_one "rgmix-qwen4b-classifier-infer2-value2" 2 2
fi
