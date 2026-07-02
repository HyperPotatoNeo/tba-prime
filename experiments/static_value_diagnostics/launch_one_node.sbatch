#!/bin/bash
#SBATCH -A m4881
#SBATCH -C "gpu&hbm80g"
#SBATCH --qos=premium
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --job-name=static-vf
#SBATCH --output=static-vf-%j.out

set -euo pipefail

module unload darshan >/dev/null 2>&1 || true

export HOME="${SCRATCH:-/pscratch/sd/s/siddart2}"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PODMANHPC_PODMAN_BIN="${PODMANHPC_PODMAN_BIN:-/global/common/shared/das/podman-4.7.0/bin/podman}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/value-functions-prime-rl/prime-rl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/static_value_diagnostics}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rgmix-qwen4b-classifier}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$EXPERIMENT_NAME/$SLURM_JOB_ID}"
DATASET_PATH="${DATASET_PATH:-$HOME/datasets/rg_mix_7500}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
WANDB_PROJECT="${WANDB_PROJECT:-prime-rl-static-value-diagnostics}"

GROUP_SIZE="${GROUP_SIZE:-16}"
NUM_TRAIN_PROMPTS="${NUM_TRAIN_PROMPTS:-64}"
NUM_VAL_PROMPTS="${NUM_VAL_PROMPTS:-16}"
NUM_TEST_PROMPTS="${NUM_TEST_PROMPTS:-16}"
SEQ_LEN="${SEQ_LEN:-8192}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-$((SEQ_LEN - 512))}"
VALUE_STEPS="${VALUE_STEPS:-50}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
VALUE_MICRO_BATCH_SIZE="${VALUE_MICRO_BATCH_SIZE:-4}"
VALUE_DISABLE_COMPILE="${VALUE_DISABLE_COMPILE:-1}"
VALUE_WARMUP_STEPS="${VALUE_WARMUP_STEPS:-50}"
VALUE_LR="${VALUE_LR:-5e-5}"
INFER_GPUS="${INFER_GPUS:-2}"
VALUE_GPUS="${VALUE_GPUS:-2}"
STAGE="${STAGE:-all}"

cd "$REPO_DIR"
mkdir -p "$RUN_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
exec > >(tee -a "$RUN_DIR/job.log") 2>&1

for env_file in "$HOME/kv-eviction/.env" "$HOME/compaction-rl/.env" "$HOME/.env"; do
  if [[ -z "${WANDB_API_KEY:-}" && -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
done
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set; set it in the submit environment or one of the sourced .env files." >&2
  exit 1
fi

gpu_list() {
  local count="$1"
  local offset="${2:-0}"
  local out=()
  for ((i = 0; i < count; i++)); do
    out+=("$((offset + i))")
  done
  local IFS=,
  echo "${out[*]}"
}

INFER_CUDA="$(gpu_list "$INFER_GPUS" 0)"
VALUE_OFFSET=$((4 - VALUE_GPUS))
VALUE_CUDA="$(gpu_list "$VALUE_GPUS" "$VALUE_OFFSET")"

cleanup() {
  if [[ -n "${COLLECT_PID:-}" ]]; then
    kill "$COLLECT_PID" >/dev/null 2>&1 || true
    wait "$COLLECT_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${INFER_PID:-}" ]]; then
    kill "$INFER_PID" >/dev/null 2>&1 || true
    wait "$INFER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

start_inference() {
  cat > "$RUN_DIR/inference.toml" <<EOF
gpu_memory_utilization = 0.9

[server]
host = "0.0.0.0"
port = 8000

[model]
name = "$MODEL"
max_model_len = $SEQ_LEN

[parallel]
tp = 1
dp = $INFER_GPUS

[vllm_extra]
generation_config = "vllm"
EOF

  CUDA_VISIBLE_DEVICES="$INFER_CUDA" uv run inference @ "$RUN_DIR/inference.toml" > "$RUN_DIR/inference.log" 2>&1 &
  INFER_PID=$!
  for _ in $(seq 1 180); do
    if curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1; then
      return
    fi
    sleep 10
  done
  echo "inference server did not become ready; tail follows" >&2
  tail -200 "$RUN_DIR/inference.log" >&2 || true
  exit 1
}

collect_rollouts_foreground() {
  uv run --with reasoning-gym python -m experiments.static_value_diagnostics.collect_static_rollouts \
    --output-dir "$RUN_DIR/data" \
    --dataset-path "$DATASET_PATH" \
    --model "$MODEL" \
    --base-url "http://127.0.0.1:8000/v1" \
    --group-size "$GROUP_SIZE" \
    --num-train-prompts "$NUM_TRAIN_PROMPTS" \
    --num-val-prompts "$NUM_VAL_PROMPTS" \
    --num-test-prompts "$NUM_TEST_PROMPTS" \
    --temperature 1.0 \
    --top-p 1.0 \
    --top-k -1 \
    --min-p 0.0 \
    --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$EXPERIMENT_NAME-collect-$SLURM_JOB_ID"
}

collect_rollouts() {
  start_inference
  collect_rollouts_foreground
  cleanup
  unset INFER_PID
}

collect_rollouts_background() {
  start_inference
  collect_rollouts_foreground &
  COLLECT_PID=$!
}

wait_for_collection() {
  if [[ -n "${COLLECT_PID:-}" ]]; then
    wait "$COLLECT_PID"
    unset COLLECT_PID
  fi
  cleanup
  unset INFER_PID
}

train_value() {
  local compile_args=()
  if [[ "$VALUE_DISABLE_COMPILE" == "1" || "$VALUE_DISABLE_COMPILE" == "true" ]]; then
    compile_args=(--disable-compile)
  fi
  CUDA_VISIBLE_DEVICES="$VALUE_CUDA" uv run torchrun --standalone --nproc-per-node "$VALUE_GPUS" \
    -m experiments.static_value_diagnostics.train_static_value \
    --rollouts "$RUN_DIR/data/rollouts.jsonl" \
    --output-dir "$RUN_DIR/value" \
    --model "$MODEL" \
    --seq-len "$SEQ_LEN" \
    --steps "$VALUE_STEPS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --micro-batch-size "$VALUE_MICRO_BATCH_SIZE" \
    --lr "$VALUE_LR" \
    --warmup-steps "$VALUE_WARMUP_STEPS" \
    --reward-range 0 1 \
    --n-bins 1 \
    "${compile_args[@]}" \
    --stream \
    --min-train-records "$GLOBAL_BATCH_SIZE" \
    --expected-val-records "$((NUM_VAL_PROMPTS * GROUP_SIZE))" \
    --expected-test-records "$((NUM_TEST_PROMPTS * GROUP_SIZE))" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$EXPERIMENT_NAME-value-${VALUE_GPUS}gpu-$SLURM_JOB_ID"
}

compute_diagnostics() {
  uv run python -m experiments.static_value_diagnostics.compute_offline_diagnostics \
    --predictions-dir "$RUN_DIR/value" \
    --output-dir "$RUN_DIR/diagnostics" \
    --group-size "$GROUP_SIZE" \
    --group-sizes 2 4 8 16 \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$EXPERIMENT_NAME-diagnostics-$SLURM_JOB_ID"
  uv run --with matplotlib python -m experiments.static_value_diagnostics.plot_offline_diagnostics \
    --diagnostics-dir "$RUN_DIR/diagnostics" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$EXPERIMENT_NAME-plots-$SLURM_JOB_ID"
}

case "$STAGE" in
  all)
    collect_rollouts_background
    train_value
    wait_for_collection
    compute_diagnostics
    ;;
  offline)
    collect_rollouts
    train_value
    compute_diagnostics
    ;;
  collect)
    collect_rollouts
    ;;
  train)
    train_value
    compute_diagnostics
    ;;
  diagnostics)
    compute_diagnostics
    ;;
  *)
    echo "unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac

echo "static value diagnostics complete: $RUN_DIR"
