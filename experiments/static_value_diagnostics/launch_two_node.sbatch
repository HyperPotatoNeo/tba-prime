#!/bin/bash
#SBATCH -A m4881
#SBATCH -C "gpu&hbm80g"
#SBATCH --qos=premium
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --job-name=static-vf-2n
#SBATCH --output=static-vf-2n-%j.out

set -euo pipefail

module unload darshan >/dev/null 2>&1 || true

export HOME="${SCRATCH:-/pscratch/sd/s/siddart2}"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/value-functions-prime-rl/prime-rl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/static_value_diagnostics}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rgmix-qwen4b-classifier-2node}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$EXPERIMENT_NAME/$SLURM_JOB_ID}"
DATASET_PATH="${DATASET_PATH:-$HOME/datasets/rg_mix_7500}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
WANDB_PROJECT="${WANDB_PROJECT:-prime-rl-static-value-diagnostics}"

GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_VAL_PROMPTS="${NUM_VAL_PROMPTS:-32}"
NUM_TEST_PROMPTS="${NUM_TEST_PROMPTS:-32}"
SEQ_LEN="${SEQ_LEN:-8192}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-$((SEQ_LEN - 512))}"
VALUE_STEPS="${VALUE_STEPS:-300}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
VALUE_UPDATES_PER_BATCH="${VALUE_UPDATES_PER_BATCH:-1}"
VALUE_DISABLE_COMPILE="${VALUE_DISABLE_COMPILE:-1}"
VALUE_DISABLE_OPTIM_CPU_OFFLOAD="${VALUE_DISABLE_OPTIM_CPU_OFFLOAD:-1}"
VALUE_DISABLE_AC_OFFLOADING="${VALUE_DISABLE_AC_OFFLOADING:-1}"
VALUE_DISABLE_RESHARD_AFTER_FORWARD="${VALUE_DISABLE_RESHARD_AFTER_FORWARD:-1}"
VALUE_DP_REPLICATE="${VALUE_DP_REPLICATE:-4}"
VALUE_LOG_EVERY="${VALUE_LOG_EVERY:-1}"
VALUE_WARMUP_STEPS="${VALUE_WARMUP_STEPS:-50}"
VALUE_LR="${VALUE_LR:-5e-5}"
VALUE_RELOAD_RECORDS_INTERVAL="${VALUE_RELOAD_RECORDS_INTERVAL:-1}"
MAX_DATASET_PROMPTS="${MAX_DATASET_PROMPTS:-7500}"

DEFAULT_TRAIN_PROMPTS=$(((VALUE_STEPS * GLOBAL_BATCH_SIZE + GROUP_SIZE - 1) / GROUP_SIZE))
MAX_TRAIN_PROMPTS=$((MAX_DATASET_PROMPTS - NUM_VAL_PROMPTS - NUM_TEST_PROMPTS))
if ((DEFAULT_TRAIN_PROMPTS > MAX_TRAIN_PROMPTS)); then
  DEFAULT_TRAIN_PROMPTS="$MAX_TRAIN_PROMPTS"
fi
NUM_TRAIN_PROMPTS="${NUM_TRAIN_PROMPTS:-$DEFAULT_TRAIN_PROMPTS}"
NUM_TRAIN_GROUPS="${NUM_TRAIN_GROUPS:-$(((VALUE_STEPS * GLOBAL_BATCH_SIZE + GROUP_SIZE - 1) / GROUP_SIZE))}"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
INFER_NODE="${NODES[0]}"
TRAIN_NODE="${NODES[1]}"

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

cleanup() {
  if [[ -n "${COLLECT_PID:-}" ]]; then
    kill "$COLLECT_PID" >/dev/null 2>&1 || true
    wait "$COLLECT_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${TRAIN_PID:-}" ]]; then
    kill "$TRAIN_PID" >/dev/null 2>&1 || true
    wait "$TRAIN_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${INFER_PID:-}" ]]; then
    kill "$INFER_PID" >/dev/null 2>&1 || true
    wait "$INFER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

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
dp = 4

[vllm_extra]
generation_config = "vllm"
EOF

echo "inference_node=$INFER_NODE trainer_node=$TRAIN_NODE run_dir=$RUN_DIR"

ssh "$INFER_NODE" "cd '$REPO_DIR' && export HOME='$HOME' PATH='$PATH' UV_CACHE_DIR='$UV_CACHE_DIR' VLLM_API_KEY='$VLLM_API_KEY' PYTHONPATH='$PYTHONPATH' PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3; uv run inference @ '$RUN_DIR/inference.toml'" \
  > "$RUN_DIR/inference.log" 2>&1 &
INFER_PID=$!

for _ in $(seq 1 180); do
  if curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" "http://$INFER_NODE:8000/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 10
done
if ! curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" "http://$INFER_NODE:8000/v1/models" >/dev/null 2>&1; then
  echo "inference server did not become ready; tail follows" >&2
  tail -200 "$RUN_DIR/inference.log" >&2 || true
  exit 1
fi

uv run --with reasoning-gym python -m experiments.static_value_diagnostics.collect_static_rollouts \
  --output-dir "$RUN_DIR/data" \
  --dataset-path "$DATASET_PATH" \
  --model "$MODEL" \
  --base-url "http://$INFER_NODE:8000/v1" \
  --group-size "$GROUP_SIZE" \
  --num-train-prompts "$NUM_TRAIN_PROMPTS" \
  --num-train-groups "$NUM_TRAIN_GROUPS" \
  --num-val-prompts "$NUM_VAL_PROMPTS" \
  --num-test-prompts "$NUM_TEST_PROMPTS" \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k -1 \
  --min-p 0.0 \
  --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$EXPERIMENT_NAME-collect-$SLURM_JOB_ID" &
COLLECT_PID=$!

value_args=()
if [[ "$VALUE_DISABLE_COMPILE" == "1" || "$VALUE_DISABLE_COMPILE" == "true" ]]; then
  value_args+=(--disable-compile)
fi
if [[ "$VALUE_DISABLE_OPTIM_CPU_OFFLOAD" == "1" || "$VALUE_DISABLE_OPTIM_CPU_OFFLOAD" == "true" ]]; then
  value_args+=(--disable-optim-cpu-offload)
fi
if [[ "$VALUE_DISABLE_AC_OFFLOADING" == "1" || "$VALUE_DISABLE_AC_OFFLOADING" == "true" ]]; then
  value_args+=(--disable-ac-offloading)
fi
if [[ "$VALUE_DISABLE_RESHARD_AFTER_FORWARD" == "1" || "$VALUE_DISABLE_RESHARD_AFTER_FORWARD" == "true" ]]; then
  value_args+=(--disable-reshard-after-forward)
fi

ssh "$TRAIN_NODE" "cd '$REPO_DIR' && export HOME='$HOME' PATH='$PATH' UV_CACHE_DIR='$UV_CACHE_DIR' WANDB_API_KEY='$WANDB_API_KEY' PYTHONPATH='$PYTHONPATH' PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC='${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}'; uv run torchrun --standalone --nproc-per-node 4 -m experiments.static_value_diagnostics.train_static_value --rollouts '$RUN_DIR/data/rollouts.jsonl' --output-dir '$RUN_DIR/value' --model '$MODEL' --seq-len '$SEQ_LEN' --steps '$VALUE_STEPS' --global-batch-size '$GLOBAL_BATCH_SIZE' --updates-per-batch '$VALUE_UPDATES_PER_BATCH' --lr '$VALUE_LR' --warmup-steps '$VALUE_WARMUP_STEPS' --dp-replicate '$VALUE_DP_REPLICATE' --reward-range 0 1 --n-bins 1 ${value_args[*]} --log-every '$VALUE_LOG_EVERY' --stream --reload-records-interval '$VALUE_RELOAD_RECORDS_INTERVAL' --min-train-records '$GLOBAL_BATCH_SIZE' --expected-val-records '$((NUM_VAL_PROMPTS * GROUP_SIZE))' --expected-test-records '$((NUM_TEST_PROMPTS * GROUP_SIZE))' --wandb-project '$WANDB_PROJECT' --wandb-run-name '$EXPERIMENT_NAME-value-4gpu-$SLURM_JOB_ID'" \
  > "$RUN_DIR/value_train.log" 2>&1 &
TRAIN_PID=$!

wait "$TRAIN_PID"
unset TRAIN_PID
wait "$COLLECT_PID"
unset COLLECT_PID
kill "$INFER_PID" >/dev/null 2>&1 || true
wait "$INFER_PID" >/dev/null 2>&1 || true
unset INFER_PID

uv run python -m experiments.static_value_diagnostics.compute_offline_diagnostics \
  --predictions-dir "$RUN_DIR/value" \
  --output-dir "$RUN_DIR/diagnostics" \
  --group-size "$GROUP_SIZE" \
  --group-sizes 2 4 8 \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$EXPERIMENT_NAME-diagnostics-$SLURM_JOB_ID"
uv run --with matplotlib python -m experiments.static_value_diagnostics.plot_offline_diagnostics \
  --diagnostics-dir "$RUN_DIR/diagnostics" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$EXPERIMENT_NAME-plots-$SLURM_JOB_ID"

echo "static value diagnostics complete: $RUN_DIR"
