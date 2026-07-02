#!/bin/bash
#SBATCH -A m4881
#SBATCH -C "gpu&hbm80g"
#SBATCH --qos=premium
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --job-name=static-vf-staged
#SBATCH --output=static-vf-staged-%j.out

set -euo pipefail

module unload darshan >/dev/null 2>&1 || true

export HOME="${SCRATCH:-/pscratch/sd/s/siddart2}"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/value-functions-prime-rl/prime-rl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/static_value_diagnostics}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rgmix-qwen4b-classifier-staged-2node}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$EXPERIMENT_NAME/$SLURM_JOB_ID}"
DATASET_PATH="${DATASET_PATH:-$HOME/datasets/rg_mix_7500}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
WANDB_PROJECT="${WANDB_PROJECT:-prime-rl-static-value-diagnostics}"

GROUP_SIZE="${GROUP_SIZE:-8}"
TRAIN_EPISODES="${TRAIN_EPISODES:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-1024}"
SEQ_LEN="${SEQ_LEN:-8192}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-$((SEQ_LEN - 512))}"
VALUE_STEPS="${VALUE_STEPS:-100}"
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
MAX_DATASET_PROMPTS="${MAX_DATASET_PROMPTS:-7500}"
EVAL_VAL_OFFSET="${EVAL_VAL_OFFSET:-7000}"
RENDERER_POOL_SIZE="${RENDERER_POOL_SIZE:-128}"
MAX_CONCURRENT_GROUPS="${MAX_CONCURRENT_GROUPS:-32}"

if ((TRAIN_EPISODES % GROUP_SIZE != 0)); then
  echo "TRAIN_EPISODES=$TRAIN_EPISODES must be divisible by GROUP_SIZE=$GROUP_SIZE" >&2
  exit 1
fi
if ((EVAL_EPISODES % GROUP_SIZE != 0)); then
  echo "EVAL_EPISODES=$EVAL_EPISODES must be divisible by GROUP_SIZE=$GROUP_SIZE" >&2
  exit 1
fi

TRAIN_GROUPS=$((TRAIN_EPISODES / GROUP_SIZE))
TRAIN_GROUPS_NODE0=$(((TRAIN_GROUPS + 1) / 2))
TRAIN_GROUPS_NODE1=$((TRAIN_GROUPS - TRAIN_GROUPS_NODE0))
EVAL_GROUPS=$((EVAL_EPISODES / GROUP_SIZE))
if ((EVAL_GROUPS % 2 != 0)); then
  echo "EVAL_EPISODES/GROUP_SIZE must be even so val/test get equal prompt counts" >&2
  exit 1
fi
VAL_GROUPS=$((EVAL_GROUPS / 2))
TEST_GROUPS=$((EVAL_GROUPS - VAL_GROUPS))
EVAL_TEST_OFFSET=$((EVAL_VAL_OFFSET + VAL_GROUPS))
if ((TRAIN_GROUPS > EVAL_VAL_OFFSET)); then
  echo "Train prompt pool overlaps held-out offset: TRAIN_GROUPS=$TRAIN_GROUPS EVAL_VAL_OFFSET=$EVAL_VAL_OFFSET" >&2
  exit 1
fi
if ((EVAL_TEST_OFFSET + TEST_GROUPS > MAX_DATASET_PROMPTS)); then
  echo "Held-out prompts exceed dataset: last=$((EVAL_TEST_OFFSET + TEST_GROUPS - 1)) max=$((MAX_DATASET_PROMPTS - 1))" >&2
  exit 1
fi

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if ((${#NODES[@]} < 2)); then
  echo "This launcher requires two allocated nodes." >&2
  exit 1
fi
NODE0="${NODES[0]}"
NODE1="${NODES[1]}"
TRAIN_NODE="$NODE0"
REMOTE_USER="$(id -un)"

cd "$REPO_DIR"
mkdir -p "$RUN_DIR"/{configs,data,logs,value,value_eval,diagnostics}
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

INFER_PIDS=()
INFER_NODES=()
INFER_CONFIGS=()
COLLECT_PIDS=()
TRAIN_PID=""
PREDICT_PID=""

cleanup_processes() {
  for pid in "${COLLECT_PIDS[@]}"; do
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
  done
  if [[ -n "$TRAIN_PID" ]]; then
    kill "$TRAIN_PID" >/dev/null 2>&1 || true
    wait "$TRAIN_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$PREDICT_PID" ]]; then
    kill "$PREDICT_PID" >/dev/null 2>&1 || true
    wait "$PREDICT_PID" >/dev/null 2>&1 || true
  fi
}

stop_inference() {
  for pid in "${INFER_PIDS[@]}"; do
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
  done
  for idx in "${!INFER_NODES[@]}"; do
    ssh "${INFER_NODES[$idx]}" "pkill -u '$REMOTE_USER' -f '${INFER_CONFIGS[$idx]}' || true" >/dev/null 2>&1 || true
  done
  INFER_PIDS=()
  INFER_NODES=()
  INFER_CONFIGS=()
}

cleanup() {
  cleanup_processes
  stop_inference
}
trap cleanup EXIT

write_inference_config() {
  local idx="$1"
  local path="$RUN_DIR/configs/inference_${idx}.toml"
  cat > "$path" <<EOF
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
  echo "$path"
}

start_inference() {
  local node="$1"
  local idx="$2"
  local config
  config="$(write_inference_config "$idx")"
  ssh "$node" "cd '$REPO_DIR' && export HOME='$HOME' PATH='$PATH' UV_CACHE_DIR='$UV_CACHE_DIR' VLLM_API_KEY='$VLLM_API_KEY' PYTHONPATH='$PYTHONPATH' PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3; uv run inference @ '$config'" \
    > "$RUN_DIR/logs/inference_${idx}.log" 2>&1 &
  INFER_PIDS+=("$!")
  INFER_NODES+=("$node")
  INFER_CONFIGS+=("$config")
}

wait_for_inference() {
  local node="$1"
  local log_path="$2"
  for _ in $(seq 1 180); do
    if curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" "http://$node:8000/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 10
  done
  echo "inference server on $node did not become ready; tail follows" >&2
  tail -200 "$log_path" >&2 || true
  return 1
}

start_all_inference() {
  echo "starting inference on $NODE0 and $NODE1"
  start_inference "$NODE0" "node0"
  start_inference "$NODE1" "node1"
  wait_for_inference "$NODE0" "$RUN_DIR/logs/inference_node0.log"
  wait_for_inference "$NODE1" "$RUN_DIR/logs/inference_node1.log"
}

common_collect_flags=(
  --dataset-path "$DATASET_PATH"
  --model "$MODEL"
  --api-key-var VLLM_API_KEY
  --group-size "$GROUP_SIZE"
  --temperature 1.0
  --top-p 1.0
  --top-k -1
  --min-p 0.0
  --max-completion-tokens "$MAX_COMPLETION_TOKENS"
  --renderer-pool-size "$RENDERER_POOL_SIZE"
  --max-concurrent-groups "$MAX_CONCURRENT_GROUPS"
  --wandb-project "$WANDB_PROJECT"
)

run_collect() {
  local node="$1"
  local name="$2"
  local output_dir="$3"
  shift 3
  ssh "$node" "cd '$REPO_DIR' && export HOME='$HOME' PATH='$PATH' UV_CACHE_DIR='$UV_CACHE_DIR' VLLM_API_KEY='$VLLM_API_KEY' WANDB_API_KEY='$WANDB_API_KEY' PYTHONPATH='$PYTHONPATH' PYTHONUNBUFFERED=1; uv run --with reasoning-gym python -m experiments.static_value_diagnostics.collect_static_rollouts --base-url 'http://127.0.0.1:8000/v1' --output-dir '$output_dir' ${common_collect_flags[*]} --wandb-run-name '$EXPERIMENT_NAME-$name-$SLURM_JOB_ID' $*" \
    > "$RUN_DIR/logs/collect_${name}.log" 2>&1 &
  COLLECT_PIDS+=("$!")
}

wait_collectors() {
  local pid
  for pid in "${COLLECT_PIDS[@]}"; do
    wait "$pid"
  done
  COLLECT_PIDS=()
}

merge_rollouts() {
  local out="$1"
  local expected="$2"
  shift 2
  : > "$out"
  local input
  for input in "$@"; do
    if [[ ! -s "$input" ]]; then
      echo "Missing rollout file $input" >&2
      exit 1
    fi
    cat "$input" >> "$out"
  done
  local actual
  actual="$(wc -l < "$out" | tr -d ' ')"
  if [[ "$actual" != "$expected" ]]; then
    echo "Merged $out has $actual rows, expected $expected" >&2
    exit 1
  fi
}

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

echo "run_dir=$RUN_DIR"
echo "train episodes=$TRAIN_EPISODES groups=$TRAIN_GROUPS split=${TRAIN_GROUPS_NODE0}+${TRAIN_GROUPS_NODE1}"
echo "eval episodes=$EVAL_EPISODES val_groups=$VAL_GROUPS test_groups=$TEST_GROUPS"

start_all_inference
run_collect "$NODE0" "collect-train-node0" "$RUN_DIR/data/train_node0" \
  --group-id-prefix node0: \
  --num-train-prompts "$TRAIN_GROUPS_NODE0" \
  --num-train-groups "$TRAIN_GROUPS_NODE0" \
  --num-val-prompts 0 \
  --num-test-prompts 0 \
  --train-offset 0
run_collect "$NODE1" "collect-train-node1" "$RUN_DIR/data/train_node1" \
  --group-id-prefix node1: \
  --num-train-prompts "$TRAIN_GROUPS_NODE1" \
  --num-train-groups "$TRAIN_GROUPS_NODE1" \
  --num-val-prompts 0 \
  --num-test-prompts 0 \
  --train-offset "$TRAIN_GROUPS_NODE0"
wait_collectors
merge_rollouts "$RUN_DIR/data/train_rollouts.jsonl" "$TRAIN_EPISODES" \
  "$RUN_DIR/data/train_node0/rollouts.jsonl" \
  "$RUN_DIR/data/train_node1/rollouts.jsonl"
stop_inference

echo "training value function offline on $TRAIN_NODE"
ssh "$TRAIN_NODE" "cd '$REPO_DIR' && export HOME='$HOME' PATH='$PATH' UV_CACHE_DIR='$UV_CACHE_DIR' WANDB_API_KEY='$WANDB_API_KEY' PYTHONPATH='$PYTHONPATH' PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC='${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}'; uv run torchrun --standalone --nproc-per-node 4 -m experiments.static_value_diagnostics.train_static_value --rollouts '$RUN_DIR/data/train_rollouts.jsonl' --output-dir '$RUN_DIR/value' --model '$MODEL' --seq-len '$SEQ_LEN' --steps '$VALUE_STEPS' --global-batch-size '$GLOBAL_BATCH_SIZE' --group-size '$GROUP_SIZE' --updates-per-batch '$VALUE_UPDATES_PER_BATCH' --lr '$VALUE_LR' --warmup-steps '$VALUE_WARMUP_STEPS' --dp-replicate '$VALUE_DP_REPLICATE' --reward-range 0 1 --n-bins 1 ${value_args[*]} --skip-predict --log-every '$VALUE_LOG_EVERY' --wandb-project '$WANDB_PROJECT' --wandb-run-name '$EXPERIMENT_NAME-value-$SLURM_JOB_ID'" \
  > "$RUN_DIR/logs/value_train.log" 2>&1 &
TRAIN_PID=$!
wait "$TRAIN_PID"
TRAIN_PID=""

start_all_inference
run_collect "$NODE0" "collect-val" "$RUN_DIR/data/eval_val" \
  --group-id-prefix val: \
  --num-train-prompts 0 \
  --num-train-groups 0 \
  --num-val-prompts "$VAL_GROUPS" \
  --num-test-prompts 0 \
  --val-offset "$EVAL_VAL_OFFSET"
run_collect "$NODE1" "collect-test" "$RUN_DIR/data/eval_test" \
  --group-id-prefix test: \
  --num-train-prompts 0 \
  --num-train-groups 0 \
  --num-val-prompts 0 \
  --num-test-prompts "$TEST_GROUPS" \
  --test-offset "$EVAL_TEST_OFFSET"
wait_collectors
merge_rollouts "$RUN_DIR/data/eval_rollouts.jsonl" "$EVAL_EPISODES" \
  "$RUN_DIR/data/eval_val/rollouts.jsonl" \
  "$RUN_DIR/data/eval_test/rollouts.jsonl"
stop_inference

echo "predicting held-out values from $RUN_DIR/value/value_checkpoint"
ssh "$TRAIN_NODE" "cd '$REPO_DIR' && export HOME='$HOME' PATH='$PATH' UV_CACHE_DIR='$UV_CACHE_DIR' WANDB_API_KEY='$WANDB_API_KEY' PYTHONPATH='$PYTHONPATH' PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC='${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}'; uv run torchrun --standalone --nproc-per-node 4 -m experiments.static_value_diagnostics.train_static_value --predict-only --load-value-checkpoint '$RUN_DIR/value/value_checkpoint' --rollouts '$RUN_DIR/data/train_rollouts.jsonl' --predict-rollouts '$RUN_DIR/data/eval_rollouts.jsonl' --output-dir '$RUN_DIR/value_eval' --model '$MODEL' --seq-len '$SEQ_LEN' --steps 0 --global-batch-size '$GLOBAL_BATCH_SIZE' --group-size '$GROUP_SIZE' --updates-per-batch '$VALUE_UPDATES_PER_BATCH' --lr '$VALUE_LR' --warmup-steps '$VALUE_WARMUP_STEPS' --dp-replicate '$VALUE_DP_REPLICATE' --reward-range 0 1 --n-bins 1 ${value_args[*]} --log-every '$VALUE_LOG_EVERY' --wandb-project '$WANDB_PROJECT' --wandb-run-name '$EXPERIMENT_NAME-predict-$SLURM_JOB_ID'" \
  > "$RUN_DIR/logs/value_predict.log" 2>&1 &
PREDICT_PID=$!
wait "$PREDICT_PID"
PREDICT_PID=""

uv run python -m experiments.static_value_diagnostics.compute_offline_diagnostics \
  --predictions-dir "$RUN_DIR/value_eval" \
  --output-dir "$RUN_DIR/diagnostics" \
  --group-size "$GROUP_SIZE" \
  --group-sizes 2 4 8 \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$EXPERIMENT_NAME-diagnostics-$SLURM_JOB_ID"
uv run --with matplotlib python -m experiments.static_value_diagnostics.plot_offline_diagnostics \
  --diagnostics-dir "$RUN_DIR/diagnostics" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$EXPERIMENT_NAME-plots-$SLURM_JOB_ID"

echo "static value staged diagnostics complete: $RUN_DIR"
