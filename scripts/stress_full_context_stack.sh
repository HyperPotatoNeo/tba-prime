#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GPUS="${GPUS:-0,1}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NPROC="${NPROC:-${#GPU_LIST[@]}}"
if [[ "$NPROC" -lt 1 ]]; then
  echo "NPROC must be >= 1" >&2
  exit 2
fi

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_IMPL="${MODEL_IMPL:-auto}"
ATTN="${ATTN:-flash_attention_2}"
SEQ_LEN="${SEQ_LEN:-16384}"
STACK_SIZES="${STACK_SIZES:-1 2 3 4}"
STACK_TOKEN_BUDGET="${STACK_TOKEN_BUDGET:-}"
GROUPS_PER_RANK="${GROUPS_PER_RANK:-1}"
STEPS="${STEPS:-3}"
GENERATE_SAMPLES="${GENERATE_SAMPLES:-false}"
AC_MODE="${AC_MODE:-full}"
AC_FREQ="${AC_FREQ:-1}"
AC_OFFLOAD="${AC_OFFLOAD:-0}"
AC_OFFLOAD_PIN_MEMORY="${AC_OFFLOAD_PIN_MEMORY:-true}"
MAX_INFLIGHT_ACTIVATIONS="${MAX_INFLIGHT_ACTIVATIONS:-1}"
FUSED_LM_HEAD_TOKEN_CHUNK_SIZE="${FUSED_LM_HEAD_TOKEN_CHUNK_SIZE:-auto}"
OPTIM_FOREACH="${OPTIM_FOREACH:-false}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"
DRY_RUN="${DRY_RUN:-0}"
KVE_MEM_TRACE="${KVE_MEM_TRACE:-0}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$PYTORCH_ALLOC_CONF}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/.local_artifacts/stress/full_context_stack_$(date +%Y%m%d_%H%M%S)}"

TORCHRUN="${TORCHRUN:-$ROOT/.venv/bin/torchrun}"
if [[ ! -x "$TORCHRUN" ]]; then
  TORCHRUN="$(command -v torchrun || true)"
fi
if [[ -z "$TORCHRUN" ]]; then
  echo "Could not find torchrun. Set TORCHRUN=/path/to/torchrun." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
SUMMARY="$OUTPUT_ROOT/summary.tsv"
printf "stack_size\tstack_token_budget\tlocal_micro_batches\tglobal_fake_batch\tseq_len\tstatus\tpeak_mem_gib\tlog\n" > "$SUMMARY"

echo "root: $ROOT"
echo "output: $OUTPUT_ROOT"
echo "gpus: $GPUS (nproc=$NPROC)"
echo "model: $MODEL impl=$MODEL_IMPL attn=$ATTN seq_len=$SEQ_LEN"
echo "stack sizes: $STACK_SIZES"
if [[ -n "$STACK_TOKEN_BUDGET" ]]; then
  echo "stack token budget: $STACK_TOKEN_BUDGET"
fi
echo
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true
echo

write_model_chunk_size() {
  local cfg="$1"
  local value="$2"
  if [[ "$value" =~ ^[0-9]+$ ]]; then
    printf 'fused_lm_head_token_chunk_size = %s\n' "$value" >> "$cfg"
  else
    printf 'fused_lm_head_token_chunk_size = "%s"\n' "$value" >> "$cfg"
  fi
}

classify_status() {
  local status="$1"
  local log="$2"
  if [[ "$status" -eq 0 ]]; then
    echo "ok"
  elif grep -qiE "out of memory|cuda.*oom|cuda error:.*memory|cublas.*alloc" "$log"; then
    echo "oom"
  else
    echo "fail"
  fi
}

extract_peak_mem() {
  local log="$1"
  grep -Eo 'Peak Mem\.: [0-9.]+ GiB|Peak memory: [0-9.]+ GiB' "$log" \
    | tail -1 \
    | grep -Eo '[0-9.]+' \
    || true
}

for stack_size in $STACK_SIZES; do
  local_micro_batches=$((stack_size * GROUPS_PER_RANK))
  fake_batch_size=$((NPROC * local_micro_batches))
  run_dir="$OUTPUT_ROOT/stack_${stack_size}"
  mkdir -p "$run_dir"
  cfg="$run_dir/trainer.toml"
  log="$run_dir/run.log"

  cat > "$cfg" <<TOML
max_steps = $STEPS
max_async_level = 0
output_dir = "$run_dir/output"
dist_timeout_seconds = 3600

[data]
micro_batch_stack_size = $stack_size
TOML
  if [[ -n "$STACK_TOKEN_BUDGET" ]]; then
    printf 'micro_batch_stack_token_budget = %s\n' "$STACK_TOKEN_BUDGET" >> "$cfg"
  fi
  cat >> "$cfg" <<TOML

[data.fake]
batch_size = $fake_batch_size
generate_samples = $GENERATE_SAMPLES

[model]
name = "$MODEL"
impl = "$MODEL_IMPL"
attn = "$ATTN"
seq_len = $SEQ_LEN
TOML
  write_model_chunk_size "$cfg" "$FUSED_LM_HEAD_TOKEN_CHUNK_SIZE"

  if [[ "$AC_MODE" != "none" ]]; then
    cat >> "$cfg" <<TOML

[model.ac]
mode = "$AC_MODE"
freq = $AC_FREQ
TOML
  fi

  if [[ "$AC_OFFLOAD" == "1" ]]; then
    cat >> "$cfg" <<TOML

[model.ac_offloading]
pin_memory = $AC_OFFLOAD_PIN_MEMORY
max_inflight_activations = $MAX_INFLIGHT_ACTIVATIONS
TOML
  fi

  cat >> "$cfg" <<TOML

[loss]
kl_tau = 0.0

[optim]
type = "adamw"
lr = 1e-6
weight_decay = 0.01
betas1 = 0.9
betas2 = 0.9
max_norm = 1.0
foreach = $OPTIM_FOREACH
TOML

  cmd=(
    "$TORCHRUN"
    --standalone
    --nproc-per-node="$NPROC"
    -m prime_rl.trainer.rl.train
    @ "$cfg"
  )

  echo "== stack=$stack_size local_micro_batches=$local_micro_batches fake_batch_size=$fake_batch_size =="
  echo "config: $cfg"
  echo "log: $log"
  printf 'command: CUDA_VISIBLE_DEVICES=%s PYTHONPATH=%s/prime-rl/src:%s/src %q' "$GPUS" "$ROOT" "$ROOT" "${cmd[0]}"
  printf ' %q' "${cmd[@]:1}"
  echo

  if [[ "$DRY_RUN" == "1" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\tdry_run\t\t%s\n" "$stack_size" "${STACK_TOKEN_BUDGET:-}" "$local_micro_batches" "$fake_batch_size" "$SEQ_LEN" "$log" >> "$SUMMARY"
    continue
  fi

  set +e
  (
    cd "$ROOT"
    export CUDA_VISIBLE_DEVICES="$GPUS"
    export PYTHONPATH="$ROOT/prime-rl/src:$ROOT/src:${PYTHONPATH:-}"
    export PYTORCH_ALLOC_CONF
    export PYTORCH_CUDA_ALLOC_CONF
    export KVE_MEM_TRACE
    "${cmd[@]}"
  ) 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e

  label="$(classify_status "$status" "$log")"
  peak_mem="$(extract_peak_mem "$log")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$stack_size" "${STACK_TOKEN_BUDGET:-}" "$local_micro_batches" "$fake_batch_size" "$SEQ_LEN" "$label" "$peak_mem" "$log" >> "$SUMMARY"

  echo "result: $label peak_mem_gib=${peak_mem:-unknown}"
  echo

  if [[ "$status" -ne 0 && "$STOP_ON_FAILURE" == "1" ]]; then
    break
  fi
done

echo "summary: $SUMMARY"
cat "$SUMMARY"
