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
STATIC_VALUE_CONFIG="${STATIC_VALUE_CONFIG:-$REPO_DIR/examples/static_value_rg_mix/static_value.toml}"

cd "$REPO_DIR"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/job.log") 2>&1

for env_file in "$HOME/kv-eviction/.env" "$HOME/compaction-rl/.env" "$HOME/.env"; do
  if [[ -z "${WANDB_API_KEY:-}" && -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
done

echo "static-value config: $STATIC_VALUE_CONFIG"
echo "static-value output: $RUN_DIR"

uv run static-value @ "$STATIC_VALUE_CONFIG" \
  --repo-dir "$REPO_DIR" \
  --output-dir "$RUN_DIR"
