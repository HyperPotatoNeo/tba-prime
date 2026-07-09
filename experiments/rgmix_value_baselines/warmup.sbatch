#!/bin/bash
# Value warmup: 100 value-only steps on the base Qwen3-4B policy over the fixed
# RG-Mix dataset. Produces the value checkpoint reused by all value baselines.
#SBATCH --job-name=rgmix-warmup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=gb200
#SBATCH --account=reasoning
#SBATCH --qos=priority-reasoning
#SBATCH --time=06:00:00
#SBATCH --output=/home/siddarth.venkatraman/rgmix_runs/logs/warmup_%j.log
#SBATCH --error=/home/siddarth.venkatraman/rgmix_runs/logs/warmup_%j.log

set -eux
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=$HOME/.uv_cache
export HF_HOME=$HOME/.hf
# Local CUDA 12.8 toolkit (flash-attn build + FA4 runtime JIT on GB200).
export CUDA_HOME="$HOME/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PATH"
[ -f "$HOME/.env" ] && source "$HOME/.env"
mkdir -p "$HOME/rgmix_runs/logs"
cd "$HOME/tba-prime"

# rl.toml defaults to the 2-node deployment; the warmup is a single-node
# value-only run (inference + value trainer co-located on one GB200), so override.
uv run --no-sync rl @ configs/rg_mix/rl.toml @ configs/rg_mix/warmup.toml \
  --deployment.type single_node \
  --wandb.name value-warmup --wandb.group warmup \
  --output-dir "$HOME/rgmix_runs/warmup"
