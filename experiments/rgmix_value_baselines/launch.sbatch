#!/bin/bash
# Single RG-Mix value-corrected GRPO run on one H100 node (4 inference / 4 trainer,
# value function co-located on the trainer GPUs). Parameterized by env vars:
#   BASELINE   one of: loo, pure_value, mixture_linear, mixture_position
#   K          group size (2, 4, 8)
#   SEED       inference sampling seed (fixes the run's stochasticity)
#   INIT_CKPT  path to the warmed value checkpoint (value baselines only; "" for loo)
#
# Submit via experiments/rgmix_value_baselines/submit_all.sh.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=h100
#SBATCH --account=reasoning
#SBATCH --qos=priority-reasoning
#SBATCH --time=24:00:00

set -eux
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=$HOME/.uv_cache
export HF_HOME=$HOME/.hf
# WANDB_API_KEY comes from ~/.env (not committed).
[ -f "$HOME/.env" ] && source "$HOME/.env"

REPO=$HOME/tba-prime
cd "$REPO"

NAME="${BASELINE}-k${K}-s${SEED}"
OVERLAY="configs/rg_mix/${BASELINE}.toml"

ARGS=( @ configs/rg_mix/rl.toml @ "$OVERLAY"
  --orchestrator.group-size "$K"
  --inference.seed "$SEED"
  --wandb.name "$NAME"
  --wandb.group "k${K}"
  --output-dir "$HOME/rgmix_runs/$NAME" )

if [ -n "${INIT_CKPT:-}" ]; then
  ARGS+=( --trainer.value-function.init-checkpoint "$INIT_CKPT" )
fi

uv run --no-sync rl "${ARGS[@]}"
