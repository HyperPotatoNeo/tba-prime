#!/bin/bash
# Render router-free 2-node RL launchers for RG-Mix runs on BLC (GB200 / aarch64).
#
# Run this ON an aarch64 compute node — the login node is x86 and cannot run the
# aarch64 venv — e.g. via srun:
#   srun --partition=gb200 --account=reasoning --qos=priority-reasoning \
#        --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:20:00 \
#        bash experiments/rgmix_value_baselines/render.sh mean-k8 tether_pos-k2 ...
#
# For each NAME = {baseline}-k{K} it writes ~/rgmix_runs/NAME/{configs,rl.sbatch}
# (a --dry-run render via the `rl` entrypoint + slurm overlay). Submit each with:
#   sbatch ~/rgmix_runs/NAME/rl.sbatch
# Value baselines (everything except `mean`) load the warmed value checkpoint.
set -e
cd "$(cd "$(dirname "$0")/../.." && pwd)"   # repo root
export CUDA_HOME=$HOME/cuda-12.8
export PATH=$HOME/.local/bin:$CUDA_HOME/bin:$PATH
source .venv/bin/activate

CKPT=$HOME/rgmix_runs/warmup/checkpoints/value_warmup_checkpoint

for name in "$@"; do
  base=${name%-k*}          # mean | pure_value | mixture_linear | mixture_position | tether | tether_pos
  K=${name##*-k}            # 2 | 4 | 8
  ckpt_arg=""
  [ "$base" != "mean" ] && ckpt_arg="--trainer.value-function.init-checkpoint $CKPT"
  rm -rf "$HOME/rgmix_runs/$name"
  uv run --no-sync rl @ configs/rg_mix/rl.toml @ "configs/rg_mix/$base.toml" @ configs/rg_mix/slurm.toml \
    --slurm.job-name "$name" --orchestrator.group-size "$K" \
    --wandb.name "$name" --wandb.group "k$K" \
    --output-dir "$HOME/rgmix_runs/$name" $ckpt_arg --dry-run
  echo "rendered $name"
done
