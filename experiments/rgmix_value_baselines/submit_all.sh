#!/bin/bash
# Submit all 24 RG-Mix value-corrected GRPO experiments:
#   4 baselines x K in {2,4,8} x 2 seeds, 500 steps each, one H100 node per run.
# W&B runs are grouped by K (group="k{K}", name="{baseline}-k{K}-s{seed}") for
# side-by-side comparison of the baselines within each group size.
#
# Value baselines (pure_value, mixture_linear, mixture_position) load the warmed
# value checkpoint from the warmup run. Pass its path via INIT_CKPT:
#   INIT_CKPT=/path/to/value_checkpoint bash experiments/rgmix_value_baselines/submit_all.sh
set -eu

REPO=$HOME/tba-prime
SB="$REPO/experiments/rgmix_value_baselines/launch.sbatch"
LOGDIR=$HOME/rgmix_runs/logs
mkdir -p "$LOGDIR"

INIT_CKPT="${INIT_CKPT:-}"
if [ -z "$INIT_CKPT" ]; then
  echo "WARNING: INIT_CKPT is empty; value baselines will train the value model from scratch (no warm start)." >&2
fi

submit () {
  local baseline=$1 k=$2 seed=$3 initckpt=$4
  local name="${baseline}-k${k}-s${seed}"
  sbatch -J "$name" \
    --output="$LOGDIR/${name}_%j.log" --error="$LOGDIR/${name}_%j.log" \
    --export=ALL,BASELINE=$baseline,K=$k,SEED=$seed,INIT_CKPT=$initckpt \
    "$SB"
}

for k in 2 4 8; do
  for seed in 1 2; do
    submit loo              "$k" "$seed" ""
    submit pure_value       "$k" "$seed" "$INIT_CKPT"
    submit mixture_linear   "$k" "$seed" "$INIT_CKPT"
    submit mixture_position "$k" "$seed" "$INIT_CKPT"
  done
done

echo "Submitted 24 runs. Monitor with: squeue --me"
