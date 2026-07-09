#!/bin/bash
# Submit all 18 RG-Mix value-corrected GRPO runs (router-free 2-node, BLC GB200):
#   6 baselines x K in {2,4,8}, single seed, 500 steps, 2 nodes each
#   (1 inference node dp=4 + 1 trainer node, policy+value FSDP-sharded).
# W&B: project `rgmix-value-grpo`, grouped by K (group="k{K}", name="{baseline}-k{K}").
#
# Run the value warmup first (produces the critic checkpoint every value baseline
# loads): sbatch experiments/rgmix_value_baselines/warmup.sbatch
#
# Rendering must run on an aarch64 node (login is x86), so this srun's render.sh,
# then sbatches each rendered job. Run from the repo root on the BLC login node.
set -eu
cd "$(cd "$(dirname "$0")/../.." && pwd)"

NAMES=""
for base in mean pure_value mixture_linear mixture_position tether tether_pos; do
  for K in 2 4 8; do NAMES="$NAMES ${base}-k${K}"; done
done

echo "Rendering $(echo "$NAMES" | wc -w) runs on an aarch64 node..."
srun --partition=gb200 --account=reasoning --qos=priority-reasoning \
     --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:20:00 \
     bash experiments/rgmix_value_baselines/render.sh $NAMES

echo "Submitting..."
for name in $NAMES; do
  sbatch "$HOME/rgmix_runs/$name/rl.sbatch"
  sleep 15   # stagger to avoid a vLLM startup thundering-herd
done
echo "Submitted 18 runs. Monitor with: squeue --me"
