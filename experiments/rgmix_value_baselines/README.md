# RG-Mix value-corrected GRPO baselines

Online RL experiments comparing group-relative and value-corrected advantage
estimators on the RG-Mix environment, following the value-corrected GRPO handoff.

## Baselines

| overlay | estimator |
| --- | --- |
| `mean` | group-mean GRPO, `A = R - B_mean` (no value function) |
| `pure_value` | pure value, `A = R - V(s_t)` (GAE, gamma=lambda=1) |
| `mixture_linear` | linear mixture, constant `rho=0.5`: `A = R - [(1-rho) B_mean + rho V_t]` |
| `mixture_position` | position-aware linear mixture, `rho(t)` linear 0->1 along the response |
| `tether` | TETHER (global): group anchor + clipped two-factor value correction, `b = clip(B_mean + alpha(V_0-B_mean) + rho(V_t-V_0), 0, 1)`, `alpha=0.5 rho=0.8` |
| `tether_pos` | TETHER (position): the same correction gated by response position `u_t` |

Grid: 6 baselines x K in {2, 4, 8}, single seed = 18 runs, 500 steps. Model
Qwen3-4B-Instruct-2507, batch 256, seq_len 8192, ancestral sampling (temp 1,
top_p 1, top_k off), policy lr 1e-6, full fine-tuning, binary-classifier value
head (reward range [0,1]), checkpoint every 50 steps. Value baselines load a
warmed critic checkpoint at launch.

## Deployment (BLC GB200)

Each run is a router-free 2-node job: 1 inference node (single vLLM server, dp=4)
+ 1 trainer node (policy + value FSDP-sharded), orchestrator co-located on the
trainer node pointing straight at the inference node — no `vllm-router` (its wheel
is x86-only). Rendered from `configs/rg_mix/slurm.toml` +
[`blc/router_free_2node.sbatch.j2`](blc/router_free_2node.sbatch.j2). Trainer uses
HSDP `dp_replicate=2`; the launcher wipes stale weight-broadcasts on (re)start and
excludes the faulted rack p4-r12.

## Prerequisites

- Env installed on a GB200 (aarch64) node: `uv sync --extra rgmix` (+ source-built
  flash-attn; see the top-level install skill). Always run with `uv run --no-sync`.
- Fixed dataset built once: `uv run python environments/rg_mix/build_dataset.py --out ~/datasets/rg_mix_10k --total 10500`.
- `~/.env` on the cluster exports `WANDB_API_KEY` (not committed).

## Run

```bash
# 1. Warm the value function once (single-node value-only run). Exports the critic
#    to ~/rgmix_runs/warmup/checkpoints/value_warmup_checkpoint.
sbatch experiments/rgmix_value_baselines/warmup.sbatch

# 2. Render (on an aarch64 node) + submit all 18 runs.
bash experiments/rgmix_value_baselines/submit_all.sh
```

To (re)launch a single run, e.g. after a node failure:

```bash
srun --partition=gb200 --account=reasoning --qos=priority-reasoning \
     --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:20:00 \
     bash experiments/rgmix_value_baselines/render.sh tether_pos-k2
sbatch ~/rgmix_runs/tether_pos-k2/rl.sbatch
```

## Reading results

Plot the wandb metric `train/rg-mix/all/reward/mean` (the mean over all rollouts),
not the orchestrator text-log `Reward` (which is the filtered effective-batch mean).
