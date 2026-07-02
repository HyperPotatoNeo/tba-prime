# RG-Mix value-corrected GRPO baselines

Online RL experiments comparing group-relative and value-corrected advantage
estimators on the RG-Mix environment, following the value-corrected GRPO handoff.

## Baselines

| overlay | estimator |
| --- | --- |
| `loo` | leave-one-out GRPO, `A = R - B_loo` (no value function) |
| `pure_value` | pure value, `A = R - V(s_t)` (GAE, gamma=lambda=1) |
| `mixture_linear` | linear mixture, constant `rho=0.5`: `A = R - [(1-rho) B_loo + rho V_t]` |
| `mixture_position` | position-aware mixture, `rho(t)` linear 0->1 along the response |

Grid: 4 baselines x K in {2, 4, 8} x 2 seeds = 24 runs, 500 steps, one H100 node
each (4 inference / 4 trainer; value function co-located on the trainer GPUs).
Batch size 128, seq_len 8192, ancestral sampling (temp 1, top_p 1, top_k off),
policy lr 3e-6, full fine-tuning, checkpoint every 50 steps.

## Prerequisites

- Env installed: `uv sync --extra rgmix --extra flash-attn`.
- Fixed dataset built once: `python environments/rg_mix/build_dataset.py --out ~/datasets/rg_mix_10k --total 10500`.
- `~/.env` on the cluster exports `WANDB_API_KEY` (not committed).

## Run

```bash
# 1. Warm up the value function once (100 value-only steps) and note the value checkpoint path.
sbatch experiments/rgmix_value_baselines/warmup.sbatch

# 2. Submit all 24 runs; value baselines load the warmed value checkpoint.
INIT_CKPT=/path/to/warmup/value_checkpoint bash experiments/rgmix_value_baselines/submit_all.sh
```

W&B: project `rgmix-value-grpo`, grouped by K (`group="k{K}"`, `name="{baseline}-k{K}-s{seed}"`).
