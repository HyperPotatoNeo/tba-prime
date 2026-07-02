# Static Value Diagnostics

Offline/static-policy diagnostics for value-function baselines on RGMix.

The policy is fixed at `Qwen/Qwen3-4B-Instruct-2507`. Rollouts are collected from the saved RGMix dataset at `/pscratch/sd/s/siddart2/datasets/rg_mix_7500` with ancestral sampling (`temperature=1`, `top_p=1`, `top_k=-1`, `min_p=0`) while a classifier value model, initialized from the same base LM trunk, trains on terminal environment reward labels for generated tokens. The policy is never updated.

Diagnostics are evaluated on held-out prompts. `rho` is selected on the validation split and reported on the test split; test rho curves are written separately as descriptive curves. The primary baseline comparison is leave-one-out group reward, not the self-including group mean.

Outputs under each run directory:

- `data/rollouts.jsonl`: collected static-policy rollouts. The launcher writes only complete groups of usable rollouts; failures go to `data/collection_errors.jsonl`.
- `value/value_checkpoint`: distributed value-model checkpoint.
- `value/predictions_{val,test}_rank*.npz`: per-token expected values and binary odds logits.
- `diagnostics/diagnostics.json`: variance proxy summaries and rho selections.
- `diagnostics/position_summary.csv`: early/middle/late and fixed-position buckets.
- `diagnostics/group_size_sensitivity.csv`: rollout-count sensitivity with resampled groups.
- `diagnostics/plots/*.png`: summary plots.

Submit the default premium-QOS sweep on Perlmutter:

```bash
cd /pscratch/sd/s/siddart2/value-functions-prime-rl/prime-rl
bash experiments/static_value_diagnostics/run_sweep.sh
```

Submit one run manually:

```bash
sbatch --export=ALL,EXPERIMENT_NAME=rgmix-qwen4b-classifier,INFER_GPUS=2,VALUE_GPUS=2 \
  experiments/static_value_diagnostics/launch_one_node.sbatch
```

The Slurm script requests one full GPU node with `-A m4881`, `-C "gpu&hbm80g"`, and `--qos=premium`. The default sweep uses 64 train prompt groups, 16 validation groups, 16 test groups, group size 16, 8192 sequence length, global value batch size 256, value microbatch size 4, disabled value-model `torch.compile`, and 50 value steps. The default `all` stage keeps inference running while value training streams from the growing rollout file; use `STAGE=offline` for collect-then-train debugging. `run_sweep.sh` submits the stable 3-inference/1-value split by default; set `INCLUDE_2GPU_VALUE_SPLIT=1` to also submit the experimental 2-inference/2-value split.
