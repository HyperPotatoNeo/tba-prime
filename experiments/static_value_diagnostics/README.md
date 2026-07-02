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

Submit the two-node throughput run, with 4 inference GPUs on one node and 4 value-trainer GPUs on the second node:

```bash
cd /pscratch/sd/s/siddart2/value-functions-prime-rl/prime-rl
bash experiments/static_value_diagnostics/run_two_node.sh
```

Submit one run manually:

```bash
sbatch --export=ALL,EXPERIMENT_NAME=rgmix-qwen4b-classifier,INFER_GPUS=2,VALUE_GPUS=2 \
  experiments/static_value_diagnostics/launch_one_node.sbatch
```

The one-node Slurm script requests one full GPU node with `-A m4881`, `-C "gpu&hbm80g"`, `--qos=premium`, and a 48h time limit. The two-node script requests two full GPU nodes with the same account/QOS. The default sweep uses group size 8, 32 validation groups, 32 test groups, 8192 sequence length, global value batch size 256, disabled value-model `torch.compile`, and 300 value steps. Train rollout groups default to `ceil(VALUE_STEPS * GLOBAL_BATCH_SIZE / GROUP_SIZE)`, so the default 300-step run targets 9600 train groups / 76800 train rollouts. The saved prompt pool is capped by the dataset size and cycled when more train groups than unique prompts are requested. Value training logs step seconds, forward-token throughput, and an approximate MFU. The default `all` stage keeps inference running while value training streams from the growing rollout file and reloads new rollouts every value step; use `STAGE=offline` for collect-then-train debugging. `run_sweep.sh` submits the stable 3-inference/1-value split by default; set `INCLUDE_2GPU_VALUE_SPLIT=1` to also submit the experimental 2-inference/2-value split.
