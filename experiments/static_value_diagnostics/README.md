# Static Value Diagnostics

Offline/static-policy diagnostics for value-function baselines on RGMix.

The policy is fixed at `Qwen/Qwen3-4B-Instruct-2507`. Rollouts are collected from the saved RGMix dataset at `/pscratch/sd/s/siddart2/datasets/rg_mix_7500` with ancestral sampling (`temperature=1`, `top_p=1`, `top_k=-1`, `min_p=0`) while a classifier value model, initialized from the same base LM trunk, trains on terminal environment reward labels for generated tokens. The policy is never updated.

Diagnostics are evaluated on held-out prompts. `rho` is selected on the validation split and reported on the test split; test rho curves are written separately as descriptive curves. The primary baseline comparison is leave-one-out group reward, not the self-including group mean.

Outputs under each run directory:

- `data/train_rollouts.jsonl`: 10,000 tokenized static-policy train rollouts collected before value training.
- `data/eval_rollouts.jsonl`: 1,024 tokenized held-out rollouts collected after value training (`512` val, `512` test by default).
- `value/value_checkpoint`: distributed value-model checkpoint, including optimizer and scheduler state.
- `value_eval/predictions_{val,test}_rank*.npz`: per-token expected values and binary odds logits loaded from the saved value checkpoint.
- `diagnostics/diagnostics.json`: variance proxy summaries and rho selections.
- `diagnostics/position_summary.csv`: early/middle/late and configured absolute-position buckets.
- `diagnostics/group_size_sensitivity.csv`: rollout-count sensitivity with resampled groups.
- `diagnostics/plots/*.png`: summary plots.

The staged runner is exposed as a first-class entrypoint:

```bash
cd /pscratch/sd/s/siddart2/value-functions-prime-rl/prime-rl
uv run static-value @ examples/static_value_rg_mix/static_value.toml --dry-run
```

The default two-node Perlmutter job uses premium QOS. It uses both nodes for bulk inference, trains the value function offline for 100 steps on one node, then uses both nodes again for held-out inference and checkpointed prediction/diagnostics:

```bash
cd /pscratch/sd/s/siddart2/value-functions-prime-rl/prime-rl
bash experiments/static_value_diagnostics/run_two_node.sh
```

The experiment contract lives in `examples/static_value_rg_mix/static_value.toml`; the Slurm script is only a NERSC wrapper. The runner writes the resolved config to `configs/static_value.toml` in the run directory, plus per-node inference configs in the same directory. To change steps, batch size, sampling, prompt offsets, value loss, W&B project, or diagnostics settings, edit/copy the TOML or pass normal config CLI overrides to `static-value`.

Submit one run manually:

```bash
sbatch --export=ALL,EXPERIMENT_NAME=rgmix-qwen4b-classifier,INFER_GPUS=2,VALUE_GPUS=2 \
  experiments/static_value_diagnostics/launch_one_node.sbatch
```

The one-node Slurm script requests one full GPU node with `-A m4881`, `-C "gpu&hbm80g"`, `--qos=premium`, and a 48h time limit. The staged two-node script requests two full GPU nodes with the same account/QOS and a 24h time limit. Defaults are group size 8, 8192 sequence length, 10,000 train episodes, 1,024 held-out episodes, global value batch size 256, disabled value-model `torch.compile`, and 100 offline value steps. The train set is fixed tokenized data and is cycled by the trainer when `train.steps * train.global_batch_size` exceeds 10,000. Held-out collection uses explicit prompt offsets near the end of the saved 7,500-example RGMix dataset to avoid train/eval overlap.
