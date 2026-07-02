# Value Functions

Value functions are optional trainer-local baselines. They are separate from
algorithm reference models such as OPD teachers: the value model lives inside
the trainer process group, trains only from environment reward, and can provide
GAE advantages for the policy-gradient loss.

Use this page for both modes:

- **RL value functions** inside `uv run rl`.
- **Standalone static-policy value training** with `uv run static-value`, used for
  offline diagnostics and warm-starting value checkpoints.

## RL Value Functions

Enable the value model by adding `[trainer.value_function]` to an RL config:

```toml
[trainer.value_function]
gamma = 1.0
gae_lambda = 1.0
updates_per_step = 1

[trainer.value_function.loss]
type = "mse"
```

`RLConfig` then enables the matching orchestrator path automatically:

- `orchestrator.value_function = true`, so rollout batches carry
  `value_rewards` and `value_dones`.
- `orchestrator.value_warmup.steps = 50` by default, unless
  `trainer.value_function.init_checkpoint` is set.
- Full trainer and orchestrator checkpoints are enabled for value warmup if no
  checkpoint config was provided.

The value model is initialized from the policy base model at startup. It uses
the same transformer trunk and replaces the language-model head with a value
head. The full value network is trained, not just the head.

### Targets and GAE

The orchestrator stamps terminal environment reward onto the last sampled token
of each rollout and zeros everywhere else. These streams are raw environment
reward only; KL penalties, algorithm-shaped advantages, OPD teacher scores, and
other policy-loss terms are not value targets.

The trainer computes GAE over sampled/action tokens inside each packed sequence:

```text
value_rewards/value_dones -> current value prediction -> GAE advantages + returns
```

By default `trainer.value_function.use_gae = true`, so the RL loss uses the
trainer-computed GAE advantage instead of the orchestrator's group-relative
advantage stream. Set it to `false` to train the value model while leaving the
policy loss on the orchestrator-provided advantages.

Defaults are intentionally Monte Carlo-like:

```toml
[trainer.value_function]
gamma = 1.0
gae_lambda = 1.0
```

With `gae_lambda = 1.0`, the estimator has no lambda bias but keeps high
variance.

### Warmup

Value warmup sends value-only batches before normal policy training:

```toml
[orchestrator.value_warmup]
steps = 50
batch_size = 256        # optional; defaults to orchestrator.batch_size
# token_batch_size = 8192  # alternative to batch_size

[trainer.value_function]
warmup_updates_per_batch = 1
```

Warmup batches do not advance `progress.step`, do not update the policy, and do
not create async trainer/inference lag. The last warmup batch requests a full
trainer checkpoint, and the orchestrator saves its checkpoint after the trainer
checkpoint is stable.

To skip warmup from a pre-trained value checkpoint:

```toml
[trainer.value_function]
init_checkpoint = "/path/to/value_checkpoint"

[orchestrator.value_warmup]
steps = 0
```

If `init_checkpoint` is set and no value warmup config is provided, warmup
defaults to `0`. A value checkpoint cannot be combined with `ckpt.resume_step`
because normal trainer resume already restores the value model, optimizer, and
scheduler from the full training checkpoint.

### Loss Modes

Two value losses are supported.

**MSE** predicts a scalar value directly:

```toml
[trainer.value_function.loss]
type = "mse"
```

**Classification** predicts a categorical distribution and uses the expected
reward of that distribution as the scalar value:

```toml
[trainer.value_function.loss]
type = "classification"
reward_range = [0.0, 1.0]
n_bins = 1
```

For `n_bins = 1`, the value head has two logits and trains binary
classification with the midpoint of `reward_range` as the class split. For
`n_bins > 1`, the bins are linearly spaced across `reward_range`. Targets
outside `reward_range` raise an error; only tiny floating-point boundary
tolerance is accepted and clamped.

### Optimizer and Scheduler

The value function has its own optimizer and scheduler:

```toml
[trainer.value_function.optim]
type = "adamw"
lr = 5e-5
weight_decay = 0.01
max_norm = 1.0

[trainer.value_function.scheduler]
type = "linear"
warmup_steps = 50
decay_steps = 0
min_lr = 0.0
```

The policy optimizer/scheduler are unchanged. During normal RL training, the
trainer first runs the policy update and then runs
`trainer.value_function.updates_per_step` value optimizer updates on the same
batch. During warmup, only value updates run.

### Checkpoints

Full trainer checkpoints include:

- policy model, optimizer, scheduler, and progress;
- value model, optimizer, and scheduler when value functions are enabled.

Warmup always saves after the final warmup batch when checkpointing is
available. Value-only checkpoints created by standalone training can be loaded
through `trainer.value_function.init_checkpoint`.

### Current Limits

Value functions currently reject:

- LoRA training;
- `max_concurrent_runs > 1`;
- VLM training;
- `ckpt.weights_only = true`.

Classification mode additionally requires bounded rewards through
`reward_range`.

## Standalone Static-Policy Value Training

`uv run static-value` is a staged offline diagnostic runner. It keeps the policy
fixed, generates tokenized rollouts, trains only the value function, then
evaluates value-baseline variance proxies on held-out rollouts.

The shipped RGMix config is:

```bash
uv run static-value @ examples/static_value_rg_mix/static_value.toml --dry-run
```

On Perlmutter, use the two-node wrapper:

```bash
bash experiments/static_value_diagnostics/run_two_node.sh
```

The wrapper requests two premium GPU nodes and delegates the experiment contract
to `examples/static_value_rg_mix/static_value.toml`. The `static-value` runner
itself writes the resolved config to `<output_dir>/configs/static_value.toml`
and per-node inference configs to the same directory.

The value-training stage can use one or more allocated nodes:

```toml
[train]
global_batch_size = 256
micro_batch_tokens = 32768  # optional; defaults to model.seq_len
num_nodes = 2               # value trainer/predict nodes

[diagnostics]
position_bucket_edges = [0, 512, 1024, 2048, 4096, 6144, 8192]
```

`micro_batch_tokens` is the packed-token budget for each value
forward/backward. Individual rollouts are still clipped by `model.seq_len`;
larger values just pack multiple shorter sequences into one varlen forward when
memory allows.

With the default `gae_lambda = 1.0`, static value targets are Monte Carlo
return-to-go values, so the trainer skips the extra no-grad value forward that
would otherwise be needed for bootstrapped lambda returns.

Position diagnostics always include fractional early/middle/late buckets. The
absolute generated-token buckets are configurable through
`diagnostics.position_bucket_edges`; the RGMix example uses 512-token and wider
buckets because generated outputs are usually thousands of tokens long.

### Stages

`static-value` runs these stages:

1. `collect_train` starts fixed-policy inference on both nodes and writes
   tokenized train rollouts.
2. `train_value` runs the value trainer from saved tokenized rollouts.
3. `collect_eval` starts fixed-policy inference again and writes held-out
   validation/test rollouts.
4. `predict` loads `value/value_checkpoint` and writes per-token value
   predictions for held-out rollouts.
5. `diagnostics` computes rho selection, variance proxy summaries, and plots.

You can run one stage with `--stage collect_train`, `--stage train_value`, etc.
The normal path is `stage = "all"`.

### Outputs

The default layout is:

```text
<output_dir>/
|-- configs/static_value.toml
|-- data/train_rollouts.jsonl
|-- data/eval_rollouts.jsonl
|-- logs/
|-- value/value_checkpoint/
|-- value_eval/predictions_{val,test}_rank*.npz
`-- diagnostics/
```

Rollout JSONL files store token ids, masks, rollout logprobs, rewards, prompt
ids, rollout ids, and group ids. The value trainer consumes these tokenized
records directly, so there is no retokenization drift between collection and
training.

### Default RGMix Contract

The default example uses:

- `Qwen/Qwen3-4B-Instruct-2507`;
- sequence length `8192`;
- FlashAttention 3 on Perlmutter H100 nodes;
- ancestral sampling: `temperature = 1.0`, `top_p = 1.0`, `top_k = -1`,
  `min_p = 0.0`;
- `10_000` train episodes;
- `1_024` held-out episodes split equally into validation and test;
- `group_size = 8`;
- classifier value loss with `reward_range = [0.0, 1.0]`, `n_bins = 1`;
- `100` value-training steps with global batch size `256`;
- two value-training nodes with `micro_batch_tokens = 32768`.
- absolute position diagnostics at generated-token edges
  `[0, 512, 1024, 2048, 4096, 6144, 8192]`.

This keeps default activation checkpointing and `model.dp_replicate = 4`.
On Qwen3-4B with 8192-token rollouts, disabling activation checkpointing OOMs
at useful packed-token budgets, and full replication (`dp_replicate = 8`) is no
faster than the default 4x2 replicate/shard mesh.

Train prompts start at dataset offset `0`. Held-out prompts start at offset
`7000` in the saved RGMix dataset to avoid train/eval overlap.

### Static Runner Limits

The staged static runner is currently intentionally narrower than the RL
launcher:

- exactly two Slurm nodes;
- no LoRA;
- no `torch.compile`;
- no activation offloading;
- no VLM or multimodal value training;
- classifier value loss only.

These limits are validated in `StaticValueConfig` so unsupported config fields
fail early instead of being ignored.
