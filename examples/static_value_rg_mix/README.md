# Static Value RGMix

This example trains and evaluates a value-function baseline with a fixed
`Qwen/Qwen3-4B-Instruct-2507` policy on saved RGMix data. It is not an RL run:
the policy is never updated. The runner first collects tokenized train
rollouts, trains only the value model, collects held-out rollouts, predicts
values from the saved value checkpoint, and writes diagnostic summaries.

Start with a dry run:

```bash
uv run static-value @ examples/static_value_rg_mix/static_value.toml --dry-run
```

On Perlmutter, the maintained launch path is the two-node premium-QOS wrapper:

```bash
bash experiments/static_value_diagnostics/run_two_node.sh
```

The config defaults to:

- `10_000` train episodes;
- `1_024` held-out episodes split evenly into validation and test;
- `group_size = 8`;
- classifier value loss with `reward_range = [0.0, 1.0]` and `n_bins = 1`;
- `100` value-training steps with global batch size `256`;
- two value-training nodes with `micro_batch_tokens = 32768`;
- FlashAttention 3 on Perlmutter H100 nodes;
- ancestral sampling: `temperature = 1.0`, `top_p = 1.0`, `top_k = -1`,
  `min_p = 0.0`.

See [Value Functions](../../docs/value-functions.md) for the full RL and
standalone value-function guide.
