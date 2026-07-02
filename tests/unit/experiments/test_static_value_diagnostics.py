import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.static_value_diagnostics.common import RolloutRecord
from experiments.static_value_diagnostics.diagnostics import (
    PredictionSet,
    bucket_masks,
    build_token_table,
    has_binary_rewards,
    method_prediction,
    mixed_methods,
    no_intercept_rho,
    rho_methods,
    select_mixed_params,
    select_rhos,
    summary_at_rhos,
    variance_proxy,
)
from prime_rl.configs.static_value import StaticValueConfig
from prime_rl.entrypoints.static_value import StaticValueRunner


def _prediction_set() -> PredictionSet:
    # Two prompts, two rollouts per prompt, two generated-token predictions per rollout.
    rewards = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    values = np.repeat(rewards, 2).astype(np.float32)
    offsets = np.asarray([0, 2, 4, 6, 8], dtype=np.int64)
    return PredictionSet(
        prompt_id=np.asarray([0, 0, 1, 1], dtype=np.int64),
        rollout_id=np.asarray([0, 1, 0, 1], dtype=np.int64),
        reward=rewards,
        offsets=offsets,
        values=values,
        logits=np.log(np.clip(values, 1e-6, 1 - 1e-6) / np.clip(1 - values, 1e-6, 1)).astype(np.float32),
        positions=np.asarray([0, 1] * 4, dtype=np.int32),
        gen_lengths=np.asarray([2, 2, 2, 2], dtype=np.int32),
        initial_value=np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        initial_logit=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def test_static_value_default_config_preserves_staged_contract():
    config = StaticValueConfig()

    assert config.model.name == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.model.seq_len == 8192
    assert config.model.compile is None
    assert config.model.ac is not None
    assert config.model.ac_offloading is None
    assert not config.model.optim_cpu_offload
    assert not config.model.reshard_after_forward
    assert config.data.group_size == 8
    assert config.data.train_groups == 1250
    assert config.data.val_groups == 64
    assert config.data.test_groups == 64
    assert config.data.eval_test_offset == 7064
    assert config.value_function.optim.lr == pytest.approx(5e-5)
    assert config.value_function.scheduler.warmup_steps == 50
    assert config.value_function.loss.n_bins == 1
    assert config.train.micro_batch_tokens is None
    assert config.train.num_nodes == 1
    assert config.diagnostics.mixed_step == pytest.approx(0.1)


def test_static_value_example_toml_loads_with_static_model_defaults():
    path = Path(__file__).parents[3] / "examples/static_value_rg_mix/static_value.toml"
    data = tomllib.loads(path.read_text())
    config = StaticValueConfig.model_validate(data)

    assert config.model.name == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.model.attn == "flash_attention_3"
    assert config.model.compile is None
    assert config.model.ac_offloading is None
    assert config.model.dp_replicate == 4
    assert config.data.train_episodes == 10_000
    assert config.train.steps == 100
    assert config.train.micro_batch_tokens == 32_768
    assert config.train.num_nodes == 2
    assert config.diagnostics.group_sizes == [2, 4, 8]
    assert config.diagnostics.position_bucket_edges == [0, 512, 1024, 2048, 4096, 6144, 8192]
    assert config.diagnostics.mixed_step == pytest.approx(0.1)


def test_static_value_config_rejects_prompt_overlap_and_bad_inference_layout():
    with pytest.raises(ValueError, match="overlaps"):
        StaticValueConfig.model_validate({"data": {"train_episodes": 64, "group_size": 8, "eval_val_offset": 4}})

    with pytest.raises(ValueError, match="tp \\* inference.dp"):
        StaticValueConfig.model_validate({"inference": {"gpus_per_node": 4, "tp": 2, "dp": 1}})

    with pytest.raises(ValueError, match="deployment.num_nodes"):
        StaticValueConfig.model_validate({"deployment": {"num_nodes": 3}})

    with pytest.raises(ValueError, match="model.compile"):
        StaticValueConfig.model_validate({"model": {"compile": {}}})

    with pytest.raises(ValueError, match="VLM"):
        StaticValueConfig.model_validate(
            {"model": {"vlm": {"vision_encoder_attr": "model.visual", "language_model_attr": "model.model"}}}
        )

    with pytest.raises(ValueError, match="train.num_nodes"):
        StaticValueConfig.model_validate({"train": {"num_nodes": 3}})

    with pytest.raises(ValueError, match="micro_batch_tokens"):
        StaticValueConfig.model_validate({"train": {"micro_batch_tokens": 1024}})

    with pytest.raises(ValueError, match="position_bucket_edges"):
        StaticValueConfig.model_validate({"diagnostics": {"position_bucket_edges": [0, 512, 512]}})


def test_static_value_runner_builds_multinode_value_torchrun(tmp_path, monkeypatch):
    config = StaticValueConfig.model_validate(
        {
            "output_dir": tmp_path / "out",
            "repo_dir": Path(__file__).parents[3],
            "train": {"num_nodes": 2, "micro_batch_tokens": 16_384},
            "wandb": None,
        }
    )
    runner = StaticValueRunner(config)
    captured: list[tuple[str, list[str], Path, str | None]] = []

    class DummyProc:
        def wait(self):
            return 0

    def fake_slurm_nodes(*, required=True):
        return ["node0", "node1"]

    def fake_popen_remote(node, cmd, log_path, *, cuda_visible_devices=None):
        captured.append((node, cmd, log_path, cuda_visible_devices))
        return DummyProc()

    monkeypatch.setattr(runner, "_slurm_nodes", fake_slurm_nodes)
    monkeypatch.setattr(runner, "_popen_remote", fake_popen_remote)
    monkeypatch.setattr(runner, "_wait_all", lambda procs: None)

    args = runner._value_train_args(rollouts=Path("train.jsonl"), output_dir=Path("value"), steps=1)
    assert args[args.index("--micro-batch-tokens") + 1] == "16384"
    runner._run_value_torchrun(args, "value_train")

    assert [item[0] for item in captured] == ["node0", "node1"]
    assert all("--nnodes" in cmd and cmd[cmd.index("--nnodes") + 1] == "2" for _, cmd, _, _ in captured)
    assert all("--standalone" not in cmd for _, cmd, _, _ in captured)
    assert captured[0][1][captured[0][1].index("--node-rank") + 1] == "0"
    assert captured[1][1][captured[1][1].index("--node-rank") + 1] == "1"
    assert all(cuda_visible_devices == "0,1,2,3" for _, _, _, cuda_visible_devices in captured)


def test_static_value_runner_writes_inference_config_without_model_chat_template(tmp_path):
    config = StaticValueConfig.model_validate({"output_dir": tmp_path / "out", "wandb": None})
    runner = StaticValueRunner(config)
    runner._prepare_output()
    path = runner._write_inference_config(0)
    data = tomllib.loads(path.read_text())

    assert data["model"]["name"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert data["model"]["max_model_len"] == 8192
    assert "chat_template" not in data["model"]


def test_loo_and_group_mean_baselines_are_distinct():
    table = build_token_table(_prediction_set(), group_size=2)

    assert np.all(table.group_mean == pytest.approx(0.5))
    assert table.loo.tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    assert variance_proxy(table, "group_mean") == pytest.approx(0.25)
    assert variance_proxy(table, "loo") == pytest.approx(1.0)
    assert variance_proxy(table, "pure_value") == pytest.approx(0.0)


def test_linear_rho_uses_no_intercept_fit():
    table = build_token_table(_prediction_set(), group_size=2)
    rho = no_intercept_rho(table.value - table.loo, table.reward - table.loo)

    assert rho == pytest.approx(1.0)
    selected = select_rhos(table, np.asarray([0.0, 0.5, 1.0]))
    assert selected["linear"] == pytest.approx(1.0)
    assert summary_at_rhos(table, selected)["linear"]["variance"] == pytest.approx(0.0)


def test_anchored_odds_uses_binary_logit_difference():
    table = build_token_table(_prediction_set(), group_size=2)
    pred0 = method_prediction(table, "anchored_odds", 0.0)
    pred1 = method_prediction(table, "anchored_odds", 1.0)

    assert pred0.tolist() == pytest.approx(table.odds_prior.tolist())
    assert pred1[:2].mean() > 0.99
    assert pred1[2:4].mean() < 0.01


def test_mixed_baselines_select_alpha_and_rho_on_validation_grid():
    table = build_token_table(_prediction_set(), group_size=2)
    methods = rho_methods(include_odds=True)
    mixed = mixed_methods(include_odds=True)
    grid = np.asarray([0.0, 1.0])
    selected = select_rhos(table, grid, methods)
    selected_mixed = select_mixed_params(table, grid, grid, mixed)
    summary = summary_at_rhos(table, selected, methods, selected_mixed)

    assert selected_mixed["mixed_add"]["alpha"] == pytest.approx(1.0)
    assert selected_mixed["mixed_add"]["rho"] == pytest.approx(1.0)
    assert summary["mixed_add"]["variance"] == pytest.approx(0.0)
    assert selected_mixed["mixed_odds"]["alpha"] == pytest.approx(1.0)
    assert selected_mixed["mixed_odds"]["rho"] == pytest.approx(1.0)
    assert summary["mixed_odds"]["variance"] < 1e-10


def test_position_buckets_scale_to_long_rollouts():
    table = build_token_table(_prediction_set(), group_size=2)
    table = table.__class__(
        reward=table.reward,
        group_mean=table.group_mean,
        loo=table.loo,
        value=table.value,
        value0=table.value0,
        logit=table.logit,
        logit0=table.logit0,
        odds_prior=table.odds_prior,
        position=np.asarray([0, 511, 512, 1023, 1024, 2047, 4096, 7679], dtype=np.int32),
        frac_position=table.frac_position,
        group_id=table.group_id,
        rollout_id=table.rollout_id,
    )
    masks = bucket_masks(table, [0, 512, 1024, 2048, 4096, 6144, 8192])

    assert "pos_0000_0512" in masks
    assert "pos_0512_1024" in masks
    assert "pos_4096_6144" in masks
    assert "pos_000_032" not in masks
    assert masks["pos_0000_0512"].sum() == 2
    assert masks["pos_4096_6144"].sum() == 1


def test_fractional_rewards_skip_binary_odds_methods():
    pred = _prediction_set()
    pred = PredictionSet(
        prompt_id=pred.prompt_id,
        rollout_id=pred.rollout_id,
        reward=np.asarray([1.0, 0.01, 1.0, 0.0], dtype=np.float32),
        offsets=pred.offsets,
        values=pred.values,
        logits=pred.logits,
        positions=pred.positions,
        gen_lengths=pred.gen_lengths,
        initial_value=pred.initial_value,
        initial_logit=pred.initial_logit,
    )
    table = build_token_table(pred, group_size=2)
    methods = rho_methods(include_odds=has_binary_rewards(pred))
    mixed = mixed_methods(include_odds=has_binary_rewards(pred))
    selected = select_rhos(table, np.asarray([0.0, 0.5, 1.0]), methods)
    selected_mixed = select_mixed_params(table, np.asarray([0.0, 0.5, 1.0]), np.asarray([0.0, 0.5, 1.0]), mixed)
    summary = summary_at_rhos(table, selected, methods, selected_mixed)

    assert "anchored_odds" not in methods
    assert "mixed_odds" not in mixed
    assert "odds_prior" not in summary
    assert "anchored_add_clipped" in summary
    assert "mixed_add" in summary


def test_training_sample_from_record_stamps_terminal_value_reward():
    pytest.importorskip("torchtitan")
    from experiments.static_value_diagnostics import train_static_value

    record = RolloutRecord(
        split="train",
        prompt_id=0,
        rollout_id=0,
        group_id="train:0",
        reward=1.0,
        token_ids=[11, 22, 33, 44],
        mask=[False, False, True, True],
        logprobs=[0.0, 0.0, -0.1, -0.2],
        num_output_tokens=2,
    )

    sample = train_static_value.training_sample_from_record(record)
    [micro_batch] = train_static_value.prepare_value_micro_batches(
        [record],
        seq_len=16,
        dp_rank=0,
        dp_world_size=1,
        bin_cost=lambda seqlens: sum(seqlens),
        pad_to_multiple_of=1,
    )

    assert sample.rl_weights == [0.0, 0.0, 0.0, 0.0]
    assert sample.value_rewards == [0.0, 0.0, 0.0, 1.0]
    assert sample.value_dones == [False, False, False, True]
    assert micro_batch["loss_mask"].tolist() == [[False, False, True, True]]
    assert micro_batch["value_rewards"].tolist() == [[0.0, 0.0, 0.0, 1.0]]


def test_prepare_value_micro_batches_uses_prime_packer_sequence_resets():
    pytest.importorskip("torchtitan")
    from experiments.static_value_diagnostics import train_static_value

    records = [
        RolloutRecord(
            split="train",
            prompt_id=0,
            rollout_id=0,
            group_id="train:0",
            reward=1.0,
            token_ids=[11, 22, 33, 44],
            mask=[False, False, True, True],
            logprobs=[0.0, 0.0, -0.1, -0.2],
            num_output_tokens=2,
        ),
        RolloutRecord(
            split="train",
            prompt_id=1,
            rollout_id=0,
            group_id="train:1",
            reward=0.0,
            token_ids=[55, 66, 77],
            mask=[False, True, True],
            logprobs=[0.0, -0.3, -0.4],
            num_output_tokens=2,
        ),
    ]

    [micro_batch] = train_static_value.prepare_value_micro_batches(
        records,
        seq_len=4,
        dp_rank=0,
        dp_world_size=1,
        bin_cost=lambda seqlens: sum(seqlens),
        pad_to_multiple_of=1,
        micro_batch_tokens=8,
    )

    assert micro_batch["input_ids"].shape == (1, 7)
    assert micro_batch["position_ids"].tolist() == [[0, 1, 2, 3, 0, 1, 2]]
    assert micro_batch["sequence_lengths"] == [4, 3]
    assert micro_batch["loss_mask"].tolist() == [[False, False, True, True, False, True, True]]
    assert micro_batch["value_dones"].tolist() == [[False, False, False, True, False, False, True]]


def test_static_value_targets_match_terminal_reward_after_packing():
    pytest.importorskip("torchtitan")
    from experiments.static_value_diagnostics import train_static_value
    from prime_rl.trainer.value import compute_gae

    records = [
        RolloutRecord(
            split="train",
            prompt_id=i,
            rollout_id=0,
            group_id=f"train:{i}",
            reward=float(i),
            token_ids=list(range(6)),
            mask=[False, True, True, True, True, True],
            logprobs=[0.0] * 6,
            num_output_tokens=5,
        )
        for i in range(2)
    ]

    [micro_batch] = train_static_value.prepare_value_micro_batches(
        records,
        seq_len=16,
        dp_rank=0,
        dp_world_size=1,
        bin_cost=lambda seqlens: sum(seqlens),
        pad_to_multiple_of=1,
    )
    values = torch.zeros_like(micro_batch["value_rewards"])
    _, returns = compute_gae(
        rewards=micro_batch["value_rewards"],
        dones=micro_batch["value_dones"],
        values=values,
        mask=micro_batch["loss_mask"],
        sequence_lengths=micro_batch["sequence_lengths"],
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert returns[micro_batch["loss_mask"]].tolist() == pytest.approx([0.0] * 5 + [1.0] * 5)


def test_rollout_validation_catches_bad_tokenized_groups():
    pytest.importorskip("torchtitan")
    from experiments.static_value_diagnostics import train_static_value
    from prime_rl.configs.trainer import ValueFunctionConfig

    vconfig = ValueFunctionConfig.model_validate(
        {"loss": {"type": "classification", "reward_range": (0.0, 1.0), "n_bins": 1}}
    )
    records = [
        RolloutRecord(
            split="train",
            prompt_id=0,
            rollout_id=i,
            group_id="node0:train:0",
            reward=float(i),
            token_ids=[1, 2, 3],
            mask=[False, True, True],
            logprobs=[0.0, -0.1, -0.2],
            num_output_tokens=2,
        )
        for i in range(2)
    ]
    train_static_value.validate_rollout_records(
        records,
        path=Path("rollouts.jsonl"),
        vconfig=vconfig,
        group_size=2,
    )

    bad = [
        RolloutRecord(
            split="train",
            prompt_id=0,
            rollout_id=0,
            group_id="node0:train:0",
            reward=1.0,
            token_ids=[1, 2, 3],
            mask=[False, True, True],
            logprobs=[0.0, -0.1],
            num_output_tokens=2,
        )
    ]
    with pytest.raises(ValueError, match="mismatched"):
        train_static_value.validate_rollout_records(
            bad,
            path=Path("rollouts.jsonl"),
            vconfig=vconfig,
            group_size=2,
        )

    with pytest.raises(ValueError, match="expected 3"):
        train_static_value.validate_rollout_records(
            records,
            path=Path("rollouts.jsonl"),
            vconfig=vconfig,
            group_size=3,
        )
