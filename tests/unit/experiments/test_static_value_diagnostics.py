import numpy as np
import pytest
import torch

from experiments.static_value_diagnostics.common import RolloutRecord
from experiments.static_value_diagnostics.diagnostics import (
    PredictionSet,
    build_token_table,
    has_binary_rewards,
    method_prediction,
    no_intercept_rho,
    rho_methods,
    select_rhos,
    summary_at_rhos,
    variance_proxy,
)


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
    selected = select_rhos(table, np.asarray([0.0, 0.5, 1.0]), methods)
    summary = summary_at_rhos(table, selected, methods)

    assert "anchored_odds" not in methods
    assert "odds_prior" not in summary
    assert "anchored_add_clipped" in summary


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
        seq_len=16,
        dp_rank=0,
        dp_world_size=1,
        bin_cost=lambda seqlens: sum(seqlens),
        pad_to_multiple_of=1,
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
