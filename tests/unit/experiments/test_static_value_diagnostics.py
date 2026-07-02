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


def test_forward_record_aligns_generated_tokens_to_previous_prefix(monkeypatch):
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

    def fake_predict_value(model, input_ids, position_ids):
        del model, position_ids
        return input_ids.float().unsqueeze(-1)

    monkeypatch.setattr(train_static_value, "predict_value", fake_predict_value)
    logits, targets, mask, mask_idxs = train_static_value.forward_record(None, record, 16, torch.device("cpu"))

    assert mask_idxs == [2, 3]
    assert mask.tolist() == [[False, False, True, True]]
    assert targets[0, 2:].tolist() == pytest.approx([1.0, 1.0])
    assert logits[0, 2, 0].item() == pytest.approx(22.0)
    assert logits[0, 3, 0].item() == pytest.approx(33.0)


def test_forward_records_packs_microbatch_with_sequence_resets(monkeypatch):
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

    def fake_predict_value(model, input_ids, position_ids):
        del model
        assert input_ids.shape == (1, 7)
        assert position_ids.tolist() == [[0, 1, 2, 3, 0, 1, 2]]
        assert position_ids.is_contiguous()
        return input_ids.float().unsqueeze(-1)

    monkeypatch.setattr(train_static_value, "predict_value", fake_predict_value)
    logits, targets, mask = train_static_value.forward_records(None, records, 16, torch.device("cpu"))

    assert mask.tolist() == [[False, False, True, True, False, True, True]]
    assert targets[0, mask[0]].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert logits[0, 5, 0].item() == pytest.approx(55.0)
    assert logits[0, 6, 0].item() == pytest.approx(66.0)
