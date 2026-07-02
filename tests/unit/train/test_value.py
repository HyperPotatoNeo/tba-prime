import pytest
import torch

from prime_rl.configs.trainer import ClassificationValueLossConfig, MSEValueLossConfig, ValueFunctionConfig
from prime_rl.trainer.value import (
    align_value_logits,
    compute_gae,
    compute_value_loss,
    predict_values,
    value_head_output_size,
    value_scheduler_max_steps,
)


def test_align_value_logits_shifts_next_token_outputs_to_token_positions():
    logits = torch.tensor([[[1.0], [2.0], [3.0]]])

    assert align_value_logits(logits).tolist() == [[[0.0], [1.0], [2.0]]]


def test_value_scheduler_max_steps_counts_warmup_and_training_updates():
    config = ValueFunctionConfig(resolved_warmup_batches=3, warmup_updates_per_batch=2, updates_per_step=4)

    assert value_scheduler_max_steps(None, config) is None
    assert value_scheduler_max_steps(5, config) == 26


def test_compute_gae_terminal_reward_with_lambda_one():
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    dones = torch.tensor([[False, False, True]])
    values = torch.zeros_like(rewards)
    mask = torch.tensor([[True, True, True]])

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        mask=mask,
        sequence_lengths=[3],
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert torch.allclose(advantages, torch.ones_like(advantages))
    assert torch.allclose(returns, torch.ones_like(returns))


def test_compute_gae_respects_packed_sequence_boundaries():
    rewards = torch.tensor([[0.0, 1.0, 0.0, 2.0]])
    dones = torch.tensor([[False, True, False, True]])
    values = torch.zeros_like(rewards)
    mask = torch.tensor([[True, True, True, True]])

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        mask=mask,
        sequence_lengths=[2, 2],
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert torch.allclose(advantages, torch.tensor([[1.0, 1.0, 2.0, 2.0]]))
    assert torch.allclose(returns, advantages)


def test_value_classification_binary_prediction_uses_reward_endpoints():
    loss_config = ClassificationValueLossConfig(n_bins=1, reward_range=(0.0, 1.0))

    assert value_head_output_size(loss_config) == 2
    predictions = predict_values(torch.tensor([[[0.0, 10.0], [10.0, 0.0]]]), loss_config)

    assert predictions[0, 0] > 0.99
    assert predictions[0, 1] < 0.01


def test_value_classification_multibin_prediction_uses_bin_expectation():
    loss_config = ClassificationValueLossConfig(n_bins=3, reward_range=(-1.0, 1.0))

    predictions = predict_values(torch.tensor([[[0.0, 10.0, 0.0]]]), loss_config)

    assert predictions.item() == pytest.approx(0.0, abs=1e-3)


def test_value_classification_rejects_out_of_range_targets():
    config = ValueFunctionConfig(loss=ClassificationValueLossConfig(reward_range=(0.0, 1.0)))

    with pytest.raises(ValueError, match="reward_range"):
        compute_value_loss(
            value_logits=torch.zeros(1, 1, 2),
            targets=torch.tensor([[2.0]]),
            mask=torch.tensor([[True]]),
            config=config,
            scale=1,
        )


def test_value_classification_allows_roundoff_at_reward_range_bounds():
    config = ValueFunctionConfig(loss=ClassificationValueLossConfig(reward_range=(0.0, 1.0), n_bins=3))

    loss, metrics = compute_value_loss(
        value_logits=torch.zeros(1, 2, 3),
        targets=torch.tensor([[-1e-7, 1.0000001]]),
        mask=torch.tensor([[True, True]]),
        config=config,
        scale=2,
    )

    assert loss.isfinite()
    assert metrics["value/loss"].numel() == 2


def test_value_mse_loss_trains_only_masked_tokens():
    config = ValueFunctionConfig(loss=MSEValueLossConfig())

    loss, metrics = compute_value_loss(
        value_logits=torch.tensor([[[1.0], [10.0]]]),
        targets=torch.tensor([[0.0, 0.0]]),
        mask=torch.tensor([[True, False]]),
        config=config,
        scale=1,
    )

    assert loss.item() == pytest.approx(1.0)
    assert metrics["value/loss"].tolist() == [1.0]
