import pytest
import torch
from torch import nn

from prime_rl.configs.trainer import (
    ClassificationValueLossConfig,
    MSEValueLossConfig,
    ValueFunctionConfig,
    ValueMixtureConfig,
)
from prime_rl.trainer import model as trainer_model
from prime_rl.trainer.model import ValueFunctionModel
from prime_rl.trainer.value import (
    action_span_position_fraction,
    align_value_logits,
    broadcast_action_span_starts,
    compute_gae,
    compute_value_loss,
    mixed_clipped_advantage,
    mixture_rho,
    predict_values,
    turn_anchor_tether_advantage,
    value_head_output_size,
    value_scheduler_max_steps,
)


class _TinyConfig:
    hidden_size = 4
    tie_word_embeddings = False
    vocab_size = 8

    def get_text_config(self) -> "_TinyConfig":
        return self


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, _TinyConfig.hidden_size)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.proj((input_ids + position_ids).float().unsqueeze(-1)),)


class _TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _TinyConfig()
        self.model = _TinyBackbone()
        self.lm_head = nn.Linear(_TinyConfig.hidden_size, _TinyConfig.vocab_size)


def test_align_value_logits_shifts_next_token_outputs_to_token_positions():
    logits = torch.tensor([[[1.0], [2.0], [3.0]]])

    assert align_value_logits(logits).tolist() == [[[0.0], [1.0], [2.0]]]


def test_align_value_logits_per_segment_seeds_each_segment_start():
    # Two packed segments of length 3 and 2. Each segment's first position must be
    # seeded with zero and shifted WITHIN the segment (no leakage across the boundary).
    logits = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])

    aligned = align_value_logits(logits, sequence_lengths=[3, 2])

    assert aligned.tolist() == [[[0.0], [1.0], [2.0], [0.0], [4.0]]]


def test_align_value_logits_single_segment_matches_global_shift():
    # With one segment spanning the whole batch, the per-segment path must be
    # byte-identical to the default global shift.
    logits = torch.tensor([[[1.0], [2.0], [3.0]]])

    assert torch.equal(align_value_logits(logits, sequence_lengths=[3]), align_value_logits(logits))


def test_align_value_logits_preserves_gradients_per_segment():
    logits = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]], requires_grad=True)

    aligned = align_value_logits(logits, sequence_lengths=[3, 2])
    aligned.sum().backward()

    # Positions that are shifted into an output receive gradient 1; segment-final
    # positions (dropped by the shift) and the seeded starts receive 0.
    assert logits.grad.squeeze().tolist() == [1.0, 1.0, 0.0, 1.0, 0.0]


def test_value_scheduler_max_steps_counts_warmup_and_training_updates():
    config = ValueFunctionConfig(resolved_warmup_batches=3, warmup_updates_per_batch=2, updates_per_step=4)

    assert value_scheduler_max_steps(None, config) is None
    assert value_scheduler_max_steps(5, config) == 26


def test_value_function_head_only_freezes_backbone_only() -> None:
    model = ValueFunctionModel(_TinyCausalLM(), output_size=3)

    assert any(param.requires_grad for param in model.model.parameters())

    model.freeze_backbone()

    assert model.backbone_is_frozen
    assert all(not param.requires_grad for param in model.model.parameters())
    assert all(param.requires_grad for param in model.value_head.parameters())


def test_setup_value_model_threads_head_only_to_setup_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_setup_model(
        config: object,
        parallel_dims: object,
        loading_from_checkpoint_later: bool,
        *,
        value_head_output_size: int | None,
        freeze_value_backbone: bool = False,
    ) -> nn.Module:
        captured["config"] = config
        captured["parallel_dims"] = parallel_dims
        captured["loading_from_checkpoint_later"] = loading_from_checkpoint_later
        captured["value_head_output_size"] = value_head_output_size
        captured["freeze_value_backbone"] = freeze_value_backbone
        return nn.Linear(1, 1)

    monkeypatch.setattr(trainer_model, "setup_model", fake_setup_model)

    trainer_model.setup_value_model(object(), object(), False, head_output_size=3, head_only=True)

    assert captured["value_head_output_size"] == 3
    assert captured["freeze_value_backbone"] is True


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


def test_action_span_position_fraction_resets_inside_each_sampled_turn():
    mask = torch.tensor([[False, True, True, False, True, True, True]])

    fraction = action_span_position_fraction(mask, sequence_lengths=[7])

    assert torch.allclose(fraction, torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0]]))


def test_broadcast_action_span_starts_anchors_each_sampled_turn():
    values = torch.tensor([[0.0, 0.2, 0.4, 0.0, 0.6, 0.8]])
    mask = torch.tensor([[False, True, True, False, True, True]])

    starts = broadcast_action_span_starts(values, mask, sequence_lengths=[6])

    assert torch.allclose(starts, torch.tensor([[0.0, 0.2, 0.2, 0.0, 0.6, 0.6]]))


def test_mixture_rho_linear_blend_uses_explicit_start_end_schedule():
    fraction = torch.tensor([[0.0, 0.5, 1.0]])
    config = ValueMixtureConfig(kind="linear", schedule="linear", rho=0.9, rho_start=0.2, rho_end=0.6)

    rho = mixture_rho(fraction, config)

    assert torch.allclose(rho, torch.tensor([[0.2, 0.4, 0.6]]))


@pytest.mark.parametrize("kind", ["mixed_clipped", "turn_anchor"])
def test_mixture_rho_linear_tether_ramps_to_configured_rho(kind: str):
    fraction = torch.tensor([[0.0, 0.5, 1.0]])
    config = ValueMixtureConfig(
        kind=kind,
        schedule="linear",
        rho=0.8,
        rho_start=0.2,
        rho_end=0.6,
    )

    rho = mixture_rho(fraction, config)

    assert torch.allclose(rho, torch.tensor([[0.0, 0.4, 0.8]]))


def test_mixed_clipped_advantage_matches_tether_formula_and_reward_clip():
    episodic_return = torch.tensor([[0.8, 0.8]])
    group_advantages = torch.tensor([[0.6, 0.4]])
    values = torch.tensor([[0.9, 1.4]])
    start_value = torch.tensor([[0.3, 0.3]])
    rho = torch.tensor([[0.25, 0.25]])

    advantages = mixed_clipped_advantage(
        group_advantages,
        episodic_return,
        values,
        start_value,
        alpha=0.5,
        rho=rho,
        reward_range=(0.0, 1.0),
    )

    b_group = episodic_return - group_advantages
    baseline = b_group + 0.5 * (start_value - b_group) + rho * (values - start_value)
    expected = episodic_return - baseline.clamp(min=0.0, max=1.0)
    assert torch.allclose(advantages, expected)


def test_turn_anchor_tether_advantage_uses_turn_return_and_turn_start_anchor():
    turn_return = torch.tensor([[0.8, 0.8, 0.2, 0.2]])
    turn_tether = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
    values = torch.tensor([[0.6, 0.7, 0.2, 0.4]])
    turn_start_value = torch.tensor([[0.55, 0.55, 0.1, 0.1]])
    rho = torch.tensor([[0.2, 0.2, 0.4, 0.4]])

    advantages = turn_anchor_tether_advantage(
        turn_return,
        turn_tether,
        values,
        turn_start_value,
        alpha=0.5,
        rho=rho,
        reward_range=(0.0, 1.0),
    )

    baseline = turn_tether + 0.5 * (turn_start_value - turn_tether) + rho * (values - turn_start_value)
    expected = turn_return - baseline.clamp(min=0.0, max=1.0)
    assert torch.allclose(advantages, expected)


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
        targets=torch.tensor([[-1e-6, 1.000001]]),
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

    # The MSE head is sigmoid-bounded (V = sigmoid(raw)); the masked token 0 has
    # raw logit 1.0, so its squared error against target 0.0 is sigmoid(1.0)**2.
    expected = torch.sigmoid(torch.tensor(1.0)).item() ** 2
    assert loss.item() == pytest.approx(expected)
    assert metrics["value/loss"].tolist() == pytest.approx([expected])


def test_value_mse_prediction_is_sigmoid_bounded():
    config = MSEValueLossConfig()
    raw = torch.tensor([[[-5.0], [0.0], [5.0]]])

    preds = predict_values(raw, config)

    assert torch.allclose(preds, torch.sigmoid(raw.squeeze(-1)))
    assert bool(((preds > 0.0) & (preds < 1.0)).all())
