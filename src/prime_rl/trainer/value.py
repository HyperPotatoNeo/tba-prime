from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor

from prime_rl.configs.trainer import (
    ClassificationValueLossConfig,
    MSEValueLossConfig,
    ValueFunctionConfig,
    ValueLossConfig,
    ValueMixtureConfig,
)


@dataclass(frozen=True)
class ValueTargets:
    advantages: Float[Tensor, "batch seq"]
    returns: Float[Tensor, "batch seq"]
    mask: Bool[Tensor, "batch seq"]
    position_fraction: Float[Tensor, "batch seq"]
    values: Float[Tensor, "batch seq"]
    start_value: Float[Tensor, "batch seq"]


@dataclass(frozen=True)
class ValueUpdateStats:
    grad_norm: Tensor | None
    zero_grad_ratio: float | None


def value_head_output_size(loss_config: ValueLossConfig) -> int:
    if isinstance(loss_config, ClassificationValueLossConfig):
        return max(loss_config.n_bins, 2)
    return 1


def value_scheduler_max_steps(max_steps: int | None, config: ValueFunctionConfig) -> int | None:
    if max_steps is None:
        return None
    return config.resolved_warmup_batches * config.warmup_updates_per_batch + max_steps * config.updates_per_step


def align_value_logits(value_logits: Float[Tensor, "batch seq output"]) -> Float[Tensor, "batch seq output"]:
    return torch.cat(
        [
            torch.zeros(
                value_logits.shape[0],
                1,
                value_logits.shape[-1],
                dtype=value_logits.dtype,
                device=value_logits.device,
            ),
            value_logits[:, :-1],
        ],
        dim=1,
    )


def _reward_bounds(loss_config: ClassificationValueLossConfig) -> tuple[float, float]:
    low, high = loss_config.reward_range
    return float(low), float(high)


def _bin_values(loss_config: ClassificationValueLossConfig, device: torch.device) -> Tensor:
    low, high = _reward_bounds(loss_config)
    return torch.linspace(low, high, value_head_output_size(loss_config), device=device, dtype=torch.float32)


def predict_values(
    value_logits: Float[Tensor, "batch seq output"],
    loss_config: ValueLossConfig,
) -> Float[Tensor, "batch seq"]:
    if isinstance(loss_config, ClassificationValueLossConfig):
        probs = value_logits.float().softmax(dim=-1)
        return probs @ _bin_values(loss_config, value_logits.device)

    return value_logits.squeeze(-1).float()


def _classification_targets(
    targets: Float[Tensor, "batch seq"],
    loss_config: ClassificationValueLossConfig,
) -> Tensor:
    low, high = _reward_bounds(loss_config)
    targets = targets.float()
    tol = 1e-5 * max(high - low, 1.0)
    out_of_range = (targets < low - tol) | (targets > high + tol)
    if bool(out_of_range.any()):
        offending = targets[out_of_range][0].item()
        raise ValueError(
            f"value_function.loss.type='classification' requires rewards in reward_range={loss_config.reward_range}; "
            f"found {offending}."
        )
    targets = targets.clamp(min=low, max=high)
    if loss_config.n_bins == 1:
        midpoint = (low + high) / 2
        return (targets >= midpoint).long()

    normalized = (targets - low) / (high - low)
    return torch.round(normalized * (loss_config.n_bins - 1)).long()


def compute_gae(
    rewards: Float[Tensor, "batch seq"],
    dones: Bool[Tensor, "batch seq"],
    values: Float[Tensor, "batch seq"],
    mask: Bool[Tensor, "batch seq"],
    sequence_lengths: list[int],
    gamma: float,
    gae_lambda: float,
) -> tuple[Float[Tensor, "batch seq"], Float[Tensor, "batch seq"]]:
    """Compute GAE over sampled/action tokens inside each packed sequence."""
    flat_rewards = rewards.reshape(-1).float()
    flat_dones = dones.reshape(-1)
    flat_values = values.reshape(-1).float()
    flat_mask = mask.reshape(-1)
    flat_advantages = torch.zeros_like(flat_values)
    flat_returns = torch.zeros_like(flat_values)

    offset = 0
    for seq_len in sequence_lengths:
        seq_slice = slice(offset, offset + seq_len)
        action_idxs = flat_mask[seq_slice].nonzero(as_tuple=False).flatten() + offset
        next_gae = flat_values.new_tensor(0.0)
        for pos in reversed(range(action_idxs.numel())):
            idx = action_idxs[pos]
            has_next = pos + 1 < action_idxs.numel()
            next_value = flat_values[action_idxs[pos + 1]] if has_next else flat_values.new_tensor(0.0)
            nonterminal = (~flat_dones[idx]).to(flat_values.dtype)
            delta = flat_rewards[idx] + gamma * next_value * nonterminal - flat_values[idx]
            next_gae = delta + gamma * gae_lambda * nonterminal * next_gae
            flat_advantages[idx] = next_gae
            flat_returns[idx] = next_gae + flat_values[idx]
        offset += seq_len

    return flat_advantages.reshape_as(values), flat_returns.reshape_as(values)


def response_position_fraction(
    mask: Bool[Tensor, "batch seq"],
    sequence_lengths: list[int],
) -> Float[Tensor, "batch seq"]:
    """Per-token position fraction along the response (action tokens) of each
    packed sequence: 0.0 at the first action token, 1.0 at the last. Non-action
    tokens and single-action responses are 0.0. Used to schedule the mixture
    weight rho(t) from the response start to its end."""
    flat_mask = mask.reshape(-1)
    frac = torch.zeros(flat_mask.numel(), dtype=torch.float32, device=mask.device)
    offset = 0
    for seq_len in sequence_lengths:
        seq_slice = slice(offset, offset + seq_len)
        action_idxs = flat_mask[seq_slice].nonzero(as_tuple=False).flatten() + offset
        n = action_idxs.numel()
        if n > 1:
            frac[action_idxs] = torch.linspace(0.0, 1.0, n, dtype=torch.float32, device=frac.device)
        offset += seq_len
    return frac.reshape_as(mask)


def mixture_rho(
    position_fraction: Float[Tensor, "batch seq"],
    config: ValueMixtureConfig,
) -> Float[Tensor, "batch seq"]:
    """Per-token mixture weight rho(t) in [0, 1]. ``constant`` uses ``rho``
    everywhere; ``linear`` ramps ``rho_start`` -> ``rho_end`` along the response
    by ``position_fraction``."""
    if config.schedule == "constant":
        return torch.full_like(position_fraction, config.rho)
    return config.rho_start + (config.rho_end - config.rho_start) * position_fraction


def mix_advantages(
    group_advantages: Float[Tensor, "batch seq"],
    value_advantages: Float[Tensor, "batch seq"],
    rho: Float[Tensor, "batch seq"],
) -> Float[Tensor, "batch seq"]:
    """Convex blend ``A = (1 - rho) * A_group + rho * A_value``. With the GRPO
    leave-one-out group advantage ``A_group = R - B_loo`` and the Monte Carlo
    value advantage ``A_value = R - V_t`` (GAE with gamma=lambda=1), this equals
    ``R - [(1 - rho) B_loo + rho V_t]`` — the linear value-corrected baseline."""
    return (1.0 - rho) * group_advantages + rho * value_advantages


def broadcast_start_value(
    values: Float[Tensor, "batch seq"],
    mask: Bool[Tensor, "batch seq"],
    sequence_lengths: list[int],
) -> Float[Tensor, "batch seq"]:
    """Per-token ``V_0``: the value prediction at the first action token of each
    packed sequence, broadcast to every position of that sequence. Sequences with
    no action tokens stay 0.0. Used by the ``mixed_clipped`` baseline's prompt-prior
    and prefix-progress split."""
    flat_values = values.reshape(-1).float()
    flat_mask = mask.reshape(-1)
    out = torch.zeros_like(flat_values)
    offset = 0
    for seq_len in sequence_lengths:
        seq_slice = slice(offset, offset + seq_len)
        action_idxs = flat_mask[seq_slice].nonzero(as_tuple=False).flatten() + offset
        if action_idxs.numel() > 0:
            out[seq_slice] = flat_values[action_idxs[0]]
        offset += seq_len
    return out.reshape_as(values)


def mixed_clipped_advantage(
    group_advantages: Float[Tensor, "batch seq"],
    returns: Float[Tensor, "batch seq"],
    values: Float[Tensor, "batch seq"],
    start_value: Float[Tensor, "batch seq"],
    gate: Float[Tensor, "batch seq"],
    alpha: float,
    rho: float,
    reward_range: tuple[float, float] = (0.0, 1.0),
) -> Float[Tensor, "batch seq"]:
    """TETHER advantage: group anchor + clipped two-factor value correction.

    ``b = clip( B_group + gate * [ alpha (V_0 - B_group) + rho (V_t - V_0) ], lo, hi )``
    and ``A = R - b``. The group baseline is recovered from the orchestrator
    advantage ``B_group = R - A_group`` (with ``returns = R`` at action tokens under
    gamma=lambda=1), ``V_t = values``, ``V_0 = start_value``. ``gate`` is 1 (global)
    or the response position fraction ``u_t`` (position-conditioned)."""
    b_group = returns - group_advantages
    correction = gate * (alpha * (start_value - b_group) + rho * (values - start_value))
    baseline = (b_group + correction).clamp(min=reward_range[0], max=reward_range[1])
    return returns - baseline


def compute_value_loss(
    value_logits: Float[Tensor, "batch seq output"],
    targets: Float[Tensor, "batch seq"],
    mask: Bool[Tensor, "batch seq"],
    config: ValueFunctionConfig,
    scale: int,
) -> tuple[Float[Tensor, ""], dict[str, Tensor]]:
    """Compute the normalized value loss over masked tokens."""
    loss_config = config.loss
    targets = targets.float()
    predictions = predict_values(value_logits, loss_config)

    if not bool(mask.any()):
        zero = value_logits.sum() * 0.0
        empty = value_logits.new_empty(0).detach()
        return zero, {
            "value/loss": empty,
            "value/prediction": empty,
            "value/target": empty,
            "value/abs_error": empty,
        }

    if isinstance(loss_config, ClassificationValueLossConfig):
        labels = _classification_targets(targets[mask], loss_config)
        masked_logits = value_logits[mask].float()
        masked_loss = F.cross_entropy(masked_logits, labels, reduction="none")
        per_token_loss = value_logits.new_zeros(targets.shape, dtype=torch.float32)
        per_token_loss[mask] = masked_loss
        accuracy = (masked_logits.argmax(dim=-1) == labels).float()
        metrics = {
            "value/accuracy": accuracy.detach(),
        }
    elif isinstance(loss_config, MSEValueLossConfig):
        per_token_loss = F.mse_loss(predictions, targets, reduction="none")
        metrics = {}
    else:
        raise ValueError(f"Unsupported value loss config: {loss_config}")

    normalized_loss = per_token_loss[mask].sum() / scale
    loss = normalized_loss * config.loss_weight
    abs_error = torch.abs(predictions - targets)

    metrics.update(
        {
            "value/loss": per_token_loss[mask].detach(),
            "value/prediction": predictions[mask].detach(),
            "value/target": targets[mask].detach(),
            "value/abs_error": abs_error[mask].detach(),
        }
    )
    return loss, metrics
