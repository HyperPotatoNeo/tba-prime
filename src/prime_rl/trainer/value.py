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
)


@dataclass(frozen=True)
class ValueTargets:
    advantages: Float[Tensor, "batch seq"]
    returns: Float[Tensor, "batch seq"]
    mask: Bool[Tensor, "batch seq"]


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
    tol = 1e-6 * max(high - low, 1.0)
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
