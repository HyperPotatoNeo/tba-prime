from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.length_penalty = config.length_penalty
        self.baseline = config.baseline

    def _baseline(self, rewards: torch.Tensor) -> torch.Tensor:
        """Per-rollout group baseline: the group mean, or the leave-one-out mean
        ``B_i = (sum_{j!=i} R_j) / (G-1)`` which excludes the rollout's own reward."""
        if self.baseline == "loo":
            group_size = rewards.numel()
            if group_size < 2:
                return rewards.mean()
            return (rewards.sum() - rewards) / (group_size - 1)
        return rewards.mean()

    async def score_group(self, group: list[Rollout]) -> None:
        rewards = torch.tensor([rollout.reward for rollout in group], dtype=torch.float32)
        length_penalty = self.length_penalty
        if length_penalty is None:
            advantages = rewards - self._baseline(rewards)
        else:
            output = torch.tensor([rollout.num_output_tokens for rollout in group], dtype=rewards.dtype)
            total = torch.tensor([rollout.num_total_tokens for rollout in group], dtype=rewards.dtype)
            turns = torch.tensor([rollout.num_turns for rollout in group], dtype=rewards.dtype)
            input = total - output
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            penalty = rewards.mean() * penalty_frac
            shaped_rewards = rewards - penalty
            advantages = shaped_rewards - self._baseline(shaped_rewards)
        for rollout, advantage in zip(group, advantages.tolist(), strict=True):
            rollout.assign_advantages(advantage)
