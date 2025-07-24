import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, Sequence

import numpy as np
import requests
from pydantic import BaseModel
from vllm import RequestOutput

from zeroband.inference.config import RewardsConfig
from zeroband.inference.genesys import TaskType, get_reward_function
from zeroband.utils.logger import get_logger


class ModelCompletion(BaseModel):
    index: int
    text: str
    token_ids: Sequence[int]


class ModelOutput(BaseModel):
    request_id: str
    outputs: list[ModelCompletion]


class RewardRequest(BaseModel):
    model_outputs: list[ModelOutput]
    ground_truths: list
    config: RewardsConfig | None = None

    def __len__(self) -> int:
        return len(self.model_outputs)

    def __iter__(self) -> Iterator[tuple[ModelOutput, dict[str, Any], TaskType]]:
        for request_output, ground_truth in zip(self.model_outputs, self.ground_truths):
            yield request_output, ground_truth


def unwrap_request_output(request_output: RequestOutput) -> ModelOutput:
    outputs = [ModelCompletion(index=o.index, text=o.text, token_ids=o.token_ids) for o in request_output.outputs]
    return ModelOutput(request_id=request_output.request_id, outputs=outputs)


def vllm_output_to_serializable(
    request_outputs: list[RequestOutput],
    ground_truths: list,
    config: RewardsConfig | None = None,
) -> RewardRequest:
    model_outputs = [unwrap_request_output(request_output) for request_output in request_outputs]
    return RewardRequest(
        model_outputs=model_outputs,
        ground_truths=ground_truths,
        config=config,
    )


class CompletionReward(BaseModel):
    completion_id: int  # type(CompletionOutput.index)
    reward: float
    task_reward: float
    advantage: float | None = None


class RequestRewards(BaseModel):
    request_id: str  # type(RequestOutput.request_id)
    rewards: list[CompletionReward]


class RewardsResponse(BaseModel):
    rewards: list[RequestRewards]


def _compute_completion_reward(
    completion_output: ModelCompletion,
    ground_truth,
    reward_fn,
) -> CompletionReward:
    """
    Computes the reward from a single vLLM completion output given the
    task type (e.g. math, code, etc.) and information on how to verify
    the output. Also supports an optional length penalty.

    Args:
        completion_output: The completion output to compute the reward for.
        verification_info: The verification info for the completion output.
        task_type: The task type for the completion output.
        config: The config for the rewards.

    Returns:
        A dictionary containing the reward, task reward, and length penalty.
    """
    # Compute task reward
    task_reward = reward_fn(completion_output.text, ground_truth)
    reward = task_reward

    return CompletionReward(
        completion_id=completion_output.index,
        reward=reward,
        task_reward=task_reward,
    )


def _compute_request_rewards(
    request_output: ModelOutput,
    ground_truth,
    reward_fn,
    config: RewardsConfig | None,
) -> RequestRewards:
    """
    Computes the rewards and advantages from a single vLLM request output given
    the task type (e.g. math, code, etc.) and information on how to verify all
    completions in the request output.

    Args:
        request_output: The request output to compute the rewards for.
        verification_info: The verification info for the request output.
        task_type: The task type for the request output.
        config: The config for the rewards.

    Returns:
        A dictionary containing the rewards, task rewards, and length penalties
        for each completion in the request output.
    """
    completion_rewards = []
    for output in request_output.outputs:
        args = (output, ground_truth, reward_fn)
        completion_rewards.append(_compute_completion_reward(*args))

    # Compute advantage (normalized rewards)
    reward_array = np.array([reward.reward for reward in completion_rewards], dtype=np.float32)

    if config:
        if config.advantage_estimation_method == "dr_grpo":
            advantage_array = reward_array - reward_array.mean()

        elif config.advantage_estimation_method == "grpo":
            advantage_array = (reward_array - reward_array.mean()) / (reward_array.std(ddof=1) + 1e-6)

        elif config.advantage_estimation_method == "opo":
            lengths = np.array([len(r.token_ids) for r in request_output.outputs], dtype=np.float32)
            weights = lengths / lengths.sum()
            weighted_mean = (reward_array * weights).sum()
            advantage_array = reward_array - weighted_mean
        else:
            raise ValueError(f"{config.advantage_estimation_method} is not supported for advantage estimation")
    else:
        advantage_array = np.zeros_like(reward_array)

    for completion_reward, advantage in zip(completion_rewards, advantage_array):
        completion_reward.advantage = float(advantage)

    return RequestRewards(request_id=request_output.request_id, rewards=completion_rewards)


def compute_rewards(
    reward_request: RewardRequest,
    reward_fn
) -> RewardsResponse:
    max_workers = min(32, len(reward_request))
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for request, info in reward_request:
            args = (request, info, reward_fn, reward_request.config)
            futures.append(executor.submit(_compute_request_rewards, *args))

    return RewardsResponse(rewards=list(future.result() for future in futures))


def compute_vllm_rewards(
    request_outputs: list[RequestOutput],
    ground_truths,
    reward_fn,
    config: RewardsConfig | None = None,
) -> list[RequestRewards]:
    """
    Computes the rewards and advantages for a list of vLLM request outputs
    given their task types and verification infos.

    Args:
        request_outputs: The request outputs to compute the rewards for.
        verification_infos: The verification infos for the request outputs.
        task_types: The task types for the request outputs.
        config: The config for the rewards.

    Returns:
        A tuple containing dictionaries mapping request IDs to lists of rewards,
        task rewards, length penalties, and advantages.
    """

    reward_request = vllm_output_to_serializable(
        request_outputs=request_outputs,
        ground_truths=ground_truths,
        config=config,
    )
    return compute_rewards(reward_request, reward_fn).rewards
