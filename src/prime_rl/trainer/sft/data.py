import hashlib
import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

import torch
from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk
from jaxtyping import Bool, Int
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.sft import (
    DataConfig,
    LossMaskConfig,
    SFTCompactionConfig,
    SFTDataConfig,
)
from prime_rl.transport.types import CallWire, CompactionEventWire
from prime_rl.trainer.world import get_world
from prime_rl.utils.chat_template import (
    build_incremental_token_mask,
    deserialize_tool_calls,
    normalize_messages,
    render_messages,
    should_add_generation_prompt,
    strip_message_content,
)
from prime_rl.utils.logger import get_logger

STACKING_DATASET_BUCKET_TIMEOUT = 10


class Sample(TypedDict):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    target_ids: list[int]
    calls: NotRequired[list[CallWire]]


class Batch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    calls: NotRequired[list[CallWire] | None]


FILLER_TOKEN_ID = 151643


def _stable_hash_index(parts: list[object], modulo: int) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "big") % modulo


def _example_compaction_policy_key(example: dict) -> str | None:
    for key in ("trace_id", "example_id", "base_example_id", "__index"):
        value = example.get(key)
        if value is not None:
            return f"{key}={value}"
    return None


def _select_sft_compaction_config(
    compaction_config: SFTCompactionConfig,
    example: dict,
    *,
    step: int,
    epoch: int,
) -> tuple[SFTCompactionConfig, str | None]:
    policies = compaction_config.policies
    if not policies:
        return compaction_config, None

    if compaction_config.policy_sampling == "fixed":
        policy_idx = 0
    else:
        trace_key = _example_compaction_policy_key(example) or f"step={step}"
        hash_parts: list[object] = [
            compaction_config.policy_seed,
            compaction_config.policy_sampling,
            trace_key,
        ]
        if compaction_config.policy_sampling == "uniform_per_access":
            hash_parts.extend([step, epoch])
        policy_idx = _stable_hash_index(hash_parts, len(policies))

    policy = policies[policy_idx]
    return (
        compaction_config.model_copy(
            update={
                "max_turns": policy.max_turns,
                "eviction_turn_stride": policy.eviction_turn_stride,
            }
        ),
        policy.name,
    )


def _pad_after_im_end(
    tokenizer: PreTrainedTokenizer,
    token_ids: list[int],
    *,
    block_size: int,
) -> list[int]:
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    out: list[int] = []
    for token_id in token_ids:
        out.append(int(token_id))
        if int(token_id) == int(im_end_id):
            n_pad = (block_size - (len(out) % block_size)) % block_size
            out.extend([FILLER_TOKEN_ID] * n_pad)
    return out


def _pad_fragment_after_im_end(
    tokenizer: PreTrainedTokenizer,
    start_offset: int,
    text: str,
    *,
    block_size: int,
) -> list[int]:
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    raw = tokenizer.encode(text, add_special_tokens=False)
    out: list[int] = []
    running = int(start_offset)
    for token_id in raw:
        out.append(int(token_id))
        running += 1
        if int(token_id) == int(im_end_id):
            n_pad = (block_size - (running % block_size)) % block_size
            out.extend([FILLER_TOKEN_ID] * n_pad)
            running += n_pad
    return out


def _completion_delta(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict],
    assistant_idx: int,
) -> list[int]:
    prompt_ids = render_messages(
        tokenizer,
        messages[:assistant_idx],
        add_generation_prompt=True,
    )
    full_ids = render_messages(
        tokenizer,
        messages[: assistant_idx + 1],
        add_generation_prompt=False,
    )
    assert prompt_ids == full_ids[: len(prompt_ids)], (
        "Mismatch in assistant completion incremental tokenization."
    )
    return [int(x) for x in full_ids[len(prompt_ids) :]]


def _build_incremental_token_mask_and_calls(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict],
    *,
    role_to_mask,
    tools: list[dict] | None,
    chat_template_kwargs: dict | None,
    compaction_config: SFTCompactionConfig,
) -> tuple[list[int], list[bool], list[CallWire]]:
    if tools not in (None, []):
        raise ValueError("SFT compaction call synthesis does not support tools")
    if chat_template_kwargs not in (None, {}):
        raise ValueError(
            "SFT compaction call synthesis does not support chat_template_kwargs"
        )

    block_size = compaction_config.block_size
    assistant_indices = [
        idx for idx, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("SFT compaction requires at least one assistant message")

    first_assistant_idx = assistant_indices[0]
    first_prompt_raw = render_messages(
        tokenizer,
        messages[:first_assistant_idx],
        add_generation_prompt=True,
    )
    first_prompt = _pad_after_im_end(
        tokenizer,
        [int(x) for x in first_prompt_raw],
        block_size=block_size,
    )
    if compaction_config.protected_prefix_tokens > 0:
        protected_prefix_len = compaction_config.protected_prefix_tokens
    else:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        first_im_end = first_prompt.index(int(im_end_id))
        protected_prefix_len = (
            ((first_im_end + 1 + block_size - 1) // block_size) * block_size
        )

    pre_trim_ids: list[int] = []
    token_mask: list[bool] = []
    calls: list[CallWire] = []
    prev_state = list(first_prompt)
    completed_turn_lengths: list[int] = []
    position_offset_after = 0

    for turn_idx, assistant_idx in enumerate(assistant_indices):
        assistant_message = messages[assistant_idx]
        if turn_idx == 0:
            submitted = list(first_prompt)
            current_user_fragment_len = len(first_prompt) - protected_prefix_len
        else:
            user_idx = assistant_idx - 1
            if user_idx < 0 or messages[user_idx].get("role") != "user":
                raise ValueError(
                    "SFT compaction expects user/assistant turns after the system prompt"
                )
            user_content = messages[user_idx].get("content") or ""
            fragment = (
                "\n<|im_start|>user\n"
                f"{user_content}"
                "<|im_end|>\n<|im_start|>assistant\n"
            )
            user_fragment = _pad_fragment_after_im_end(
                tokenizer,
                len(prev_state),
                fragment,
                block_size=block_size,
            )
            submitted = list(prev_state) + user_fragment
            current_user_fragment_len = len(user_fragment)

        events: list[CompactionEventWire] = []
        kept_prompt = list(submitted)
        if len(completed_turn_lengths) + 1 > compaction_config.max_turns:
            evict_turns = min(
                compaction_config.eviction_turn_stride,
                len(completed_turn_lengths),
            )
            tokens_evicted = sum(completed_turn_lengths[:evict_turns])
            if tokens_evicted > 0:
                evict_start = protected_prefix_len
                position_offset_after += tokens_evicted
                kept_prompt = (
                    submitted[:evict_start]
                    + submitted[evict_start + tokens_evicted :]
                )
                events.append(
                    CompactionEventWire(
                        num_output_tokens_at_compaction=0,
                        tokens_evicted=tokens_evicted,
                        position_offset_after=position_offset_after,
                        num_prompt_tokens=len(kept_prompt),
                        evict_start=evict_start,
                        new_user_fragment_len=max(1, current_user_fragment_len),
                        last_turn_evicted=evict_turns - 1,
                        num_turns_evicted_after=evict_turns,
                    )
                )
                del completed_turn_lengths[:evict_turns]

        completion_ids = _completion_delta(tokenizer, messages, assistant_idx)
        state_without_pad = kept_prompt + completion_ids
        trailing_pad_len = (block_size - (len(state_without_pad) % block_size)) % block_size
        trailing_pad = [FILLER_TOKEN_ID] * trailing_pad_len

        calls.append(
            CallWire(
                submitted_prompt_ids=list(submitted),
                completion_ids=list(completion_ids),
                completion_logprobs=[0.0] * len(completion_ids),
                completion_temperatures=[1.0] * len(completion_ids),
                compaction_events=events,
                trailing_pad_ids=trailing_pad,
            )
        )

        if turn_idx == 0:
            new_prompt_piece = submitted
            prompt_mask = [False] * len(new_prompt_piece)
            current_turn_len = (
                len(submitted) - protected_prefix_len
                + len(completion_ids)
                + trailing_pad_len
            )
        else:
            prior_post_len = len(prev_state)
            new_prompt_piece = submitted[prior_post_len:]
            prompt_mask = [False] * len(new_prompt_piece)
            current_turn_len = (
                len(new_prompt_piece) + len(completion_ids) + trailing_pad_len
            )

        pre_trim_ids.extend(new_prompt_piece)
        token_mask.extend(prompt_mask)
        assistant_mask = bool(role_to_mask(assistant_message))
        pre_trim_ids.extend(completion_ids)
        token_mask.extend([assistant_mask] * len(completion_ids))
        pre_trim_ids.extend(trailing_pad)
        token_mask.extend([False] * trailing_pad_len)

        completed_turn_lengths.append(current_turn_len)
        prev_state = state_without_pad + trailing_pad

    return pre_trim_ids, token_mask, calls


def _trim_last_call_to_input_length(calls: list[CallWire], trim_tokens: int) -> list[CallWire]:
    if trim_tokens <= 0:
        return calls
    if not calls:
        return calls
    out = list(calls)
    last = out[-1]
    keep = max(0, len(last.completion_ids) - trim_tokens)
    out[-1] = CallWire(
        submitted_prompt_ids=list(last.submitted_prompt_ids),
        completion_ids=list(last.completion_ids[:keep]),
        completion_logprobs=list(last.completion_logprobs[:keep]),
        completion_temperatures=list(last.completion_temperatures[:keep]),
        compaction_events=list(last.compaction_events),
        trailing_pad_ids=[],
    )
    return out


class StatefulIterableDataset(Stateful, IterableDataset):
    """SFT dataset are iterable (infinite) and stateful (can be checkpointed)."""

    def __init__(self):
        self.step, self.epoch = 0, 0
        self.num_samples = defaultdict(int)
        self.num_tokens = defaultdict(int)
        self.fast_forward = False
        self._setup_world_info()

    def state_dict(self) -> dict:
        return {"step": self.step, "epoch": self.epoch}

    def load_state_dict(self, state_dict: dict):
        assert "step" in state_dict and "epoch" in state_dict
        self.fast_forward = True
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]

    def _setup_world_info(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1
        self.data_rank = get_world().rank * num_workers + worker_id
        self.data_world_size = get_world().world_size * num_workers


class FakeDataset(StatefulIterableDataset):
    """A dataset of fake tokens"""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        length: Literal["fixed", "variable"] = "fixed",
        input_ids: Literal["increasing", "random"] = "random",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length
        self.input_ids = input_ids

    def __iter__(self):
        while True:
            self.step += 1

            # Skip samples that don't belong to this data rank
            if (self.step - 1) % self.data_world_size != self.data_rank:
                continue

            seq_len = int(torch.randint(1, self.seq_len, (1,)).item()) if self.length == "variable" else self.seq_len
            input_ids = (
                [self.step - 1] * (seq_len + 1)
                if self.input_ids == "increasing"
                else torch.randint(0, self.vocab_size, (self.seq_len + 1,)).long().tolist()
            )
            position_ids = list(range(seq_len))
            loss_mask = [True] * seq_len
            fake_sample = {
                "input_ids": input_ids[:-1],
                "target_ids": input_ids[1:],
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            self.num_samples["fake"] += 1
            self.num_tokens["fake"] += len(input_ids)
            yield fake_sample


class SFTDataset(StatefulIterableDataset):
    """A dataset wrapping a HF SFT dataset with prompt/completion or raw messages format."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer | None,
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 128,
        non_dp_size: int = 1,
        loss_mask_config: LossMaskConfig = LossMaskConfig(),
        max_examples: int | None = None,
        max_epochs: int | None = None,
        compaction_config: SFTCompactionConfig | None = None,
    ):
        super().__init__()
        self.logger = get_logger()
        self.dataset = dataset
        self.num_examples = len(self.dataset)
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.seed = seed
        self.seq_len = seq_len
        self.loss_mask_config = loss_mask_config
        self.max_examples = max_examples
        self.max_epochs = max_epochs
        self.compaction_config = compaction_config
        self.num_compaction_policy_samples = defaultdict(int)

        if self.tokenizer is None:
            self.logger.warning("No tokenizer provided, will not process examples")

        # If specified, select a subset of the dataset
        if self.max_examples is not None:
            self.num_examples = min(self.num_examples, self.max_examples)
            self.dataset = self.dataset.take(self.max_examples)

        # Get the data rank and world size
        worker_info = get_worker_info()
        worker_id, num_workers = 0, 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        assert get_world().world_size % non_dp_size == 0, "world_size must be divisible by non_dp_size"
        self.data_rank = get_world().rank // non_dp_size * num_workers + worker_id
        self.data_world_size = get_world().world_size // non_dp_size * num_workers

    def _process(self, example: dict) -> dict | None:
        # Skip processing if no tokenizer was provided
        if self.tokenizer is None:
            return example

        def resolve_messages(example: dict) -> list[dict]:
            # `messages` takes precedence over explicit split fields and is interpreted
            # as a whole-chat training sample with an empty prompt.
            if "messages" in example:
                messages = normalize_messages(example["messages"], default_role="assistant")
            elif "prompt" in example and "completion" in example:
                messages = normalize_messages(example["prompt"], default_role="user") + normalize_messages(
                    example["completion"], default_role="assistant"
                )
            else:
                raise ValueError(
                    "All examples in the dataset must have either a 'messages' column "
                    "or both 'prompt' and 'completion' columns for SFT"
                )

            # Deserialize tool call arguments from message list, if present - assumes OAI format
            # Reference: https://platform.openai.com/docs/guides/function-calling#handling-function-calls
            messages = deserialize_tool_calls(messages)

            # Strip content from all messages so that incremental tokenization works
            # NOTE: This has the side effect that we do never train on leading or trailing whitespace
            return strip_message_content(messages)

        messages = resolve_messages(example)

        # Parse available tools, if present - assumes OAI format
        # Reference: https://platform.openai.com/docs/guides/function-calling#function-tool-example
        tools = json.loads(example.get("tools") or "[]")

        def should_mask(message: dict) -> bool:
            assert "role" in message, "Message must have a role"
            match message["role"]:
                case "user":
                    return True if self.loss_mask_config.user else False
                case "assistant":
                    return True if self.loss_mask_config.assistant else False
                case "system":
                    return True if self.loss_mask_config.system else False
                case "tool":
                    return True if self.loss_mask_config.tool else False
                case _:
                    raise ValueError(f"Invalid message role: {message['role']}")

        if self.compaction_config is not None and self.compaction_config.enabled:
            compaction_config, policy_name = _select_sft_compaction_config(
                self.compaction_config,
                example,
                step=self.step,
                epoch=self.epoch,
            )
            input_ids, token_mask, calls = _build_incremental_token_mask_and_calls(
                self.tokenizer,
                messages,
                role_to_mask=should_mask,
                tools=tools,
                chat_template_kwargs=example.get("chat_template_kwargs", {}),
                compaction_config=compaction_config,
            )
            if len(input_ids) > self.seq_len:
                self.logger.warning(
                    f"Skipping compaction example {example.get('__index', '')} "
                    f"because it has {len(input_ids)} tokens, exceeding seq_len={self.seq_len}."
                )
                return None
            if policy_name is not None:
                self.num_compaction_policy_samples[policy_name] += 1
            target_ids = input_ids[1:] + [0]
            loss_mask = token_mask[1:] + [False]
            return {
                "input_ids": input_ids,
                "target_ids": target_ids,
                "loss_mask": loss_mask,
                "position_ids": list(range(len(input_ids))),
                "calls": calls,
            }

        input_ids, loss_mask = build_incremental_token_mask(
            self.tokenizer,
            messages,
            role_to_mask=should_mask,
            tools=tools,
            chat_template_kwargs=example.get("chat_template_kwargs", {}),
            collapse_consecutive_tool_messages=True,
        )

        # If EOS token is not found, manually append it
        if not self.tokenizer.eos_token_id in input_ids:
            self.logger.warning(
                f"Did not find EOS token ID {self.tokenizer.eos_token_id} in input_ids. Is something wrong with the chat template? Manually appending EOS token..."
            )
            input_ids.append(cast(int, self.tokenizer.eos_token_id))
            loss_mask.append(True)

        # Prepare inputs
        target_ids = input_ids.copy()[1:]
        loss_mask = loss_mask[1:]
        input_ids = input_ids[:-1]

        if sum(loss_mask[: self.seq_len]) == 0:
            self.logger.warning(
                f"Skipping example {example.get('__index', '')} because no trainable tokens were found within the context window ({self.seq_len}). This is to prevent NaN loss."
            )
            return

        assert len(input_ids) == len(loss_mask) == len(target_ids), (
            f"input_ids, loss_mask and target_ids must have the same length, but got {len(input_ids)=}, {len(loss_mask)=}, {len(target_ids)=}"
        )
        assert sum(loss_mask) > 0, "There are no tokens in this sample that contribute to the loss"
        assert self.tokenizer.eos_token_id in target_ids, "EOS token ID must be present in target_ids"

        # Create sample (with one fake target for the last token)
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "position_ids": list(range(len(input_ids))),
        }

    def __iter__(self):
        """
        Apply chat template and tokenize a single example in prompt + completion format (https://github.com/huggingface/trl/blob/de27d612b026526ba39b88eee348994d7636e033/trl/trainer/sft_trainer.py#L661)
        """
        dataset = self.dataset.shuffle(seed=self.epoch + self.seed) if self.shuffle else self.dataset
        while True:
            self.step += 1

            # Determine epoch from current step
            epoch = (self.step - 1) // self.num_examples

            # Break if max epochs is reached
            if self.max_epochs is not None and epoch >= self.max_epochs:
                break

            # Update stored epoch if new epoch is reached, optionally shuffle
            if epoch > self.epoch:
                self.epoch = epoch
                dataset = self.dataset.shuffle(seed=self.epoch + self.seed) if self.shuffle else self.dataset

            # Skip samples that don't belong to this data rank
            if (self.step - 1) % self.data_world_size != self.data_rank:
                continue

            # Get example
            example = dataset[(self.step - 1) % self.num_examples]

            # Process example
            processed_example = self._process(cast(dict, example))

            # If processed example is None, skip it (e.g. if tokenized sample exceeds context window)
            if processed_example is None:
                continue

            # Yield the example
            example = cast(dict, example)
            subset_or_split = example.get("__subset") or example.get("__split")
            self.logger.debug(
                f"Yield example {example.get('__index', '')}"
                + (f" from {subset_or_split} " if subset_or_split else " ")
                + f"with {len(processed_example.get('input_ids', []))} tokens ({sum(processed_example.get('loss_mask', []))} trainable tokens)"
            )
            self.num_samples[subset_or_split] += 1
            self.num_tokens[subset_or_split] += len(processed_example.get("input_ids", []))
            yield processed_example


class CatDataset(StatefulIterableDataset):
    """A dataset that concatenates samples into a single sequence with a fixed length."""

    def __init__(self, dataset: StatefulIterableDataset, seq_len: int):
        self.logger = get_logger()
        self.dataset = dataset
        self.seq_len = seq_len

    def state_dict(self) -> dict:
        return {"dataset": self.dataset.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict["dataset"])

    def __iter__(self):
        packed_samples, seq_len = defaultdict(list), 0
        for sample in self.dataset:
            # Add sample to packed samples
            for key, value in sample.items():
                assert isinstance(value, list), f"Value for key {key} must be a list"
                packed_samples[key].extend(value)

            # Update sequence length
            seq_len += len(sample["input_ids"])

            # If batch is full, truncate and yield it
            if seq_len >= self.seq_len:
                for key, value in packed_samples.items():
                    assert isinstance(value, list), f"Value for key {key} must be a list"
                    packed_samples[key] = value[: self.seq_len]
                yield packed_samples
                packed_samples, seq_len = defaultdict(list), 0


class PadCompactionDataset(StatefulIterableDataset):
    """Pad one synthetic compaction sample to a fixed length without packing."""

    def __init__(self, dataset: StatefulIterableDataset, seq_len: int):
        self.dataset = dataset
        self.seq_len = seq_len

    def state_dict(self) -> dict:
        return {"dataset": self.dataset.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict["dataset"])

    def __iter__(self):
        for sample in self.dataset:
            pad_len = self.seq_len - len(sample["input_ids"])
            if pad_len < 0:
                continue
            yield {
                "input_ids": sample["input_ids"] + [0] * pad_len,
                "target_ids": sample["target_ids"] + [0] * pad_len,
                "loss_mask": sample["loss_mask"] + [False] * pad_len,
                "position_ids": sample["position_ids"] + [0] * pad_len,
                "calls": sample["calls"],
            }


class StackDataset(StatefulIterableDataset):
    """A dataset that stacks samples into batch with a fixed area"""

    def __init__(self, dataset: StatefulIterableDataset, max_area: int):
        self.logger = get_logger()
        self.dataset = dataset
        self.max_area = max_area
        assert self.max_area % 256 == 0
        self.bucket_sizes = []
        while max_area % 256 == 0:
            self.bucket_sizes.insert(0, max_area)
            max_area //= 2
        self.logger.debug(f"Initialized {len(self.bucket_sizes)} buckets (bucket_sizes={self.bucket_sizes})")
        # Checkpoint state
        self.step = 0
        self.buckets = [[] for _ in range(len(self.bucket_sizes))]
        self.bucket_timers: list[int | None] = [None] * len(self.buckets)

    def state_dict(self) -> dict:
        return {
            "dataset": self.dataset.state_dict(),
            "step": self.step,
            "buckets": self.buckets,
            "bucket_timers": self.bucket_timers,
        }

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict["dataset"])
        self.step = state_dict["step"]
        self.buckets = state_dict["buckets"]
        self.bucket_timers = state_dict["bucket_timers"]

    def __iter__(self):
        for sample in self.dataset:
            # Truncate sample if it's longer than max area
            len_sample = len(sample["input_ids"])
            if len_sample > self.max_area:
                for key, value in sample.items():
                    assert isinstance(value, list)
                    sample[key] = sample[key][: self.max_area]
                len_sample = self.max_area

            # Add sample to bucket
            def find_bucket_idx(len_sample: int) -> int:
                bucket_idx = 0
                while bucket_idx < len(self.bucket_sizes) - 1 and len_sample > self.bucket_sizes[bucket_idx]:
                    bucket_idx += 1
                return bucket_idx

            bucket_idx = find_bucket_idx(len_sample)
            self.buckets[bucket_idx].append(sample)

            # Check if bucket has timed out
            bucket_timer = self.bucket_timers[bucket_idx]
            if bucket_timer is not None:
                hit_timeout = bucket_timer + STACKING_DATASET_BUCKET_TIMEOUT < self.step
            else:
                hit_timeout = False

            # Check if bucket is full
            is_full = self.bucket_sizes[bucket_idx] * len(self.buckets[bucket_idx]) >= self.max_area

            if is_full or hit_timeout:
                if hit_timeout:
                    while bucket_idx < len(self.buckets) - 1:
                        if (
                            self.bucket_sizes[bucket_idx + 1]
                            * (len(self.buckets[bucket_idx]) + len(self.buckets[bucket_idx + 1]))
                            < self.max_area
                        ):
                            self.buckets[bucket_idx + 1].extend(self.buckets[bucket_idx])
                            self.buckets[bucket_idx] = []
                            self.bucket_timers[bucket_idx] = None
                            bucket_idx += 1
                        else:
                            break

                    while self.bucket_sizes[bucket_idx] * len(self.buckets[bucket_idx]) < self.max_area:
                        dummy_sample = {}
                        for key, value in sample.items():
                            dummy_sample[key] = [0]
                        self.buckets[bucket_idx].append(dummy_sample)

                packed_samples = defaultdict(list)
                num_samples, num_tokens, num_trainable_tokens, num_pad_tokens = 0, 0, 0, 0
                for bucket_item in self.buckets[bucket_idx]:
                    num_samples += 1
                    for key, value in bucket_item.items():
                        pad_tokens = [0] * (self.bucket_sizes[bucket_idx] - len(value))
                        if key == "loss_mask":
                            num_tokens += len(value)
                            num_trainable_tokens += sum(value)
                            num_pad_tokens += len(pad_tokens)
                        packed_samples[key].append(value + pad_tokens)
                reason = "bucket is full" if is_full else "because bucket timed out"
                reason += " and " if is_full and hit_timeout else ""
                reason += "bucket timed out" if hit_timeout else ""
                self.logger.debug(
                    f"Yield bucket {bucket_idx} because {reason} with {num_samples=}, {num_tokens=}, {num_trainable_tokens=}, {num_pad_tokens=}"
                )
                yield packed_samples
                self.step += 1
                self.buckets[bucket_idx] = []
                self.bucket_timers[bucket_idx] = None
            else:
                if self.bucket_timers[bucket_idx] is None:
                    self.bucket_timers[bucket_idx] = self.step


def stack_collate(samples: list[Sample]) -> Batch:
    batch = {
        "input_ids": torch.tensor(samples[0]["input_ids"], dtype=torch.long, device="cuda"),
        "position_ids": torch.tensor(samples[0]["position_ids"], dtype=torch.long, device="cuda"),
        "loss_mask": torch.tensor(samples[0]["loss_mask"], dtype=torch.bool, device="cuda"),
        "target_ids": torch.tensor(samples[0]["target_ids"], dtype=torch.long, device="cuda"),
    }
    if "calls" in samples[0]:
        batch["calls"] = samples[0]["calls"]
    return batch


def cat_collate(samples: list[Sample]) -> Batch:
    batch = {
        "input_ids": torch.stack([torch.tensor(sample["input_ids"]) for sample in samples], dim=0).long().to("cuda"),
        "position_ids": torch.stack([torch.tensor(sample["position_ids"]) for sample in samples], dim=0)
        .long()
        .to("cuda"),
        "loss_mask": torch.stack([torch.tensor(sample["loss_mask"]) for sample in samples], dim=0).bool().to("cuda"),
        "target_ids": torch.stack([torch.tensor(sample["target_ids"]) for sample in samples], dim=0).long().to("cuda"),
    }
    if "calls" in samples[0]:
        batch["calls"] = samples[0]["calls"]
    return batch


def setup_and_interleave_datasets(
    dataset_name: str,
    subsets_and_splits: list[tuple[str | None, str]],
    probabilities: list[float] | None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"],
    seed: int = 0,
) -> Dataset:
    logger = get_logger()
    datasets = []
    for subset, split in subsets_and_splits:
        logger.debug(f"Loading dataset {dataset_name} with {subset=} and {split=}")
        dataset = cast(Dataset, load_dataset(dataset_name, subset, split=split))
        num_examples = len(dataset)
        dataset = dataset.add_column("__subset", [subset] * num_examples, new_fingerprint=str(uuid.uuid4()))
        dataset = dataset.add_column("__split", [split] * num_examples, new_fingerprint=str(uuid.uuid4()))
        dataset = dataset.add_column("__index", list(range(num_examples)), new_fingerprint=str(uuid.uuid4()))
        datasets.append(dataset)
    if len(datasets) > 1:
        logger.debug(f"Interleaving datasets with {probabilities=} and {stopping_strategy=}")
        dataset = interleave_datasets(
            datasets,
            probabilities=probabilities,
            stopping_strategy=stopping_strategy,
            seed=seed,
        )
    else:
        dataset = datasets[0]

    return dataset


def load_sft_dataset(config: SFTDataConfig) -> Dataset:
    """Load and interleave the raw HF dataset. This is the expensive I/O step."""
    logger = get_logger()
    dataset_path = Path(config.name)
    if dataset_path.exists() and (dataset_path / "dataset_info.json").exists():
        return cast(Dataset, load_from_disk(str(dataset_path)))
    if config.subsets is None and config.splits is None:
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(None, "train")],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )
    elif config.subsets is not None and config.splits is None:
        logger.debug(f"Loading datasets for subsets {config.subsets} with default split 'train'")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(subset, "train") for subset in config.subsets],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )
    elif config.subsets is None and config.splits is not None:
        logger.debug(f"Loading datasets for splits {config.splits} with default subset 'None'")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(None, split) for split in config.splits],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )
    else:
        assert config.subsets is not None and config.splits is not None
        logger.debug(f"Loading datasets for subsets {config.subsets} with splits {config.splits}")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=list(zip(config.subsets, config.splits)),
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )


def setup_dataset(
    tokenizer: PreTrainedTokenizer,
    config: DataConfig,
    non_dp_size: int = 1,
    *,
    max_epochs: int | None = None,
    raw_dataset: Dataset | None = None,
    compaction_config: SFTCompactionConfig | None = None,
) -> StatefulIterableDataset:
    if config.type == "fake":
        return FakeDataset(
            vocab_size=tokenizer.vocab_size, seq_len=config.seq_len, length=config.length, input_ids=config.input_ids
        )
    elif config.type == "sft":
        if raw_dataset is None:
            raw_dataset = load_sft_dataset(config)
        return SFTDataset(
            raw_dataset,
            tokenizer,
            shuffle=config.shuffle,
            seed=config.seed,
            seq_len=config.seq_len,
            loss_mask_config=config.loss_mask,
            non_dp_size=non_dp_size,
            max_epochs=max_epochs,
            compaction_config=compaction_config,
        )
    else:
        raise ValueError(f"Invalid dataset type: {config.type}")


def setup_dataloader(
    dataset: StatefulIterableDataset,
    config: DataConfig,
    compaction_config: SFTCompactionConfig | None = None,
) -> StatefulDataLoader:
    if compaction_config is not None and compaction_config.enabled:
        padded_dataset = PadCompactionDataset(dataset, config.seq_len)
        return StatefulDataLoader(padded_dataset, batch_size=1, collate_fn=cat_collate)
    if config.pack_function == "stack":
        stacking_dataset = StackDataset(dataset, config.seq_len * config.micro_batch_size)
        return StatefulDataLoader(stacking_dataset, batch_size=1, collate_fn=stack_collate)
    elif config.pack_function == "cat":
        packing_dataset = CatDataset(dataset, config.seq_len * config.micro_batch_size)
        return StatefulDataLoader(packing_dataset, batch_size=1, collate_fn=cat_collate)
    else:
        raise ValueError(f"Invalid pack function: {config.pack_function}")
