import base64
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

# When the orchestrator enables Markovian Thinker, each extension-break
# starts a fresh TrainingSample from a re-truncated prompt. The system
# prefix must survive truncation — a silent drop would let the model
# train without its instructions. We cheaply assert that every new
# sample's prompt_ids begins with the same first token as the very
# first sample's prompt_ids (typically `<|im_start|>`). Off by default
# (gated by the env var the orchestrator sets when markovian is enabled).
_MARKOVIAN_ASSERT_ENABLED = os.environ.get("KV_EVICTION_MARKOVIAN_ENABLED") == "1"

import torch
import verifiers as vf
from PIL import Image
from transformers.tokenization_utils import PreTrainedTokenizer

# Compatibility shim: newer verifiers exposes `RolloutOutput` as the type of
# a completed rollout, but older versions (0.1.9.post3, pinned by some
# downstream integrators including kv-eviction's container setup) only have
# `State`. `RolloutOutput` is behaviorally identical to `State` for all the
# dict accesses this file performs, so aliasing is safe.
if not hasattr(vf, "RolloutOutput"):
    vf.RolloutOutput = vf.State  # type: ignore[attr-defined]

from prime_rl.transport import TrainingSample
from prime_rl.transport.types import CallWire, CompactionEventWire
from prime_rl.utils.chat_template import (
    common_prefix_len,
    deserialize_tool_calls,
    normalize_messages,
    render_messages,
    strip_message_content,
)
from prime_rl.utils.logger import get_logger

# We use list() instead of deepcopy() for flat lists (token IDs, logprobs) - safe because
# primitives are immutable. pixel_values/image_grid_thw are not mutated after creation.


def _align_routed_experts(
    routed_experts: list[list[list[int]]] | None,
    expected_len: int,
) -> list[list[list[int]]] | None:
    """Align routed_experts length with the expected token count.

    VLLM's capturer uses `num_tokens - 1` slot mappings because the final
    generated token was never fed as input to a forward pass and has no
    routing decision. Append zero-filled entries for the missing positions.
    """
    if routed_experts is None or not routed_experts:
        return routed_experts
    deficit = expected_len - len(routed_experts)
    if deficit <= 0:
        return routed_experts
    num_layers = len(routed_experts[0])
    topk = len(routed_experts[0][0])
    zero_entry = [[0] * topk for _ in range(num_layers)]
    return routed_experts + [zero_entry for _ in range(deficit)]


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    return common_prefix_len(a, b)


def _normalize_messages(messages: Any, default_role: str) -> list[dict[str, Any]]:
    return normalize_messages(messages, default_role)


def _deserialize_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return deserialize_tool_calls(messages)


def _strip_message_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return strip_message_content(messages)


def _render_messages(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = False,
    tools: list[dict[str, Any]] | None = None,
    processor=None,
) -> list[int]:
    return render_messages(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        processor=processor,
    )


def _prepare_messages_for_processor(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert messages to the format expected by the VLM processor.

    - Converts image_url items to image items with loaded PIL Images
    - Strips extra fields (e.g. image_url on text items) that confuse the processor
    - Ensures all message content is in list format (processor requires this)
    """
    prepared = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            prepared.append({**msg, "content": [{"type": "text", "text": content}]})
            continue

        if not isinstance(content, list):
            prepared.append(msg)
            continue

        new_content = []
        for item in content:
            if item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url.startswith(_FILE_URL_PREFIX):
                    img = _load_file_image(url)
                elif url.startswith("data:image"):
                    b64_data = url.split(",", 1)[1]
                    img = Image.open(BytesIO(base64.b64decode(b64_data)))
                else:
                    new_content.append(item)
                    continue
                new_content.append({"type": "image", "image": img})
            elif item.get("type") == "text":
                new_content.append({"type": "text", "text": item.get("text", "")})
            else:
                new_content.append(item)
        prepared.append({**msg, "content": new_content})

    return prepared


def _tokenize_step_from_messages(
    step: vf.TrajectoryStep,
    tokenizer: PreTrainedTokenizer,
    tools: list[dict[str, Any]] | None = None,
    processor=None,
) -> dict[str, Any]:
    prompt = _normalize_messages(step.get("prompt"), default_role="user")
    completion = _normalize_messages(step.get("completion"), default_role="assistant")

    prompt = _strip_message_content(_deserialize_tool_calls(prompt))
    completion = _strip_message_content(_deserialize_tool_calls(completion))

    assert all(m.get("role") == "assistant" for m in completion), (
        "Expected all completion messages to be assistant role for SFT distillation, "
        f"got roles: {[m.get('role') for m in completion]}"
    )

    if processor is not None:
        prompt = _prepare_messages_for_processor(prompt)
        completion = _prepare_messages_for_processor(completion)

    all_messages = prompt + completion
    prompt_has_assistant_completion = len(completion) > 0 and completion[0].get("role") == "assistant"
    prompt_ids = _render_messages(
        tokenizer,
        prompt,
        add_generation_prompt=prompt_has_assistant_completion,
        tools=tools,
        processor=processor,
    )
    full_ids = _render_messages(
        tokenizer,
        all_messages,
        tools=tools,
        processor=processor,
    )

    split_idx = _common_prefix_len(prompt_ids, full_ids)
    original_prompt_len = len(prompt_ids)

    prompt_ids = full_ids[:split_idx]
    completion_ids = full_ids[split_idx:]
    completion_mask = [True] * len(completion_ids)
    completion_logprobs = [0.0] * len(completion_ids)

    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": [False] * len(prompt_ids),
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "completion_logprobs": completion_logprobs,
        "routed_experts": None,
        "prompt_prefix_len": split_idx,
        "original_prompt_len": original_prompt_len,
    }


def _convert_tools_to_oai_format(tool_defs: list) -> list[dict[str, Any]] | None:
    """Convert verifiers Tool objects or dicts to OAI function-calling format."""
    if not tool_defs:
        return None

    def _get(tool: Any, key: str) -> Any:
        if isinstance(tool, dict):
            return tool.get(key)
        return getattr(tool, key, None)

    return [
        {
            "type": "function",
            "function": {
                "name": _get(tool, "name"),
                "description": _get(tool, "description"),
                "parameters": _get(tool, "parameters"),
                **({} if _get(tool, "strict") is None else {"strict": _get(tool, "strict")}),
            },
        }
        for tool in tool_defs
    ]


def pretokenize_rollout_trajectory(
    output: vf.RolloutOutput,
    tokenizer: PreTrainedTokenizer,
    processor=None,
) -> bool:
    """Populate missing step tokens from prompt/completion messages."""
    logger = get_logger()
    tools = _convert_tools_to_oai_format(output.get("tool_defs", []))

    for step_idx, step in enumerate(output["trajectory"]):
        if step["tokens"] is not None:
            continue

        reconstructed = _tokenize_step_from_messages(step, tokenizer, tools=tools, processor=processor)
        if reconstructed["prompt_prefix_len"] < reconstructed["original_prompt_len"]:
            logger.debug(
                f"Prompt tokenization was non-prefix for example {output['example_id']} step {step_idx}. "
                f"Using longest common prefix length {reconstructed['prompt_prefix_len']} "
                f"(original prompt had {reconstructed['original_prompt_len']} tokens)."
            )

        reconstructed.pop("prompt_prefix_len")
        reconstructed.pop("original_prompt_len")
        step["tokens"] = reconstructed

    return True


def _build_summary_sample(
    step: vf.TrajectoryStep,
    *,
    temperature: float,
    has_error: bool,
) -> TrainingSample | None:
    """Construct a standalone TrainingSample from a summary payload
    stashed on a trajectory step's extras.

    Returns None when the step has no summary payload, or when the
    payload is malformed (missing prompt/completion tokens, or logprobs
    length mismatch with completion tokens). The regular per-step
    sample emission is unaffected.

    Full-credit: ``completion_mask`` is all-True when the rollout did
    not error; for errored rollouts we zero the mask (same policy as
    make_sample above). ``compaction_events`` is populated from the
    payload in ``mode="eviction"`` (vLLM-side eviction can fire during
    the summary call's own prefill/decode); ``None`` in
    ``mode="markovian"`` where the summary call runs with vLLM
    compaction off.
    """
    extras = step.get("extras") if isinstance(step, dict) else None
    if not extras:
        return None
    payload = extras.get("summary_trainsample")
    if not isinstance(payload, dict):
        return None
    try:
        prompt_ids = [int(x) for x in (payload.get("prompt_token_ids") or [])]
        completion_ids = [
            int(x) for x in (payload.get("completion_token_ids") or [])
        ]
        completion_logprobs = [
            float(x) for x in (payload.get("completion_logprobs") or [])
        ]
    except (TypeError, ValueError):
        return None
    if not prompt_ids or not completion_ids:
        return None
    if len(completion_logprobs) != len(completion_ids):
        return None

    # Coerce payload's ``compaction_events`` list to
    # ``CompactionEventWire`` instances. Mirrors the same
    # dict / list-tuple / typed handling as the regular per-step path
    # in ``_compaction_events_from_step`` — summary payloads traverse
    # the same msgspec boundary and can arrive as any of those shapes.
    raw_events = payload.get("compaction_events")
    summary_events: list[CompactionEventWire] | None = None
    if raw_events:
        coerced: list[CompactionEventWire] = []
        for e in raw_events:
            if isinstance(e, CompactionEventWire):
                coerced.append(e)
            elif isinstance(e, dict):
                try:
                    coerced.append(
                        CompactionEventWire(
                            num_output_tokens_at_compaction=int(
                                e["num_output_tokens_at_compaction"]
                            ),
                            tokens_evicted=int(e["tokens_evicted"]),
                            position_offset_after=int(e["position_offset_after"]),
                            num_prompt_tokens=int(e.get("num_prompt_tokens", 0)),
                            evict_start=int(e.get("evict_start", 0)),
                            new_user_fragment_len=int(
                                e.get("new_user_fragment_len", 0)
                            ),
                            kept_indices=[
                                int(x) for x in (e.get("kept_indices") or [])
                            ],
                            kept_token_ids=[
                                int(x) for x in (e.get("kept_token_ids") or [])
                            ],
                            last_turn_evicted=int(
                                e.get("last_turn_evicted", -1)
                            ),
                            num_turns_evicted_after=int(
                                e.get("num_turns_evicted_after", 0)
                            ),
                            archived_span_ids=[
                                str(x) for x in (e.get("archived_span_ids") or [])
                            ],
                            archived_span_bounds=[
                                int(x)
                                for x in (e.get("archived_span_bounds") or [])
                            ],
                            event_kind=int(e.get("event_kind", 0)),
                            restored_span_ids=[
                                str(x)
                                for x in (e.get("restored_span_ids") or [])
                            ],
                            visibility_boundary_computed=int(
                                e.get("visibility_boundary_computed", -1)
                            ),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            elif isinstance(e, (list, tuple)) and len(e) >= 3:
                try:
                    coerced.append(
                        CompactionEventWire(
                            num_output_tokens_at_compaction=int(e[0]),
                            tokens_evicted=int(e[1]),
                            position_offset_after=int(e[2]),
                            num_prompt_tokens=int(e[3]) if len(e) >= 4 else 0,
                            evict_start=int(e[4]) if len(e) >= 5 else 0,
                            new_user_fragment_len=int(e[5]) if len(e) >= 6 else 0,
                            kept_indices=[int(x) for x in e[6]]
                            if len(e) >= 7 and e[6]
                            else [],
                            kept_token_ids=[int(x) for x in e[7]]
                            if len(e) >= 8 and e[7]
                            else [],
                            last_turn_evicted=int(e[8]) if len(e) >= 9 else -1,
                            num_turns_evicted_after=int(e[9]) if len(e) >= 10 else 0,
                            archived_span_ids=[str(x) for x in e[10]]
                            if len(e) >= 11 and e[10]
                            else [],
                            archived_span_bounds=[int(x) for x in e[11]]
                            if len(e) >= 12 and e[11]
                            else [],
                            event_kind=int(e[12]) if len(e) >= 13 else 0,
                            restored_span_ids=[str(x) for x in e[13]]
                            if len(e) >= 14 and e[13]
                            else [],
                            visibility_boundary_computed=int(e[14])
                            if len(e) >= 15
                            else -1,
                        )
                    )
                except (TypeError, ValueError):
                    continue
        summary_events = coerced or None

    completion_mask = (
        [False] * len(completion_ids)
        if has_error
        else [True] * len(completion_ids)
    )
    return TrainingSample(
        prompt_ids=prompt_ids,
        prompt_mask=[False] * len(prompt_ids),
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_logprobs=completion_logprobs,
        completion_temperatures=[temperature] * len(completion_ids),
        teacher_logprobs=None,
        advantage=None,
        routed_experts=None,
        compaction_events=summary_events,
    )


def interleave_rollout(
    output: vf.RolloutOutput,
    vlm_cache: "VLMImageCache | None" = None,
    cache_key: int | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    log_evicted_text: bool = False,
) -> list[TrainingSample] | None:
    """
    Convert vf.RolloutOutput to trainable rollouts by interleaving trajectory steps
    where the extension property holds.

    When consecutive steps share token prefixes (extension property), they are
    merged into a single sample. When extension breaks (e.g., due to context
    compaction or a change in control-flow), a new sample is started.

    Supports multi-prefix matching to handle interleaved agents. For example,
    [agent1-step1, agent1-step2, agent2-step1, agent1-step3] produces two samples:
    agent1 steps merged together, agent2 step separate.

    Returns a list of samples - could be 1 (extension always held) or up to T
    (extension never held).

    For VLM models, pass vlm_cache to attach cumulative pixel_values per sample.
    Each sample gets the images accumulated up to its last merged step.

    Args:
        output: vf.RolloutOutput containing trajectory data
        vlm_cache: Pre-computed VLM image cache for multimodal training
        cache_key: Cache key to use when retrieving images from the VLM cache
    """
    logger = get_logger()

    trajectory = output["trajectory"]
    if len(trajectory) == 0:
        error = output.get("error")
        stop = output.get("stop_condition")
        logger.warning(
            f"No trajectory steps for example {output['example_id']} (error={error}, stop={stop}). Skipping rollout."
        )
        return None

    has_error = output["error"] is not None
    # this field should be guaranteed because we set temperature in get_sampling_args
    temperature = output["sampling_args"]["temperature"]

    def prepare_step_tokens(step: vf.TrajectoryStep, step_idx: int) -> dict[str, Any] | None:
        tokens = step["tokens"]
        if tokens is not None:
            prompt_ids = list(tokens["prompt_ids"])
            prompt_mask = [bool(i) for i in tokens["prompt_mask"]]

            # Block-aligned padding mode: the kv_eviction interceptor
            # stashed the EXACT padded token stream vLLM ran on in the
            # step's extras. Override the verifiers-re-tokenized prompt_ids
            # (which is unpadded) so the trainer sees what inference saw.
            # prompt_mask is rebuilt as all-False because (a) every prompt
            # position is no-gradient by construction and (b) the existing
            # per-token verifiers mask applies to the unpadded sequence
            # and can't be re-aligned cheaply.
            extras = step.get("extras")
            padded_ids = extras.get("prompt_token_ids") if extras else None
            if padded_ids:
                prompt_ids = [int(x) for x in padded_ids]
                prompt_mask = [False] * len(prompt_ids)

            # Capture the PRE-trim submitted prompt + the full event list
            # (admission + mid-gen) BEFORE _apply_admission_trim mutates
            # them. The per-call trainer rebuild (Phase B+) needs the
            # pre-trim prompt for phase-1 of the two-phase forward.
            submitted_pre_trim: list[int] = list(prompt_ids)
            step_events = _compaction_events_from_step(step)
            all_events_pre_trim: list[CompactionEventWire] = (
                list(step_events) if step_events else []
            )

            # Admission-time compaction trim: the vLLM response carries
            # the ORIGINAL (pre-trim) prompt_token_ids, but inference ran
            # on the trimmed prompt. Replay the scheduler's token
            # deletions so the trainer sees identical tokens + positions.
            if step_events and padded_ids:
                orig_len = len(prompt_ids)
                prompt_ids, prompt_mask, step_events = _apply_admission_trim(
                    prompt_ids, prompt_mask, step_events,
                    step_idx=step_idx,
                    example_id=output.get("example_id", "?"),
                )
                if len(prompt_ids) < orig_len:
                    logger.info(
                        f"[ADMISSION-TRIM] step {step_idx}: "
                        f"prompt {orig_len} -> {len(prompt_ids)} tokens, "
                        f"{len(step_events)} mid-gen events remaining"
                    )
                # Write surviving events back to the step's extras so the
                # downstream step_compaction_events list sees the trimmed
                # set (admission events consumed, only mid-gen remain).
                if extras is not None:
                    if step_events:
                        extras["compaction_events"] = [
                            {
                                "num_output_tokens_at_compaction": e.num_output_tokens_at_compaction,
                                "tokens_evicted": e.tokens_evicted,
                                "position_offset_after": e.position_offset_after,
                                "num_prompt_tokens": e.num_prompt_tokens,
                                "evict_start": e.evict_start,
                                "new_user_fragment_len": e.new_user_fragment_len,
                                "kept_indices": list(e.kept_indices),
                                "kept_token_ids": list(e.kept_token_ids),
                                "last_turn_evicted": e.last_turn_evicted,
                                "num_turns_evicted_after": e.num_turns_evicted_after,
                                "archived_span_ids": list(e.archived_span_ids),
                            }
                            for e in step_events
                        ]
                    else:
                        extras["compaction_events"] = None

            # vLLM auto-pad: filler token ids appended to this call's KV
            # cache after sampling stopped (so the trailing block lands in
            # the prefix cache). The trainer needs these per-call so its
            # persistent_cache layout matches vLLM's. Empty list when
            # auto-pad did not fire for this call.
            trailing_pad_ids: list[int] = []
            if extras is not None:
                raw_pad = extras.get("padding_token_ids")
                if raw_pad:
                    try:
                        trailing_pad_ids = [int(x) for x in raw_pad]
                    except (TypeError, ValueError):
                        trailing_pad_ids = []

            return {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": list(tokens["completion_ids"]),
                "completion_mask": [bool(i) for i in tokens["completion_mask"]],
                "completion_logprobs": list(tokens["completion_logprobs"]),
                "routed_experts": tokens.get("routed_experts"),
                # Per-call rebuild: pre-trim submitted prompt + all events
                # (admission + mid-gen). The trainer's two-phase forward
                # needs the pre-trim prompt to run phase 1 over
                # [0, evict_end).
                "submitted_prompt_ids_pre_trim": submitted_pre_trim,
                "all_compaction_events_pre_trim": all_events_pre_trim,
                "trailing_pad_ids": trailing_pad_ids,
            }

        logger.warning(f"Missing rollout tokens for example {output['example_id']} step {step_idx}.")
        return None

    def _apply_admission_trim(
        prompt_ids: list[int],
        prompt_mask: list[bool],
        events: list[CompactionEventWire],
        step_idx: int = -1,
        example_id: Any = "?",
    ) -> tuple[list[int], list[bool], list[CompactionEventWire]]:
        """Apply admission-time compaction events to the prompt.

        Admission events (num_output_tokens_at_compaction == 0) represent
        token deletions that the vLLM scheduler applied to the prompt
        BEFORE prefill. The response carries the original (pre-trim)
        prompt, so we replay the deletions here so the trainer sees the
        same tokens inference ran on.

        Events are applied in order (oldest-first), matching the
        scheduler's _apply_trim sequence. Each event's evict_start is
        relative to the CURRENT (already-partially-trimmed) prompt.

        Returns (trimmed_prompt, trimmed_mask, remaining_events) where
        remaining_events contains only mid-generation events (output > 0).
        """
        trimmed_ids = list(prompt_ids)
        trimmed_mask = list(prompt_mask)
        remaining: list[CompactionEventWire] = []
        ev_idx = 0
        for evt in events:
            if evt.num_output_tokens_at_compaction == 0:
                start = evt.evict_start
                end = start + evt.tokens_evicted
                if log_evicted_text and tokenizer is not None:
                    evicted_ids = trimmed_ids[start:end]
                    text = tokenizer.decode(
                        evicted_ids, skip_special_tokens=False
                    )
                    framed = "\n".join(
                        f"  | {line}" for line in text.split("\n")
                    )
                    logger.info(
                        f"[COMPACT-TEXT] ex={example_id} step={step_idx} "
                        f"#{ev_idx} ADMISSION evict=[{start},{end}) "
                        f"({evt.tokens_evicted} tokens, {len(text)} chars)\n"
                        f"  {'-' * 70}\n"
                        f"{framed}\n"
                        f"  {'-' * 70}"
                    )
                del trimmed_ids[start:end]
                del trimmed_mask[start:end]
                ev_idx += 1
            else:
                remaining.append(evt)
        return trimmed_ids, trimmed_mask, remaining

    def _compaction_events_from_step(
        step: vf.TrajectoryStep,
    ) -> list[CompactionEventWire] | None:
        """Read compaction events from a step's extras dict, handling both
        already-typed CompactionEventWire instances and the dict/list forms
        that can arrive after msgspec roundtrip. None when no events present.
        """
        extras = step.get("extras")
        if not extras:
            return None
        raw = extras.get("compaction_events")
        if not raw:
            return None
        out: list[CompactionEventWire] = []
        for e in raw:
            if isinstance(e, CompactionEventWire):
                out.append(e)
            elif isinstance(e, dict):
                out.append(
                    CompactionEventWire(
                        num_output_tokens_at_compaction=int(
                            e["num_output_tokens_at_compaction"]
                        ),
                        tokens_evicted=int(e["tokens_evicted"]),
                        position_offset_after=int(e["position_offset_after"]),
                        num_prompt_tokens=int(e.get("num_prompt_tokens", 0)),
                        evict_start=int(e.get("evict_start", 0)),
                        new_user_fragment_len=int(
                            e.get("new_user_fragment_len", 0)
                        ),
                        kept_indices=[
                            int(x) for x in (e.get("kept_indices") or [])
                        ],
                        kept_token_ids=[
                            int(x) for x in (e.get("kept_token_ids") or [])
                        ],
                        last_turn_evicted=int(
                            e.get("last_turn_evicted", -1)
                        ),
                        num_turns_evicted_after=int(
                            e.get("num_turns_evicted_after", 0)
                        ),
                        archived_span_ids=[
                            str(x) for x in (e.get("archived_span_ids") or [])
                        ],
                        archived_span_bounds=[
                            int(x)
                            for x in (e.get("archived_span_bounds") or [])
                        ],
                        event_kind=int(e.get("event_kind", 0)),
                        restored_span_ids=[
                            str(x) for x in (e.get("restored_span_ids") or [])
                        ],
                        visibility_boundary_computed=int(
                            e.get("visibility_boundary_computed", -1)
                        ),
                        restored_span_token_ids=[
                            int(x)
                            for x in (e.get("restored_span_token_ids") or [])
                        ],
                        restored_span_pos_start=int(
                            e.get("restored_span_pos_start", -1)
                        ),
                    )
                )
            elif isinstance(e, (list, tuple)) and len(e) >= 3:
                # msgspec array_like form
                out.append(
                    CompactionEventWire(
                        num_output_tokens_at_compaction=int(e[0]),
                        tokens_evicted=int(e[1]),
                        position_offset_after=int(e[2]),
                        num_prompt_tokens=int(e[3]) if len(e) >= 4 else 0,
                        evict_start=int(e[4]) if len(e) >= 5 else 0,
                        new_user_fragment_len=int(e[5]) if len(e) >= 6 else 0,
                        kept_indices=[int(x) for x in e[6]]
                        if len(e) >= 7 and e[6]
                        else [],
                        kept_token_ids=[int(x) for x in e[7]]
                        if len(e) >= 8 and e[7]
                        else [],
                        last_turn_evicted=int(e[8]) if len(e) >= 9 else -1,
                        num_turns_evicted_after=int(e[9]) if len(e) >= 10 else 0,
                        archived_span_ids=[str(x) for x in e[10]]
                        if len(e) >= 11 and e[10]
                        else [],
                        archived_span_bounds=[int(x) for x in e[11]]
                        if len(e) >= 12 and e[11]
                        else [],
                        event_kind=int(e[12]) if len(e) >= 13 else 0,
                        restored_span_ids=[str(x) for x in e[13]]
                        if len(e) >= 14 and e[13]
                        else [],
                        visibility_boundary_computed=int(e[14])
                        if len(e) >= 15
                        else -1,
                        restored_span_token_ids=[int(x) for x in e[15]]
                        if len(e) >= 16 and e[15]
                        else [],
                        restored_span_pos_start=int(e[16])
                        if len(e) >= 17
                        else -1,
                    )
                )
        return out or None

    prepared_steps: list[dict[str, Any]] = []
    step_compaction_events: list[list[CompactionEventWire] | None] = []
    for step_idx, step in enumerate(trajectory):
        prepared = prepare_step_tokens(step, step_idx)
        if prepared is None:
            return None
        prepared_steps.append(prepared)
        step_compaction_events.append(_compaction_events_from_step(step))

    # --- Diagnostic: detect coordinate mismatch between compaction events
    # and completion_ids BEFORE the merge loop runs (and potentially asserts).
    example_id = output.get("example_id", "?")
    for diag_idx, (diag_tokens, diag_events) in enumerate(
        zip(prepared_steps, step_compaction_events)
    ):
        if diag_events:
            max_raw = max(e.num_output_tokens_at_compaction for e in diag_events)
            clen = len(diag_tokens["completion_ids"])
            plen = len(diag_tokens["prompt_ids"])
            total_evicted = sum(e.tokens_evicted for e in diag_events)
            # Notebook-style per-event log (one line per CompactionEvent).
            # Mirrors experiments/debug_balrog/compaction_test.ipynb output:
            #   #0 evicted=352 offset_after=352 prompt_tokens=996 ...
            logger.info(
                f"[COMPACT] ex={example_id} step={diag_idx} prompt_len={plen} "
                f"completion_len={clen} events={len(diag_events)}"
            )
            for ev_idx, ev in enumerate(diag_events):
                logger.info(
                    f"[COMPACT]   #{ev_idx} evicted={ev.tokens_evicted} "
                    f"offset_after={ev.position_offset_after} "
                    f"prompt_tokens={ev.num_prompt_tokens} "
                    f"output_at_compact={ev.num_output_tokens_at_compaction}"
                )
            # Debug-only: decode each event's evicted token range from the
            # orchestrator's local (prompt + completion) buffer and log a
            # framed `| ... |` block. Mirrors the notebook trace format
            # (experiments/debug_balrog/compaction_test.ipynb / out_turn.txt).
            #
            # Range math (mirrors vLLM scheduler.py:_effective_prompt_tokens
            # + evict_start computation):
            #   1. protected_prefix_len = first <|im_end|> position + 1.
            #      vLLM auto-detects this by scanning for eos_token_id; in
            #      turn mode the protected region is exactly the system
            #      message.
            #   2. evict_base = align_up(protected_prefix_len, block_size).
            #   3. position_offset_after is the cumulative count of evicted
            #      tokens after this event. Events on a step are contiguous
            #      and ordered, so for event N:
            #         start = evict_base + (offset_after - tokens_evicted)
            #         end   = evict_base + offset_after
            #      The system prompt is never evicted, so the indices stay
            #      valid into the orchestrator's (prompt+completion) buffer.
            if log_evicted_text and tokenizer is not None:
                # Use the EXACT padded prompt_token_ids vLLM ran on, taken
                # from the kv_eviction interceptor's stash on extras. The
                # `prepare_step_tokens` override should already substitute
                # this — but reading extras directly avoids any ambiguity
                # about which buffer (padded vs unpadded) we're slicing.
                step_extras = trajectory[diag_idx].get("extras") or {}
                padded_prompt_ids = step_extras.get("prompt_token_ids")
                if padded_prompt_ids:
                    prompt_ids_for_decode = [int(x) for x in padded_prompt_ids]
                else:
                    prompt_ids_for_decode = list(diag_tokens["prompt_ids"])
                full_ids = prompt_ids_for_decode + list(
                    diag_tokens["completion_ids"]
                )
                eos_id = tokenizer.eos_token_id
                protected_prefix_len = len(prompt_ids_for_decode)  # fallback
                if eos_id is not None:
                    for i, tok_id in enumerate(prompt_ids_for_decode):
                        if tok_id == eos_id:
                            protected_prefix_len = i + 1
                            break
                # One-shot diagnostic per step: confirm we're slicing the
                # padded buffer (filler count > 0 expected on multi-turn
                # rollouts). Under AFTER-padding, <|im_end|> sits at its
                # natural position; fillers occupy slots between
                # <|im_end|> and the next block boundary. The useful
                # invariant is therefore "first_boundary is block-
                # aligned", not "first_im_end is aligned".
                _filler_count = sum(
                    1 for t in prompt_ids_for_decode if t == 151643
                )
                _first_im_end = (
                    protected_prefix_len - 1
                    if protected_prefix_len < len(prompt_ids_for_decode)
                    else -1
                )
                _first_boundary = (
                    ((_first_im_end + 1 + 15) // 16) * 16
                    if _first_im_end >= 0 else -1
                )
                logger.info(
                    f"[COMPACT-DEBUG] ex={example_id} step={diag_idx} "
                    f"prompt_len={len(prompt_ids_for_decode)} "
                    f"src={'extras' if padded_prompt_ids else 'prep'} "
                    f"fillers={_filler_count} first_im_end={_first_im_end} "
                    f"first_boundary={_first_boundary} "
                    f"block_aligned={_first_boundary % 16 == 0 if _first_boundary >= 0 else 'n/a'}"
                )
                # block_size is the same as inference.vllm_extra.block_size.
                # Hardcoded to 16 here to match the only supported value
                # (PagedAttention block_size is fixed across the run).
                block_size = 16
                evict_base = (
                    (protected_prefix_len + block_size - 1) // block_size
                ) * block_size
                for ev_idx, ev in enumerate(diag_events):
                    start = evict_base + ev.position_offset_after - ev.tokens_evicted
                    end = evict_base + ev.position_offset_after
                    if start < 0 or end > len(full_ids) or start >= end:
                        logger.warning(
                            f"[COMPACT-TEXT] ex={example_id} step={diag_idx} "
                            f"#{ev_idx} skipped: range [{start},{end}) outside "
                            f"buffer (len={len(full_ids)})"
                        )
                        continue
                    text = tokenizer.decode(
                        full_ids[start:end], skip_special_tokens=False
                    )
                    framed = "\n".join(f"  | {line}" for line in text.split("\n"))
                    logger.info(
                        f"[COMPACT-TEXT] ex={example_id} step={diag_idx} "
                        f"#{ev_idx} evict=[{start},{end}) "
                        f"({ev.tokens_evicted} tokens, {len(text)} chars)\n"
                        f"  {'-' * 70}\n"
                        f"{framed}\n"
                        f"  {'-' * 70}"
                    )
            events_summary = [(e.num_output_tokens_at_compaction, e.tokens_evicted)
                              for e in diag_events]
            logger.warning(
                f"[DIAG] step {diag_idx}: max_event_raw={max_raw}  "
                f"completion_ids_len={clen}  total_evicted={total_evicted}  "
                f"gap={max_raw - clen}  events={events_summary}"
            )
            if max_raw > clen:
                logger.error(
                    f"[DIAG] MISMATCH step {diag_idx}: max_event_raw={max_raw} > "
                    f"completion_ids_len={clen} (gap={max_raw - clen}). This will "
                    f"cause non-monotonic boundary assertion in merge."
                )

    def make_sample(
        tokens: dict[str, Any],
        compaction_events: list[CompactionEventWire] | None = None,
    ) -> TrainingSample:
        """Create a new TrainingSample from a trajectory step."""
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]
        completion_ids = list(tokens["completion_ids"])

        routed_experts = _align_routed_experts(
            tokens.get("routed_experts"),
            len(tokens["prompt_ids"]) + len(tokens["completion_ids"]),
        )

        return TrainingSample(
            prompt_ids=list(tokens["prompt_ids"]),
            prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=list(tokens["completion_logprobs"]),
            completion_temperatures=[temperature] * len(completion_ids),
            teacher_logprobs=None,
            advantage=None,
            routed_experts=routed_experts,
            compaction_events=compaction_events,
        )

    def extend_sample(sample: TrainingSample, prefix_len: int, step_idx: int) -> None:
        """Extend an existing sample with a new trajectory step (extension property holds).

        ``prefix_len`` is in PRE-TRIM step coord (= length of the matched
        stored prefix, which is the prior cumulative POST-TRIM state and
        equal to V's ``prev_kept_state`` for this step). Admission
        deletions for this step (if any) must have already been applied
        to ``sample``'s cumulative arrays by the caller; the slice
        ``submitted_prompt_ids_pre_trim[prefix_len:]`` is just the
        new_user_fragment (untouched by admissions, which evict from the
        prior region).
        """
        tokens = prepared_steps[step_idx]

        # Extend with new prompt tokens (mask=False, no gradient). Use
        # PRE-TRIM here so admission steps slice the new_user_fragment
        # correctly — admissions delete from prior cumulative content
        # (applied earlier) and never touch the tail.
        step_pre_trim = tokens["submitted_prompt_ids_pre_trim"]
        new_prompt_ids = step_pre_trim[prefix_len:]
        sample.completion_ids.extend(new_prompt_ids)
        sample.completion_mask.extend([False] * len(new_prompt_ids))
        sample.completion_logprobs.extend([0.0] * len(new_prompt_ids))
        sample.completion_temperatures.extend([temperature] * len(new_prompt_ids))

        # Extend with new completion tokens
        completion_ids = tokens["completion_ids"]
        sample.completion_ids.extend(completion_ids)
        if has_error:
            sample.completion_mask.extend([False] * len(tokens["completion_mask"]))
        else:
            sample.completion_mask.extend(bool(i) for i in tokens["completion_mask"])
        sample.completion_logprobs.extend(tokens["completion_logprobs"])
        sample.completion_temperatures.extend([temperature] * len(completion_ids))

        if tokens.get("routed_experts") is not None and sample.routed_experts is not None:
            step_routed = tokens["routed_experts"]
            # The previous step's last routing entry was zero-padded by _align_routed_experts
            # (vLLM only captures num_tokens-1 routings per request). This step actually
            # processed that boundary token as part of its prompt, so replace the zero-fill
            # with the real routing decision before appending new entries.
            if prefix_len > 0 and prefix_len <= len(step_routed):
                sample.routed_experts[prefix_len - 1] = step_routed[prefix_len - 1]
            sample.routed_experts.extend(step_routed[prefix_len:])
            expected_len = len(sample.prompt_ids) + len(sample.completion_ids)
            sample.routed_experts = _align_routed_experts(sample.routed_experts, expected_len)

    # Track [prefix_tokens, sample, last_step_idx, step_idx_list] per
    # active sample. The 4th element accumulates ALL merged step indices
    # so we can emit one CallWire per call after merging completes.
    active_samples: list[list] = []

    first_tokens = prepared_steps[0]
    first_prefix = first_tokens["prompt_ids"] + first_tokens["completion_ids"]
    # Phase A.4: sample.compaction_events is consumed by the legacy
    # segmented_forward path (mid-gen handling). Admission/synthetic
    # events live PER-CALL on CallWire.compaction_events and must NOT
    # leak into the merged list.
    first_events_raw = step_compaction_events[0]
    first_events_seed = (
        [e for e in first_events_raw
         if int(e.num_output_tokens_at_compaction) > 0]
        if first_events_raw else None
    ) or None
    active_samples.append(
        [first_prefix, make_sample(first_tokens, first_events_seed), 0, [0]]
    )

    for step_idx, _step in enumerate(trajectory[1:], start=1):
        tokens = prepared_steps[step_idx]
        step_pre_trim = tokens["submitted_prompt_ids_pre_trim"]
        # step_compaction_events[step_idx] holds ONLY surviving (mid-gen)
        # events after prepare_step_tokens stripped admission entries via
        # _apply_admission_trim. To see admission events here we need the
        # untouched PRE-TRIM list captured before the strip.
        step_events_all = tokens.get("all_compaction_events_pre_trim") or []
        step_events_midgen = step_compaction_events[step_idx]

        # Phase A.2: when a step has any mid-generation compaction event,
        # don't merge — start a new sample. Mid-gen samples are routed to
        # legacy segmented_forward (its segment_boundaries machinery is
        # incompatible with per-call admission splicing) and we keep the
        # admission-only-stays-per-call invariant clean.
        has_midgen = step_events_midgen is not None and any(
            int(e.num_output_tokens_at_compaction) > 0
            for e in step_events_midgen
        )

        # Phase A.1: extension check uses PRE-TRIM step prompt against the
        # stored POST-TRIM cumulative prefix. They agree up through prior
        # cumulative content because step N's pre-trim prompt =
        # prev_kept_state_N + new_user_N, and prev_kept_state_N is exactly
        # the sample's POST-TRIM cumulative state (V's KV cache view at
        # the start of step N). POST-TRIM step_prompt_ids fails this check
        # whenever step N also has an admission event — that's the bug.
        # Strict-length (>) prevents over-merging on zero-delta extensions.
        matched_idx = None
        if not has_midgen:
            for idx, entry in enumerate(active_samples):
                prefix_tokens = entry[0]
                if (
                    len(step_pre_trim) > len(prefix_tokens)
                    and step_pre_trim[: len(prefix_tokens)] == prefix_tokens
                ):
                    matched_idx = idx
                    break

        if matched_idx is not None:
            # Extension holds - merge into matched sample
            prefix_tokens = active_samples[matched_idx][0]
            sample = active_samples[matched_idx][1]
            # IMMUTABLE: where step's new_user_fragment begins in PRE-TRIM
            # coord. Used by extend_sample to slice the new content.
            orig_prefix_len = len(prefix_tokens)

            # Phase A.3 (REVERTED): we do NOT delete admission ranges
            # from the sample's cumulative arrays. Keeping mb.input_ids
            # in PRE-TRIM cumulative coord matches the trainer's
            # per-call pre_trim_ids 1:1, so seg_loss_fn aligns logits
            # against the right labels. V's cache state is mirrored on
            # the trainer side by splicing persistent_cache during call
            # N>0's admission (Phase B), which only affects the K
            # context — not the input tokens or the loss positions.
            admission_evts = [
                e for e in step_events_all
                if int(e.num_output_tokens_at_compaction) == 0
                and int(e.tokens_evicted) > 0
            ]

            # Offset mid-gen events into merged completion coord.
            # Admission/synthetic events live PER-CALL on CallWire and
            # are NOT added to sample.compaction_events (Phase A.4) —
            # that field is the legacy segmented_forward's input and it
            # consumes only mid-generation boundaries.
            midgen_events = (
                [e for e in step_events_midgen
                 if int(e.num_output_tokens_at_compaction) > 0]
                if step_events_midgen else []
            )
            if midgen_events:
                # new content in pre-trim = step_pre_trim[orig_prefix_len:]
                # = new_user_fragment. Admissions don't touch the tail,
                # so this length is the new content's POST-TRIM length too.
                new_prompt_ext_len = (
                    len(step_pre_trim) - orig_prefix_len
                )
                # current_completion_len: POST admission deletions, BEFORE
                # the upcoming new_user_fragment append.
                current_completion_len = len(sample.completion_ids)
                generation_offset = (
                    current_completion_len + new_prompt_ext_len
                )
                existing = sample.compaction_events or []
                for e in midgen_events:
                    offsetted = (
                        int(e.num_output_tokens_at_compaction)
                        + generation_offset
                    )
                    if existing:
                        prev = existing[-1].num_output_tokens_at_compaction
                        assert offsetted >= prev, (
                            f"Non-monotonic compaction boundary at step "
                            f"{step_idx}: offsetted={offsetted} < prev={prev} "
                            f"(raw={e.num_output_tokens_at_compaction}, "
                            f"generation_offset={generation_offset})"
                        )
                        if offsetted == prev:
                            existing[-1] = CompactionEventWire(
                                num_output_tokens_at_compaction=offsetted,
                                tokens_evicted=existing[-1].tokens_evicted
                                + e.tokens_evicted,
                                position_offset_after=e.position_offset_after,
                                num_prompt_tokens=e.num_prompt_tokens,
                            )
                            continue
                    existing.append(
                        CompactionEventWire(
                            num_output_tokens_at_compaction=offsetted,
                            tokens_evicted=e.tokens_evicted,
                            position_offset_after=e.position_offset_after,
                            num_prompt_tokens=e.num_prompt_tokens,
                        )
                    )
                sample.compaction_events = existing

            if admission_evts:
                logger.info(
                    f"[ADM-MERGE] step {step_idx}: "
                    f"merged_into_sample_idx={matched_idx} "
                    f"call_idx={len(active_samples[matched_idx][3])} "
                    f"n_admission_evts={len(admission_evts)} "
                    f"total_evicted={sum(int(e.tokens_evicted) for e in admission_evts)} "
                    f"new_user_len={len(step_pre_trim) - orig_prefix_len}"
                )

            extend_sample(sample, orig_prefix_len, step_idx=step_idx)
            active_samples[matched_idx][0] = tokens["prompt_ids"] + tokens["completion_ids"]
            active_samples[matched_idx][2] = step_idx
            active_samples[matched_idx][3].append(step_idx)
        else:
            # No prefix matches - start a new sample
            step_prompt_ids = tokens["prompt_ids"]
            logger.debug(
                f"Extension property broke at step {step_idx + 1} for example {output['example_id']}. "
                f"Starting new sample (active_prefixes={len(active_samples)}, step_prompt_len={len(step_prompt_ids)})."
            )

            # Markovian Thinker sanity check: every post-truncation
            # sample must begin with the same system-prefix token as the
            # first sample. Catches silent regressions where truncation
            # accidentally drops the system message.
            if (
                _MARKOVIAN_ASSERT_ENABLED
                and step_prompt_ids
                and first_tokens["prompt_ids"]
            ):
                expected_first = first_tokens["prompt_ids"][0]
                assert step_prompt_ids[0] == expected_first, (
                    f"[MARKOVIAN] system prefix missing at step {step_idx}: "
                    f"expected first prompt token {expected_first}, "
                    f"got {step_prompt_ids[0]}. This usually means "
                    f"truncate_messages_to_last_k_turns dropped the "
                    f"system message."
                )

            new_prefix = tokens["prompt_ids"] + tokens["completion_ids"]
            active_samples.append(
                [new_prefix, make_sample(tokens, step_compaction_events[step_idx]), step_idx, [step_idx]]
            )

    # Attach images once per sample using only the last merged step
    if vlm_cache is not None:
        key = output["example_id"] if cache_key is None else cache_key
        for entry in active_samples:
            sample = entry[1]
            last_step_idx = entry[2]
            pv, shape, grids = vlm_cache.get_for_step(key, last_step_idx)
            sample.pixel_values = pv
            sample.pixel_values_shape = shape
            sample.image_grid_thw = grids

    # Emit per-call breakdown (Phase B of plans/two_phase_per_call_trainer.md).
    # One CallWire per vLLM chat() call (= per merged step) per sample.
    # Used by the trainer's per-call segmented forward in Phase C+.
    for entry in active_samples:
        sample = entry[1]
        step_idx_list = entry[3]
        calls: list[CallWire] = []
        for sidx in step_idx_list:
            prepared = prepared_steps[sidx]
            calls.append(
                CallWire(
                    submitted_prompt_ids=list(
                        prepared.get("submitted_prompt_ids_pre_trim")
                        or prepared["prompt_ids"]
                    ),
                    completion_ids=list(prepared["completion_ids"]),
                    completion_logprobs=list(prepared["completion_logprobs"]),
                    completion_temperatures=[temperature]
                    * len(prepared["completion_ids"]),
                    compaction_events=list(
                        prepared.get("all_compaction_events_pre_trim") or []
                    ),
                    trailing_pad_ids=list(
                        prepared.get("trailing_pad_ids") or []
                    ),
                )
            )
        sample.calls = calls

        # Self-consistency check: sum of all call completion lengths +
        # cross-call prefix-extension prompt deltas should equal the
        # merged sample's completion length. We don't have the
        # extension deltas trivially here (they're computed inline in
        # extend_sample as new_prompt_ids), so we just verify the
        # weaker invariant that the number of calls matches the merge
        # bookkeeping.
        assert len(sample.calls) == len(step_idx_list), (
            f"call/step_idx mismatch: {len(sample.calls)} calls vs "
            f"{len(step_idx_list)} merged steps"
        )

    # Markovian Summary extension: for every trajectory step that carries
    # an `extras["summary_trainsample"]` payload, emit a separate
    # TrainingSample. These samples are full-credit (completion_mask
    # all-True) and their logprobs come from the summary call itself.
    # They inherit the rollout's advantage via the orchestrator's
    # per-rollout advantage assignment after this function returns.
    summary_samples: list[TrainingSample] = []
    for step in trajectory:
        s = _build_summary_sample(step, temperature=temperature, has_error=has_error)
        if s is not None:
            summary_samples.append(s)

    regular_samples = [entry[1] for entry in active_samples]
    # Summary samples appended AFTER the regular samples so they don't
    # disturb per-rollout prefix-matching / merge invariants. Ordering
    # within the return list does not affect training semantics (each
    # sample trains independently); sharing the rollout's advantage
    # assignment happens downstream.
    return regular_samples + summary_samples


# =============================================================================
# VLM-specific functions
# =============================================================================


_FILE_URL_PREFIX = "file://"


def offload_images_to_disk(rollouts: list[vf.RolloutOutput], output_dir: Path) -> int:
    """Replace base64 image data in rollout trajectories with file paths on disk.

    Scans all trajectory step prompts for data:image URLs, writes the decoded
    image bytes to ``{output_dir}/assets/images/{hash}.png``, and replaces the
    URL in-place with ``file://{path}``.  Deduplicates by content hash so each
    unique image is written only once.

    Returns the number of unique images written to disk.
    """
    images_dir = output_dir / "assets" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    written: set[str] = set()

    for output in rollouts:
        for step in output.get("trajectory", []):
            prompt = step.get("prompt")
            if not prompt or not isinstance(prompt, list):
                continue
            for msg in prompt:
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                for item in content:
                    if item.get("type") != "image_url":
                        continue
                    url = item.get("image_url", {}).get("url", "")
                    if not url.startswith("data:image"):
                        continue
                    b64_data = url.split(",", 1)[1]
                    content_hash = hashlib.sha256(b64_data.encode()).hexdigest()[:16]
                    path = images_dir / f"{content_hash}.png"
                    if content_hash not in written:
                        if not path.exists():
                            path.write_bytes(base64.b64decode(b64_data))
                        written.add(content_hash)
                    item["image_url"]["url"] = f"{_FILE_URL_PREFIX}{path}"

    return len(written)


def _load_file_image(path_str: str) -> Image.Image:
    """Load an image from a file:// path."""
    return Image.open(path_str.removeprefix(_FILE_URL_PREFIX))


def _extract_images_from_messages(messages: list) -> list[tuple[Image.Image, str]]:
    """Extract (image, key) pairs from OpenAI-style chat messages.

    Handles both base64 data URLs and file:// paths from disk offloading.
    """
    images = []
    if not messages or not isinstance(messages, list):
        return images

    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith(_FILE_URL_PREFIX):
                        img = _load_file_image(url)
                        images.append((img, url))
                    elif url.startswith("data:image"):
                        b64_data = url.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_data)
                        img = Image.open(BytesIO(img_bytes))
                        images.append((img, b64_data))
    return images


def _collect_image_keys_from_messages(messages: list) -> list[str]:
    """Extract image keys from OpenAI-style chat messages without decoding.

    Handles both base64 data URLs and file:// paths from disk offloading.
    """
    keys = []
    if not messages or not isinstance(messages, list):
        return keys
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        keys.append(url.split(",", 1)[1])
                    elif url.startswith(_FILE_URL_PREFIX):
                        keys.append(url)
    return keys


def _decode_image(key: str) -> Image.Image:
    """Decode an image from a base64 string or load from a file:// path."""
    if key.startswith(_FILE_URL_PREFIX):
        return _load_file_image(key)
    return Image.open(BytesIO(base64.b64decode(key)))


_PARALLEL_DECODE_THRESHOLD = 4


_IMAGE_STRIPPED_PLACEHOLDER = "[preprocessed image]"


def strip_base64_images(examples: list[tuple[int, vf.RolloutOutput]]) -> None:
    """Strip image data from rollout prompts to free memory.

    Handles both base64 data URLs and file:// paths from disk offloading.
    The images have been decoded and indexed; the original data is no longer needed.
    """
    for _, output in examples:
        for step in output.get("trajectory", []):
            prompt = step.get("prompt")
            if not prompt or not isinstance(prompt, list):
                continue
            for msg in prompt:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if url.startswith("data:image") or url.startswith(_FILE_URL_PREFIX):
                                item["image_url"]["url"] = _IMAGE_STRIPPED_PLACEHOLDER


def _extract_images_from_examples(
    examples: list[tuple[int, vf.RolloutOutput]],
) -> tuple[list[Image.Image], dict[int, list[list[int]]]]:
    """
    Extract images from all trajectory steps of each example.

    Two-pass approach: first collects unique base64 keys (fast, string-only),
    then decodes unique images in parallel via ThreadPoolExecutor.

    Args:
        examples: List of (cache_key, output) tuples where output contains a "trajectory"
            list with steps that have "prompt" messages in OpenAI chat format.

    Returns:
        Tuple of (all_images, step_image_indices_per_example)
        - all_images: deduplicated flat list of decoded PIL images
        - step_image_indices_per_example: dict mapping cache_key to per-step lists of
          indices into all_images (e.g., [[0], [0, 1], [1]] for the decreasing-images case)
    """
    # Pass 1: collect unique b64 keys and build step indices
    unique_keys: list[str] = []
    key_to_index: dict[str, int] = {}
    step_image_indices_per_example: dict[int, list[list[int]]] = {}

    for eid, output in examples:
        trajectory = output.get("trajectory", [])
        if not trajectory:
            step_image_indices_per_example[eid] = []
            continue

        step_image_indices = []
        for step in trajectory:
            prompt = step.get("prompt")
            image_keys = _collect_image_keys_from_messages(prompt)
            indices = []
            for key in image_keys:
                if key not in key_to_index:
                    key_to_index[key] = len(unique_keys)
                    unique_keys.append(key)
                indices.append(key_to_index[key])
            step_image_indices.append(indices)

        step_image_indices_per_example[eid] = step_image_indices

    # Pass 2: decode unique images (parallel when worthwhile)
    if len(unique_keys) > _PARALLEL_DECODE_THRESHOLD:
        with ThreadPoolExecutor(max_workers=min(len(unique_keys), 16)) as pool:
            all_images = list(pool.map(_decode_image, unique_keys))
    else:
        all_images = [_decode_image(k) for k in unique_keys]
    del unique_keys, key_to_index

    strip_base64_images(examples)

    return all_images, step_image_indices_per_example


_DEFAULT_IMAGE_CHUNK_SIZE = 32


class _ImageStore:
    """Holds per-unique-image data, assembled lazily on demand.

    Instead of duplicating pixel bytes for every step that references an image,
    we store each image's bytes once and assemble the concatenation at retrieval time.
    """

    def __init__(
        self,
        image_bytes: list[bytes],
        image_num_patches: list[int],
        patch_dim: int,
        image_grids: list[list[int]],
    ):
        self.image_bytes = image_bytes
        self.image_num_patches = image_num_patches
        self.patch_dim = patch_dim
        self.image_grids = image_grids
        self._cache: dict[tuple[int, ...], tuple[bytes, list[int], list[list[int]]]] = {}

    def assemble(self, indices: list[int]) -> tuple[bytes, list[int], list[list[int]]]:
        """Assemble pixel bytes, shape, and grids for a set of image indices.

        Results are cached by index tuple — multi-turn rollouts with the same
        cumulative image set (common across rollouts of the same example) hit
        the cache and skip the join.
        """
        cache_key = tuple(indices)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        total_patches = sum(self.image_num_patches[i] for i in indices)
        pixel_bytes = b"".join(self.image_bytes[i] for i in indices)
        shape = [total_patches, self.patch_dim]
        grids = [self.image_grids[i] for i in indices]
        result = (pixel_bytes, shape, grids)
        self._cache[cache_key] = result
        return result


def _preprocess_images_batched(
    images: list[Image.Image],
    step_image_indices_per_example: dict[int, list[list[int]]],
    processor,
    chunk_size: int = _DEFAULT_IMAGE_CHUNK_SIZE,
) -> tuple["_ImageStore | None", dict[int, list[list[int]]]]:
    """
    Preprocess all images in chunked batches, returning an _ImageStore and step indices.

    Images are processed in chunks to avoid OOM on large batches. Per-image bytes are
    stored once in the _ImageStore and assembled lazily at retrieval time.

    Returns:
        Tuple of (_ImageStore or None, step_image_indices_per_example).
        The store is None when there are no images or no processor.
    """
    if not images or processor is None:
        return None, step_image_indices_per_example

    logger = get_logger()
    image_sizes = [(img.width, img.height) for img in images]

    # Process images in chunks to avoid OOM, parallelized across threads
    # (PIL/numpy release the GIL so threads give real concurrency here)
    chunks = [images[i : i + chunk_size] for i in range(0, len(images), chunk_size)]

    def _process_chunk(chunk: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        processed = processor.image_processor(images=chunk, return_tensors="pt")
        return processed["pixel_values"], processed["image_grid_thw"]

    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=min(len(chunks), 8)) as pool:
            results = list(pool.map(_process_chunk, chunks))
    else:
        results = [_process_chunk(chunks[0])]

    # Free PIL images now that preprocessing is done
    del chunks
    images.clear()

    all_pixel_values_list = [r[0] for r in results]
    all_grid_thw_list = [r[1] for r in results]

    all_pixel_values = torch.cat(all_pixel_values_list, dim=0)
    all_grid_thw = torch.cat(all_grid_thw_list, dim=0)
    del all_pixel_values_list, all_grid_thw_list, results

    logger.debug(
        f"VLM image processing: {len(image_sizes)} images, sizes={image_sizes}, "
        f"pixel_values={all_pixel_values.shape}, grid_thw={all_grid_thw.tolist()}"
    )

    # Pre-compute patch start offset for each image
    patch_starts = [0]
    for g in all_grid_thw:
        patch_starts.append(patch_starts[-1] + int(g[0] * g[1] * g[2]))

    patch_dim = all_pixel_values.shape[1]

    # Convert to bytes per-image and free the tensor immediately after
    image_bytes_list: list[bytes] = []
    image_num_patches_list: list[int] = []
    image_grids_list: list[list[int]] = []
    for i in range(len(image_sizes)):
        img_slice = all_pixel_values[patch_starts[i] : patch_starts[i + 1]]
        image_bytes_list.append(img_slice.numpy().tobytes())
        image_num_patches_list.append(img_slice.shape[0])
        image_grids_list.append(all_grid_thw[i].tolist())
    del all_pixel_values, all_grid_thw

    store = _ImageStore(
        image_bytes=image_bytes_list,
        image_num_patches=image_num_patches_list,
        patch_dim=patch_dim,
        image_grids=image_grids_list,
    )

    return store, step_image_indices_per_example


class VLMImageCache:
    """Result of building VLM image cache with per-step image data."""

    def __init__(
        self,
        cache: dict[int, list[tuple[bytes | None, list[int] | None, list[list[int]] | None]]],
        num_unique_examples: int,
        extract_time: float,
        preprocess_time: float,
    ):
        self._store: _ImageStore | None = None
        self._step_indices: dict[int, list[list[int]]] | None = None
        self.cache = cache
        self.num_unique_examples = num_unique_examples
        self.num_unique_images = 0
        self.extract_time = extract_time
        self.preprocess_time = preprocess_time

    @classmethod
    def from_store(
        cls,
        store: _ImageStore | None,
        step_indices: dict[int, list[list[int]]],
        num_unique_examples: int,
        num_unique_images: int,
        extract_time: float,
        preprocess_time: float,
    ) -> "VLMImageCache":
        """Create a store-backed cache that assembles bytes lazily."""
        obj = cls.__new__(cls)
        obj._store = store
        obj._step_indices = step_indices
        obj.cache = {}
        obj.num_unique_examples = num_unique_examples
        obj.num_unique_images = num_unique_images
        obj.extract_time = extract_time
        obj.preprocess_time = preprocess_time
        return obj

    def _assemble(self, indices: list[int]) -> tuple[bytes | None, list[int] | None, list[list[int]] | None]:
        if not indices:
            return (None, None, None)
        return self._store.assemble(indices)

    def get_for_step(
        self, cache_key: int, step_idx: int
    ) -> tuple[bytes | None, list[int] | None, list[list[int]] | None]:
        """Get cumulative images up to and including the given step."""
        if self._store is not None:
            steps = self._step_indices.get(cache_key, [])
            if not steps or step_idx >= len(steps):
                return (None, None, None)
            return self._assemble(steps[step_idx])

        steps = self.cache.get(cache_key, [])
        if not steps or step_idx >= len(steps):
            return (None, None, None)
        return steps[step_idx]

    def get_all(self, cache_key: int) -> tuple[bytes | None, list[int] | None, list[list[int]] | None]:
        """Get all images for the cache key (last step's cumulative images)."""
        if self._store is not None:
            steps = self._step_indices.get(cache_key, [])
            if not steps:
                return (None, None, None)
            return self._assemble(steps[-1])

        steps = self.cache.get(cache_key, [])
        if not steps:
            return (None, None, None)
        return steps[-1]


def build_vlm_image_cache(rollouts: list[vf.RolloutOutput], processor) -> VLMImageCache:
    """
    Build image cache for VLM training by extracting and preprocessing images.

    Caches per rollout to keep images aligned with divergent multi-turn trajectories.
    """
    examples = [(idx, rollout) for idx, rollout in enumerate(rollouts)]
    unique_example_ids = {rollout["example_id"] for rollout in rollouts}

    # Extract images (also strips base64 data from rollout prompts to free memory)
    extract_start = time.perf_counter()
    all_images, images_per_example = _extract_images_from_examples(examples)
    num_unique_images = len(all_images)
    extract_time = time.perf_counter() - extract_start

    # Preprocess images (clears PIL image list when done)
    preprocess_start = time.perf_counter()
    store, step_indices = _preprocess_images_batched(all_images, images_per_example, processor)
    preprocess_time = time.perf_counter() - preprocess_start

    return VLMImageCache.from_store(
        store=store,
        step_indices=step_indices,
        num_unique_examples=len(unique_example_ids),
        num_unique_images=num_unique_images,
        extract_time=extract_time,
        preprocess_time=preprocess_time,
    )
