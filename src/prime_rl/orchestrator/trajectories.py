import base64
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

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
from prime_rl.transport.types import CompactionEventWire
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


def _coerce_int_list(value: Any) -> list[int] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        return [int(x) for x in value]
    except (TypeError, ValueError):
        return None


def _normalize_selected_indices(raw: Any) -> list[list[list[int]]] | None:
    if raw is None:
        return None
    try:
        out: list[list[list[int]]] = []
        for layer in raw:
            layer_out: list[list[int]] = []
            for head in layer:
                layer_out.append([int(idx) for idx in head])
            out.append(layer_out)
        return out or None
    except (TypeError, ValueError):
        return None


def _coerce_float_list(value: Any) -> list[float] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def _compaction_events_from_raw(
    raw: Any,
) -> list[CompactionEventWire] | None:
    if not raw:
        return None
    out: list[CompactionEventWire] = []
    for e in raw:
        try:
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
                        compaction_strategy=str(e.get("compaction_strategy", "fifo")),
                        source_len=int(e.get("source_len", 0)),
                        target_len=int(e.get("target_len", 0)),
                        protected_prefix_len=int(e.get("protected_prefix_len", 0)),
                        synthetic_prefix_len=int(e.get("synthetic_prefix_len", 0)),
                        exact_kept_tokens=int(e.get("exact_kept_tokens", 0)),
                        attention_matching_query_source=str(
                            e.get("attention_matching_query_source", "")
                        ),
                        attention_matching_max_queries_per_kv_head=int(
                            e.get("attention_matching_max_queries_per_kv_head", 0)
                        ),
                        attention_matching_query_seed=int(
                            e.get("attention_matching_query_seed", 0)
                        ),
                        attention_matching_zerobeta=bool(
                            e.get("attention_matching_zerobeta", False)
                        ),
                        attention_matching_pre_sample=bool(
                            e.get("attention_matching_pre_sample", False)
                        ),
                        attention_matching_replay_steps=(
                            e.get("attention_matching_replay_steps")
                        ),
                        attention_matching_cache_hit_tokens=int(
                            e.get("attention_matching_cache_hit_tokens", 0) or 0
                        ),
                        attention_matching_selected_indices=(
                            _normalize_selected_indices(
                                e.get("attention_matching_selected_indices")
                            )
                        ),
                        attention_matching_forget_gate_enabled=bool(
                            e.get("attention_matching_forget_gate_enabled", False)
                        ),
                        attention_matching_forget_gate_alpha=float(
                            e.get("attention_matching_forget_gate_alpha", 0.5)
                        ),
                        attention_matching_forget_gate_applied=bool(
                            e.get("attention_matching_forget_gate_applied", False)
                        ),
                        attention_matching_hidden_tail_token_ids=_coerce_int_list(
                            e.get("attention_matching_hidden_tail_token_ids")
                        ),
                    )
                )
            elif isinstance(e, (list, tuple)) and len(e) >= 3:
                out.append(
                    CompactionEventWire(
                        num_output_tokens_at_compaction=int(e[0]),
                        tokens_evicted=int(e[1]),
                        position_offset_after=int(e[2]),
                        num_prompt_tokens=int(e[3]) if len(e) >= 4 else 0,
                        evict_start=int(e[4]) if len(e) >= 5 else 0,
                        compaction_strategy=str(e[5]) if len(e) >= 6 else "fifo",
                        source_len=int(e[6]) if len(e) >= 7 else 0,
                        target_len=int(e[7]) if len(e) >= 8 else 0,
                        protected_prefix_len=int(e[8]) if len(e) >= 9 else 0,
                        synthetic_prefix_len=int(e[9]) if len(e) >= 10 else 0,
                        exact_kept_tokens=int(e[10]) if len(e) >= 11 else 0,
                        attention_matching_query_source=(
                            str(e[11]) if len(e) >= 12 else ""
                        ),
                        attention_matching_max_queries_per_kv_head=(
                            int(e[12]) if len(e) >= 13 else 0
                        ),
                        attention_matching_query_seed=(
                            int(e[13]) if len(e) >= 14 else 0
                        ),
                        attention_matching_zerobeta=(
                            bool(e[14]) if len(e) >= 15 else False
                        ),
                        attention_matching_pre_sample=(
                            bool(e[15]) if len(e) >= 16 else False
                        ),
                        attention_matching_replay_steps=(
                            e[16] if len(e) >= 17 else None
                        ),
                        attention_matching_cache_hit_tokens=(
                            int(e[17]) if len(e) >= 18 else 0
                        ),
                        attention_matching_selected_indices=(
                            _normalize_selected_indices(
                                e[18] if len(e) >= 19 else None
                            )
                        ),
                        attention_matching_forget_gate_enabled=(
                            bool(e[19]) if len(e) >= 20 else False
                        ),
                        attention_matching_forget_gate_alpha=(
                            float(e[20]) if len(e) >= 21 else 0.5
                        ),
                        attention_matching_forget_gate_applied=(
                            bool(e[21]) if len(e) >= 22 else False
                        ),
                        attention_matching_hidden_tail_token_ids=_coerce_int_list(
                            e[22] if len(e) >= 23 else None
                        ),
                    )
                )
        except (KeyError, TypeError, ValueError):
            continue
    return out or None


def _am_prompt_replay_event(event: CompactionEventWire) -> bool:
    """Whether an event is a prompt-time AM turn-chain replay event.

    These are the events emitted by the compressed cross-turn AM path. Unlike
    generation-time AM events, they can be merged across turns because the
    compressed prefix cache intentionally makes the request behave like a
    recurrent state carried across environment turns.

    A prompt-time AM event with zero compressed-cache hit tokens still produces
    a valid AM state: vLLM missed the cache for this request, restored the full
    prompt, and privately replayed AM from raw text before sampling. Later turns
    can hit the compressed state written by this request. The orchestrator uses
    such no-hit events as state producers, but only cache-hit prompt events as
    state consumers when merging across turns.
    """
    return (
        event.compaction_strategy == "attention_matching"
        and event.attention_matching_pre_sample
        and event.num_output_tokens_at_compaction == 0
        and bool(event.attention_matching_replay_steps)
    )


def _am_prompt_cache_hit_event(event: CompactionEventWire) -> bool:
    """Whether a prompt-time AM event actually reused compressed KV."""
    return _am_prompt_replay_event(event) and event.attention_matching_cache_hit_tokens > 0


def _coalesce_am_prompt_replay_events(
    events: list[CompactionEventWire] | None,
    *,
    prompt_len: int | None = None,
) -> CompactionEventWire | None:
    """Represent prompt-time AM as one rewrite at the true pre-sample point.

    vLLM computes the full request prompt/warmup first, then applies the
    turn-chain AM replay internally with suffix-preservation. Splitting that
    internal chain into multiple trainer events is not faithful because it
    re-feeds suffix tokens under intermediate compacted states that vLLM never
    used. The trainer should instead forward to the same pre-sample boundary
    and apply the whole AM replay chain in one segmented_forward rewrite.
    """
    if not events or not all(_am_prompt_replay_event(event) for event in events):
        return None

    source_prompt_len = (
        int(prompt_len)
        if prompt_len is not None
        else int(events[-1].num_prompt_tokens or events[-1].source_len)
    )
    if source_prompt_len <= 0:
        raise RuntimeError("AM prompt replay coalescing needs a positive prompt length.")

    replay_steps = _merged_am_prompt_replay_steps(events)

    if not replay_steps:
        return None

    total_evicted = sum(
        int(step.get("source_len", 0) or 0)
        - int(step.get("target_len", 0) or 0)
        for step in replay_steps
    )
    if total_evicted <= 0:
        raise RuntimeError("AM prompt replay coalescing found no evicted tokens.")

    target_len = source_prompt_len - total_evicted
    last_event = events[-1]
    last_step = replay_steps[-1]
    protected = int(last_step.get("protected_prefix_len", 0) or 0)
    synthetic = int(last_step.get("synthetic_prefix_len", 0) or 0)
    exact = target_len - protected - synthetic
    if exact <= 0:
        raise RuntimeError(
            "AM prompt replay coalescing produced non-positive exact tail: "
            f"target={target_len}, protected={protected}, synthetic={synthetic}."
        )

    selected = _normalize_selected_indices(
        last_step.get("attention_matching_selected_indices")
        if isinstance(last_step, dict)
        else None
    ) or last_event.attention_matching_selected_indices

    cache_hit_tokens = int(last_event.attention_matching_cache_hit_tokens or 0)
    if cache_hit_tokens < 0 or cache_hit_tokens >= target_len:
        raise RuntimeError(
            "AM prompt replay coalescing found invalid cache-hit length: "
            f"hit={cache_hit_tokens}, target={target_len}."
        )
    if cache_hit_tokens and cache_hit_tokens < protected + synthetic:
        raise RuntimeError(
            "AM prompt replay coalescing found cache-hit length that does not "
            "cover protected+synthetic memory: "
            f"hit={cache_hit_tokens}, protected={protected}, synthetic={synthetic}."
        )

    return CompactionEventWire(
        num_output_tokens_at_compaction=0,
        tokens_evicted=total_evicted,
        position_offset_after=total_evicted,
        num_prompt_tokens=source_prompt_len,
        evict_start=last_event.evict_start,
        compaction_strategy=last_event.compaction_strategy,
        source_len=source_prompt_len,
        target_len=target_len,
        protected_prefix_len=protected,
        synthetic_prefix_len=synthetic,
        exact_kept_tokens=exact,
        attention_matching_query_source=last_event.attention_matching_query_source,
        attention_matching_max_queries_per_kv_head=(
            last_event.attention_matching_max_queries_per_kv_head
        ),
        attention_matching_query_seed=int(
            last_step.get(
                "attention_matching_query_seed",
                last_event.attention_matching_query_seed,
            )
            or 0
        ),
        attention_matching_zerobeta=last_event.attention_matching_zerobeta,
        attention_matching_pre_sample=True,
        attention_matching_replay_steps=replay_steps,
        # Standalone prompt-time replay has no previous trainer segment to
        # carry vLLM's compressed cache hit. Preserve the actual hit length so
        # segmented replay rebuilds the cached prefix, then warms only the
        # uncached exact-tail suffix under that compressed state.
        attention_matching_cache_hit_tokens=cache_hit_tokens,
        attention_matching_selected_indices=selected,
        attention_matching_forget_gate_enabled=bool(
            last_step.get(
                "attention_matching_forget_gate_enabled",
                last_event.attention_matching_forget_gate_enabled,
            )
            if isinstance(last_step, dict)
            else last_event.attention_matching_forget_gate_enabled
        ),
        attention_matching_forget_gate_alpha=float(
            last_step.get(
                "attention_matching_forget_gate_alpha",
                last_event.attention_matching_forget_gate_alpha,
            )
            if isinstance(last_step, dict)
            else last_event.attention_matching_forget_gate_alpha
        ),
        attention_matching_forget_gate_applied=bool(
            last_step.get(
                "attention_matching_forget_gate_applied",
                last_event.attention_matching_forget_gate_applied,
            )
            if isinstance(last_step, dict)
            else last_event.attention_matching_forget_gate_applied
        ),
        attention_matching_hidden_tail_token_ids=(
            _am_hidden_tail_token_ids(events) or None
        ),
    )


def _merged_am_prompt_replay_steps(
    events: list[CompactionEventWire],
) -> list[dict[str, Any]]:
    """Merge prompt-time AM replay step metadata into one ordered chain."""
    replay_steps: list[dict[str, Any]] = []
    for event in events:
        current = list(event.attention_matching_replay_steps or [])
        if not current:
            continue
        if replay_steps:
            common = _am_common_replay_prefix_len(replay_steps, current)
            if common == len(replay_steps):
                replay_steps.extend(current[common:])
            elif common == 0:
                # vLLM can emit one prompt event for a compressed-cache hit
                # and a later event for private suffix replay. The latter is
                # already relative to the restored compressed state, so its
                # replay_steps are a suffix rather than a full chain.
                replay_steps.extend(current)
            else:
                raise RuntimeError(
                    "Cannot merge AM prompt replay events whose replay chains "
                    "diverge before the accumulated prefix."
                )
        else:
            replay_steps.extend(current)
    return replay_steps


def _am_completed_request_replay_steps(
    events: list[CompactionEventWire] | None,
) -> list[dict[str, Any]] | None:
    """Return the AM replay chain represented by a completed request.

    A TextWorld turn can contain one prompt-time AM cache event followed by
    generation-time AM events. The next turn's prompt-time cache event refers to
    the whole chain, so the active state producer must include both kinds.
    """
    if not events:
        return None
    am_events = [
        event
        for event in events
        if event.compaction_strategy == "attention_matching"
        and bool(event.attention_matching_replay_steps)
    ]
    if not am_events:
        return None
    return _merged_am_prompt_replay_steps(am_events) or None


def _am_completed_request_offset_after(
    events: list[CompactionEventWire] | None,
) -> int:
    if not events:
        return 0
    for event in reversed(events):
        if event.compaction_strategy == "attention_matching":
            return int(event.position_offset_after)
    return 0


def _am_generation_events(
    events: list[CompactionEventWire] | None,
) -> list[CompactionEventWire]:
    if not events:
        return []
    return [
        event
        for event in events
        if event.compaction_strategy == "attention_matching"
        and not _am_prompt_replay_event(event)
        and int(event.num_output_tokens_at_compaction) > 0
    ]


def _am_replay_step_signature(step: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(step.get("attention_matching_prefix_cache_key") or ""),
        int(step.get("source_len", 0) or 0),
        int(step.get("target_len", 0) or 0),
        int(step.get("protected_prefix_len", 0) or 0),
        int(step.get("synthetic_prefix_len", 0) or 0),
        int(step.get("exact_kept_tokens", 0) or 0),
        int(step.get("attention_matching_query_seed", 0) or 0),
        int(bool(step.get("attention_matching_forget_gate_enabled", False))),
        int(
            round(
                float(step.get("attention_matching_forget_gate_alpha", 0.5))
                * 1_000_000
            )
        ),
        int(bool(step.get("attention_matching_forget_gate_applied", False))),
    )


def _am_common_replay_prefix_len(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> int:
    n = min(len(left), len(right))
    for i in range(n):
        if _am_replay_step_signature(left[i]) != _am_replay_step_signature(right[i]):
            return i
    return n


def _optional_prefix_padding_token_ids() -> set[int]:
    token_ids = {151643}
    env_value = os.environ.get("KV_EVICTION_PADDING_FILLER_ID")
    if env_value:
        try:
            token_ids.add(int(env_value))
        except ValueError:
            pass
    return token_ids


def _am_hidden_tail_token_ids(
    events: list[CompactionEventWire] | None,
) -> list[int]:
    """Return deterministic hidden tail tokens vLLM prefetched for AM cache.

    These tokens are not visible response tokens and carry no loss, but they
    are real KV entries when vLLM admits a cross-turn compressed prefix cache
    block. The trainer must replay them before the next turn's prompt suffix.
    """
    if not events:
        return []
    for event in reversed(events):
        hidden_tail = getattr(event, "attention_matching_hidden_tail_token_ids", None)
        if hidden_tail:
            return [int(token_id) for token_id in hidden_tail]
    return []


def _prefix_match_len_with_optional_padding(
    prefix_tokens: list[int],
    prompt_tokens: list[int],
) -> int | None:
    """Return consumed prompt length when ``prompt_tokens`` extends prefix.

    Block-aligned turn padding inserts filler tokens before a request's
    generation prefix. Once the assistant message is closed and re-rendered on
    the next turn, those temporary filler tokens can disappear from that
    location. We may skip only filler tokens already present in the old trace
    prefix. Filler tokens newly present in the next prompt are real warmup
    tokens that vLLM prefills to close partial blocks, so they must remain in
    the returned prompt suffix with ``completion_mask=False``.
    """
    if prompt_tokens[: len(prefix_tokens)] == prefix_tokens:
        return len(prefix_tokens)

    padding_ids = _optional_prefix_padding_token_ids()
    prefix_idx = 0
    prompt_idx = 0
    while prefix_idx < len(prefix_tokens) and prompt_idx < len(prompt_tokens):
        if prefix_tokens[prefix_idx] == prompt_tokens[prompt_idx]:
            prefix_idx += 1
            prompt_idx += 1
            continue
        if prefix_tokens[prefix_idx] in padding_ids:
            prefix_idx += 1
            continue
        return None

    while prefix_idx < len(prefix_tokens) and prefix_tokens[prefix_idx] in padding_ids:
        prefix_idx += 1

    if prefix_idx != len(prefix_tokens):
        return None
    return prompt_idx


def _prefix_match_len_with_optional_hidden_tail(
    prefix_tokens: list[int],
    prompt_tokens: list[int],
    hidden_tail_token_ids: list[int],
) -> int | None:
    """Match an AM stateful prefix against a later visible prompt.

    Hidden-tail finalization tokens are real KV entries used to admit the final
    partial block into the compressed prefix cache, but they are not necessarily
    visible text in the next rendered prompt. Prefer the exact match when the
    next prompt contains those tokens; otherwise allow removing only the known
    hidden-tail suffix from the old prefix before matching.
    """
    prefix_len = _prefix_match_len_with_optional_padding(
        prefix_tokens,
        prompt_tokens,
    )
    if prefix_len is not None:
        return prefix_len
    if not hidden_tail_token_ids:
        return None
    if len(hidden_tail_token_ids) > len(prefix_tokens):
        return None
    if prefix_tokens[-len(hidden_tail_token_ids) :] != hidden_tail_token_ids:
        return None
    return _prefix_match_len_with_optional_padding(
        prefix_tokens[: -len(hidden_tail_token_ids)],
        prompt_tokens,
    )


def _make_stateful_am_prompt_event(
    *,
    event: CompactionEventWire,
    replay_steps: list[dict[str, Any]],
    global_boundary: int,
    global_prompt_len: int,
    offset_before: int,
    source_len_override: int | None = None,
    cache_hit_tokens: int = 0,
    pre_sample: bool = True,
) -> CompactionEventWire:
    """Convert a request-local AM prompt event into a merged-trace event."""
    tokens_evicted = sum(
        int(step.get("source_len", 0) or 0)
        - int(step.get("target_len", 0) or 0)
        for step in replay_steps
    )
    if tokens_evicted <= 0:
        raise ValueError("AM replay suffix did not evict any tokens")

    source_len = (
        global_prompt_len + global_boundary - offset_before
        if source_len_override is None
        else int(source_len_override)
    )
    target_len = source_len - tokens_evicted
    if source_len <= 0 or target_len <= 0:
        raise ValueError(
            "AM stateful event produced invalid source/target length: "
            f"source={source_len}, target={target_len}, "
            f"boundary={global_boundary}, prompt={global_prompt_len}, "
            f"offset_before={offset_before}"
        )
    first_replay_source = int(replay_steps[0].get("source_len", 0) or 0)
    if first_replay_source <= 0 or first_replay_source > source_len:
        raise ValueError(
            "AM stateful event replay source must lie inside the trainer KV: "
            f"trainer_source={source_len}, replay_source={first_replay_source}, "
            f"boundary={global_boundary}, prompt={global_prompt_len}, "
            f"offset_before={offset_before}"
        )
    source_boundary = source_len + offset_before - global_prompt_len
    if pre_sample:
        if source_boundary != global_boundary:
            raise ValueError(
                "Pre-sample AM stateful event must compact at the logprob "
                f"boundary: source_boundary={source_boundary}, "
                f"boundary={global_boundary}."
            )
    elif source_boundary >= global_boundary:
        raise ValueError(
            "Post-sample AM stateful event must have warmup tokens between "
            f"source_boundary={source_boundary} and boundary={global_boundary}."
        )

    last = replay_steps[-1]
    protected = int(last.get("protected_prefix_len", 0) or 0)
    synthetic = int(last.get("synthetic_prefix_len", 0) or 0)
    exact = target_len - protected - synthetic
    if exact <= 0:
        raise ValueError(
            "AM stateful event produced non-positive exact tail: "
            f"target={target_len}, protected={protected}, synthetic={synthetic}"
        )
    if cache_hit_tokens:
        if cache_hit_tokens > target_len:
            raise ValueError(
                "AM stateful event cache hit exceeds target length: "
                f"hit={cache_hit_tokens}, target={target_len}"
            )
        if cache_hit_tokens < protected + synthetic:
            raise ValueError(
                "AM stateful event cache hit does not cover synthetic prefix: "
                f"hit={cache_hit_tokens}, protected={protected}, "
                f"synthetic={synthetic}"
            )

    return CompactionEventWire(
        num_output_tokens_at_compaction=global_boundary,
        tokens_evicted=tokens_evicted,
        position_offset_after=offset_before + tokens_evicted,
        num_prompt_tokens=global_prompt_len,
        evict_start=event.evict_start,
        compaction_strategy=event.compaction_strategy,
        source_len=source_len,
        target_len=target_len,
        protected_prefix_len=protected,
        synthetic_prefix_len=synthetic,
        exact_kept_tokens=exact,
        attention_matching_query_source=event.attention_matching_query_source,
        attention_matching_max_queries_per_kv_head=(
            event.attention_matching_max_queries_per_kv_head
        ),
        attention_matching_query_seed=int(
            last.get(
                "attention_matching_query_seed",
                event.attention_matching_query_seed,
            )
            or 0
        ),
        attention_matching_zerobeta=event.attention_matching_zerobeta,
        attention_matching_pre_sample=pre_sample,
        attention_matching_replay_steps=replay_steps,
        attention_matching_cache_hit_tokens=int(cache_hit_tokens),
        attention_matching_selected_indices=(
            _normalize_selected_indices(
                last.get("attention_matching_selected_indices")
                if isinstance(last, dict)
                else None
            )
            or event.attention_matching_selected_indices
        ),
        attention_matching_forget_gate_enabled=bool(
            last.get(
                "attention_matching_forget_gate_enabled",
                event.attention_matching_forget_gate_enabled,
            )
            if isinstance(last, dict)
            else event.attention_matching_forget_gate_enabled
        ),
        attention_matching_forget_gate_alpha=float(
            last.get(
                "attention_matching_forget_gate_alpha",
                event.attention_matching_forget_gate_alpha,
            )
            if isinstance(last, dict)
            else event.attention_matching_forget_gate_alpha
        ),
        attention_matching_forget_gate_applied=bool(
            last.get(
                "attention_matching_forget_gate_applied",
                event.attention_matching_forget_gate_applied,
            )
            if isinstance(last, dict)
            else event.attention_matching_forget_gate_applied
        ),
        attention_matching_hidden_tail_token_ids=(
            getattr(event, "attention_matching_hidden_tail_token_ids", None)
        ),
    )


def _build_summary_sample(
    step: vf.TrajectoryStep,
    *,
    temperature: float,
    has_error: bool,
) -> TrainingSample | None:
    """Build a standalone TrainingSample from a summary_trainsample payload."""
    extras = step.get("extras")
    if not extras:
        return None
    payload = extras.get("summary_trainsample")
    if not isinstance(payload, dict):
        return None

    prompt_ids = _coerce_int_list(payload.get("prompt_token_ids"))
    completion_ids = _coerce_int_list(payload.get("completion_token_ids"))
    completion_logprobs = _coerce_float_list(payload.get("completion_logprobs"))
    if not prompt_ids or not completion_ids or completion_logprobs is None:
        return None
    if len(completion_logprobs) != len(completion_ids):
        return None

    return TrainingSample(
        prompt_ids=prompt_ids,
        prompt_mask=[False] * len(prompt_ids),
        completion_ids=completion_ids,
        completion_mask=[False if has_error else True] * len(completion_ids),
        completion_logprobs=completion_logprobs,
        completion_temperatures=[temperature] * len(completion_ids),
        teacher_logprobs=None,
        advantage=None,
        routed_experts=None,
        compaction_events=_compaction_events_from_raw(payload.get("compaction_events")),
    )


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


def interleave_rollout(
    output: vf.RolloutOutput,
    vlm_cache: "VLMImageCache | None" = None,
    cache_key: int | None = None,
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
    disable_stateful_am_trace = os.environ.get(
        "KVE_DISABLE_STATEFUL_AM_TRACE", ""
    ).lower() in {"1", "true", "yes", "on"}

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
            extras = step.get("extras") or {}
            override_prompt_ids = _coerce_int_list(extras.get("prompt_token_ids"))
            if override_prompt_ids:
                prompt_ids = override_prompt_ids
                prompt_mask = [False] * len(prompt_ids)
            return {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": list(tokens["completion_ids"]),
                "completion_mask": [bool(i) for i in tokens["completion_mask"]],
                "completion_logprobs": list(tokens["completion_logprobs"]),
                "routed_experts": tokens.get("routed_experts"),
            }

        logger.warning(f"Missing rollout tokens for example {output['example_id']} step {step_idx}.")
        return None

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
        return _compaction_events_from_raw(raw)

    prepared_steps: list[dict[str, Any]] = []
    step_compaction_events: list[list[CompactionEventWire] | None] = []
    for step_idx, step in enumerate(trajectory):
        prepared = prepare_step_tokens(step, step_idx)
        if prepared is None:
            return None
        prepared_steps.append(prepared)
        step_compaction_events.append(_compaction_events_from_step(step))

    def make_sample(
        tokens: dict[str, Any],
        compaction_events: list[CompactionEventWire] | None = None,
    ) -> TrainingSample:
        """Create a new TrainingSample from a trajectory step."""
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]
        prompt_ids = list(tokens["prompt_ids"])
        prompt_mask = [bool(i) for i in tokens["prompt_mask"]]
        completion_ids = list(tokens["completion_ids"])
        completion_logprobs = list(tokens["completion_logprobs"])
        expanded_events = compaction_events

        if compaction_events and all(
            _am_prompt_replay_event(event) for event in compaction_events
        ):
            coalesced_event = _coalesce_am_prompt_replay_events(
                compaction_events,
                prompt_len=len(prompt_ids),
            )
            expanded_events = (
                [coalesced_event] if coalesced_event is not None else None
            )

        hidden_tail_ids = _am_hidden_tail_token_ids(compaction_events)
        if hidden_tail_ids:
            completion_ids.extend(hidden_tail_ids)
            completion_mask.extend([False] * len(hidden_tail_ids))
            completion_logprobs.extend([0.0] * len(hidden_tail_ids))

        routed_experts = _align_routed_experts(
            tokens.get("routed_experts"),
            len(prompt_ids) + len(completion_ids),
        )

        return TrainingSample(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=completion_logprobs,
            completion_temperatures=[temperature] * len(completion_ids),
            teacher_logprobs=None,
            advantage=None,
            routed_experts=routed_experts,
            compaction_events=expanded_events,
        )

    def extend_sample(sample: TrainingSample, prefix_len: int, step_idx: int) -> None:
        """Extend an existing sample with a new trajectory step (extension property holds)."""
        tokens = prepared_steps[step_idx]

        # Extend with new prompt tokens (mask=False, no gradient)
        new_prompt_ids = tokens["prompt_ids"][prefix_len:]
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

    def single_am_prompt_event(
        events: list[CompactionEventWire] | None,
    ) -> CompactionEventWire | None:
        if not events:
            return None
        prompt_events = [
            event for event in events if _am_prompt_replay_event(event)
        ]
        return _coalesce_am_prompt_replay_events(prompt_events)

    def extend_stateful_am_sample(
        sample: TrainingSample,
        prefix_len: int,
        step_idx: int,
        *,
        step_event: CompactionEventWire,
        step_events: list[CompactionEventWire] | None,
        active_replay_steps: list[dict[str, Any]],
        offset_before: int,
    ) -> tuple[list[dict[str, Any]], int, list[int]]:
        """Extend a sample while preserving AM compressed state across turns."""
        tokens = prepared_steps[step_idx]
        current_replay_steps = list(step_event.attention_matching_replay_steps or [])
        common = _am_common_replay_prefix_len(
            active_replay_steps, current_replay_steps
        )
        if common != len(active_replay_steps):
            raise RuntimeError(
                "Internal AM merge error: active replay state is not a prefix "
                "of the next turn's replay chain."
            )

        new_prompt_ids = tokens["prompt_ids"][prefix_len:]
        replay_suffix = current_replay_steps[common:]
        offset_after = offset_before

        def append_prompt_warmup(prompt_ids: list[int]) -> None:
            sample.completion_ids.extend(prompt_ids)
            sample.completion_mask.extend([False] * len(prompt_ids))
            sample.completion_logprobs.extend([0.0] * len(prompt_ids))
            sample.completion_temperatures.extend([temperature] * len(prompt_ids))

        if replay_suffix:
            suffix_tokens_evicted = sum(
                int(step.get("source_len", 0) or 0)
                - int(step.get("target_len", 0) or 0)
                for step in replay_suffix
            )
            if suffix_tokens_evicted <= 0:
                raise RuntimeError(
                    "Internal AM merge error: replay suffix did not evict any "
                    "tokens."
                )
            local_target_len = int(step_event.target_len or 0)
            local_source_len = local_target_len + suffix_tokens_evicted
            global_boundary = (
                local_source_len + offset_before - len(sample.prompt_ids)
            )
            prompt_warmup_before_am = global_boundary - len(sample.completion_ids)
            if (
                local_target_len <= 0
                or local_source_len <= 0
                or prompt_warmup_before_am < 0
                or prompt_warmup_before_am > len(new_prompt_ids)
            ):
                raise RuntimeError(
                    "Internal AM merge error: prompt-time replay suffix cannot "
                    "be aligned to the vLLM source boundary. "
                    f"local_source={local_source_len}, "
                    f"local_target={local_target_len}, "
                    f"offset_before={offset_before}, "
                    f"global_prompt={len(sample.prompt_ids)}, "
                    f"current_completion={len(sample.completion_ids)}, "
                    f"new_prompt={len(new_prompt_ids)}, "
                    f"boundary={global_boundary}."
                )

            append_prompt_warmup(new_prompt_ids[:prompt_warmup_before_am])
            if global_boundary != len(sample.completion_ids):
                raise RuntimeError(
                    "Internal AM merge error: prompt warmup split did not land "
                    f"on AM boundary: boundary={global_boundary}, "
                    f"completion={len(sample.completion_ids)}."
                )
            source_len = (
                len(sample.prompt_ids) + len(sample.completion_ids) - offset_before
            )
            new_event = _make_stateful_am_prompt_event(
                event=step_event,
                replay_steps=replay_suffix,
                global_boundary=global_boundary,
                global_prompt_len=len(sample.prompt_ids),
                offset_before=offset_before,
                source_len_override=source_len,
                # The cached prefix is already represented by the active
                # trainer-side state. Only the newly required replay suffix
                # becomes a segment boundary in the merged trace.
                cache_hit_tokens=0,
                pre_sample=True,
            )
            if sample.compaction_events is None:
                sample.compaction_events = []
            sample.compaction_events.append(new_event)
            offset_after = new_event.position_offset_after
            append_prompt_warmup(new_prompt_ids[prompt_warmup_before_am:])
        elif int(step_event.position_offset_after) != offset_after:
            raise RuntimeError(
                "Internal AM merge error: prompt cache event offset does not "
                "match the carried active state. "
                f"event_offset={step_event.position_offset_after}, "
                f"active_offset={offset_after}."
            )
        else:
            append_prompt_warmup(new_prompt_ids)

        completion_base = len(sample.completion_ids)
        for event in _am_generation_events(step_events):
            replay_steps = list(event.attention_matching_replay_steps or [])
            if not replay_steps:
                raise RuntimeError(
                    "Cannot merge AM generation event without replay_steps."
                )
            global_boundary = (
                completion_base + int(event.num_output_tokens_at_compaction)
            )
            new_event = _make_stateful_am_prompt_event(
                event=event,
                replay_steps=replay_steps,
                global_boundary=global_boundary,
                global_prompt_len=len(sample.prompt_ids),
                offset_before=offset_after,
                cache_hit_tokens=0,
                pre_sample=bool(event.attention_matching_pre_sample),
            )
            if sample.compaction_events is None:
                sample.compaction_events = []
            sample.compaction_events.append(new_event)
            offset_after = new_event.position_offset_after

        completion_ids = tokens["completion_ids"]
        sample.completion_ids.extend(completion_ids)
        if has_error:
            sample.completion_mask.extend([False] * len(tokens["completion_mask"]))
        else:
            sample.completion_mask.extend(bool(i) for i in tokens["completion_mask"])
        sample.completion_logprobs.extend(tokens["completion_logprobs"])
        sample.completion_temperatures.extend([temperature] * len(completion_ids))

        hidden_tail_ids = _am_hidden_tail_token_ids(step_compaction_events[step_idx])
        if hidden_tail_ids:
            sample.completion_ids.extend(hidden_tail_ids)
            sample.completion_mask.extend([False] * len(hidden_tail_ids))
            sample.completion_logprobs.extend([0.0] * len(hidden_tail_ids))
            sample.completion_temperatures.extend(
                [temperature] * len(hidden_tail_ids)
            )

        if tokens.get("routed_experts") is not None and sample.routed_experts is not None:
            step_routed = tokens["routed_experts"]
            if prefix_len > 0 and prefix_len <= len(step_routed):
                sample.routed_experts[prefix_len - 1] = step_routed[prefix_len - 1]
            sample.routed_experts.extend(step_routed[prefix_len:])
            expected_len = len(sample.prompt_ids) + len(sample.completion_ids)
            sample.routed_experts = _align_routed_experts(sample.routed_experts, expected_len)

        completed_replay_steps = _am_completed_request_replay_steps(step_events)
        if completed_replay_steps is None:
            completed_replay_steps = current_replay_steps
        completed_offset_after = _am_completed_request_offset_after(step_events)
        if completed_offset_after and completed_offset_after != offset_after:
            raise RuntimeError(
                "Internal AM merge error: remapped generation events ended at "
                "a different offset than vLLM reported. "
                f"remapped_offset={offset_after}, "
                f"event_offset={completed_offset_after}."
            )

        return completed_replay_steps, offset_after, hidden_tail_ids

    # Track [prefix_tokens, sample, last_step_idx, replay_steps,
    # offset_after, hidden_tail_ids] per active sample.
    active_samples: list[list] = []

    first_tokens = prepared_steps[0]
    first_hidden_tail = _am_hidden_tail_token_ids(step_compaction_events[0])
    first_prefix = (
        first_tokens["prompt_ids"]
        + first_tokens["completion_ids"]
        + first_hidden_tail
    )
    first_sample = make_sample(first_tokens, step_compaction_events[0])
    first_replay_steps = _am_completed_request_replay_steps(
        step_compaction_events[0]
    )
    first_offset_after = _am_completed_request_offset_after(
        step_compaction_events[0]
    )
    active_samples.append(
        [
            first_prefix,
            first_sample,
            0,
            first_replay_steps,
            first_offset_after,
            first_hidden_tail,
        ]
    )

    for step_idx, _step in enumerate(trajectory[1:], start=1):
        tokens = prepared_steps[step_idx]
        step_prompt_ids = tokens["prompt_ids"]

        # Check if this step extends ANY active prefix.
        #
        # Request-local compaction events generally cannot be merged across
        # turns: a fresh vLLM request with a full text prompt would recompute
        # them from text. Cross-turn AM prompt replay events are the exception.
        # In am_full mode vLLM intentionally reuses compressed prefix state
        # across turns, so the trainer must preserve that state by merging the
        # compatible prompt-time AM chain and adding only newly required replay
        # suffixes as later segment boundaries.
        step_events = step_compaction_events[step_idx]
        step_am_event = single_am_prompt_event(step_events)
        step_am_has_cache_hit = any(
            _am_prompt_cache_hit_event(event) for event in (step_events or [])
        )
        matched_idx = None
        matched_prefix_len = None
        step_am_can_consume_state = (
            not disable_stateful_am_trace
            and step_am_event is not None
            and step_am_has_cache_hit
        )
        if step_events is None or step_am_can_consume_state:
            for idx, active in enumerate(active_samples):
                prefix_tokens, sample = active[0], active[1]
                active_replay_steps = active[3]
                active_hidden_tail_ids = active[5] if len(active) > 5 else []
                if step_events is None:
                    if sample.compaction_events is not None:
                        # A later request that truly reuses compressed AM state
                        # records a prompt-time AM event with a cache hit. With
                        # no event, the trainer has no replay metadata proving
                        # vLLM used compressed state, so do not silently carry
                        # the old AM state across the turn.
                        continue
                else:
                    if active_replay_steps is None:
                        continue
                    current_replay_steps = list(
                        step_am_event.attention_matching_replay_steps or []
                    )
                    common = _am_common_replay_prefix_len(
                        active_replay_steps, current_replay_steps
                    )
                    if common != len(active_replay_steps):
                        continue
                if step_events is None:
                    prefix_len = (
                        len(prefix_tokens)
                        if step_prompt_ids[: len(prefix_tokens)] == prefix_tokens
                        else None
                    )
                else:
                    prefix_len = _prefix_match_len_with_optional_hidden_tail(
                        prefix_tokens,
                        step_prompt_ids,
                        active_hidden_tail_ids,
                    )
                if prefix_len is not None:
                    matched_idx = idx
                    matched_prefix_len = prefix_len
                    break

        if matched_idx is not None and step_events is None:
            # Extension holds - merge into matched sample
            prefix_tokens, sample, *_ = active_samples[matched_idx]
            assert matched_prefix_len is not None
            extend_sample(sample, matched_prefix_len, step_idx=step_idx)
            active_samples[matched_idx][0] = tokens["prompt_ids"] + tokens["completion_ids"]
            active_samples[matched_idx][2] = step_idx
        elif matched_idx is not None and step_am_can_consume_state:
            prefix_tokens, sample = active_samples[matched_idx][0], active_samples[matched_idx][1]
            active_replay_steps = active_samples[matched_idx][3]
            offset_before = int(active_samples[matched_idx][4])
            assert matched_prefix_len is not None
            next_replay_steps, offset_after, hidden_tail_ids = extend_stateful_am_sample(
                sample,
                matched_prefix_len,
                step_idx=step_idx,
                step_event=step_am_event,
                step_events=step_events,
                active_replay_steps=active_replay_steps,
                offset_before=offset_before,
            )
            active_samples[matched_idx][0] = (
                tokens["prompt_ids"] + tokens["completion_ids"] + hidden_tail_ids
            )
            active_samples[matched_idx][2] = step_idx
            active_samples[matched_idx][3] = next_replay_steps
            active_samples[matched_idx][4] = offset_after
            active_samples[matched_idx][5] = hidden_tail_ids
        else:
            # No prefix matches - start a new sample
            logger.debug(
                f"Extension property broke at step {step_idx + 1} for example {output['example_id']}. "
                f"Starting new sample (active_prefixes={len(active_samples)}, step_prompt_len={len(step_prompt_ids)})."
            )
            hidden_tail_ids = _am_hidden_tail_token_ids(step_compaction_events[step_idx])
            new_prefix = tokens["prompt_ids"] + tokens["completion_ids"] + hidden_tail_ids
            new_sample = make_sample(tokens, step_compaction_events[step_idx])
            new_replay_steps = _am_completed_request_replay_steps(
                step_compaction_events[step_idx]
            )
            new_offset_after = _am_completed_request_offset_after(
                step_compaction_events[step_idx]
            )
            active_samples.append(
                [
                    new_prefix,
                    new_sample,
                    step_idx,
                    new_replay_steps,
                    new_offset_after,
                    hidden_tail_ids,
                ]
            )

    # Attach images once per sample using only the last merged step
    if vlm_cache is not None:
        key = output["example_id"] if cache_key is None else cache_key
        for active in active_samples:
            sample = active[1]
            last_step_idx = active[2]
            pv, shape, grids = vlm_cache.get_for_step(key, last_step_idx)
            sample.pixel_values = pv
            sample.pixel_values_shape = shape
            sample.image_grid_thw = grids

    samples = [active[1] for active in active_samples]
    for step in trajectory:
        summary_sample = _build_summary_sample(
            step, temperature=temperature, has_error=has_error
        )
        if summary_sample is not None:
            samples.append(summary_sample)

    return samples


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
