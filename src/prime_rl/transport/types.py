import hashlib
import json
import math
import re
from collections.abc import Sequence

import msgspec


def training_effective_temperature(requested_temperature: float) -> float:
    """Return the temperature used to replay inference logits in training."""
    try:
        temperature = float(requested_temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"training temperature must be a finite non-negative number, got "
            f"{requested_temperature!r}"
        ) from exc
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError(
            f"training temperature must be finite and non-negative, got "
            f"{requested_temperature!r}"
        )
    return 1.0 if temperature == 0.0 else temperature


COMPACTION_REPLAY_MODE_LEGACY = 0
COMPACTION_REPLAY_MODE_PREFILL_TRIM = 1

TURN_COMPACTION_STATE_VERSION = 1
TURN_COMPACTION_STATE_HASH_DOMAIN = b"sglang.turn_compaction_state.v1\0"
_TURN_COMPACTION_STATE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TURN_COMPACTION_STATE_FIELDS = (
    "version",
    "position_offset",
    "protected_prefix_len",
    "num_turns_evicted",
    "carried_prefix_num_live_turns",
    "carried_prefix_len",
    "state_id",
)


def validate_survivor_metadata(
    kept_indices: object,
    kept_token_ids: object,
    *,
    context: str,
) -> tuple[list[int], list[int]]:
    def validate(values: object, field: str) -> list[int]:
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{context} {field} must be a concrete list or tuple")
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(f"{context} {field} must contain only non-negative Python ints")
        return list(values)

    return (
        validate(kept_indices, "kept_indices"),
        validate(kept_token_ids, "kept_token_ids"),
    )


class CompactionEventWire(
    msgspec.Struct, array_like=True, gc=False, omit_defaults=True
):
    """Wire-format view of a vLLM KV cache compaction event.

    Mirrors vllm.v1.core.compaction.types.CompactionEvent. Defined here (not
    imported from vllm) so prime-rl stays free of vllm runtime dependencies.
    The kv-eviction integration layer re-exports this as
    kv_eviction.types.CompactionEventWire for env/trainer code.

    Wire semantics: array_like + omit_defaults means extending this struct
    with a new optional trailing field (e.g. evicted_block_indices for a
    non-FIFO eviction strategy) is backwards-compatible on the wire.
    """

    # Monotonic count of tokens generated at the moment this event fired.
    # The trainer uses this as the cumulative boundary in completion-token
    # space where segmented_forward drops KV between segments.
    num_output_tokens_at_compaction: int

    # Number of tokens physically evicted. For FIFO sliding-window eviction
    # this equals stride_blocks * block_size.
    tokens_evicted: int

    # Cumulative position_offset after this event. Legacy replay requires this
    # to equal the event's deletion delta; client-carried SGLang replay checks
    # it against TurnCompactionStateWire's cumulative offset.
    position_offset_after: int

    # Prompt length (in tokens) of the vLLM request that produced this event.
    # Multi-turn: each turn is a separate vLLM request whose prompt grows with
    # the conversation. The trainer uses this to verify eviction boundaries
    # when protected_prefix_tokens is set. Default 0 preserves backward
    # compatibility (omit_defaults=True on the struct).
    num_prompt_tokens: int = 0

    # Start position of the eviction range in the current (possibly
    # already-partially-trimmed) token sequence. For admission-time events
    # (num_output_tokens_at_compaction == 0) the orchestrator replays the
    # same del prompt_ids[evict_start : evict_start + tokens_evicted] that
    # the vLLM scheduler applied, so the trainer sees the trimmed prompt.
    evict_start: int = 0

    # Length of the new_user_fragment forwarded under post-eviction K/V in
    # vLLM's phase-2 prefill. Admission events only (mid-gen events emit 0).
    # The trainer needs this to set the phase-1/phase-2 split boundary
    # inside the per-call two-phase forward (see
    # plans/two_phase_per_call_trainer.md Phase D).
    new_user_fragment_len: int = 0

    # Pre-event indices of tokens that physically survive this eviction.
    # vLLM's authoritative "what's left" view. Length = pre-event_len -
    # tokens_evicted. Sorted ascending. Default empty (omit_defaults).
    kept_indices: list[int] = msgspec.field(default_factory=list)

    # Token IDs at kept_indices positions, in order. Length matches
    # kept_indices. Used by the orchestrator's Phase4 path
    # (orchestrator.compaction_padding.phase4_enabled) to assemble the next
    # call's submitted prompt as [kept_token_ids + asst_out + new_user].
    kept_token_ids: list[int] = msgspec.field(default_factory=list)

    # Turn-mode only: 0-indexed turn that this event evicted (inclusive).
    # -1 for block-FIFO eviction mode. Used by the trainer to verify turn
    # boundary semantics in turn-mode tests.
    last_turn_evicted: int = -1

    # Cumulative count of completed turns evicted from this request after
    # this event. 0 for block-FIFO mode.
    num_turns_evicted_after: int = 0

    # Managed-context extension: IDs of scheduler-local archived spans created
    # by this eviction. Empty unless managed context is explicitly enabled.
    archived_span_ids: list[str] = msgspec.field(default_factory=list)

    # Managed-context extension: per-span [start, end) bounds for each entry
    # in archived_span_ids, flattened pairs in the same pre-event frame as
    # evict_start/kept_indices. Length = 2 * len(archived_span_ids). Maps a
    # span id (referenced later by restore events) to the exact rows that
    # died at this event.
    archived_span_bounds: list[int] = msgspec.field(default_factory=list)

    # Managed-context restore lifecycle. 0 = eviction (default); 1 = hidden
    # restore ATTACH (restored_span_ids became visible to subsequent
    # queries); 2 = restore RELEASE (they left visibility). Kind 1/2 events
    # carry zeros/defaults in the eviction fields; the trainer's interval-
    # visibility mask consumes them, and the death-index (eviction-only)
    # paths must hard-error if one appears.
    event_kind: int = 0

    # Kind 1/2 only: span ids attached/released by this event.
    restored_span_ids: list[str] = msgspec.field(default_factory=list)

    # Kind 1/2 only: the request's num_computed_tokens (current frame) at
    # the visibility change. Queries at positions >= this boundary see
    # (kind 1) / stop seeing (kind 2) the spans. -1 on eviction events.
    visibility_boundary_computed: int = -1

    # Kind 1 recall token-surfacing: the single restored span's token ids and
    # the absolute RoPE position of its first token (rest contiguous). The
    # trainer reconstructs a recalled span whose birth turn is not in this
    # sample by forwarding these at restored_span_pos_start.. positions.
    restored_span_token_ids: list[int] = msgspec.field(default_factory=list)
    restored_span_pos_start: int = -1


_PREFILL_TRIM_EVENT_INT_FIELDS = (
    ("num_output_tokens_at_compaction", 0),
    ("tokens_evicted", 1),
    ("position_offset_after", 2),
    ("num_prompt_tokens", 3),
    ("evict_start", 4),
    ("new_user_fragment_len", 5),
    ("last_turn_evicted", 8),
    ("num_turns_evicted_after", 9),
)


def validate_prefill_trim_event_field_types(
    event: object,
    *,
    context: str,
) -> None:
    """Reject mode-1 event scalar coercions before constructing its wire type."""
    missing = object()
    for field, array_index in _PREFILL_TRIM_EVENT_INT_FIELDS:
        if isinstance(event, CompactionEventWire):
            value = getattr(event, field)
        elif type(event) is dict:
            value = event.get(field, missing)
        elif isinstance(event, (list, tuple)):
            value = event[array_index] if len(event) > array_index else missing
        else:
            value = getattr(event, field, missing)
        if type(value) is not int:
            raise ValueError(f"{context} event {field} must be an exact int")


class TurnCompactionStateWire(
    msgspec.Struct, array_like=True, gc=False, omit_defaults=True
):
    """Validated client-carried SGLang turn-compaction state."""

    version: int
    position_offset: int
    protected_prefix_len: int
    num_turns_evicted: int
    carried_prefix_num_live_turns: int
    carried_prefix_len: int
    state_id: str


def compute_turn_compaction_state_id(
    *,
    version: int,
    position_offset: int,
    protected_prefix_len: int,
    num_turns_evicted: int,
    carried_prefix_num_live_turns: int,
    carried_prefix_len: int,
    carried_prefix_token_ids: Sequence[int],
) -> str:
    """Hash the compact canonical state anchor shared with SGLang."""
    token_ids = list(carried_prefix_token_ids)
    if len(token_ids) != carried_prefix_len:
        raise ValueError(
            "carried-prefix token count does not match carried_prefix_len"
        )
    if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        raise ValueError("carried-prefix token IDs must be nonnegative integers")
    payload = [
        version,
        position_offset,
        protected_prefix_len,
        num_turns_evicted,
        carried_prefix_num_live_turns,
        carried_prefix_len,
        token_ids,
    ]
    canonical = TURN_COMPACTION_STATE_HASH_DOMAIN + json.dumps(
        payload, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    digest = hashlib.sha256(canonical).hexdigest()
    return f"sha256:{digest}"


def turn_compaction_state_to_json(
    state: TurnCompactionStateWire,
) -> dict[str, int | str]:
    return {field: getattr(state, field) for field in _TURN_COMPACTION_STATE_FIELDS}


def validate_turn_compaction_state(
    raw_state: object,
    prompt_ids: object,
    *,
    context: str,
) -> TurnCompactionStateWire:
    """Strictly validate a state and bind its hash to its carried prompt prefix."""
    if isinstance(raw_state, TurnCompactionStateWire):
        values = {
            field: getattr(raw_state, field)
            for field in _TURN_COMPACTION_STATE_FIELDS
        }
    elif type(raw_state) is dict:
        unknown = set(raw_state) - set(_TURN_COMPACTION_STATE_FIELDS)
        missing = set(_TURN_COMPACTION_STATE_FIELDS) - set(raw_state)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if unknown:
                details.append(f"unknown {sorted(unknown)}")
            raise ValueError(
                f"{context} turn_compaction_state has invalid fields: "
                + ", ".join(details)
            )
        values = {field: raw_state[field] for field in _TURN_COMPACTION_STATE_FIELDS}
    else:
        raise ValueError(f"{context} turn_compaction_state must be a JSON object")

    positive_fields = (
        "position_offset",
        "protected_prefix_len",
        "num_turns_evicted",
        "carried_prefix_len",
    )
    if type(values["version"]) is not int or values["version"] != TURN_COMPACTION_STATE_VERSION:
        raise ValueError(
            f"{context} turn_compaction_state version must be "
            f"{TURN_COMPACTION_STATE_VERSION}"
        )
    for field in positive_fields:
        value = values[field]
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"{context} turn_compaction_state {field} must be a positive int"
            )
    live_turns = values["carried_prefix_num_live_turns"]
    if type(live_turns) is not int or live_turns < 0:
        raise ValueError(
            f"{context} turn_compaction_state "
            "carried_prefix_num_live_turns must be a nonnegative int"
        )
    state_id = values["state_id"]
    if type(state_id) is not str or _TURN_COMPACTION_STATE_ID_RE.fullmatch(
        state_id
    ) is None:
        raise ValueError(
            f"{context} turn_compaction_state state_id must be "
            "'sha256:' followed by 64 lowercase hex characters"
        )

    if not isinstance(prompt_ids, (list, tuple)):
        raise ValueError(f"{context} state anchor prompt must be a token ID list")
    if any(type(token_id) is not int or token_id < 0 for token_id in prompt_ids):
        raise ValueError(f"{context} state anchor prompt contains an invalid token ID")
    carried_prefix_len = values["carried_prefix_len"]
    protected_prefix_len = values["protected_prefix_len"]
    if carried_prefix_len > len(prompt_ids):
        raise ValueError(
            f"{context} turn_compaction_state carried_prefix_len exceeds "
            "the physical prompt length"
        )
    if protected_prefix_len > carried_prefix_len:
        raise ValueError(
            f"{context} turn_compaction_state protected_prefix_len exceeds "
            "carried_prefix_len"
        )

    carried_prefix_token_ids = list(prompt_ids[:carried_prefix_len])
    expected_state_id = compute_turn_compaction_state_id(
        version=values["version"],
        position_offset=values["position_offset"],
        protected_prefix_len=protected_prefix_len,
        num_turns_evicted=values["num_turns_evicted"],
        carried_prefix_num_live_turns=live_turns,
        carried_prefix_len=carried_prefix_len,
        carried_prefix_token_ids=carried_prefix_token_ids,
    )
    if state_id != expected_state_id:
        raise ValueError(
            f"{context} turn_compaction_state state_id does not match "
            "the carried prompt prefix"
        )

    return TurnCompactionStateWire(
        version=values["version"],
        position_offset=values["position_offset"],
        protected_prefix_len=protected_prefix_len,
        num_turns_evicted=values["num_turns_evicted"],
        carried_prefix_num_live_turns=live_turns,
        carried_prefix_len=carried_prefix_len,
        state_id=state_id,
    )


class CallWire(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A single inference chat() call within a rollout (Phase B of
    plans/two_phase_per_call_trainer.md).

    With Phase4 prefix-caching mode (orchestrator.compaction_padding.
    phase4_enabled=True), each call's submitted_prompt_ids is the
    incremental form ``prev_kept_state + padded_new_user_fragment``. The
    trainer iterates calls in order; each call runs (a) a single HF
    forward when admission did not fire, or (b) a two-phase forward
    (phase 1 over [0, evict_end) + cache splice + phase 2 over
    [evict_end, len(submitted_prompt))) when admission fired.

    ``submitted_prompt_ids`` is the physical prompt received by the inference
    server: pre-event when admission compacts it, unchanged when a carried call
    has no new event. Legacy vLLM replay uses it for phase 1. SGLang mode 1
    retains it only for validation; the batch adapter clears ``calls`` before
    trainer dispatch.
    """

    # Tokens the inference server received as the prompt (pre-eviction).
    submitted_prompt_ids: list[int]

    # Tokens sampled during this call.
    completion_ids: list[int]

    # Per-token logprobs of the sampled tokens. Length matches completion_ids.
    completion_logprobs: list[float]

    # Per-token sampling temperatures used during generation. Length matches
    # completion_ids.
    completion_temperatures: list[float]

    # Compaction events that fired during this call (admission AND mid-gen).
    # An empty list means no eviction during this call's prefill or decode.
    # Admission events: num_output_tokens_at_compaction == 0.
    # Mid-gen events: num_output_tokens_at_compaction > 0 (offset is
    # relative to THIS call's completion start, not the merged sample's).
    compaction_events: list[CompactionEventWire] = msgspec.field(default_factory=list)

    # KV cache compaction auto-pad: filler token ids vLLM appended to its
    # KV cache at the END of this call's request (so the trailing block
    # lands in the prefix cache). These are NOT sampled tokens — they sit
    # between completion_ids and the next call's submitted_prompt_ids:
    #   - V's next call inherits these blocks via prefix cache (so they
    #     appear at the head of next_call.submitted_prompt_ids).
    #   - The trainer appends them to this call's pre_trim K-cache
    #     contribution so its persistent cache layout matches V's.
    # Empty when auto-pad did not fire for this call.
    trailing_pad_ids: list[int] = msgspec.field(default_factory=list)

    # 0 = legacy vLLM warmup/KV splice; 1 = SGLang post-trim prompt replay.
    compaction_replay_mode: int = COMPACTION_REPLAY_MODE_LEGACY

    # Client-carried cumulative turn-compaction state. Trailing and optional so
    # CallWire arrays emitted before this field continue to decode unchanged.
    turn_compaction_state: TurnCompactionStateWire | None = None


# Orchestrator -> Packer
class TrainingSample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A single training example."""

    prompt_ids: list[int]
    prompt_mask: list[bool]
    completion_ids: list[int]
    completion_mask: list[bool]
    completion_logprobs: list[float]
    completion_temperatures: list[float]  # Per-token temperatures used during generation
    teacher_logprobs: list[float] | None = None
    advantage: float | None = None
    reward: float | None = None

    # Multimodal fields (Qwen3-VL) — pixel_values stored as raw float32 bytes for efficient serialization
    pixel_values: bytes | None = None
    pixel_values_shape: list[int] | None = None  # [num_patches, patch_dim]
    # image_grid_thw: grid dimensions [num_images, 3] where each entry is [temporal, height, width]
    image_grid_thw: list[list[int]] | None = None

    routed_experts: list[list[list[int]]] | None = None  # [seq_len, layers, topk]

    # KV cache compaction events from the inference engine. None when
    # compaction is disabled or no events fired for this sample. The trainer
    # dispatches to segmented_forward when this is non-empty.
    compaction_events: list[CompactionEventWire] | None = None

    # Per-call breakdown for the per-call trainer rebuild (Phase B+).
    # One CallWire per vLLM chat() call merged into this sample. None when
    # compaction is disabled (the trainer uses the merged path instead).
    calls: list[CallWire] | None = None

    # Aggregate replay contract. Mode 1 samples contain exactly one mode 1 call.
    compaction_replay_mode: int = COMPACTION_REPLAY_MODE_LEGACY


class TrainingBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A batch of training examples with metadata for transport."""

    examples: list[TrainingSample]
    step: int
    run_idx: int | None = None


# Packer -> Trainer
class MicroBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A micro batch of data for training."""

    input_ids: list[int]
    loss_mask: list[bool]
    advantages: list[float]
    inference_logprobs: list[float]
    position_ids: list[int]
    temperatures: list[float]  # Per-token temperatures used during generation
    teacher_logprobs: list[float] | None = None
    lora_num_tokens: list[int] | None = None
    routed_experts: list[list[list[int]]] | None = None

    # Multimodal fields (Qwen3-VL) — pixel_values stored as raw float32 bytes for efficient serialization
    pixel_values: bytes | None = None
    pixel_values_shape: list[int] | None = None  # [num_patches, patch_dim]
    # image_grid_thw: grid dimensions [num_images, 3] where each entry is [temporal, height, width]
    image_grid_thw: list[list[int]] | None = None

    # KV cache compaction events for the single sample in this micro-batch.
    # Invariant: when non-None, the packer has NOT bin-packed other samples
    # alongside this one (see SinglePacker + compaction assertion), so there
    # is exactly one sample per micro-batch and one list of events for it.
    # None for non-compaction samples (can be packed freely).
    compaction_events: list[CompactionEventWire] | None = None

    # Prompt length (= len(TrainingSample.prompt_ids), after any truncation).
    # Only set for compaction samples; the trainer needs this to compute
    # prompt_aligned_len = ceil(prompt_len / block_size) * block_size for
    # segmented_forward's drop boundary. None for non-compaction samples.
    prompt_len: int | None = None

    # Per-call breakdown forwarded from TrainingSample.calls. The trainer's
    # per-call segmented forward iterates these. None when the sample
    # didn't come from a compaction run.
    calls: list[CallWire] | None = None

    # Retained after mode 1 events/calls are cleared before trainer dispatch.
    compaction_replay_mode: int = COMPACTION_REPLAY_MODE_LEGACY
