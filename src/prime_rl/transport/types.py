import msgspec


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

    # Cumulative position_offset after this event (not currently used by the
    # trainer; position_ids can be computed as a plain arange since gen[N]'s
    # RoPE always equals prompt_len + N, but carried for debugging and for
    # future non-FIFO strategies).
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


class CallWire(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A single vLLM chat() call within a rollout (Phase B of
    plans/two_phase_per_call_trainer.md).

    With Phase4 prefix-caching mode (orchestrator.compaction_padding.
    phase4_enabled=True), each call's submitted_prompt_ids is the
    incremental form ``prev_kept_state + padded_new_user_fragment``. The
    trainer iterates calls in order; each call runs (a) a single HF
    forward when admission did not fire, or (b) a two-phase forward
    (phase 1 over [0, evict_end) + cache splice + phase 2 over
    [evict_end, len(submitted_prompt))) when admission fired.

    ``submitted_prompt_ids`` is the PRE-eviction prompt (what vLLM
    received). The trainer's two-phase forward needs this to run
    phase 1 over [0, evict_end). The post-eviction view used as
    ``TrainingSample.prompt_ids`` is the same sequence with
    ``[evict_start, evict_end)`` deleted — derivable from this struct
    by replaying ``compaction_events``.
    """

    # Tokens vLLM received as the prompt for this call (pre-eviction).
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
