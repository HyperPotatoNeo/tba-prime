from typing import Any

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

    # Prompt token count and eviction start are optional trailing metadata.
    # Keeping defaults preserves msgspec array_like wire compatibility with
    # older 3-field events while allowing turn/summary compaction code to
    # retain the extra alignment diagnostics when present.
    num_prompt_tokens: int = 0
    evict_start: int = 0
    compaction_strategy: str = "fifo"
    source_len: int = 0
    target_len: int = 0
    protected_prefix_len: int = 0
    synthetic_prefix_len: int = 0
    exact_kept_tokens: int = 0
    attention_matching_query_source: str = ""
    attention_matching_max_queries_per_kv_head: int = 0
    attention_matching_query_seed: int = 0
    attention_matching_zerobeta: bool = False
    attention_matching_pre_sample: bool = False
    attention_matching_replay_steps: list[dict[str, Any]] | None = None
    attention_matching_cache_hit_tokens: int = 0
    attention_matching_selected_indices: list[list[list[int]]] | None = None
    attention_matching_forget_gate_enabled: bool = False
    attention_matching_forget_gate_alpha: float = 0.5
    attention_matching_forget_gate_applied: bool = False
    attention_matching_hidden_tail_token_ids: list[int] | None = None


class CallWire(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """One inference call inside an interleaved multi-turn rollout trace."""

    submitted_prompt_ids: list[int]
    completion_ids: list[int]
    compaction_events: list[CompactionEventWire] | None = None
    trailing_pad_ids: list[int] | None = None


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

    # Optional per-call trace for multi-turn replay. When present, the trainer
    # can preserve the same request boundaries vLLM used at rollout time rather
    # than scoring the merged episode as one dense full-context forward.
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

    # Optional per-call trace for the single sample in this micro-batch.
    # Invariant: when non-None, the packer has NOT bin-packed other samples
    # alongside this one, matching compaction-style micro-batch isolation.
    calls: list[CallWire] | None = None
