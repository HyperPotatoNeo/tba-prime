import copy

from prime_rl.transport.types import CompactionEventWire, MicroBatch, TrainingSample


def _clamp_compaction_events(
    events: list[CompactionEventWire] | None,
    completion_len_after_trunc: int,
) -> list[CompactionEventWire] | None:
    """Drop compaction events that refer to generated tokens lost to truncation.

    Each event's `num_output_tokens_at_compaction` is a position in
    completion-space (pre-truncation). If truncation cut the completion to
    `completion_len_after_trunc` tokens, any event with
    `num_output_tokens_at_compaction > completion_len_after_trunc` refers
    to a compaction that happened on tokens that no longer exist in the
    training sample — drop it. Events at or before the cut are kept as-is.

    Returns None when no events remain (either input was None/empty, or all
    were dropped).
    """
    if not events:
        return None
    kept = [
        e for e in events if e.num_output_tokens_at_compaction <= completion_len_after_trunc
    ]
    return kept or None


def prepare_sample(training_example: TrainingSample, seq_len: int) -> MicroBatch:
    """
    Prepare a problem for sequence packing training.
    Tokenize and prepare tensors.
    """
    input_ids = training_example.prompt_ids + training_example.completion_ids
    loss_mask = training_example.prompt_mask + training_example.completion_mask
    inference_logprobs = [0.0] * len(training_example.prompt_ids) + training_example.completion_logprobs
    advantages = [training_example.advantage] * len(input_ids)
    position_ids = list(range(len(input_ids)))

    # Per-token temperatures: prompt tokens use first completion temp (masked out anyway)
    # Default to 1.0 if completion is empty (e.g., model generated only tool calls with no text)
    prompt_temp = training_example.completion_temperatures[0] if training_example.completion_temperatures else 1.0
    temperatures = [prompt_temp] * len(training_example.prompt_ids) + training_example.completion_temperatures

    # Teacher logprobs already cover the full sequence (prompt + completion),
    # computed via prefill in the orchestrator when a teacher model is configured
    teacher_logprobs = training_example.teacher_logprobs
    routed_experts = training_example.routed_experts
    # Compaction events (kv-eviction). Passed through to the MicroBatch and
    # clamped below if the completion was truncated to fit seq_len.
    compaction_events = training_example.compaction_events

    if len(input_ids) > seq_len:
        input_ids = input_ids[:seq_len]
        loss_mask = loss_mask[:seq_len]
        inference_logprobs = inference_logprobs[:seq_len]
        position_ids = position_ids[:seq_len]
        advantages = advantages[:seq_len]
        temperatures = temperatures[:seq_len]
        if teacher_logprobs is not None:
            teacher_logprobs = teacher_logprobs[:seq_len]
        if routed_experts is not None:
            routed_experts = routed_experts[:seq_len]
        # Clamp compaction events to the post-truncation completion length.
        # completion_len_after_trunc = min(full_completion, seq_len - prompt_len).
        # If the prompt itself fills/exceeds seq_len, no completion tokens
        # remain and all events are dropped.
        prompt_len = len(training_example.prompt_ids)
        completion_len_after_trunc = max(0, seq_len - prompt_len)
        compaction_events = _clamp_compaction_events(
            compaction_events, completion_len_after_trunc
        )

    assert (
        len(input_ids)
        == len(advantages)
        == len(loss_mask)
        == len(position_ids)
        == len(inference_logprobs)
        == len(temperatures)
    ), (
        f"input_ids: {len(input_ids)}, advantages: {len(advantages)}, loss_mask: {len(loss_mask)}, position_ids: {len(position_ids)}, inference_logprobs: {len(inference_logprobs)}, temperatures: {len(temperatures)}"
    )
    if teacher_logprobs is not None:
        assert len(teacher_logprobs) == len(input_ids), f"teacher_logprobs: {len(teacher_logprobs)}"

    if routed_experts is not None:
        assert len(routed_experts) == len(input_ids), (
            f"routed_experts: {len(routed_experts)}, input_ids: {len(input_ids)}"
        )

    # Set prompt_len on the MicroBatch ONLY for compaction samples (the
    # trainer needs it to compute prompt_aligned_len for segmented_forward).
    # Non-compaction samples leave it None so the msgspec wire format stays
    # backwards-compatible with existing pipelines.
    prompt_len: int | None = None
    if compaction_events:
        # Post-truncation prompt length. If truncation cut INTO the prompt
        # itself (extreme case), use the new length; otherwise it's the
        # original prompt length.
        prompt_len = min(len(training_example.prompt_ids), len(input_ids))

    return MicroBatch(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask,
        position_ids=position_ids,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=teacher_logprobs,
        temperatures=temperatures,
        routed_experts=routed_experts,
        # Multimodal fields (Qwen3-VL) - passed through without modification
        pixel_values=training_example.pixel_values,
        pixel_values_shape=training_example.pixel_values_shape,
        image_grid_thw=training_example.image_grid_thw,
        # kv-eviction compaction events + prompt_len
        compaction_events=compaction_events,
        prompt_len=prompt_len,
    )


def _is_multimodal_sample(sample: MicroBatch) -> bool:
    """Check if a sample contains multimodal data (images)."""
    return sample.pixel_values is not None


def _is_compaction_sample(sample: MicroBatch) -> bool:
    """Check if a sample carries KV cache compaction events.

    Compaction samples CANNOT be bin-packed with any other sample because:
    1. The trainer dispatches to segmented_forward per-sample, and
       segmented_forward expects batch_size=1 with a single contiguous
       prompt + completion layout.
    2. The compaction event list is per-sample; there's no slot in MicroBatch
       for a list-of-lists (one per packed sample).
    Each compaction sample therefore becomes its own micro batch, same
    pattern as multimodal samples.
    """
    return sample.compaction_events is not None and len(sample.compaction_events) > 0


def packed_samples_into_micro_bs(
    samples: list[tuple[int, MicroBatch]], max_seq_len: int, num_loras: int
) -> list[MicroBatch]:
    """
    Pack samples into micro_batch efficiently.
    We follow the First Fit Decreasing algorithm to pack the samples into bins and minimize potential padding while never truncating.
    With per-token temperatures, samples can be packed together regardless of their temperature values.

    NOTE: Multimodal samples (with pixel_values) are NOT packed together as they have variable-sized
    vision data that doesn't pack well. Each multimodal sample becomes its own micro batch.

    NOTE: Compaction samples (with compaction_events) are also NOT packed
    together. See _is_compaction_sample docstring for rationale.
    """
    # Sort by (lora_idx, -length) for packing efficiency
    samples.sort(key=lambda x: (x[0], -len(x[1].input_ids)))

    ## we create bins
    micro_batches: list[MicroBatch] = []

    for idx, sample in samples:
        # Multimodal and compaction samples cannot be packed - each becomes
        # its own micro batch.
        if _is_multimodal_sample(sample) or _is_compaction_sample(sample):
            sample.lora_num_tokens = [0] * num_loras
            sample.lora_num_tokens[idx] = len(sample.input_ids)
            micro_batches.append(sample)
            continue

        # Try to find a bin that can fit this sequence (only pack text-only samples)
        for bin_content in micro_batches:
            # Don't pack into multimodal or compaction micro batches
            if _is_multimodal_sample(bin_content) or _is_compaction_sample(bin_content):
                continue
            # Check if sequence fits in this bin
            if len(bin_content.input_ids) + len(sample.input_ids) <= max_seq_len:
                bin_content.input_ids.extend(sample.input_ids)
                bin_content.loss_mask.extend(sample.loss_mask)
                bin_content.advantages.extend(sample.advantages)
                bin_content.inference_logprobs.extend(sample.inference_logprobs)
                bin_content.temperatures.extend(sample.temperatures)
                if sample.teacher_logprobs is not None:
                    if bin_content.teacher_logprobs is None:
                        bin_content.teacher_logprobs = []
                    bin_content.teacher_logprobs.extend(sample.teacher_logprobs)
                if sample.routed_experts is not None:
                    if bin_content.routed_experts is None:
                        bin_content.routed_experts = []
                    bin_content.routed_experts.extend(sample.routed_experts)
                bin_content.position_ids.extend(sample.position_ids)
                bin_content.lora_num_tokens[idx] += len(sample.input_ids)
                break
        else:
            sample.lora_num_tokens = [0] * num_loras
            sample.lora_num_tokens[idx] = len(sample.input_ids)
            micro_batches.append(sample)

    return micro_batches


def pad_micro_batch(micro_batch: MicroBatch, pad_to_multiple_of: int) -> MicroBatch:
    """
    Pad a micro batch with the given padding size sample
    Return the padded micro batch.
    Args:
        micro_batch: The micro batch to pad.
        padding_size: The number of padding tokens to add.
    Returns:
        The padded micro batch.
    """

    padding_size = (pad_to_multiple_of - (len(micro_batch.input_ids) % pad_to_multiple_of)) % pad_to_multiple_of

    if not (pad_to_multiple_of > 1 and padding_size > 0):
        return micro_batch

    micro_batch.input_ids.extend([1] * padding_size)
    micro_batch.advantages.extend([0.0] * padding_size)
    micro_batch.loss_mask.extend([False] * padding_size)
    # Compaction samples require continuous position_ids (segmented_forward
    # uses them as absolute RoPE positions over the whole concatenated
    # sequence). For packed text/multimodal samples, padding position_ids
    # restart at 0 to mark a new cu_seqlens boundary.
    if _is_compaction_sample(micro_batch):
        last_pos = micro_batch.position_ids[-1] if micro_batch.position_ids else -1
        micro_batch.position_ids.extend(
            list(range(last_pos + 1, last_pos + 1 + padding_size))
        )
    else:
        micro_batch.position_ids.extend(list(range(padding_size)))
    micro_batch.inference_logprobs.extend([0.0] * padding_size)
    # Use temperature 1.0 for padding tokens (doesn't matter since loss_mask is False)
    micro_batch.temperatures.extend([1.0] * padding_size)
    if micro_batch.teacher_logprobs is not None:
        micro_batch.teacher_logprobs.extend([0.0] * padding_size)
    micro_batch.lora_num_tokens[-1] += (
        padding_size  # We send padding to the last lora so that tokens have ascending lora idx
    )

    return micro_batch


def _make_dummy_batch(source: MicroBatch) -> MicroBatch:
    """Create a zero-loss dummy batch from an existing batch, preserving its modality."""
    dummy = copy.deepcopy(source)
    dummy.advantages = [0.0] * len(dummy.input_ids)
    dummy.loss_mask = [False] * len(dummy.input_ids)
    return dummy


def _pad_group_for_distribution(group: list[MicroBatch], num_train_workers: int) -> list[MicroBatch]:
    """Pad a group of micro batches so its length is divisible by num_train_workers."""
    num_padding = -len(group) % num_train_workers
    if num_padding > 0 and len(group) > 0:
        dummy = _make_dummy_batch(group[0])
        group.extend([dummy] * num_padding)
    return group


def prepare_batch(
    rollouts: list[TrainingSample],
    seq_len: int,
    num_train_workers: int,
    idxs: list[int],
    num_loras: int,
    pad_to_multiple_of: int = 1,
) -> list[list[MicroBatch]]:
    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    Each micro batch is shape [1, seq_len], the number of samples is not fixed per micro batch.

    FSDP requires all ranks to execute the same operations at each step. If one rank
    processes a multimodal batch (triggering the vision encoder) while another processes
    a text-only batch, the all-gather will hang. We separate micro batches by modality
    and distribute them so that at each step index, all ranks see the same modality.

    Three modality buckets:
    1. Multimodal (pixel_values set) — triggers vision encoder.
    2. Compaction (compaction_events set) — triggers segmented_forward with
       multiple forward passes per micro-batch instead of the standard single
       pass. If one rank takes the segmented branch and another the standard
       branch at the same step index, they diverge in FSDP all-gather counts
       and NCCL deadlocks.
    3. Text (neither) — the default single-pass path.
    """
    all_samples = [(idx, prepare_sample(rollout, seq_len)) for idx, rollout in zip(idxs, rollouts)]

    micro_batches = packed_samples_into_micro_bs(all_samples, seq_len, num_loras)
    micro_batches = [pad_micro_batch(micro_batch, pad_to_multiple_of) for micro_batch in micro_batches]

    # Separate by modality so each step index has uniform modality across all ranks.
    # Order of checks matters: compaction samples may technically also be
    # multimodal in the future, but today they're mutually exclusive.
    mm_batches = [b for b in micro_batches if _is_multimodal_sample(b)]
    compaction_batches = [
        b for b in micro_batches
        if not _is_multimodal_sample(b) and _is_compaction_sample(b)
    ]
    text_batches = [
        b for b in micro_batches
        if not _is_multimodal_sample(b) and not _is_compaction_sample(b)
    ]

    # Pad each group independently so its count is divisible by num_train_workers
    mm_batches = _pad_group_for_distribution(mm_batches, num_train_workers)
    compaction_batches = _pad_group_for_distribution(compaction_batches, num_train_workers)
    text_batches = _pad_group_for_distribution(text_batches, num_train_workers)

    # Combine: all multimodal first, then all compaction, then all text-only.
    # Each group's length is divisible by num_train_workers, so modality
    # boundaries align with distribution rows — every column (rank) sees
    # exactly the same modality at step index i.
    ordered = mm_batches + compaction_batches + text_batches

    assert len(ordered) % num_train_workers == 0, "Number of micro batches is not divisible by number of data ranks"

    # Distribute in strided order so each step index has the same modality across ranks
    batches_per_gpu: list[list[MicroBatch]] = [[] for _ in range(num_train_workers)]
    for i, batch in enumerate(ordered):
        batches_per_gpu[i % num_train_workers].append(batch)

    return batches_per_gpu
