from __future__ import annotations

import torch

from prime_rl.trainer.rl.data import TensorMicroBatch
from prime_rl.trainer.utils import get_response_lengths


def _flatten_one_level(items):
    flat = []
    for item in items or []:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _is_list_of_lists(items) -> bool:
    return bool(items) and isinstance(items, list) and all(isinstance(item, list) for item in items)


def compaction_metric_events_by_row(
    compaction_events,
    calls=None,
    *,
    include_call_events: bool,
):
    """Return compaction metric events as one list per logical row.

    A stacked flex-compaction micro-batch stores ``calls`` as
    ``list[list[Call]]``. Keep those rows separate so logging reports
    per-sample event counts instead of one inflated count per stacked group.
    """
    batched_calls = include_call_events and _is_list_of_lists(calls)
    if batched_calls:
        compaction_rows = (
            compaction_events
            if _is_list_of_lists(compaction_events) and len(compaction_events) == len(calls)
            else [[] for _ in calls]
        )
        rows = []
        for outer_events, row_calls in zip(compaction_rows, calls, strict=True):
            row_events = _flatten_one_level(outer_events)
            for call in row_calls or []:
                row_events.extend(getattr(call, "compaction_events", None) or [])
            rows.append(row_events)
        return rows

    events = _flatten_one_level(compaction_events)
    if include_call_events:
        for call in _flatten_one_level(calls):
            events.extend(getattr(call, "compaction_events", None) or [])
    return [events]


def is_standard_stackable_micro_batch(
    micro_batch: TensorMicroBatch,
    *,
    compaction_enabled: bool,
    cp_enabled: bool,
    lora_enabled: bool,
    multi_run_enabled: bool,
) -> bool:
    """Return whether a micro-batch can be stacked along batch dimension.

    This is intentionally limited to the standard text path. Compaction,
    multimodal, routed-expert replay, LoRA, and multi-run batches all carry
    side metadata whose current trainer dispatch assumes a single micro-batch.
    """
    if compaction_enabled or cp_enabled or lora_enabled or multi_run_enabled:
        return False
    if micro_batch.get("pixel_values") is not None:
        return False
    if micro_batch.get("image_grid_thw") is not None:
        return False
    if micro_batch.get("routed_experts") is not None:
        return False
    if micro_batch.get("compaction_events"):
        return False
    if micro_batch.get("prompt_len") is not None:
        return False
    if micro_batch.get("calls") is not None:
        return False
    return True


def _calls_have_midgen_compaction(calls) -> bool:
    for call in calls or []:
        for event in getattr(call, "compaction_events", None) or []:
            if int(getattr(event, "num_output_tokens_at_compaction", 0)) > 0:
                return True
    return False


def is_flex_compaction_stackable_micro_batch(
    micro_batch: TensorMicroBatch,
    *,
    compaction_enabled: bool,
    flex_compaction_enabled: bool,
    cp_enabled: bool,
    lora_enabled: bool,
    multi_run_enabled: bool,
) -> bool:
    """Return whether a compaction micro-batch can use batched flex replay."""
    if (
        not compaction_enabled
        or not flex_compaction_enabled
        or cp_enabled
        or lora_enabled
        or multi_run_enabled
    ):
        return False
    if micro_batch.get("pixel_values") is not None:
        return False
    if micro_batch.get("image_grid_thw") is not None:
        return False
    if micro_batch.get("routed_experts") is not None:
        return False
    calls = micro_batch.get("calls")
    if not calls:
        return False
    if _calls_have_midgen_compaction(calls):
        return False
    return True


def make_micro_batch_groups(
    micro_batches: list[TensorMicroBatch],
    *,
    stack_size: int,
    stack_token_budget: int | None = None,
    flex_compaction_stack_mode: str = "vertical",
    seq_lens: list[int] | None = None,
    flex_compaction_enabled: bool = False,
    flex_compaction_stackable: list[bool] | None = None,
    compaction_enabled: bool,
    cp_enabled: bool,
    lora_enabled: bool,
    multi_run_enabled: bool,
) -> list[list[TensorMicroBatch]]:
    """Group adjacent stackable standard micro-batches into stack groups.

    When stack_token_budget is unset, grouping is fixed-count. When it is set,
    each stacked standard/vertical forward is capped by rows *
    max_seq_len_in_stack. Horizontal flex-compaction stacking instead caps by
    sum(seq_lens), where seq_lens should be writer lengths. In distributed
    training, pass seq_lens as the per-index maximum length across data-parallel
    ranks so every rank produces identical group boundaries.
    """
    stack_size = max(1, int(stack_size))
    if stack_token_budget is not None:
        stack_token_budget = max(1, int(stack_token_budget))
    if flex_compaction_stack_mode not in {"vertical", "horizontal"}:
        raise ValueError(
            "flex_compaction_stack_mode must be 'vertical' or 'horizontal', "
            f"got {flex_compaction_stack_mode!r}"
        )
    if seq_lens is None:
        seq_lens = [int(mb["input_ids"].shape[1]) for mb in micro_batches]
    if len(seq_lens) != len(micro_batches):
        raise ValueError(
            "seq_lens must match micro_batches length, got "
            f"{len(seq_lens)} lengths for {len(micro_batches)} micro-batches"
        )
    if (
        flex_compaction_stackable is not None
        and len(flex_compaction_stackable) != len(micro_batches)
    ):
        raise ValueError(
            "flex_compaction_stackable must match micro_batches length, got "
            f"{len(flex_compaction_stackable)} flags for "
            f"{len(micro_batches)} micro-batches"
        )

    groups: list[list[TensorMicroBatch]] = []
    pending: list[TensorMicroBatch] = []
    pending_kind: str | None = None
    pending_max_len = 0
    pending_sum_len = 0

    def flush_pending() -> None:
        nonlocal pending, pending_kind, pending_max_len, pending_sum_len
        if pending:
            groups.append(pending)
            pending = []
            pending_kind = None
            pending_max_len = 0
            pending_sum_len = 0

    def stack_kind(idx: int, micro_batch: TensorMicroBatch) -> str | None:
        if is_standard_stackable_micro_batch(
            micro_batch,
            compaction_enabled=compaction_enabled,
            cp_enabled=cp_enabled,
            lora_enabled=lora_enabled,
            multi_run_enabled=multi_run_enabled,
        ):
            return "standard"
        globally_flex_stackable = (
            True
            if flex_compaction_stackable is None
            else bool(flex_compaction_stackable[idx])
        )
        if is_flex_compaction_stackable_micro_batch(
            micro_batch,
            compaction_enabled=compaction_enabled,
            flex_compaction_enabled=(
                flex_compaction_enabled and globally_flex_stackable
            ),
            cp_enabled=cp_enabled,
            lora_enabled=lora_enabled,
            multi_run_enabled=multi_run_enabled,
        ):
            return "flex_compaction"
        return None

    for idx, (micro_batch, seq_len) in enumerate(
        zip(micro_batches, seq_lens, strict=True)
    ):
        kind = stack_kind(idx, micro_batch)
        if kind is None:
            flush_pending()
            groups.append([micro_batch])
            continue

        if pending_kind is not None and pending_kind != kind:
            flush_pending()

        seq_len = max(1, int(seq_len))
        if pending:
            next_count = len(pending) + 1
            next_max_len = max(pending_max_len, seq_len)
            next_sum_len = pending_sum_len + seq_len
            over_count = next_count > stack_size
            if kind == "flex_compaction" and flex_compaction_stack_mode == "horizontal":
                budget_tokens = next_sum_len
            else:
                budget_tokens = next_count * next_max_len
            over_budget = (
                stack_token_budget is not None
                and budget_tokens > stack_token_budget
            )
            if over_count or over_budget:
                flush_pending()

        pending.append(micro_batch)
        pending_kind = kind
        pending_max_len = max(pending_max_len, seq_len)
        pending_sum_len += seq_len
        if len(pending) >= stack_size:
            flush_pending()

    flush_pending()
    return groups


def _pad_2d(
    tensor: torch.Tensor,
    target_len: int,
    *,
    value: int | float | bool,
) -> torch.Tensor:
    if tensor.ndim != 2 or tensor.shape[0] != 1:
        raise ValueError(
            f"expected an unstacked [1, seq] tensor, got shape={tuple(tensor.shape)}"
        )
    pad_len = target_len - int(tensor.shape[1])
    if pad_len < 0:
        raise ValueError(
            f"target_len={target_len} is shorter than tensor length={tensor.shape[1]}"
        )
    if pad_len == 0:
        return tensor
    pad = torch.full(
        (1, pad_len),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=1)


def _pad_position_ids(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    if tensor.ndim != 2 or tensor.shape[0] != 1:
        raise ValueError(
            f"expected an unstacked [1, seq] tensor, got shape={tuple(tensor.shape)}"
        )
    pad_len = target_len - int(tensor.shape[1])
    if pad_len <= 0:
        return tensor
    pad = torch.arange(pad_len, dtype=tensor.dtype, device=tensor.device).unsqueeze(0)
    return torch.cat([tensor, pad], dim=1)


def stack_standard_micro_batches(
    group: list[TensorMicroBatch],
) -> TensorMicroBatch:
    """Stack standard text micro-batches into one [B, S] batch."""
    if len(group) == 0:
        raise ValueError("cannot stack an empty micro-batch group")
    if len(group) == 1:
        return group[0]

    target_len = max(int(mb["input_ids"].shape[1]) for mb in group)

    teacher_present = [mb["teacher_logprobs"] is not None for mb in group]
    if any(teacher_present) and not all(teacher_present):
        raise ValueError("cannot stack mixed teacher_logprobs presence")

    lora_num_tokens = None
    if group[0]["lora_num_tokens"] is not None:
        lora_parts = []
        for mb in group:
            counts = mb["lora_num_tokens"].clone()
            counts[-1] += target_len - int(mb["input_ids"].shape[1])
            lora_parts.append(counts)
        lora_num_tokens = torch.stack(lora_parts, dim=0).sum(dim=0)

    return TensorMicroBatch(
        input_ids=torch.cat(
            [_pad_2d(mb["input_ids"], target_len, value=1) for mb in group],
            dim=0,
        ),
        position_ids=torch.cat(
            [_pad_position_ids(mb["position_ids"], target_len) for mb in group],
            dim=0,
        ),
        advantages=torch.cat(
            [_pad_2d(mb["advantages"], target_len, value=0.0) for mb in group],
            dim=0,
        ),
        inference_logprobs=torch.cat(
            [
                _pad_2d(mb["inference_logprobs"], target_len, value=0.0)
                for mb in group
            ],
            dim=0,
        ),
        teacher_logprobs=(
            torch.cat(
                [
                    _pad_2d(mb["teacher_logprobs"], target_len, value=0.0)
                    for mb in group
                ],
                dim=0,
            )
            if all(teacher_present)
            else None
        ),
        loss_mask=torch.cat(
            [_pad_2d(mb["loss_mask"], target_len, value=False) for mb in group],
            dim=0,
        ),
        temperatures=torch.cat(
            [_pad_2d(mb["temperatures"], target_len, value=1.0) for mb in group],
            dim=0,
        ),
        lora_num_tokens=lora_num_tokens,
        routed_experts=None,
        pixel_values=None,
        image_grid_thw=None,
        compaction_events=None,
        prompt_len=None,
        calls=None,
    )


def stack_flex_compaction_micro_batches(
    group: list[TensorMicroBatch],
) -> TensorMicroBatch:
    """Stack flex-compaction micro-batches into a padded [B, S] batch."""
    if len(group) == 0:
        raise ValueError("cannot stack an empty micro-batch group")
    if len(group) == 1:
        return group[0]

    target_len = max(int(mb["input_ids"].shape[1]) for mb in group)

    teacher_present = [mb["teacher_logprobs"] is not None for mb in group]
    if any(teacher_present) and not all(teacher_present):
        raise ValueError("cannot stack mixed teacher_logprobs presence")

    calls_batch = []
    prompt_lens = []
    compaction_events_batch = []
    sequence_lengths = []
    for mb in group:
        calls = mb.get("calls")
        if not calls:
            raise ValueError("flex compaction stacking requires calls on every row")
        calls_batch.append(calls)
        prompt_lens.append(mb.get("prompt_len"))
        compaction_events_batch.append(mb.get("compaction_events") or [])
        sequence_lengths.append(int(mb["input_ids"].shape[1]))

    lora_num_tokens = None
    if group[0]["lora_num_tokens"] is not None:
        lora_parts = []
        for mb in group:
            counts = mb["lora_num_tokens"].clone()
            counts[-1] += target_len - int(mb["input_ids"].shape[1])
            lora_parts.append(counts)
        lora_num_tokens = torch.stack(lora_parts, dim=0).sum(dim=0)

    stacked = TensorMicroBatch(
        input_ids=torch.cat(
            [_pad_2d(mb["input_ids"], target_len, value=1) for mb in group],
            dim=0,
        ),
        position_ids=torch.cat(
            [_pad_position_ids(mb["position_ids"], target_len) for mb in group],
            dim=0,
        ),
        advantages=torch.cat(
            [_pad_2d(mb["advantages"], target_len, value=0.0) for mb in group],
            dim=0,
        ),
        inference_logprobs=torch.cat(
            [
                _pad_2d(mb["inference_logprobs"], target_len, value=0.0)
                for mb in group
            ],
            dim=0,
        ),
        teacher_logprobs=(
            torch.cat(
                [
                    _pad_2d(mb["teacher_logprobs"], target_len, value=0.0)
                    for mb in group
                ],
                dim=0,
            )
            if all(teacher_present)
            else None
        ),
        loss_mask=torch.cat(
            [_pad_2d(mb["loss_mask"], target_len, value=False) for mb in group],
            dim=0,
        ),
        temperatures=torch.cat(
            [_pad_2d(mb["temperatures"], target_len, value=1.0) for mb in group],
            dim=0,
        ),
        lora_num_tokens=lora_num_tokens,
        routed_experts=None,
        pixel_values=None,
        image_grid_thw=None,
        compaction_events=compaction_events_batch,
        prompt_len=prompt_lens,
        calls=calls_batch,
    )
    stacked["sequence_lengths"] = sequence_lengths
    return stacked


def pack_horizontal_flex_compaction_micro_batches(
    group: list[TensorMicroBatch],
) -> TensorMicroBatch:
    """Pack flex-compaction micro-batches into one unpadded [1, sum(S)] row."""
    if len(group) == 0:
        raise ValueError("cannot stack an empty micro-batch group")
    if len(group) == 1:
        return group[0]

    teacher_present = [mb["teacher_logprobs"] is not None for mb in group]
    if any(teacher_present) and not all(teacher_present):
        raise ValueError("cannot stack mixed teacher_logprobs presence")
    lora_num_tokens = None
    if group[0]["lora_num_tokens"] is not None:
        lora_num_tokens = torch.stack(
            [mb["lora_num_tokens"] for mb in group],
            dim=0,
        ).sum(dim=0)

    calls_batch = []
    prompt_lens = []
    compaction_events_batch = []
    sequence_lengths = []
    sequence_offsets = []
    offset = 0
    for mb in group:
        calls = mb.get("calls")
        if not calls:
            raise ValueError("flex compaction stacking requires calls on every row")
        seq_len = int(mb["input_ids"].shape[1])
        calls_batch.append(calls)
        prompt_lens.append(mb.get("prompt_len"))
        compaction_events_batch.append(mb.get("compaction_events") or [])
        sequence_lengths.append(seq_len)
        sequence_offsets.append(offset)
        offset += seq_len

    packed = TensorMicroBatch(
        input_ids=torch.cat([mb["input_ids"] for mb in group], dim=1),
        position_ids=torch.cat([mb["position_ids"] for mb in group], dim=1),
        advantages=torch.cat([mb["advantages"] for mb in group], dim=1),
        inference_logprobs=torch.cat(
            [mb["inference_logprobs"] for mb in group],
            dim=1,
        ),
        teacher_logprobs=(
            torch.cat([mb["teacher_logprobs"] for mb in group], dim=1)
            if all(teacher_present)
            else None
        ),
        loss_mask=torch.cat([mb["loss_mask"] for mb in group], dim=1),
        temperatures=torch.cat([mb["temperatures"] for mb in group], dim=1),
        lora_num_tokens=lora_num_tokens,
        routed_experts=None,
        pixel_values=None,
        image_grid_thw=None,
        compaction_events=compaction_events_batch,
        prompt_len=prompt_lens,
        calls=calls_batch,
    )
    packed["sequence_lengths"] = sequence_lengths
    packed["sequence_offsets"] = sequence_offsets
    packed["flex_stack_mode"] = "horizontal"
    return packed


def split_packed_batch(
    values: torch.Tensor,
    position_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Split [B, S] packed values using the existing position-id convention."""
    if values.ndim != 2 or position_ids.ndim != 2:
        raise ValueError(
            "split_packed_batch expects [batch, seq] tensors, got "
            f"values={tuple(values.shape)} position_ids={tuple(position_ids.shape)}"
        )
    if values.shape != position_ids.shape:
        raise ValueError(
            "values and position_ids must have identical [batch, seq] shapes, got "
            f"values={tuple(values.shape)} position_ids={tuple(position_ids.shape)}"
        )

    chunks: list[torch.Tensor] = []
    for row_values, row_positions in zip(values, position_ids, strict=True):
        chunks.extend(row_values.split(get_response_lengths(row_positions)))
    return chunks
