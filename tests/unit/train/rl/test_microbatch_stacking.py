import torch

from prime_rl.trainer.rl.microbatch_stacking import (
    compaction_metric_events_by_row,
    is_flex_compaction_stackable_micro_batch,
    make_micro_batch_groups,
    pack_horizontal_flex_compaction_micro_batches,
    split_packed_batch,
    stack_flex_compaction_micro_batches,
    stack_standard_micro_batches,
)


def _mb(length: int, *, compaction: bool = False):
    input_ids = torch.arange(length, dtype=torch.long).unsqueeze(0)
    position_ids = torch.arange(length, dtype=torch.long).unsqueeze(0)
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "advantages": torch.ones(1, length),
        "inference_logprobs": torch.zeros(1, length),
        "teacher_logprobs": None,
        "loss_mask": torch.ones(1, length, dtype=torch.bool),
        "temperatures": torch.ones(1, length),
        "lora_num_tokens": torch.tensor([length], dtype=torch.int32),
        "routed_experts": None,
        "pixel_values": None,
        "image_grid_thw": None,
        "compaction_events": [object()] if compaction else None,
        "prompt_len": 1 if compaction else None,
        "calls": None,
    }


class _Event:
    def __init__(self, num_output_tokens_at_compaction: int, tokens_evicted: int = 0):
        self.num_output_tokens_at_compaction = num_output_tokens_at_compaction
        self.tokens_evicted = tokens_evicted


class _Call:
    def __init__(self, events=None):
        self.compaction_events = events or []


def _flex_mb(length: int, *, midgen: bool = False):
    mb = _mb(length, compaction=True)
    mb["calls"] = [_Call([_Event(1)] if midgen else [])]
    return mb


def test_stack_standard_micro_batches_pads_teacher_logprobs():
    a = _mb(3)
    b = _mb(5)
    a["teacher_logprobs"] = torch.arange(3, dtype=torch.float32).unsqueeze(0)
    b["teacher_logprobs"] = torch.arange(5, dtype=torch.float32).unsqueeze(0)

    stacked = stack_standard_micro_batches([a, b])

    assert stacked["teacher_logprobs"] is not None
    assert stacked["teacher_logprobs"].tolist() == [
        [0.0, 1.0, 2.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
    ]


def test_make_micro_batch_groups_stacks_adjacent_standard_batches():
    batches = [_mb(3), _mb(4), _mb(5), _mb(6), _mb(7)]

    groups = make_micro_batch_groups(
        batches,
        stack_size=2,
        compaction_enabled=False,
        cp_enabled=False,
        lora_enabled=False,
        multi_run_enabled=False,
    )

    assert [len(g) for g in groups] == [2, 2, 1]


def test_make_micro_batch_groups_caps_stacked_token_slots():
    batches = [_mb(10), _mb(12), _mb(8), _mb(7)]

    groups = make_micro_batch_groups(
        batches,
        stack_size=16,
        stack_token_budget=32,
        compaction_enabled=False,
        cp_enabled=False,
        lora_enabled=False,
        multi_run_enabled=False,
    )

    assert [len(g) for g in groups] == [2, 2]


def test_make_micro_batch_groups_uses_global_seq_lens_for_budget():
    batches = [_mb(10), _mb(10), _mb(10)]

    groups = make_micro_batch_groups(
        batches,
        stack_size=16,
        stack_token_budget=50,
        seq_lens=[10, 20, 20],
        compaction_enabled=False,
        cp_enabled=False,
        lora_enabled=False,
        multi_run_enabled=False,
    )

    assert [len(g) for g in groups] == [2, 1]


def test_make_micro_batch_groups_token_budget_caps_full_context_rows():
    batches = [_mb(16_384) for _ in range(5)]

    groups = make_micro_batch_groups(
        batches,
        stack_size=16,
        stack_token_budget=65_536,
        compaction_enabled=False,
        cp_enabled=False,
        lora_enabled=False,
        multi_run_enabled=False,
    )

    assert [len(g) for g in groups] == [4, 1]


def test_make_micro_batch_groups_stacks_flex_compaction_batches():
    batches = [_flex_mb(10), _flex_mb(12), _flex_mb(8)]

    groups = make_micro_batch_groups(
        batches,
        stack_size=16,
        stack_token_budget=32,
        flex_compaction_enabled=True,
        compaction_enabled=True,
        cp_enabled=False,
        lora_enabled=False,
        multi_run_enabled=False,
    )

    assert [len(g) for g in groups] == [2, 1]


def test_make_micro_batch_groups_horizontal_flex_budget_uses_sum():
    batches = [_flex_mb(10), _flex_mb(12), _flex_mb(8)]

    groups = make_micro_batch_groups(
        batches,
        stack_size=16,
        stack_token_budget=32,
        flex_compaction_stack_mode="horizontal",
        seq_lens=[20, 20, 10],
        flex_compaction_enabled=True,
        compaction_enabled=True,
        cp_enabled=False,
        lora_enabled=False,
        multi_run_enabled=False,
    )

    assert [len(g) for g in groups] == [1, 2]


def test_make_micro_batch_groups_keeps_midgen_compaction_isolated():
    batches = [_flex_mb(10), _flex_mb(12, midgen=True), _flex_mb(8)]

    groups = make_micro_batch_groups(
        batches,
        stack_size=16,
        stack_token_budget=32,
        flex_compaction_enabled=True,
        compaction_enabled=True,
        cp_enabled=False,
        lora_enabled=False,
        multi_run_enabled=False,
    )

    assert [len(g) for g in groups] == [1, 1, 1]
    assert not is_flex_compaction_stackable_micro_batch(
        batches[1],
        compaction_enabled=True,
        flex_compaction_enabled=True,
        cp_enabled=False,
        lora_enabled=False,
        multi_run_enabled=False,
    )


def test_make_micro_batch_groups_uses_global_flex_stackable_mask():
    batches = [_flex_mb(10), _flex_mb(12), _flex_mb(8)]

    groups = make_micro_batch_groups(
        batches,
        stack_size=16,
        stack_token_budget=32,
        flex_compaction_enabled=True,
        flex_compaction_stackable=[True, False, True],
        compaction_enabled=True,
        cp_enabled=False,
        lora_enabled=False,
        multi_run_enabled=False,
    )

    assert [len(g) for g in groups] == [1, 1, 1]


def test_stack_flex_compaction_micro_batches_keeps_per_row_metadata():
    stacked = stack_flex_compaction_micro_batches([_flex_mb(3), _flex_mb(5)])

    assert stacked["input_ids"].shape == (2, 5)
    assert stacked["loss_mask"][0].tolist() == [True, True, True, False, False]
    assert stacked["position_ids"][0].tolist() == [0, 1, 2, 0, 1]
    assert [len(calls) for calls in stacked["calls"]] == [1, 1]
    assert stacked["prompt_len"] == [1, 1]
    assert stacked["sequence_lengths"] == [3, 5]


def test_pack_horizontal_flex_compaction_micro_batches_keeps_offsets():
    packed = pack_horizontal_flex_compaction_micro_batches(
        [_flex_mb(3), _flex_mb(5)]
    )

    assert packed["input_ids"].shape == (1, 8)
    assert packed["input_ids"][0].tolist() == [0, 1, 2, 0, 1, 2, 3, 4]
    assert packed["loss_mask"][0].tolist() == [True] * 8
    assert packed["position_ids"][0].tolist() == [0, 1, 2, 0, 1, 2, 3, 4]
    assert [len(calls) for calls in packed["calls"]] == [1, 1]
    assert packed["prompt_len"] == [1, 1]
    assert packed["sequence_lengths"] == [3, 5]
    assert packed["sequence_offsets"] == [0, 3]
    assert packed["flex_stack_mode"] == "horizontal"


def test_compaction_metric_events_by_row_keeps_stacked_rows_separate():
    row0_event = _Event(0, tokens_evicted=10)
    row1_events = [_Event(0, tokens_evicted=20), _Event(0, tokens_evicted=30)]

    rows = compaction_metric_events_by_row(
        [[], []],
        [[_Call([row0_event])], [_Call(row1_events)]],
        include_call_events=True,
    )

    assert rows == [[row0_event], row1_events]
    assert [sum(event.tokens_evicted for event in row) for row in rows] == [10, 50]


def test_compaction_metric_events_by_row_handles_singleton_outer_and_call_events():
    outer_event = _Event(0, tokens_evicted=10)
    call_event = _Event(0, tokens_evicted=20)

    rows = compaction_metric_events_by_row(
        [outer_event],
        [_Call([call_event])],
        include_call_events=True,
    )

    assert rows == [[outer_event, call_event]]


def test_make_micro_batch_groups_keeps_unstackable_batches_isolated():
    batches = [_mb(3), _mb(4, compaction=True), _mb(5), _mb(6)]

    groups = make_micro_batch_groups(
        batches,
        stack_size=2,
        compaction_enabled=False,
        cp_enabled=False,
        lora_enabled=False,
        multi_run_enabled=False,
    )

    assert [len(g) for g in groups] == [1, 1, 2]
    assert groups[1][0]["compaction_events"]


def test_make_micro_batch_groups_disables_stacking_for_cp():
    batches = [_mb(3), _mb(4)]

    groups = make_micro_batch_groups(
        batches,
        stack_size=2,
        compaction_enabled=False,
        cp_enabled=True,
        lora_enabled=False,
        multi_run_enabled=False,
    )

    assert [len(g) for g in groups] == [1, 1]


def test_stack_standard_micro_batches_pads_to_group_max_len():
    stacked = stack_standard_micro_batches([_mb(3), _mb(5)])

    assert stacked["input_ids"].shape == (2, 5)
    assert stacked["input_ids"][0].tolist() == [0, 1, 2, 1, 1]
    assert stacked["loss_mask"][0].tolist() == [True, True, True, False, False]
    assert stacked["position_ids"][0].tolist() == [0, 1, 2, 0, 1]
    assert stacked["temperatures"][0].tolist() == [1, 1, 1, 1, 1]
    assert stacked["lora_num_tokens"].tolist() == [10]


def test_split_packed_batch_handles_real_batch_dimension():
    values = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1], [0, 1, 2, 3, 0]])

    chunks = split_packed_batch(values, position_ids)

    assert [c.tolist() for c in chunks] == [
        [10, 11, 12],
        [13, 14],
        [20, 21, 22, 23, 24],
    ]
