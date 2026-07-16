import pytest

from prime_rl.trainer.batch import (
    _is_compaction_sample,
    prepare_batch,
    prepare_sample,
    requires_segmented_forward,
)
from prime_rl.transport.types import (
    CallWire,
    CompactionEventWire,
    TrainingSample,
)


@pytest.fixture
def make_training_example():
    def _make_training_example(temperature: float = 1.0) -> TrainingSample:
        return TrainingSample(
            prompt_ids=[1, 2],
            prompt_mask=[False, False],
            completion_ids=[3, 4],
            completion_mask=[True, True],
            completion_logprobs=[-0.1, -0.2],
            completion_temperatures=[temperature, temperature],  # Per-token temperatures
            teacher_logprobs=[0.0, 0.0, 0.0, 0.0],
            advantage=1.0,
        )

    return _make_training_example


def _make_prefill_trim_sample() -> TrainingSample:
    submitted = list(range(24))
    kept_indices = list(range(4)) + list(range(8, 24))
    kept_token_ids = [submitted[index] for index in kept_indices]
    completion_ids = [24, 25]
    completion_logprobs = [-0.1, -0.2]
    completion_temperatures = [0.7, 0.7]
    event = CompactionEventWire(
        num_output_tokens_at_compaction=0,
        tokens_evicted=4,
        position_offset_after=4,
        num_prompt_tokens=20,
        evict_start=4,
        new_user_fragment_len=2,
        kept_indices=kept_indices,
        kept_token_ids=kept_token_ids,
        last_turn_evicted=0,
        num_turns_evicted_after=1,
    )
    call = CallWire(
        submitted_prompt_ids=submitted,
        completion_ids=completion_ids,
        completion_logprobs=completion_logprobs,
        completion_temperatures=completion_temperatures,
        compaction_events=[event],
        compaction_replay_mode=1,
    )
    return TrainingSample(
        prompt_ids=kept_token_ids,
        prompt_mask=[False] * len(kept_token_ids),
        completion_ids=completion_ids,
        completion_mask=[True, True],
        completion_logprobs=completion_logprobs,
        completion_temperatures=completion_temperatures,
        advantage=1.0,
        calls=[call],
        compaction_replay_mode=1,
    )


def _invalid_survivor_metadata(valid, malformation):
    if malformation == "scalar-string":
        return "".join(str(value) for value in valid)
    if malformation == "scalar-bytes":
        return bytes(valid)
    if malformation == "scalar-int":
        return valid[0]
    if malformation == "integral-float":
        return [float(valid[0]), *valid[1:]]
    if malformation == "truncatable-float":
        return [valid[0] + 0.9, *valid[1:]]
    if malformation == "bool":
        return [False, *valid[1:]]
    if malformation == "negative":
        return [-1, *valid[1:]]
    if malformation == "mixed-string":
        return [*valid[:-1], str(valid[-1])]
    raise AssertionError(f"unknown malformation: {malformation}")


@pytest.mark.parametrize(
    ("rollout_count", "num_train_workers", "expected_batches_per_worker"), [(4, 2, 2), (5, 2, 3), (7, 1, 7), (11, 4, 3)]
)
def test_prepare_batch_balances_micro_batches_across_workers(
    make_training_example, rollout_count, num_train_workers, expected_batches_per_worker
):
    examples = [make_training_example() for i in range(rollout_count)]

    batches_per_gpu = prepare_batch(
        rollouts=examples,
        seq_len=4,
        num_train_workers=num_train_workers,
        idxs=[0] * rollout_count,
        num_loras=1,
    )

    assert all(len(worker_batches) == expected_batches_per_worker for worker_batches in batches_per_gpu)

    flat_batches = [batch for worker_batches in batches_per_gpu for batch in worker_batches]
    assert len(examples) <= len(flat_batches) < len(examples) + num_train_workers
    print(flat_batches)

    # Verify real rollouts have expected non-zero advantages and loss mask
    for batch in flat_batches[: len(examples)]:
        print(batch)
        assert sum(1 for advantage in batch.advantages if advantage != 0.0) == 4
        assert sum(1 for loss_mask in batch.loss_mask if loss_mask) == 2

    # Verify padded batches have zero advantages and loss mask
    for batch in flat_batches[len(examples) :]:
        assert sum(1 for advantage in batch.advantages if advantage != 0.0) == 0
        assert sum(1 for loss_mask in batch.loss_mask if loss_mask) == 0


def test_prepare_batch_packs_different_temperatures(make_training_example):
    """With per-token temperatures, samples can be packed together regardless of their temperature values."""
    example1 = make_training_example(temperature=0.7)
    example2 = make_training_example(temperature=1.1)

    batches_per_gpu = prepare_batch(
        rollouts=[example1, example2],
        seq_len=16,
        num_train_workers=1,
        idxs=[0, 0],
        num_loras=1,
    )

    flat_batches = [batch for worker_batches in batches_per_gpu for batch in worker_batches]
    # With per-token temperatures, samples can now be packed together
    assert len(flat_batches) == 1
    # Each sample has 4 tokens (2 prompt + 2 completion), so 8 total tokens
    assert len(flat_batches[0].temperatures) == 8
    # First sample (4 tokens): all get temp 0.7
    assert flat_batches[0].temperatures[:4] == [0.7, 0.7, 0.7, 0.7]
    # Second sample (4 tokens): all get temp 1.1
    assert flat_batches[0].temperatures[4:8] == [1.1, 1.1, 1.1, 1.1]


def test_prepare_sample_with_routed_experts():
    """Routed experts are passed through prepare_sample and match input_ids length."""
    # 2 prompt + 2 completion = 4 tokens, 2 layers, topk=2
    routed_experts = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[0, 2], [1, 3]], [[1, 0], [3, 2]]]
    sample = TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4],
        completion_mask=[True, True],
        completion_logprobs=[-0.1, -0.2],
        completion_temperatures=[1.0, 1.0],
        advantage=1.0,
        routed_experts=routed_experts,
    )

    micro_batch = prepare_sample(sample, seq_len=8)
    assert micro_batch.routed_experts is not None
    assert len(micro_batch.routed_experts) == 4
    assert micro_batch.routed_experts == routed_experts


def test_prepare_sample_truncates_routed_experts():
    """Routed experts are truncated to seq_len when input exceeds it."""
    routed_experts = [[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]]]
    sample = TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4],
        completion_mask=[True, True],
        completion_logprobs=[-0.1, -0.2],
        completion_temperatures=[1.0, 1.0],
        advantage=1.0,
        routed_experts=routed_experts,
    )

    micro_batch = prepare_sample(sample, seq_len=3)
    assert micro_batch.routed_experts is not None
    assert len(micro_batch.routed_experts) == 3
    assert micro_batch.routed_experts == routed_experts[:3]


def test_prepare_sample_none_routed_experts():
    """When routed_experts is None, micro_batch.routed_experts is None."""
    sample = TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4],
        completion_mask=[True, True],
        completion_logprobs=[-0.1, -0.2],
        completion_temperatures=[1.0, 1.0],
        advantage=1.0,
    )

    micro_batch = prepare_sample(sample, seq_len=8)
    assert micro_batch.routed_experts is None


def test_legacy_prepare_sample_keeps_arange_events_and_calls():
    sample = _make_prefill_trim_sample()
    sample.compaction_replay_mode = 0
    assert sample.calls is not None
    sample.calls[0].compaction_replay_mode = 0
    sample.calls[0].compaction_events[0].kept_indices = []
    sample.compaction_events = list(sample.calls[0].compaction_events)

    micro_batch = prepare_sample(
        sample,
        seq_len=64,
        compaction_enabled=True,
    )

    assert micro_batch.position_ids == list(range(22))
    assert micro_batch.compaction_events == sample.compaction_events
    assert micro_batch.calls == sample.calls
    assert micro_batch.compaction_replay_mode == 0


@pytest.mark.parametrize(
    ("sample_mode", "call_mode"),
    [(0, 1), (1, 0)],
)
def test_prepare_sample_rejects_sample_call_replay_mode_mismatch(
    sample_mode,
    call_mode,
):
    sample = _make_prefill_trim_sample()
    sample.compaction_replay_mode = sample_mode
    assert sample.calls is not None
    sample.calls[0].compaction_replay_mode = call_mode

    with pytest.raises(ValueError, match="must match its TrainingSample"):
        prepare_sample(sample, seq_len=64)


def test_prepare_sample_validates_every_call_replay_mode():
    sample = _make_prefill_trim_sample()
    sample.compaction_replay_mode = 0
    assert sample.calls is not None
    sample.calls[0].compaction_replay_mode = 0
    sample.calls.append(
        CallWire(
            submitted_prompt_ids=[],
            completion_ids=[],
            completion_logprobs=[],
            completion_temperatures=[],
            compaction_replay_mode=2,
        )
    )

    with pytest.raises(ValueError, match="unsupported compaction_replay_mode"):
        prepare_sample(sample, seq_len=64)


def test_prefill_trim_prepare_sample_uses_piecewise_positions_and_clears_replay():
    sample = _make_prefill_trim_sample()

    micro_batch = prepare_sample(sample, seq_len=8)

    assert micro_batch.input_ids == list(range(4)) + list(range(8, 26))
    assert micro_batch.position_ids == list(range(4)) + list(range(8, 26))
    assert micro_batch.position_ids[-2:] == [24, 25]
    assert micro_batch.loss_mask == [False] * 20 + [True, True]
    assert micro_batch.inference_logprobs[-2:] == [-0.1, -0.2]
    assert micro_batch.prompt_len == 20
    assert micro_batch.compaction_events is None
    assert micro_batch.calls is None
    assert micro_batch.compaction_replay_mode == 1
    assert _is_compaction_sample(micro_batch)


@pytest.mark.parametrize("field", ["kept_indices", "kept_token_ids"])
@pytest.mark.parametrize(
    "malformation",
    [
        "scalar-string",
        "scalar-bytes",
        "scalar-int",
        "integral-float",
        "truncatable-float",
        "bool",
        "negative",
        "mixed-string",
    ],
)
def test_prefill_trim_prepare_sample_rejects_invalid_survivor_metadata(
    field,
    malformation,
):
    sample = _make_prefill_trim_sample()
    assert sample.calls is not None
    event = sample.calls[0].compaction_events[0]
    valid = getattr(event, field)
    setattr(event, field, _invalid_survivor_metadata(valid, malformation))

    with pytest.raises(ValueError, match=rf"{field} must"):
        prepare_sample(sample, seq_len=64)


def test_prefill_trim_prepare_sample_accepts_tuple_survivor_metadata():
    sample = _make_prefill_trim_sample()
    assert sample.calls is not None
    event = sample.calls[0].compaction_events[0]
    event.kept_indices = tuple(event.kept_indices)
    event.kept_token_ids = tuple(event.kept_token_ids)

    micro_batch = prepare_sample(sample, seq_len=64)

    assert micro_batch.compaction_replay_mode == 1
    assert micro_batch.position_ids == list(range(4)) + list(range(8, 26))


def test_legacy_prepare_sample_keeps_permissive_survivor_metadata():
    sample = _make_prefill_trim_sample()
    sample.compaction_replay_mode = 0
    assert sample.calls is not None
    sample.calls[0].compaction_replay_mode = 0
    event = sample.calls[0].compaction_events[0]
    event.kept_indices = "legacy-indices"
    event.kept_token_ids = b"legacy-token-ids"
    sample.compaction_events = [event]

    micro_batch = prepare_sample(sample, seq_len=64, compaction_enabled=True)

    assert micro_batch.compaction_events == [event]
    assert micro_batch.calls == sample.calls
    assert micro_batch.compaction_replay_mode == 0


def test_prefill_trim_prepare_sample_allows_zero_length_survivor_prefix():
    sample = _make_prefill_trim_sample()
    assert sample.calls is not None
    call = sample.calls[0]
    event = call.compaction_events[0]
    event.evict_start = 0
    event.kept_indices = list(range(4, 24))
    event.kept_token_ids = list(range(4, 24))
    sample.prompt_ids = list(event.kept_token_ids)

    micro_batch = prepare_sample(sample, seq_len=8)

    assert micro_batch.input_ids == list(range(4, 26))
    assert micro_batch.position_ids == list(range(4, 26))
    assert micro_batch.compaction_replay_mode == 1


def test_prefill_trim_isolated_and_forces_unified_segmented_dispatch():
    mode_one = _make_prefill_trim_sample()
    ordinary = TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3],
        completion_mask=[True],
        completion_logprobs=[-0.1],
        completion_temperatures=[1.0],
        advantage=1.0,
    )

    batches = prepare_batch(
        rollouts=[mode_one, ordinary],
        seq_len=64,
        num_train_workers=1,
        idxs=[0, 0],
        num_loras=1,
        compaction_enabled=False,
    )[0]

    assert len(batches) == 2
    replay_batch = next(
        batch for batch in batches if batch.compaction_replay_mode == 1
    )
    assert replay_batch.calls is None
    assert replay_batch.compaction_events is None
    assert requires_segmented_forward(0, replay_batch.compaction_replay_mode)


def test_prefill_trim_distribution_rows_do_not_mix_replay_modes():
    mode_one = _make_prefill_trim_sample()
    ordinary = TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3],
        completion_mask=[True],
        completion_logprobs=[-0.1],
        completion_temperatures=[1.0],
        advantage=1.0,
    )

    batches = prepare_batch(
        rollouts=[mode_one, ordinary],
        seq_len=64,
        num_train_workers=2,
        idxs=[0, 0],
        num_loras=1,
        compaction_enabled=True,
    )

    assert [len(worker) for worker in batches] == [2, 2]
    for same_step in zip(*batches):
        assert len({batch.compaction_replay_mode for batch in same_step}) == 1


@pytest.mark.parametrize(
    "malformation",
    [
        "mid_generation",
        "multiple_calls",
        "prompt_mismatch",
        "kept_indices",
        "empty_kept_indices",
        "short_kept_indices",
        "out_of_range_kept_indices",
        "unexpected_kept_indices",
        "teacher_logprobs",
        "routed_experts",
        "nonfinite_logprob",
        "temperature_mismatch",
        "zero_temperature",
    ],
)
def test_prefill_trim_prepare_sample_rejects_malformed_inputs(malformation):
    sample = _make_prefill_trim_sample()
    assert sample.calls is not None
    event = sample.calls[0].compaction_events[0]
    if malformation == "mid_generation":
        event.num_output_tokens_at_compaction = 1
    elif malformation == "multiple_calls":
        sample.calls.append(sample.calls[0])
    elif malformation == "prompt_mismatch":
        sample.prompt_ids[-1] = 999
    elif malformation == "kept_indices":
        event.kept_indices[-2:] = list(reversed(event.kept_indices[-2:]))
    elif malformation == "empty_kept_indices":
        event.kept_indices = []
    elif malformation == "short_kept_indices":
        event.kept_indices.pop()
    elif malformation == "out_of_range_kept_indices":
        event.kept_indices[-1] = len(sample.calls[0].submitted_prompt_ids)
    elif malformation == "unexpected_kept_indices":
        event.kept_indices[4] = 7
    elif malformation == "teacher_logprobs":
        sample.teacher_logprobs = [0.0] * 22
    elif malformation == "routed_experts":
        sample.routed_experts = [[[0]]] * 22
    elif malformation == "nonfinite_logprob":
        sample.completion_logprobs[0] = float("nan")
        sample.calls[0].completion_logprobs[0] = float("nan")
    elif malformation == "temperature_mismatch":
        sample.completion_temperatures.pop()
    elif malformation == "zero_temperature":
        sample.completion_temperatures[:] = [0.0, 0.0]
        sample.calls[0].completion_temperatures[:] = [0.0, 0.0]

    with pytest.raises(ValueError, match="prefill_trim"):
        prepare_sample(sample, seq_len=64)
