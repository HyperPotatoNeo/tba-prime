"""Phase B tests: per-call breakdown emission in interleave_rollout.

After Phase B, every TrainingSample carries a `calls: list[CallWire]`
populated from the trajectory's merged steps. Each call captures the
PRE-trim submitted prompt (what vLLM received) + the FULL compaction
event list (admission + mid-gen) for that vLLM chat() call.

Covers:
  1. Single-call rollout (no merging) → 1 call.
  2. Multi-call rollout where extension property holds → N calls in
     one sample.
  3. Rollout with extension-break → multiple samples, each with its
     own call list whose lengths sum to N.
  4. Per-call submitted_prompt_ids is the PRE-trim form (when
     admission events trim the prompt in prepare_step_tokens, the
     CallWire still carries the original tokens).
  5. Per-call compaction_events list contains BOTH admission and
     mid-gen events (the post-trim event writeback to step extras
     only carries mid-gen, but CallWire captures both).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import verifiers as vf

from prime_rl.orchestrator.trajectories import (
    interleave_rollout,
    pretokenize_rollout_trajectory,
)
from prime_rl.transport.types import (
    TurnCompactionStateWire,
    compute_turn_compaction_state_id,
)


def _make_step(
    *,
    prompt_ids,
    completion_ids,
    extras=None,
):
    """Build a TrajectoryStep with minimal scaffolding."""
    extras = dict(extras or {})
    if (
        extras.get("compaction_replay_mode") == "prefill_trim"
        and "submitted_prompt_token_ids" not in extras
    ):
        extras["submitted_prompt_token_ids"] = list(prompt_ids)
    return vf.TrajectoryStep(
        prompt=[{"role": "user", "content": "U"}],
        completion=[{"role": "assistant", "content": "A"}],
        response=MagicMock(),
        tokens=vf.TrajectoryStepTokens(
            prompt_ids=list(prompt_ids),
            prompt_mask=[0] * len(prompt_ids),
            completion_ids=list(completion_ids),
            completion_mask=[1] * len(completion_ids),
            completion_logprobs=[-0.1] * len(completion_ids),
            overlong_prompt=False,
            is_truncated=False,
        ),
        reward=None,
        advantage=None,
        is_truncated=False,
        trajectory_id="1",
        extras=extras,
    )


def _make_output(steps, *, temperature=1.0):
    return vf.RolloutOutput(
        example_id=0,
        trajectory=list(steps),
        sampling_args={"temperature": temperature},
        error=None,
    )


def _prefill_trim_event(prompt_ids, evict_start=2, tokens_evicted=2, **overrides):
    evict_end = evict_start + tokens_evicted
    kept_indices = list(range(evict_start)) + list(
        range(evict_end, len(prompt_ids))
    )
    event = {
        "num_output_tokens_at_compaction": 0,
        "tokens_evicted": tokens_evicted,
        "position_offset_after": tokens_evicted,
        "num_prompt_tokens": len(kept_indices),
        "evict_start": evict_start,
        "new_user_fragment_len": 1,
        "kept_indices": kept_indices,
        "kept_token_ids": [prompt_ids[index] for index in kept_indices],
        "last_turn_evicted": 0,
        "num_turns_evicted_after": 1,
    }
    event.update(overrides)
    return event


def _turn_compaction_state(
    prompt_ids,
    *,
    position_offset=8,
    protected_prefix_len=2,
    num_turns_evicted=2,
    carried_prefix_num_live_turns=1,
    carried_prefix_len=None,
):
    if carried_prefix_len is None:
        carried_prefix_len = len(prompt_ids)
    state = {
        "version": 1,
        "position_offset": position_offset,
        "protected_prefix_len": protected_prefix_len,
        "num_turns_evicted": num_turns_evicted,
        "carried_prefix_num_live_turns": carried_prefix_num_live_turns,
        "carried_prefix_len": carried_prefix_len,
    }
    state["state_id"] = compute_turn_compaction_state_id(
        **state,
        carried_prefix_token_ids=prompt_ids[:carried_prefix_len],
    )
    return state


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


def test_single_step_emits_one_call():
    """A single-step rollout has exactly one chat() call → one CallWire."""
    out = _make_output(
        [_make_step(prompt_ids=[1, 2], completion_ids=[3, 4])]
    )
    rollouts = interleave_rollout(out)
    assert rollouts is not None and len(rollouts) == 1
    sample = rollouts[0]
    assert sample.calls is not None
    assert len(sample.calls) == 1
    call = sample.calls[0]
    # No extras["prompt_token_ids"] -> submitted_prompt_ids_pre_trim
    # falls back to the post-prepare prompt_ids ([1, 2]).
    assert call.submitted_prompt_ids == [1, 2]
    assert call.completion_ids == [3, 4]
    assert call.completion_logprobs == [-0.1, -0.1]
    assert call.completion_temperatures == [1.0, 1.0]
    assert call.compaction_events == []


def test_two_step_extending_emits_two_calls_in_one_sample():
    """Two steps where step 2's prompt extends step 1's full sequence
    merge into ONE sample with TWO calls."""
    out = _make_output([
        _make_step(prompt_ids=[1, 2], completion_ids=[3, 4]),
        # Step 2 prefix = step 1's [prompt + completion] = [1, 2, 3, 4]
        _make_step(prompt_ids=[1, 2, 3, 4, 5, 6], completion_ids=[7, 8]),
    ])
    rollouts = interleave_rollout(out)
    assert rollouts is not None and len(rollouts) == 1
    sample = rollouts[0]
    assert sample.calls is not None
    assert len(sample.calls) == 2

    # Call 0: step 1's submitted prompt + completion
    assert sample.calls[0].submitted_prompt_ids == [1, 2]
    assert sample.calls[0].completion_ids == [3, 4]

    # Call 1: step 2's submitted prompt + completion
    assert sample.calls[1].submitted_prompt_ids == [1, 2, 3, 4, 5, 6]
    assert sample.calls[1].completion_ids == [7, 8]


def test_extension_break_emits_separate_samples_each_with_own_calls():
    """When the extension property breaks, each new sample gets its own
    independent call list."""
    out = _make_output([
        _make_step(prompt_ids=[1, 2], completion_ids=[3, 4]),
        # Step 2 prefix [1, 2, 3, 4] = step 1 full -> extends sample 0
        _make_step(prompt_ids=[1, 2, 3, 4, 5], completion_ids=[6, 7]),
        # Step 3 starts fresh (no prefix match) -> new sample
        _make_step(prompt_ids=[99, 98], completion_ids=[97, 96]),
    ])
    rollouts = interleave_rollout(out)
    assert rollouts is not None and len(rollouts) == 2

    s0, s1 = rollouts
    assert s0.calls is not None and len(s0.calls) == 2
    assert s1.calls is not None and len(s1.calls) == 1

    assert s0.calls[0].submitted_prompt_ids == [1, 2]
    assert s0.calls[1].submitted_prompt_ids == [1, 2, 3, 4, 5]
    assert s1.calls[0].submitted_prompt_ids == [99, 98]


def test_compaction_events_pre_trim_carried_on_callwire():
    """When a step's extras has compaction events that trim the prompt,
    the CallWire's compaction_events list still carries the FULL set
    (admission + mid-gen), and submitted_prompt_ids is the PRE-trim form.
    """
    # Step 1 has padded prompt_token_ids = [1, 2, 9, 9, 3, 4] and an
    # admission event that evicts indices [2, 4) = [9, 9]. Post-trim
    # prompt is [1, 2, 3, 4]. CallWire should carry the PRE-trim
    # [1, 2, 9, 9, 3, 4] and the original event.
    pre_trim_padded = [1, 2, 9, 9, 3, 4]
    admission_event = {
        "num_output_tokens_at_compaction": 0,
        "tokens_evicted": 2,
        "position_offset_after": 2,
        "num_prompt_tokens": 6,
        "evict_start": 2,
        # Phase A fields default to safe values; orchestrator-side
        # _compaction_events_from_step parses them as defaults.
    }
    step = _make_step(
        prompt_ids=pre_trim_padded,
        completion_ids=[10, 11],
        extras={
            "prompt_token_ids": pre_trim_padded,
            "compaction_events": [admission_event],
        },
    )
    out = _make_output([step])
    rollouts = interleave_rollout(out)
    assert rollouts is not None and len(rollouts) == 1
    sample = rollouts[0]

    # The TrainingSample's prompt_ids is the POST-trim version.
    assert sample.prompt_ids == [1, 2, 3, 4]

    # The CallWire keeps the PRE-trim submitted prompt.
    assert sample.calls is not None and len(sample.calls) == 1
    call = sample.calls[0]
    assert call.submitted_prompt_ids == [1, 2, 9, 9, 3, 4]

    # The CallWire's event list carries the admission event (the
    # event was consumed by _apply_admission_trim for the trainer's
    # main path, but we capture it on the CallWire for Phase D).
    assert len(call.compaction_events) == 1
    ev = call.compaction_events[0]
    assert ev.num_output_tokens_at_compaction == 0
    assert ev.tokens_evicted == 2
    assert ev.evict_start == 2


def test_non_compaction_rollout_still_has_calls_list():
    """Without any compaction extras, every step still emits a CallWire
    with empty compaction_events. Lets the trainer dispatch through the
    per-call path uniformly."""
    out = _make_output([
        _make_step(prompt_ids=[1, 2], completion_ids=[3, 4]),
        _make_step(prompt_ids=[1, 2, 3, 4, 5], completion_ids=[6, 7]),
    ])
    rollouts = interleave_rollout(out)
    assert rollouts is not None and len(rollouts) == 1
    sample = rollouts[0]
    assert sample.calls is not None and len(sample.calls) == 2
    for call in sample.calls:
        assert call.compaction_events == []


def test_prefill_trim_uses_authoritative_survivors_and_mode_one_call():
    submitted = list(range(24))
    event = _prefill_trim_event(
        submitted,
        evict_start=4,
        tokens_evicted=4,
    )
    step = _make_step(
        prompt_ids=submitted,
        completion_ids=[24, 25],
        extras={
            "compaction_events": [event],
            "compaction_replay_mode": "prefill_trim",
        },
    )

    rollouts = interleave_rollout(_make_output([step]))

    assert rollouts is not None and len(rollouts) == 1
    sample = rollouts[0]
    assert sample.prompt_ids == event["kept_token_ids"]
    assert sample.completion_ids == [24, 25]
    assert sample.completion_logprobs == [-0.1, -0.1]
    assert sample.completion_mask == [True, True]
    assert sample.compaction_events is None
    assert sample.compaction_replay_mode == 1
    assert sample.calls is not None and len(sample.calls) == 1
    assert sample.completion_temperatures == [1.0, 1.0]
    assert sample.calls[0].submitted_prompt_ids == submitted
    assert sample.calls[0].completion_temperatures == [1.0, 1.0]
    assert sample.calls[0].compaction_replay_mode == 1
    assert sample.calls[0].compaction_events[0].kept_token_ids == sample.prompt_ids


def test_prefill_trim_state_only_emits_one_state_aware_call():
    prompt_ids = list(range(10))
    state = _turn_compaction_state(prompt_ids, carried_prefix_len=8)
    step = _make_step(
        prompt_ids=prompt_ids,
        completion_ids=[10],
        extras={
            "compaction_events": [],
            "compaction_replay_mode": "prefill_trim",
            "turn_compaction_state": state,
        },
    )

    samples = interleave_rollout(_make_output([step]))

    assert samples is not None and len(samples) == 1
    sample = samples[0]
    assert sample.prompt_ids == prompt_ids
    assert sample.calls is not None and len(sample.calls) == 1
    call = sample.calls[0]
    assert call.compaction_events == []
    assert isinstance(call.turn_compaction_state, TurnCompactionStateWire)
    assert call.turn_compaction_state.position_offset == 8


def test_prefill_trim_cumulative_event_emits_state_and_delta_event():
    submitted = list(range(12))
    event = _prefill_trim_event(
        submitted,
        evict_start=2,
        tokens_evicted=4,
        position_offset_after=8,
        last_turn_evicted=1,
        num_turns_evicted_after=2,
    )
    kept = event["kept_token_ids"]
    state = _turn_compaction_state(kept)
    step = _make_step(
        prompt_ids=submitted,
        completion_ids=[12],
        extras={
            "compaction_events": [event],
            "compaction_replay_mode": "prefill_trim",
            "turn_compaction_state": state,
        },
    )

    samples = interleave_rollout(_make_output([step]))

    assert samples is not None and len(samples) == 1
    sample = samples[0]
    assert sample.prompt_ids == kept
    assert sample.calls is not None and len(sample.calls) == 1
    call = sample.calls[0]
    assert call.turn_compaction_state is not None
    assert call.turn_compaction_state.position_offset == 8
    assert len(call.compaction_events) == 1
    assert call.compaction_events[0].tokens_evicted == 4
    assert call.compaction_events[0].position_offset_after == 8


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
def test_prefill_trim_sample_rejects_invalid_survivor_metadata(
    field,
    malformation,
):
    submitted = list(range(10))
    event = _prefill_trim_event(submitted)
    event[field] = _invalid_survivor_metadata(event[field], malformation)
    step = _make_step(
        prompt_ids=submitted,
        completion_ids=[10],
        extras={
            "compaction_events": [event],
            "compaction_replay_mode": "prefill_trim",
        },
    )

    with pytest.raises(ValueError, match=rf"{field} must"):
        interleave_rollout(_make_output([step]))


def test_prefill_trim_sample_accepts_tuple_survivor_metadata():
    submitted = list(range(10))
    event = _prefill_trim_event(submitted)
    event["kept_indices"] = tuple(event["kept_indices"])
    event["kept_token_ids"] = tuple(event["kept_token_ids"])
    step = _make_step(
        prompt_ids=submitted,
        completion_ids=[10],
        extras={
            "compaction_events": [event],
            "compaction_replay_mode": "prefill_trim",
        },
    )

    samples = interleave_rollout(_make_output([step]))

    assert samples is not None
    assert samples[0].prompt_ids == [0, 1, 4, 5, 6, 7, 8, 9]


def test_legacy_sample_keeps_permissive_survivor_coercion():
    submitted = list(range(10))
    event = _prefill_trim_event(submitted)
    event["kept_indices"] = [0.0, "1", False]
    event["kept_token_ids"] = [0.0, "1", True]
    step = _make_step(
        prompt_ids=submitted,
        completion_ids=[10],
        extras={"compaction_events": [event]},
    )

    samples = interleave_rollout(_make_output([step]))

    assert samples is not None
    assert samples[0].calls is not None
    converted = samples[0].calls[0].compaction_events[0]
    assert converted.kept_indices == [0, 1, 0]
    assert converted.kept_token_ids == [0, 1, 1]
    assert samples[0].compaction_replay_mode == 0


def test_prefill_trim_greedy_temperature_uses_effective_one():
    from prime_rl.trainer.batch import prepare_sample

    submitted = list(range(10))
    step = _make_step(
        prompt_ids=submitted,
        completion_ids=[10],
        extras={
            "compaction_events": [_prefill_trim_event(submitted)],
            "compaction_replay_mode": "prefill_trim",
        },
    )

    samples = interleave_rollout(_make_output([step], temperature=0.0))

    assert samples is not None and len(samples) == 1
    sample = samples[0]
    assert sample.completion_temperatures == [1.0]
    assert sample.calls is not None
    assert sample.calls[0].completion_temperatures == [1.0]
    assert prepare_sample(sample, seq_len=32).temperatures == [1.0] * 9


@pytest.mark.parametrize(
    "temperature",
    [-0.1, float("nan"), float("inf"), float("-inf")],
)
def test_prefill_trim_rejects_invalid_training_temperature(temperature):
    submitted = list(range(10))
    step = _make_step(
        prompt_ids=submitted,
        completion_ids=[10],
        extras={
            "compaction_events": [_prefill_trim_event(submitted)],
            "compaction_replay_mode": "prefill_trim",
        },
    )

    with pytest.raises(ValueError, match="training temperature"):
        interleave_rollout(_make_output([step], temperature=temperature))


def test_legacy_rollout_preserves_zero_temperature():
    samples = interleave_rollout(
        _make_output(
            [_make_step(prompt_ids=[1, 2], completion_ids=[3])],
            temperature=0.0,
        )
    )

    assert samples is not None
    assert samples[0].completion_temperatures == [0.0]
    assert samples[0].calls is not None
    assert samples[0].calls[0].completion_temperatures == [0.0]


def test_prefill_trim_steps_never_merge_even_when_prefix_extends():
    first_prompt = list(range(10))
    first = _make_step(
        prompt_ids=first_prompt,
        completion_ids=[10],
        extras={
            "compaction_events": [_prefill_trim_event(first_prompt)],
            "compaction_replay_mode": "prefill_trim",
        },
    )
    first_post_trim = [0, 1, 4, 5, 6, 7, 8, 9, 10]
    second_prompt = first_post_trim + [11, 12]
    second = _make_step(
        prompt_ids=second_prompt,
        completion_ids=[13],
        extras={
            "compaction_events": [_prefill_trim_event(second_prompt)],
            "compaction_replay_mode": "prefill_trim",
        },
    )

    rollouts = interleave_rollout(_make_output([first, second]))

    assert rollouts is not None and len(rollouts) == 2
    assert all(sample.compaction_replay_mode == 1 for sample in rollouts)
    assert all(sample.calls is not None and len(sample.calls) == 1 for sample in rollouts)


@pytest.mark.parametrize(
    "events",
    [
        [],
        [
            _prefill_trim_event(
                list(range(10)),
                num_output_tokens_at_compaction=1,
            )
        ],
        [
            _prefill_trim_event(list(range(10))),
            _prefill_trim_event(list(range(10))),
        ],
        [
            _prefill_trim_event(
                list(range(10)),
                kept_token_ids=[999] * 8,
            )
        ],
        [{key: value for key, value in _prefill_trim_event(list(range(10))).items() if key != "kept_indices"}],
        [
            _prefill_trim_event(
                list(range(10)),
                kept_indices=[],
            )
        ],
    ],
)
def test_prefill_trim_rejects_malformed_events(events):
    step = _make_step(
        prompt_ids=list(range(10)),
        completion_ids=[10],
        extras={
            "compaction_events": events,
            "compaction_replay_mode": "prefill_trim",
        },
    )

    with pytest.raises(ValueError, match="prefill_trim"):
        interleave_rollout(_make_output([step]))


def test_prefill_trim_rejects_submitted_native_prompt_mismatch():
    prompt_ids = list(range(10))
    step = _make_step(
        prompt_ids=prompt_ids,
        completion_ids=[10],
        extras={
            "compaction_events": [_prefill_trim_event(prompt_ids)],
            "compaction_replay_mode": "prefill_trim",
            "submitted_prompt_token_ids": [999, *prompt_ids[1:]],
        },
    )

    with pytest.raises(ValueError, match="submitted/native prompt mismatch"):
        interleave_rollout(_make_output([step]))


def test_prefill_trim_rejects_missing_native_tokens_before_retokenization():
    prompt_ids = list(range(10))
    step = _make_step(
        prompt_ids=prompt_ids,
        completion_ids=[10],
        extras={
            "compaction_events": [_prefill_trim_event(prompt_ids)],
            "compaction_replay_mode": "prefill_trim",
        },
    )
    step["tokens"] = None
    output = _make_output([step])

    with pytest.raises(ValueError, match="cannot fall back to retokenization"):
        pretokenize_rollout_trajectory(output, tokenizer=object())
    with pytest.raises(ValueError, match="requires native trajectory tokens"):
        interleave_rollout(output)


def test_prefill_trim_rejects_missing_completion_token_ids():
    prompt_ids = list(range(10))
    step = _make_step(
        prompt_ids=prompt_ids,
        completion_ids=[10],
        extras={
            "compaction_events": [_prefill_trim_event(prompt_ids)],
            "compaction_replay_mode": "prefill_trim",
        },
    )
    del step["tokens"]["completion_ids"]

    with pytest.raises(ValueError, match="completion metadata"):
        interleave_rollout(_make_output([step]))


@pytest.mark.parametrize("logprobs", [[-0.1, -0.2], [float("nan")]])
def test_prefill_trim_rejects_misaligned_or_nonfinite_logprobs(logprobs):
    prompt_ids = list(range(10))
    step = _make_step(
        prompt_ids=prompt_ids,
        completion_ids=[10],
        extras={
            "compaction_events": [_prefill_trim_event(prompt_ids)],
            "compaction_replay_mode": "prefill_trim",
        },
    )
    step["tokens"]["completion_logprobs"] = logprobs

    with pytest.raises(ValueError, match="completion"):
        interleave_rollout(_make_output([step]))


@pytest.mark.parametrize(
    "history",
    [
        "valid-plus-none",
        "valid-plus-malformed-dict",
        "valid-plus-malformed-list",
        "valid-plus-malformed-object",
        "two-valid",
        "sole-malformed",
    ],
)
def test_prefill_trim_rejects_invalid_raw_event_history(history):
    prompt_ids = list(range(10))
    valid = _prefill_trim_event(prompt_ids)
    malformed = {
        "valid-plus-none": None,
        "valid-plus-malformed-dict": {"tokens_evicted": "bad"},
        "valid-plus-malformed-list": [0, "bad", 0],
        "valid-plus-malformed-object": object(),
    }
    if history == "two-valid":
        events = [valid, dict(valid)]
    elif history == "sole-malformed":
        events = [{"tokens_evicted": "bad"}]
    else:
        events = [valid, malformed[history]]
    step = _make_step(
        prompt_ids=prompt_ids,
        completion_ids=[10],
        extras={
            "compaction_events": events,
            "compaction_replay_mode": "prefill_trim",
        },
    )
    error = (
        "exactly one valid compaction event"
        if history == "sole-malformed"
        else "exactly one raw compaction event"
    )

    with pytest.raises(ValueError, match=error):
        interleave_rollout(_make_output([step]))


def test_legacy_replay_keeps_permissive_unrecognized_event_filtering():
    prompt_ids = list(range(10))
    valid = _prefill_trim_event(prompt_ids)
    valid.pop("kept_indices")
    step = _make_step(
        prompt_ids=prompt_ids,
        completion_ids=[10],
        extras={"compaction_events": [valid, None, object()]},
    )

    rollouts = interleave_rollout(_make_output([step]))

    assert rollouts is not None and len(rollouts) == 1
    sample = rollouts[0]
    assert sample.compaction_replay_mode == 0
    assert sample.calls is not None
    assert sample.calls[0].compaction_events[0].kept_token_ids == valid["kept_token_ids"]
    assert sample.calls[0].compaction_events[0].kept_indices == []
    assert len(sample.calls[0].compaction_events) == 1
