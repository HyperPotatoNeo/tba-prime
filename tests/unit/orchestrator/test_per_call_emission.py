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

from prime_rl.orchestrator.trajectories import interleave_rollout


def _make_step(
    *,
    prompt_ids,
    completion_ids,
    extras=None,
):
    """Build a TrajectoryStep with minimal scaffolding."""
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
        extras=extras if extras is not None else {},
    )


def _make_output(steps):
    return vf.RolloutOutput(
        example_id=0,
        trajectory=list(steps),
        sampling_args={"temperature": 1.0},
        error=None,
    )


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
