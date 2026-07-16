"""Trajectory emission tests for Markovian Summary integration.

Covers :func:`prime_rl.orchestrator.trajectories._build_summary_sample`
and the extension to :func:`interleave_rollout` that emits one extra
:class:`TrainingSample` per trajectory step carrying a
``summary_trainsample`` payload on its extras dict.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import verifiers as vf

from prime_rl.orchestrator.trajectories import (
    _build_summary_sample,
    interleave_rollout,
)
from prime_rl.transport.types import compute_turn_compaction_state_id


def _step(
    *,
    prompt_ids,
    completion_ids,
    completion_mask=None,
    logprobs=None,
    extras=None,
):
    return vf.TrajectoryStep(
        prompt=[{"role": "user", "content": "U"}],
        completion=[{"role": "assistant", "content": "A"}],
        response=MagicMock(),
        tokens=vf.TrajectoryStepTokens(
            prompt_ids=list(prompt_ids),
            prompt_mask=[0] * len(prompt_ids),
            completion_ids=list(completion_ids),
            completion_mask=(
                completion_mask
                if completion_mask is not None
                else [1] * len(completion_ids)
            ),
            completion_logprobs=(
                logprobs
                if logprobs is not None
                else [-0.1] * len(completion_ids)
            ),
            overlong_prompt=False,
            is_truncated=False,
        ),
        reward=None,
        advantage=None,
        is_truncated=False,
        trajectory_id="1",
        extras=extras or {},
    )


def _rollout(steps):
    return vf.RolloutOutput(
        example_id=0,
        trajectory=list(steps),
        sampling_args={"temperature": 1.0},
        error=None,
    )


def _summary_payload(
    *,
    prompt_ids,
    completion_ids,
    logprobs=None,
    model="test-model",
):
    return {
        "prompt_token_ids": list(prompt_ids),
        "completion_token_ids": list(completion_ids),
        "completion_logprobs": (
            list(logprobs)
            if logprobs is not None
            else [-0.1] * len(completion_ids)
        ),
        "model": model,
    }


def _prefill_trim_summary_payload():
    submitted = [10, 11, 12, 13, 14, 15]
    kept = [10, 11, 14, 15]
    payload = _summary_payload(
        prompt_ids=kept,
        completion_ids=[20, 21],
        logprobs=[-0.1, -0.2],
    )
    payload.update(
        {
            "compaction_replay_mode": "prefill_trim",
            "submitted_prompt_token_ids": submitted,
            "native_prompt_token_ids": submitted,
            "completion_temperature": 0.25,
            "compaction_events": [
                {
                    "num_output_tokens_at_compaction": 0,
                    "tokens_evicted": 2,
                    "position_offset_after": 2,
                    "num_prompt_tokens": len(kept),
                    "evict_start": 2,
                    "new_user_fragment_len": 1,
                    "kept_indices": [0, 1, 4, 5],
                    "kept_token_ids": kept,
                    "last_turn_evicted": 0,
                    "num_turns_evicted_after": 1,
                }
            ],
        }
    )
    return payload


# ─── _build_summary_sample (pure helper) ───


def test_build_summary_sample_happy_path():
    step = _step(
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        extras={
            "summary_trainsample": _summary_payload(
                prompt_ids=[10, 11, 12],
                completion_ids=[20, 21],
                logprobs=[-0.1, -0.2],
            )
        },
    )
    sample = _build_summary_sample(step, temperature=0.7, has_error=False)
    assert sample is not None
    assert sample.prompt_ids == [10, 11, 12]
    assert sample.prompt_mask == [False, False, False]
    assert sample.completion_ids == [20, 21]
    assert sample.completion_mask == [True, True]
    assert sample.completion_logprobs == [-0.1, -0.2]
    assert sample.completion_temperatures == [0.7, 0.7]
    assert sample.advantage is None
    assert sample.compaction_events is None
    assert sample.routed_experts is None
    assert sample.calls is None
    assert sample.compaction_replay_mode == 0


def test_build_summary_sample_prefill_trim_uses_authoritative_survivors():
    from prime_rl.trainer.batch import prepare_sample
    from prime_rl.transport.types import COMPACTION_REPLAY_MODE_PREFILL_TRIM

    payload = _prefill_trim_summary_payload()
    step = _step(
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        extras={"summary_trainsample": payload},
    )

    sample = _build_summary_sample(step, temperature=0.7, has_error=False)

    assert sample is not None
    assert sample.compaction_replay_mode == COMPACTION_REPLAY_MODE_PREFILL_TRIM
    assert sample.prompt_ids == [10, 11, 14, 15]
    assert sample.compaction_events is None
    assert sample.calls is not None
    assert len(sample.calls) == 1
    call = sample.calls[0]
    assert call.compaction_replay_mode == COMPACTION_REPLAY_MODE_PREFILL_TRIM
    assert call.submitted_prompt_ids == [10, 11, 12, 13, 14, 15]
    assert call.completion_ids == sample.completion_ids == [20, 21]
    assert call.completion_logprobs == sample.completion_logprobs
    assert call.completion_temperatures == sample.completion_temperatures == [
        0.25,
        0.25,
    ]
    assert len(call.compaction_events) == 1
    assert call.compaction_events[0].kept_token_ids == sample.prompt_ids

    micro_batch = prepare_sample(sample, seq_len=32)
    assert micro_batch.position_ids == [0, 1, 4, 5, 6, 7]
    assert micro_batch.compaction_events is None
    assert micro_batch.calls is None
    assert (
        micro_batch.compaction_replay_mode
        == COMPACTION_REPLAY_MODE_PREFILL_TRIM
    )


def test_build_summary_sample_prefill_trim_supports_state_only_replay():
    from prime_rl.trainer.batch import prepare_sample

    prompt_ids = [10, 11, 12, 13, 14, 15]
    carried_prefix_len = 4
    state = {
        "version": 1,
        "position_offset": 8,
        "protected_prefix_len": 2,
        "num_turns_evicted": 2,
        "carried_prefix_num_live_turns": 1,
        "carried_prefix_len": carried_prefix_len,
    }
    state["state_id"] = compute_turn_compaction_state_id(
        **state,
        carried_prefix_token_ids=prompt_ids[:carried_prefix_len],
    )
    payload = _summary_payload(
        prompt_ids=prompt_ids,
        completion_ids=[20, 21],
        logprobs=[-0.1, -0.2],
    )
    payload.update(
        {
            "compaction_replay_mode": "prefill_trim",
            "submitted_prompt_token_ids": prompt_ids,
            "native_prompt_token_ids": prompt_ids,
            "completion_temperature": 0.25,
            "compaction_events": [],
            "turn_compaction_state": state,
        }
    )
    step = _step(
        prompt_ids=[1],
        completion_ids=[2],
        extras={"summary_trainsample": payload},
    )

    sample = _build_summary_sample(step, temperature=0.7, has_error=False)

    assert sample is not None
    assert sample.prompt_ids == prompt_ids
    assert sample.calls is not None and len(sample.calls) == 1
    call = sample.calls[0]
    assert call.compaction_events == []
    assert call.turn_compaction_state is not None
    assert call.turn_compaction_state.position_offset == 8
    assert prepare_sample(sample, seq_len=32).position_ids == [
        0,
        1,
        10,
        11,
        12,
        13,
        14,
        15,
    ]


def test_build_summary_sample_prefill_trim_supports_cumulative_event_state():
    from prime_rl.trainer.batch import prepare_sample

    payload = _prefill_trim_summary_payload()
    event = payload["compaction_events"][0]
    event["position_offset_after"] = 4
    event["last_turn_evicted"] = 1
    event["num_turns_evicted_after"] = 2
    kept = payload["prompt_token_ids"]
    state = {
        "version": 1,
        "position_offset": 4,
        "protected_prefix_len": 2,
        "num_turns_evicted": 2,
        "carried_prefix_num_live_turns": 1,
        "carried_prefix_len": len(kept),
    }
    state["state_id"] = compute_turn_compaction_state_id(
        **state,
        carried_prefix_token_ids=kept,
    )
    payload["turn_compaction_state"] = state
    step = _step(
        prompt_ids=[1],
        completion_ids=[2],
        extras={"summary_trainsample": payload},
    )

    sample = _build_summary_sample(step, temperature=0.7, has_error=False)

    assert sample is not None
    assert sample.calls is not None
    call = sample.calls[0]
    assert call.compaction_events[0].tokens_evicted == 2
    assert call.compaction_events[0].position_offset_after == 4
    assert call.turn_compaction_state is not None
    assert prepare_sample(sample, seq_len=32).position_ids == [0, 1, 6, 7, 8, 9]


@pytest.mark.parametrize(
    "malformation",
    [
        "wrong_survivors",
        "wrong_indices",
        "missing_submitted_prompt",
        "native_prompt_mismatch",
        "missing_temperature",
    ],
)
def test_build_summary_sample_prefill_trim_rejects_malformed_survivors(
    malformation,
):
    payload = _prefill_trim_summary_payload()
    if malformation == "wrong_survivors":
        payload["prompt_token_ids"][-1] = 99
    elif malformation == "wrong_indices":
        payload["compaction_events"][0]["kept_indices"] = [0, 1, 3, 5]
    elif malformation == "missing_submitted_prompt":
        payload["submitted_prompt_token_ids"] = []
    elif malformation == "native_prompt_mismatch":
        payload["native_prompt_token_ids"][-1] = 99
    else:
        payload.pop("completion_temperature")
    step = _step(
        prompt_ids=[1],
        completion_ids=[2],
        extras={"summary_trainsample": payload},
    )

    with pytest.raises(ValueError, match="prefill_trim summary replay"):
        _build_summary_sample(step, temperature=1.0, has_error=False)


def test_build_summary_sample_uses_payload_temperature_with_legacy_fallback():
    payload = _summary_payload(
        prompt_ids=[10],
        completion_ids=[20, 21],
    )
    payload["completion_temperature"] = 0.2
    step = _step(
        prompt_ids=[1],
        completion_ids=[2],
        extras={"summary_trainsample": payload},
    )

    sample = _build_summary_sample(step, temperature=0.9, has_error=False)
    assert sample is not None
    assert sample.completion_temperatures == [0.2, 0.2]

    payload.pop("completion_temperature")
    legacy = _build_summary_sample(step, temperature=0.9, has_error=False)
    assert legacy is not None
    assert legacy.completion_temperatures == [0.9, 0.9]


def test_prefill_trim_summary_greedy_temperature_is_effective_everywhere():
    from prime_rl.trainer.batch import prepare_sample

    payload = _prefill_trim_summary_payload()
    payload["completion_temperature"] = 0.0
    step = _step(
        prompt_ids=[1],
        completion_ids=[2],
        extras={"summary_trainsample": payload},
    )

    sample = _build_summary_sample(step, temperature=0.7, has_error=False)

    assert sample is not None
    assert sample.completion_temperatures == [1.0, 1.0]
    assert sample.calls is not None
    assert sample.calls[0].completion_temperatures == [1.0, 1.0]
    assert prepare_sample(sample, seq_len=32).temperatures == [1.0] * 6


@pytest.mark.parametrize(
    "temperature",
    [-0.1, float("nan"), float("inf"), float("-inf")],
)
def test_summary_sample_rejects_invalid_training_temperature(temperature):
    payload = _prefill_trim_summary_payload()
    payload["completion_temperature"] = temperature
    step = _step(
        prompt_ids=[1],
        completion_ids=[2],
        extras={"summary_trainsample": payload},
    )

    with pytest.raises(ValueError, match="training temperature"):
        _build_summary_sample(step, temperature=0.7, has_error=False)


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
def test_build_summary_sample_prefill_trim_rejects_invalid_raw_event_history(
    history,
):
    payload = _prefill_trim_summary_payload()
    valid = payload["compaction_events"][0]
    malformed = {
        "valid-plus-none": None,
        "valid-plus-malformed-dict": {"tokens_evicted": "bad"},
        "valid-plus-malformed-list": [0, "bad", 0],
        "valid-plus-malformed-object": object(),
    }
    if history == "two-valid":
        payload["compaction_events"] = [valid, dict(valid)]
    elif history == "sole-malformed":
        payload["compaction_events"] = [{"tokens_evicted": "bad"}]
    else:
        payload["compaction_events"] = [valid, malformed[history]]
    step = _step(
        prompt_ids=[1],
        completion_ids=[2],
        extras={"summary_trainsample": payload},
    )
    error = (
        "exactly one valid compaction event"
        if history == "sole-malformed"
        else "exactly one raw compaction event"
    )

    with pytest.raises(ValueError, match=error):
        _build_summary_sample(step, temperature=1.0, has_error=False)


def test_build_summary_sample_unknown_replay_mode_fails_closed():
    payload = _summary_payload(prompt_ids=[10], completion_ids=[20])
    payload["compaction_replay_mode"] = "future_mode"
    step = _step(
        prompt_ids=[1],
        completion_ids=[2],
        extras={"summary_trainsample": payload},
    )

    with pytest.raises(ValueError, match="unsupported compaction_replay_mode"):
        _build_summary_sample(step, temperature=1.0, has_error=False)


def test_build_summary_sample_no_payload_returns_none():
    step = _step(prompt_ids=[1], completion_ids=[2], extras={})
    assert _build_summary_sample(step, temperature=1.0, has_error=False) is None


def test_build_summary_sample_non_dict_payload_returns_none():
    step = _step(
        prompt_ids=[1],
        completion_ids=[2],
        extras={"summary_trainsample": "not a dict"},
    )
    assert _build_summary_sample(step, temperature=1.0, has_error=False) is None


def test_build_summary_sample_missing_prompt_returns_none():
    step = _step(
        prompt_ids=[1],
        completion_ids=[2],
        extras={
            "summary_trainsample": _summary_payload(
                prompt_ids=[], completion_ids=[20]
            )
        },
    )
    assert _build_summary_sample(step, temperature=1.0, has_error=False) is None


def test_build_summary_sample_missing_completion_returns_none():
    step = _step(
        prompt_ids=[1],
        completion_ids=[2],
        extras={
            "summary_trainsample": _summary_payload(
                prompt_ids=[10], completion_ids=[]
            )
        },
    )
    assert _build_summary_sample(step, temperature=1.0, has_error=False) is None


def test_build_summary_sample_logprob_length_mismatch_returns_none():
    step = _step(
        prompt_ids=[1],
        completion_ids=[2],
        extras={
            "summary_trainsample": _summary_payload(
                prompt_ids=[10],
                completion_ids=[20, 21],
                logprobs=[-0.1],  # one short
            )
        },
    )
    assert _build_summary_sample(step, temperature=1.0, has_error=False) is None


def test_build_summary_sample_error_zeros_completion_mask():
    step = _step(
        prompt_ids=[1],
        completion_ids=[2],
        extras={
            "summary_trainsample": _summary_payload(
                prompt_ids=[10], completion_ids=[20, 21]
            )
        },
    )
    sample = _build_summary_sample(step, temperature=1.0, has_error=True)
    assert sample is not None
    assert sample.completion_mask == [False, False]


def test_build_summary_sample_coerces_numeric_types():
    step = _step(
        prompt_ids=[1],
        completion_ids=[2],
        extras={
            "summary_trainsample": {
                "prompt_token_ids": ["10", 11.0],
                "completion_token_ids": (20, 21.0),
                "completion_logprobs": ("-0.1", -0.2),
                "model": "m",
            }
        },
    )
    sample = _build_summary_sample(step, temperature=1.0, has_error=False)
    assert sample is not None
    assert sample.prompt_ids == [10, 11]
    assert sample.completion_ids == [20, 21]


# ─── interleave_rollout emission ───


def test_interleave_rollout_no_summary_yields_single_sample():
    step = _step(
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        logprobs=[-0.1, -0.2],
    )
    out = _rollout([step])
    samples = interleave_rollout(out)
    assert samples is not None
    assert len(samples) == 1


def test_interleave_rollout_step_with_summary_yields_two_samples():
    step = _step(
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        logprobs=[-0.1, -0.2],
        extras={
            "summary_trainsample": _summary_payload(
                prompt_ids=[100, 101, 102],
                completion_ids=[200, 201],
                logprobs=[-0.5, -0.6],
            )
        },
    )
    out = _rollout([step])
    samples = interleave_rollout(out)
    assert samples is not None
    assert len(samples) == 2
    # Regular sample comes first, summary second (ordering in the return
    # list does not affect training semantics, but the contract is
    # stable — see interleave_rollout docstring).
    regular, summary = samples
    assert regular.prompt_ids == [1, 2]
    assert regular.completion_ids == [3, 4]
    # Summary sample: full-credit completion, zero-credit prompt.
    assert summary.prompt_ids == [100, 101, 102]
    assert summary.completion_ids == [200, 201]
    assert summary.completion_mask == [True, True]
    assert summary.prompt_mask == [False, False, False]
    assert summary.completion_logprobs == [-0.5, -0.6]


def test_interleave_rollout_multiple_summaries():
    # 2 steps, both with a summary payload → 2 regular samples
    # (extension breaks between prompt [1,2] and prompt [5,6,7]
    # since they share no prefix) + 2 summary samples = 4.
    step1 = _step(
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        logprobs=[-0.1, -0.2],
        extras={
            "summary_trainsample": _summary_payload(
                prompt_ids=[100], completion_ids=[200], logprobs=[-0.5]
            )
        },
    )
    step2 = _step(
        prompt_ids=[5, 6, 7],  # doesn't extend step1 — new sample
        completion_ids=[8, 9],
        logprobs=[-0.3, -0.4],
        extras={
            "summary_trainsample": _summary_payload(
                prompt_ids=[300, 301], completion_ids=[400], logprobs=[-0.7]
            )
        },
    )
    out = _rollout([step1, step2])
    samples = interleave_rollout(out)
    assert samples is not None
    assert len(samples) == 4


def test_interleave_rollout_malformed_summary_is_skipped():
    step = _step(
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        logprobs=[-0.1, -0.2],
        extras={"summary_trainsample": {"prompt_token_ids": []}},  # malformed
    )
    out = _rollout([step])
    samples = interleave_rollout(out)
    assert samples is not None
    # Only the regular sample; malformed summary dropped.
    assert len(samples) == 1


def test_interleave_rollout_errored_output_zeros_summary_mask():
    step = _step(
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        logprobs=[-0.1, -0.2],
        extras={
            "summary_trainsample": _summary_payload(
                prompt_ids=[10], completion_ids=[20, 21]
            )
        },
    )
    out = _rollout([step])
    out["error"] = "boom"
    samples = interleave_rollout(out)
    assert samples is not None
    assert len(samples) == 2
    summary = samples[-1]
    assert summary.completion_mask == [False, False]


# ─── compaction_events forwarding (eviction mode) ───


def test_build_summary_sample_compaction_events_from_dict():
    """Eviction-mode summary payloads carry ``compaction_events`` as a
    list of dicts (how the vLLM fork emits them). The builder must
    coerce them to ``CompactionEventWire`` so the trainer's dispatcher
    routes the sample through ``segmented_forward``."""
    from prime_rl.transport.types import CompactionEventWire

    payload = _summary_payload(
        prompt_ids=[10, 11, 12], completion_ids=[20, 21]
    )
    payload["compaction_events"] = [
        {
            "num_output_tokens_at_compaction": 128,
            "tokens_evicted": 512,
            "position_offset_after": 4096,
            "num_prompt_tokens": 2048,
            "evict_start": 0,
        },
    ]
    step = _step(
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        extras={"summary_trainsample": payload},
    )
    sample = _build_summary_sample(step, temperature=0.7, has_error=False)
    assert sample is not None
    assert sample.compaction_events is not None
    assert len(sample.compaction_events) == 1
    evt = sample.compaction_events[0]
    assert isinstance(evt, CompactionEventWire)
    assert evt.num_output_tokens_at_compaction == 128
    assert evt.tokens_evicted == 512
    assert evt.position_offset_after == 4096
    assert evt.num_prompt_tokens == 2048
    assert evt.evict_start == 0


def test_build_summary_sample_compaction_events_empty_list():
    """Empty events list (markovian mode, or eviction mode without
    events during the summary call) → compaction_events is None."""
    payload = _summary_payload(
        prompt_ids=[10], completion_ids=[20]
    )
    payload["compaction_events"] = []
    step = _step(
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        extras={"summary_trainsample": payload},
    )
    sample = _build_summary_sample(step, temperature=1.0, has_error=False)
    assert sample is not None
    assert sample.compaction_events is None


def test_build_summary_sample_compaction_events_absent():
    """No ``compaction_events`` key (legacy payload) → None."""
    payload = _summary_payload(
        prompt_ids=[10], completion_ids=[20]
    )
    step = _step(
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        extras={"summary_trainsample": payload},
    )
    sample = _build_summary_sample(step, temperature=1.0, has_error=False)
    assert sample is not None
    assert sample.compaction_events is None


def test_build_summary_sample_compaction_events_skips_malformed():
    """Malformed events are dropped, not raised — mirrors the
    per-step path's defensive coercion."""
    from prime_rl.transport.types import CompactionEventWire

    payload = _summary_payload(
        prompt_ids=[10], completion_ids=[20]
    )
    payload["compaction_events"] = [
        {"tokens_evicted": 512},  # missing required keys — dropped
        {
            "num_output_tokens_at_compaction": 64,
            "tokens_evicted": 256,
            "position_offset_after": 1024,
        },  # valid
    ]
    step = _step(
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        extras={"summary_trainsample": payload},
    )
    sample = _build_summary_sample(step, temperature=1.0, has_error=False)
    assert sample is not None
    assert sample.compaction_events is not None
    assert len(sample.compaction_events) == 1
    assert isinstance(sample.compaction_events[0], CompactionEventWire)
    assert sample.compaction_events[0].tokens_evicted == 256
