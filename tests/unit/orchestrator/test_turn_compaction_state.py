from __future__ import annotations

import pytest

from prime_rl.trainer.batch import prepare_sample
from prime_rl.transport.types import (
    CallWire,
    CompactionEventWire,
    TrainingSample,
    TurnCompactionStateWire,
    compute_turn_compaction_state_id,
    validate_turn_compaction_state,
)


def _state(
    prompt_ids: list[int],
    *,
    position_offset: int = 8,
    protected_prefix_len: int = 2,
    num_turns_evicted: int = 2,
    carried_prefix_num_live_turns: int = 1,
    carried_prefix_len: int | None = None,
) -> dict[str, int | str]:
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


def _sample(
    *,
    submitted_prompt_ids: list[int],
    prompt_ids: list[int],
    state: dict[str, int | str] | TurnCompactionStateWire | None,
    event: CompactionEventWire | None,
) -> TrainingSample:
    completion_ids = [30, 31]
    completion_logprobs = [-0.1, -0.2]
    completion_temperatures = [0.7, 0.7]
    call = CallWire(
        submitted_prompt_ids=submitted_prompt_ids,
        completion_ids=completion_ids,
        completion_logprobs=completion_logprobs,
        completion_temperatures=completion_temperatures,
        compaction_events=[] if event is None else [event],
        compaction_replay_mode=1,
        turn_compaction_state=state,
    )
    return TrainingSample(
        prompt_ids=prompt_ids,
        prompt_mask=[False] * len(prompt_ids),
        completion_ids=completion_ids,
        completion_mask=[True] * len(completion_ids),
        completion_logprobs=completion_logprobs,
        completion_temperatures=completion_temperatures,
        advantage=1.0,
        calls=[call],
        compaction_replay_mode=1,
    )


def _event(
    submitted_prompt_ids: list[int],
    *,
    tokens_evicted: int = 4,
    position_offset_after: int = 8,
    num_turns_evicted_after: int = 2,
) -> tuple[CompactionEventWire, list[int]]:
    evict_start = 2
    evict_end = evict_start + tokens_evicted
    kept_indices = list(range(evict_start)) + list(range(evict_end, len(submitted_prompt_ids)))
    kept_token_ids = [submitted_prompt_ids[index] for index in kept_indices]
    return (
        CompactionEventWire(
            num_output_tokens_at_compaction=0,
            tokens_evicted=tokens_evicted,
            position_offset_after=position_offset_after,
            num_prompt_tokens=len(kept_token_ids),
            evict_start=evict_start,
            new_user_fragment_len=1,
            kept_indices=kept_indices,
            kept_token_ids=kept_token_ids,
            last_turn_evicted=num_turns_evicted_after - 1,
            num_turns_evicted_after=num_turns_evicted_after,
        ),
        kept_token_ids,
    )


def test_turn_compaction_state_hash_matches_sglang_canonical_contract():
    assert (
        compute_turn_compaction_state_id(
            version=1,
            position_offset=8,
            protected_prefix_len=2,
            num_turns_evicted=2,
            carried_prefix_num_live_turns=1,
            carried_prefix_len=4,
            carried_prefix_token_ids=[10, 11, 12, 13],
        )
        == "sha256:3f33ed186ada1e0b7f2f590b6d31da109bb5c0911c337cabb83ad00c8f353067"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("position_offset", "8"),
        ("protected_prefix_len", 2.0),
        ("num_turns_evicted", False),
        ("carried_prefix_num_live_turns", "1"),
        ("carried_prefix_len", 4.0),
    ],
)
def test_turn_compaction_state_rejects_non_exact_integer_values(field, value):
    prompt_ids = [10, 11, 12, 13]
    state = _state(prompt_ids)
    state[field] = value

    with pytest.raises(ValueError, match=field):
        validate_turn_compaction_state(state, prompt_ids, context="test")


@pytest.mark.parametrize(
    "state_id",
    [
        "sha256:" + "A" * 64,
        "sha256:" + "a" * 63,
        "a" * 64,
    ],
)
def test_turn_compaction_state_rejects_bad_state_id(state_id):
    prompt_ids = [10, 11, 12, 13]
    state = _state(prompt_ids)
    state["state_id"] = state_id

    with pytest.raises(ValueError, match="state_id"):
        validate_turn_compaction_state(state, prompt_ids, context="test")


def test_turn_compaction_state_rejects_stale_prompt_anchor():
    prompt_ids = [10, 11, 12, 13, 14]
    state = _state(prompt_ids, carried_prefix_len=4)

    with pytest.raises(ValueError, match="carried prompt prefix"):
        validate_turn_compaction_state(
            state,
            [10, 11, 99, 13, 14],
            context="test",
        )


def test_eventless_state_uses_cumulative_piecewise_positions():
    prompt_ids = [10, 11, 12, 13, 14, 15]
    state = _state(prompt_ids, carried_prefix_len=4)
    sample = _sample(
        submitted_prompt_ids=prompt_ids,
        prompt_ids=prompt_ids,
        state=state,
        event=None,
    )

    micro_batch = prepare_sample(sample, seq_len=4)

    assert micro_batch.input_ids == prompt_ids + [30, 31]
    assert micro_batch.position_ids == [0, 1, 10, 11, 12, 13, 14, 15]
    assert micro_batch.calls is None
    assert micro_batch.compaction_events is None


def test_second_cumulative_event_uses_state_offset_not_deletion_delta():
    submitted_prompt_ids = list(range(10, 22))
    event, kept_token_ids = _event(submitted_prompt_ids)
    state = _state(kept_token_ids)
    sample = _sample(
        submitted_prompt_ids=submitted_prompt_ids,
        prompt_ids=kept_token_ids,
        state=state,
        event=event,
    )

    micro_batch = prepare_sample(sample, seq_len=4)

    assert event.tokens_evicted == 4
    assert event.position_offset_after == 8
    assert micro_batch.position_ids == [0, 1, 10, 11, 12, 13, 14, 15, 16, 17]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("position_offset_after", 7, "cumulative position offset"),
        ("num_turns_evicted_after", 1, "cumulative turn count"),
    ],
)
def test_state_event_cumulative_mismatch_rejects(field, value, error):
    submitted_prompt_ids = list(range(10, 22))
    event, kept_token_ids = _event(submitted_prompt_ids)
    setattr(event, field, value)
    sample = _sample(
        submitted_prompt_ids=submitted_prompt_ids,
        prompt_ids=kept_token_ids,
        state=_state(kept_token_ids),
        event=event,
    )

    with pytest.raises(ValueError, match=error):
        prepare_sample(sample, seq_len=32)


def test_state_event_rejects_scalar_coercion():
    submitted_prompt_ids = list(range(10, 22))
    event, kept_token_ids = _event(submitted_prompt_ids)
    event.position_offset_after = "8"
    sample = _sample(
        submitted_prompt_ids=submitted_prompt_ids,
        prompt_ids=kept_token_ids,
        state=_state(kept_token_ids),
        event=event,
    )

    with pytest.raises(ValueError, match="must be an exact int"):
        prepare_sample(sample, seq_len=32)


def test_legacy_event_only_replay_still_uses_delta_offset():
    submitted_prompt_ids = list(range(10, 22))
    event, kept_token_ids = _event(
        submitted_prompt_ids,
        position_offset_after=4,
        num_turns_evicted_after=1,
    )
    sample = _sample(
        submitted_prompt_ids=submitted_prompt_ids,
        prompt_ids=kept_token_ids,
        state=None,
        event=event,
    )

    micro_batch = prepare_sample(sample, seq_len=32)

    assert micro_batch.position_ids == [0, 1, 6, 7, 8, 9, 10, 11, 12, 13]
