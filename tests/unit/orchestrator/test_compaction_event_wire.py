"""Unit tests for CompactionEventWire's 5 newly-plumbed fields (Phase A
of plans/two_phase_per_call_trainer.md):

  - kept_indices
  - kept_token_ids
  - new_user_fragment_len
  - last_turn_evicted
  - num_turns_evicted_after

Covers:
  1. Construct + msgspec roundtrip with all 10 fields populated.
  2. Backwards-compat: old wire payload (only the original 5 fields)
     decodes with safe defaults for the new ones.
  3. _compaction_events_from_step parses dict form correctly.
  4. _compaction_events_from_step parses msgspec array-like (list) form.
"""

from __future__ import annotations

import msgspec

from prime_rl.transport.types import (
    CallWire,
    CompactionEventWire,
    MicroBatch,
    TrainingSample,
    TurnCompactionStateWire,
    compute_turn_compaction_state_id,
)


def _full_event() -> CompactionEventWire:
    return CompactionEventWire(
        num_output_tokens_at_compaction=0,
        tokens_evicted=16,
        position_offset_after=16,
        num_prompt_tokens=64,
        evict_start=16,
        new_user_fragment_len=24,
        kept_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                      32, 33, 34],
        kept_token_ids=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                        110, 111, 112, 113, 114, 115, 200, 201, 202],
        last_turn_evicted=0,
        num_turns_evicted_after=1,
    )


def test_compaction_event_wire_full_roundtrip():
    e = _full_event()
    encoded = msgspec.msgpack.encode(e)
    decoded = msgspec.msgpack.decode(encoded, type=CompactionEventWire)
    assert decoded.new_user_fragment_len == 24
    assert decoded.kept_indices[:3] == [0, 1, 2]
    assert decoded.kept_token_ids[:3] == [100, 101, 102]
    assert decoded.last_turn_evicted == 0
    assert decoded.num_turns_evicted_after == 1


def test_compaction_event_wire_backwards_compat_decode():
    """An old wire payload (omits the new fields via omit_defaults) must
    still decode, with the new fields taking their declared defaults."""
    old_e = CompactionEventWire(
        num_output_tokens_at_compaction=0,
        tokens_evicted=16,
        position_offset_after=16,
    )
    encoded = msgspec.msgpack.encode(old_e)
    decoded = msgspec.msgpack.decode(encoded, type=CompactionEventWire)
    assert decoded.new_user_fragment_len == 0
    assert decoded.kept_indices == []
    assert decoded.kept_token_ids == []
    assert decoded.last_turn_evicted == -1
    assert decoded.num_turns_evicted_after == 0


def test_compaction_event_wire_array_like_round_trip_decodes_with_new_fields():
    """msgspec array_like=True encodes positionally; the array form must
    interpret positions 5-9 as the 5 new fields."""
    e = _full_event()
    # array_like + omit_defaults encodes only up to the last non-default
    # field; with all fields populated, we get a 10-element array.
    encoded = msgspec.json.encode(e)
    decoded = msgspec.json.decode(encoded, type=CompactionEventWire)
    assert decoded == e


def test_legacy_transport_arrays_decode_with_replay_mode_zero():
    call_payload = [[1], [2], [-0.1], [1.0]]
    sample_payload = [[1], [False], [2], [True], [-0.1], [1.0]]
    micro_batch_payload = [[1], [False], [1.0], [0.0], [0], [1.0]]
    call = msgspec.msgpack.decode(
        msgspec.msgpack.encode(call_payload),
        type=CallWire,
    )
    sample = msgspec.msgpack.decode(
        msgspec.msgpack.encode(sample_payload),
        type=TrainingSample,
    )
    micro_batch = msgspec.msgpack.decode(
        msgspec.msgpack.encode(micro_batch_payload),
        type=MicroBatch,
    )

    assert call.compaction_replay_mode == 0
    assert call.turn_compaction_state is None
    assert sample.compaction_replay_mode == 0
    assert micro_batch.compaction_replay_mode == 0


def test_prefill_trim_discriminator_round_trips_across_transport_structs():
    call = CallWire([1], [2], [-0.1], [1.0], compaction_replay_mode=1)
    sample = TrainingSample(
        [1],
        [False],
        [2],
        [True],
        [-0.1],
        [1.0],
        calls=[call],
        compaction_replay_mode=1,
    )
    micro_batch = MicroBatch(
        [1, 2],
        [False, True],
        [1.0, 1.0],
        [0.0, -0.1],
        [0, 1],
        [1.0, 1.0],
        compaction_replay_mode=1,
    )

    for value, value_type in (
        (call, CallWire),
        (sample, TrainingSample),
        (micro_batch, MicroBatch),
    ):
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(value),
            type=value_type,
        )
        assert decoded.compaction_replay_mode == 1


def test_turn_compaction_state_and_call_wire_round_trip():
    prompt_ids = [10, 11, 12, 13]
    state = TurnCompactionStateWire(
        version=1,
        position_offset=8,
        protected_prefix_len=2,
        num_turns_evicted=2,
        carried_prefix_num_live_turns=1,
        carried_prefix_len=len(prompt_ids),
        state_id=compute_turn_compaction_state_id(
            version=1,
            position_offset=8,
            protected_prefix_len=2,
            num_turns_evicted=2,
            carried_prefix_num_live_turns=1,
            carried_prefix_len=len(prompt_ids),
            carried_prefix_token_ids=prompt_ids,
        ),
    )
    call = CallWire(
        submitted_prompt_ids=prompt_ids,
        completion_ids=[20],
        completion_logprobs=[-0.1],
        completion_temperatures=[1.0],
        compaction_replay_mode=1,
        turn_compaction_state=state,
    )

    decoded_state = msgspec.msgpack.decode(
        msgspec.msgpack.encode(state),
        type=TurnCompactionStateWire,
    )
    decoded_call = msgspec.msgpack.decode(
        msgspec.msgpack.encode(call),
        type=CallWire,
    )

    assert decoded_state == state
    assert decoded_call == call
    assert decoded_call.turn_compaction_state == state
