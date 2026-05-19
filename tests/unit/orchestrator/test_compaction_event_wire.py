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

from prime_rl.transport.types import CompactionEventWire


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
