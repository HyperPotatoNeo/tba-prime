"""Wire-field stamping for the per-token streams.

The training loss is a sum of three components — ``rl`` (importance-weighted
PG + KL), ``ce`` (masked NLL), and ``ref_kl`` (reverse KL to a reference model
as the PG signal) — each normalized by its own global token count in the
trainer. The algorithm decides which component the action tokens feed
and the per-token advantages the rl component consumes; these helpers write
the component weight streams and the advantage stream onto the
``TrainingSample`` wire fields at group finalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import ActionLossType
from prime_rl.transport import TrainingSample

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


def stamp_loss_routing(sample: TrainingSample, action_loss_type: ActionLossType) -> None:
    """Stamp the algorithm's loss routing onto one sample's component weight
    streams: action tokens (the trainable completion tokens, per the loss
    mask) feed the algorithm's declared component.

    ``rl`` is the default and ships nothing (absent streams mean rl weight
    1.0 on the loss mask — the hot path); ``ce``/``ref_kl`` weight the action
    tokens into that component's stream and zero the rl stream. Streams an
    algorithm wrote directly (echo's observation ce weights) are merged, not
    clobbered — env-provided tokens stay out of the loss ``mask``, so the
    component an algorithm weights them into is the only one that trains
    them.
    """
    if action_loss_type == "rl":
        return

    seq_len = len(sample.token_ids)
    sample.rl_weights = [0.0] * seq_len
    action_weights = (
        sample.ce_weights if action_loss_type == "ce" and sample.ce_weights is not None else [0.0] * seq_len
    )
    for i, trains in enumerate(sample.mask):
        if trains:
            action_weights[i] = 1.0
    if action_loss_type == "ce":
        sample.ce_weights = action_weights
    else:
        assert action_loss_type == "ref_kl"
        sample.ref_kl_weights = action_weights


def stamp_advantages(rollout: Rollout) -> None:
    """Stamp the rollout's per-token advantage stream onto its samples' wire
    fields. The stream is full-length-N — aligned to the samples' ``token_ids``
    concatenated in order, 0.0 on non-trainable positions — and sliced across
    them. Rollouts with no credit assigned (``advantages=None``, e.g. opd/opsd)
    ship no advantage stream.
    """
    advantages = rollout.advantages
    if advantages is None:
        return
    total = sum(len(sample.token_ids) for sample in rollout.samples)
    if len(advantages) != total:
        raise ValueError(
            f"advantage stream must align with the rollout's tokens: "
            f"got {len(advantages)}, expected {total} (env '{rollout.env_name}')."
        )
    offset = 0
    for sample in rollout.samples:
        num_tokens = len(sample.token_ids)
        sample.advantages = list(advantages[offset : offset + num_tokens])
        offset += num_tokens


@dataclass(frozen=True)
class _ValueTurn:
    start: int
    terminal: int
    reward: float | None


def _sampled_spans(mask: list[bool], offset: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    idx = 0
    while idx < len(mask):
        if not mask[idx]:
            idx += 1
            continue
        start = offset + idx
        while idx + 1 < len(mask) and mask[idx + 1]:
            idx += 1
        spans.append((start, offset + idx))
        idx += 1
    return spans


def _extract_value_turns(rollout: Rollout) -> tuple[int, list[_ValueTurn], bool]:
    total = sum(len(sample.token_ids) for sample in rollout.samples)
    sampled_branches = [branch for branch in rollout.branches if any(branch.sampled_mask)]
    have_branches = len(sampled_branches) == len(rollout.samples)

    turns: list[_ValueTurn] = []
    misaligned = False
    offset = 0
    for sample_idx, sample in enumerate(rollout.samples):
        spans = _sampled_spans(sample.mask, offset)
        if have_branches:
            turn_rewards = [node.reward for node in sampled_branches[sample_idx].nodes if node.sampled]
        else:
            turn_rewards = []
        if len(turn_rewards) == len(spans):
            turns.extend(
                _ValueTurn(start=start, terminal=terminal, reward=reward)
                for (start, terminal), reward in zip(spans, turn_rewards, strict=True)
            )
        else:
            misaligned = True
            turns.extend(_ValueTurn(start=start, terminal=terminal, reward=None) for start, terminal in spans)
        offset += len(sample.token_ids)
    return total, turns, misaligned


def _turn_returns(rollout: Rollout, turns: list[_ValueTurn], misaligned: bool) -> tuple[list[float], bool]:
    use_return_to_go = len(turns) >= 2 and not misaligned and any(turn.reward is not None for turn in turns)
    if not turns:
        return [], use_return_to_go
    if not use_return_to_go:
        return [float(rollout.reward)] * len(turns), False

    returns = [0.0] * len(turns)
    running = 0.0
    for idx, turn in reversed(list(enumerate(turns))):
        running += float(turn.reward) if turn.reward is not None else 0.0
        returns[idx] = running
    head = returns[0]
    if abs(head - float(rollout.reward)) > 1e-3 * (1.0 + abs(float(rollout.reward))):
        raise ValueError(
            f"return-to-go head {head} disagrees with episodic reward {rollout.reward} "
            f"(env '{rollout.env_name}'): per-turn rewards misaligned."
        )
    return returns, True


def _assign_token_stream(rollout: Rollout, name: str, stream: list[float] | list[bool]) -> None:
    offset = 0
    for sample in rollout.samples:
        num_tokens = len(sample.token_ids)
        setattr(sample, name, stream[offset : offset + num_tokens])
        offset += num_tokens


def stamp_value_returns(rollout: Rollout) -> None:
    """Stamp per-token environment reward streams for trainer-local value learning.

    Value training uses terminal rewards/dones. For envs with per-turn rewards,
    each sampled turn receives its suffix return-to-go at the turn terminal; for
    terminal-only envs, the final sampled token receives the episodic reward.
    """
    total, turns, misaligned = _extract_value_turns(rollout)
    if total == 0 or not turns:
        return

    rewards = [0.0] * total
    dones = [False] * total
    turn_returns, use_return_to_go = _turn_returns(rollout, turns, misaligned)

    if use_return_to_go:
        for turn, turn_return in zip(turns, turn_returns, strict=True):
            rewards[turn.terminal] = turn_return
            dones[turn.terminal] = True
    else:
        terminal = turns[-1].terminal
        rewards[terminal] = float(rollout.reward)
        dones[terminal] = True

    position: list[float] | None = None
    if len(turns) >= 2:
        sampled_offsets: list[int] = []
        for turn in turns:
            sampled_offsets.extend(range(turn.start, turn.terminal + 1))
        if len(sampled_offsets) >= 2:
            position = [0.0] * total
            denom = len(sampled_offsets) - 1
            for rank, off in enumerate(sampled_offsets):
                position[off] = rank / denom

    episodic_return: list[float] | None = None
    if len(turns) >= 2:
        episodic_return = [0.0] * total
        for turn in turns:
            for off in range(turn.start, turn.terminal + 1):
                episodic_return[off] = float(rollout.reward)

    _assign_token_stream(rollout, "value_rewards", rewards)
    _assign_token_stream(rollout, "value_dones", dones)
    if position is not None:
        _assign_token_stream(rollout, "value_position_fraction", position)
    if episodic_return is not None:
        _assign_token_stream(rollout, "value_episodic_return", episodic_return)


def stamp_turn_anchor_tethers(group: list[Rollout]) -> None:
    """Stamp G_m and the full-mean turn tether T_m for turn-anchor TETHER."""
    per_rollout: list[tuple[Rollout, int, list[_ValueTurn], list[float]]] = []
    max_turns = 0
    for rollout in group:
        total, turns, misaligned = _extract_value_turns(rollout)
        turn_returns, _ = _turn_returns(rollout, turns, misaligned)
        per_rollout.append((rollout, total, turns, turn_returns))
        max_turns = max(max_turns, len(turn_returns))

    turn_means: list[float] = []
    for turn_idx in range(max_turns):
        reached = [turn_returns[turn_idx] for _, _, _, turn_returns in per_rollout if turn_idx < len(turn_returns)]
        assert reached, f"No rollouts reached turn {turn_idx} while max_turns={max_turns}"
        turn_means.append(sum(reached) / len(reached))

    for rollout, total, turns, turn_returns in per_rollout:
        if total == 0 or not turns:
            continue
        turn_return_stream = [0.0] * total
        turn_tether_stream = [0.0] * total
        for turn_idx, (turn, turn_return) in enumerate(zip(turns, turn_returns, strict=True)):
            turn_tether = turn_means[turn_idx]
            for off in range(turn.start, turn.terminal + 1):
                turn_return_stream[off] = turn_return
                turn_tether_stream[off] = turn_tether
        _assign_token_stream(rollout, "value_turn_return", turn_return_stream)
        _assign_token_stream(rollout, "value_turn_tether", turn_tether_stream)
