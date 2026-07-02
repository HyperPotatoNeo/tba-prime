from __future__ import annotations

import gzip
import json
from json import JSONDecodeError
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class RolloutRecord:
    split: str
    prompt_id: int
    rollout_id: int
    group_id: str
    reward: float
    token_ids: list[int]
    mask: list[bool]
    logprobs: list[float]
    num_output_tokens: int
    stop_reason: str | None = None
    has_error: bool = False
    error: str | None = None

    @property
    def usable(self) -> bool:
        return not self.has_error and any(self.mask) and len(self.token_ids) == len(self.mask)


def open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding=None if "b" in mode else "utf-8")
    return path.open(mode, encoding=None if "b" in mode else "utf-8")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open_text(path, "rt") as f:
        for raw_line in f:
            if not raw_line.endswith("\n"):
                # Streaming readers can observe the writer's current partial line.
                continue
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except JSONDecodeError:
                raise


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "at") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
        f.flush()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_rollout_records(path: Path, split: str | None = None) -> list[RolloutRecord]:
    records = [RolloutRecord(**row) for row in iter_jsonl(path)]
    if split is not None:
        records = [r for r in records if r.split == split]
    return records


def rollout_to_dict(record: RolloutRecord) -> dict[str, Any]:
    return asdict(record)


def split_prompt_ids(
    *,
    num_train_prompts: int,
    num_val_prompts: int,
    num_test_prompts: int,
    train_offset: int = 0,
    val_offset: int | None = None,
    test_offset: int | None = None,
) -> dict[str, list[int]]:
    val_offset = train_offset + num_train_prompts if val_offset is None else val_offset
    test_offset = val_offset + num_val_prompts if test_offset is None else test_offset
    return {
        "train": list(range(train_offset, train_offset + num_train_prompts)),
        "val": list(range(val_offset, val_offset + num_val_prompts)),
        "test": list(range(test_offset, test_offset + num_test_prompts)),
    }


def action_indices(mask: list[bool], seq_len: int | None = None) -> list[int]:
    if seq_len is None:
        seq_len = len(mask)
    return [idx for idx, is_action in enumerate(mask[:seq_len]) if is_action]


def clipped_record_arrays(record: RolloutRecord, seq_len: int) -> tuple[list[int], list[bool], list[float]]:
    cut = min(seq_len, len(record.token_ids), len(record.mask), len(record.logprobs))
    return record.token_ids[:cut], record.mask[:cut], record.logprobs[:cut]
