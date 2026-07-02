"""Build a fixed RG-Mix dataset for reuse across runs.

Generates a mixture of the five reasoning-gym task variants used by RG-Mix and
saves it in the on-disk format ``rg_mix.load_environment`` expects:

    <out>/dataset/          # HF dataset with {question, answer=global_index} rows
    <out>/metadata.json     # {entry_map: [[variant_id, local_idx], ...], entries_cache: {idx: entry}}

Rows are shuffled (seeded) so variants are interleaved. The environment slices
the first ``num_train_examples`` rows for training and the next
``num_eval_examples`` for eval, so build a total >= train + eval.

Usage:
    uv run --extra rgmix python environments/rg_mix/build_dataset.py --out <dir> --total 10500
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import reasoning_gym as rg
from datasets import Dataset

from rg_mix import TASK_VARIANTS


def build_dataset(total: int, out_dir: str, seed: int) -> None:
    n_variants = len(TASK_VARIANTS)
    per_variant = -(-total // n_variants)  # ceil division

    entry_map: list[list] = []
    entries_cache: dict[int, dict] = {}
    rows: list[dict] = []
    global_idx = 0
    for variant in TASK_VARIANTS:
        ds = rg.create_dataset(variant["task"], seed=seed, size=per_variant, **variant["config"])
        for local_idx in range(len(ds)):
            entry = ds[local_idx]
            entry_map.append([variant["id"], local_idx])
            entries_cache[global_idx] = entry
            rows.append({"question": entry["question"], "answer": global_idx})
            global_idx += 1

    random.Random(seed).shuffle(rows)

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(root / "dataset"))
    with (root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"entry_map": entry_map, "entries_cache": entries_cache}, f)

    print(f"Wrote {len(rows)} RG-Mix examples to {root} ({per_variant}/variant x {n_variants} variants)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed RG-Mix dataset.")
    parser.add_argument("--out", type=str, required=True, help="Output dataset root directory.")
    parser.add_argument("--total", type=int, default=10500, help="Total examples to generate (train + eval).")
    parser.add_argument("--seed", type=int, default=42, help="Generation + shuffle seed.")
    args = parser.parse_args()
    build_dataset(args.total, args.out, args.seed)


if __name__ == "__main__":
    main()
