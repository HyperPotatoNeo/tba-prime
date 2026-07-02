from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.static_value_diagnostics.common import write_json

EPS = 1e-8


@dataclass(frozen=True)
class PredictionSet:
    prompt_id: np.ndarray
    rollout_id: np.ndarray
    reward: np.ndarray
    offsets: np.ndarray
    values: np.ndarray
    logits: np.ndarray
    positions: np.ndarray
    gen_lengths: np.ndarray
    initial_value: np.ndarray
    initial_logit: np.ndarray

    @property
    def num_records(self) -> int:
        return int(self.reward.shape[0])


@dataclass(frozen=True)
class TokenTable:
    reward: np.ndarray
    group_mean: np.ndarray
    loo: np.ndarray
    value: np.ndarray
    value0: np.ndarray
    logit: np.ndarray
    logit0: np.ndarray
    odds_prior: np.ndarray
    position: np.ndarray
    frac_position: np.ndarray
    group_id: np.ndarray
    rollout_id: np.ndarray


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def no_intercept_rho(x: np.ndarray, y: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> float:
    denom = float(np.dot(x, x))
    if denom <= EPS:
        return 0.0
    return float(np.clip(float(np.dot(x, y)) / (denom + EPS), lo, hi))


def load_prediction_set(path: Path, split: str) -> PredictionSet:
    files = sorted(path.glob(f"predictions_{split}_rank*.npz"))
    if not files:
        files = sorted(path.glob(f"predictions_{split}.npz"))
    if not files:
        raise FileNotFoundError(f"No prediction files for split={split!r} under {path}")

    parts = [np.load(f) for f in files]
    record_keys = [
        "prompt_id",
        "rollout_id",
        "reward",
        "gen_lengths",
        "initial_value",
        "initial_logit",
    ]
    flat_keys = ["values", "logits", "positions"]
    records = {k: np.concatenate([p[k] for p in parts], axis=0) for k in record_keys}
    flats = {k: np.concatenate([p[k] for p in parts], axis=0) for k in flat_keys}
    lengths = [np.diff(p["offsets"]) for p in parts]
    offsets = np.concatenate([[0], np.cumsum(np.concatenate(lengths, axis=0))]).astype(np.int64)
    order = np.lexsort((records["rollout_id"], records["prompt_id"]))
    return _reorder_prediction_set(
        PredictionSet(offsets=offsets, **records, **flats),
        order,
    )


def has_binary_rewards(pred: PredictionSet, low: float = 0.0, high: float = 1.0) -> bool:
    rewards = pred.reward.astype(np.float64)
    bad = rewards[(np.abs(rewards - low) > 1e-6) & (np.abs(rewards - high) > 1e-6)]
    return bad.size == 0


def rho_methods(include_odds: bool) -> list[str]:
    methods = ["linear", "anchored_add", "anchored_add_clipped"]
    if include_odds:
        methods.append("anchored_odds")
    return methods


def _reorder_prediction_set(pred: PredictionSet, order: np.ndarray) -> PredictionSet:
    if np.all(order == np.arange(pred.num_records)):
        return pred
    flat_values: list[np.ndarray] = []
    flat_logits: list[np.ndarray] = []
    flat_positions: list[np.ndarray] = []
    offsets = [0]
    for idx in order:
        sl = slice(int(pred.offsets[idx]), int(pred.offsets[idx + 1]))
        flat_values.append(pred.values[sl])
        flat_logits.append(pred.logits[sl])
        flat_positions.append(pred.positions[sl])
        offsets.append(offsets[-1] + (sl.stop - sl.start))
    return PredictionSet(
        prompt_id=pred.prompt_id[order],
        rollout_id=pred.rollout_id[order],
        reward=pred.reward[order],
        offsets=np.asarray(offsets, dtype=np.int64),
        values=np.concatenate(flat_values).astype(np.float32),
        logits=np.concatenate(flat_logits).astype(np.float32),
        positions=np.concatenate(flat_positions).astype(np.int32),
        gen_lengths=pred.gen_lengths[order],
        initial_value=pred.initial_value[order],
        initial_logit=pred.initial_logit[order],
    )


def build_token_table(
    pred: PredictionSet,
    *,
    group_size: int,
    alpha: float = 0.5,
    beta: float = 0.5,
    prompt_ids: np.ndarray | None = None,
    group_records: dict[int, list[int]] | None = None,
) -> TokenTable:
    if group_size < 2:
        raise ValueError("group_size must be >= 2 for leave-one-out baselines.")

    selected_prompt_ids = set(prompt_ids.tolist()) if prompt_ids is not None else None
    group_map: dict[int, list[int]] = {}
    for idx, prompt_id in enumerate(pred.prompt_id.astype(int).tolist()):
        if selected_prompt_ids is not None and prompt_id not in selected_prompt_ids:
            continue
        group_map.setdefault(prompt_id, []).append(idx)

    arrays: dict[str, list[np.ndarray]] = {
        "reward": [],
        "group_mean": [],
        "loo": [],
        "value": [],
        "value0": [],
        "logit": [],
        "logit0": [],
        "odds_prior": [],
        "position": [],
        "frac_position": [],
        "group_id": [],
        "rollout_id": [],
    }
    kept_group = 0
    for prompt_id in sorted(group_map):
        if group_records is not None:
            idxs = group_records.get(prompt_id, [])
        else:
            idxs = sorted(group_map[prompt_id], key=lambda i: int(pred.rollout_id[i]))[:group_size]
        if len(idxs) < group_size:
            continue
        rewards = pred.reward[idxs].astype(np.float64)
        reward_sum = float(rewards.sum())
        group_mean = reward_sum / group_size
        for local, idx in enumerate(idxs):
            sl = slice(int(pred.offsets[idx]), int(pred.offsets[idx + 1]))
            n = sl.stop - sl.start
            if n <= 0:
                continue
            reward = float(rewards[local])
            loo = (reward_sum - reward) / (group_size - 1)
            p_loo = ((reward_sum - reward) + alpha) / ((group_size - 1) + alpha + beta)
            gen_len = max(int(pred.gen_lengths[idx]), 1)
            pos = pred.positions[sl].astype(np.float64)
            arrays["reward"].append(np.full(n, reward, dtype=np.float64))
            arrays["group_mean"].append(np.full(n, group_mean, dtype=np.float64))
            arrays["loo"].append(np.full(n, loo, dtype=np.float64))
            arrays["value"].append(pred.values[sl].astype(np.float64))
            arrays["value0"].append(np.full(n, float(pred.initial_value[idx]), dtype=np.float64))
            arrays["logit"].append(pred.logits[sl].astype(np.float64))
            arrays["logit0"].append(np.full(n, float(pred.initial_logit[idx]), dtype=np.float64))
            arrays["odds_prior"].append(np.full(n, p_loo, dtype=np.float64))
            arrays["position"].append(pred.positions[sl].astype(np.int32))
            arrays["frac_position"].append(((pos + 0.5) / gen_len).astype(np.float64))
            arrays["group_id"].append(np.full(n, kept_group, dtype=np.int32))
            arrays["rollout_id"].append(np.full(n, int(pred.rollout_id[idx]), dtype=np.int32))
        kept_group += 1

    if not arrays["reward"]:
        raise ValueError("No complete groups with non-empty token predictions.")

    return TokenTable(**{k: np.concatenate(v) for k, v in arrays.items()})


def method_prediction(table: TokenTable, method: str, rho: float) -> np.ndarray:
    if method == "group_mean":
        return table.group_mean
    if method == "loo":
        return table.loo
    if method == "pure_value":
        return table.value
    if method == "linear":
        return table.loo + rho * (table.value - table.loo)
    if method == "anchored_add":
        return table.loo + rho * (table.value - table.value0)
    if method == "anchored_add_clipped":
        return np.clip(table.loo + rho * (table.value - table.value0), 0.0, 1.0)
    if method == "odds_prior":
        return table.odds_prior
    if method == "anchored_odds":
        return sigmoid(logit(table.odds_prior) + rho * (table.logit - table.logit0))
    raise ValueError(f"unknown method {method!r}")


def variance_proxy(table: TokenTable, method: str, rho: float = 0.0, mask: np.ndarray | None = None) -> float:
    pred = method_prediction(table, method, rho)
    err = (table.reward - pred) ** 2
    if mask is not None:
        err = err[mask]
    return float(err.mean()) if err.size else math.nan


def bucket_masks(table: TokenTable) -> dict[str, np.ndarray]:
    pos = table.position
    frac = table.frac_position
    return {
        "early": frac < 0.25,
        "middle": (frac >= 0.25) & (frac < 0.75),
        "late": frac >= 0.75,
        "pos_000_032": (pos >= 0) & (pos < 32),
        "pos_032_064": (pos >= 32) & (pos < 64),
        "pos_064_128": (pos >= 64) & (pos < 128),
        "pos_128_plus": pos >= 128,
    }


def rho_curves(table: TokenTable, rhos: np.ndarray, methods: list[str] | None = None) -> dict[str, dict[str, float]]:
    methods = methods or rho_methods(include_odds=True)
    curves: dict[str, dict[str, float]] = {}
    for method in methods:
        curves[method] = {f"{rho:.2f}": variance_proxy(table, method, float(rho)) for rho in rhos}
    return curves


def select_rhos(table: TokenTable, rhos: np.ndarray, methods: list[str] | None = None) -> dict[str, float]:
    selected: dict[str, float] = {}
    methods = methods or rho_methods(include_odds=True)
    for method, curve in rho_curves(table, rhos, methods).items():
        selected[method] = float(min(curve.items(), key=lambda kv: kv[1])[0])
    selected["linear_closed_form"] = no_intercept_rho(table.value - table.loo, table.reward - table.loo)
    selected["anchored_add_closed_form"] = no_intercept_rho(table.value - table.value0, table.reward - table.loo)
    return selected


def summary_at_rhos(
    table: TokenTable,
    selected_rhos: dict[str, float],
    methods: list[str] | None = None,
) -> dict[str, Any]:
    methods = methods or rho_methods(include_odds=True)
    summary = {
        "group_mean": {"variance": variance_proxy(table, "group_mean")},
        "loo": {"variance": variance_proxy(table, "loo")},
        "pure_value": {"variance": variance_proxy(table, "pure_value")},
    }
    if "anchored_odds" in methods:
        summary["odds_prior"] = {"variance": variance_proxy(table, "odds_prior")}
    for method in methods:
        rho = float(selected_rhos[method])
        metric = variance_proxy(table, method, rho)
        entry: dict[str, Any] = {"rho": rho, "variance": metric}
        if method == "anchored_add_clipped":
            unclipped = method_prediction(table, "anchored_add", rho)
            entry["clip_fraction"] = float(((unclipped < 0.0) | (unclipped > 1.0)).mean())
        summary[method] = entry
    loo = summary["loo"]["variance"]
    for entry in summary.values():
        entry["delta_vs_loo"] = entry["variance"] - loo
        entry["relative_delta_vs_loo"] = entry["delta_vs_loo"] / max(loo, EPS)
    return summary


def position_summary(
    val_table: TokenTable,
    test_table: TokenTable,
    selected_rhos: dict[str, float],
    rhos: np.ndarray,
    methods: list[str] | None = None,
) -> list[dict[str, Any]]:
    methods = methods or rho_methods(include_odds=True)
    rows: list[dict[str, Any]] = []
    val_masks = bucket_masks(val_table)
    test_masks = bucket_masks(test_table)
    for bucket in val_masks:
        row: dict[str, Any] = {"bucket": bucket, "test_tokens": int(test_masks[bucket].sum())}
        row["loo_variance"] = variance_proxy(test_table, "loo", mask=test_masks[bucket])
        for method in methods:
            val_curve = {
                float(rho): variance_proxy(val_table, method, float(rho), mask=val_masks[bucket]) for rho in rhos
            }
            rho_bucket = min(val_curve.items(), key=lambda kv: kv[1])[0]
            rho_overall = float(selected_rhos[method])
            row[f"{method}_rho_overall"] = rho_overall
            row[f"{method}_rho_bucket"] = rho_bucket
            row[f"{method}_variance_overall_rho"] = variance_proxy(
                test_table, method, rho_overall, mask=test_masks[bucket]
            )
            row[f"{method}_variance_bucket_rho"] = variance_proxy(
                test_table, method, rho_bucket, mask=test_masks[bucket]
            )
        rows.append(row)
    return rows


def group_size_sensitivity(
    val_pred: PredictionSet,
    test_pred: PredictionSet,
    *,
    group_sizes: list[int],
    actual_group_size: int,
    rhos: np.ndarray,
    draws: int,
    seed: int,
    methods: list[str] | None = None,
) -> list[dict[str, Any]]:
    methods = methods or rho_methods(include_odds=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    val_prompts = sorted(set(val_pred.prompt_id.astype(int).tolist()))
    test_prompts = sorted(set(test_pred.prompt_id.astype(int).tolist()))
    for k in group_sizes:
        if k > actual_group_size:
            continue
        draw_metrics: dict[str, list[float]] = {method: [] for method in ["loo", "pure_value", *methods]}
        for draw_idx in range(draws):
            val_groups = _subsample_groups(val_pred, val_prompts, k, rng)
            test_groups = _subsample_groups(test_pred, test_prompts, k, rng)
            val_table = build_token_table(val_pred, group_size=k, group_records=val_groups)
            test_table = build_token_table(test_pred, group_size=k, group_records=test_groups)
            selected = select_rhos(val_table, rhos, methods)
            summary = summary_at_rhos(test_table, selected, methods)
            for method in draw_metrics:
                draw_metrics[method].append(summary[method]["variance"])
        for method, values in draw_metrics.items():
            arr = np.asarray(values, dtype=np.float64)
            rows.append(
                {
                    "group_size": k,
                    "method": method,
                    "draws": draws,
                    "mean_variance": float(arr.mean()),
                    "std_variance": float(arr.std(ddof=1)) if draws > 1 else 0.0,
                    "ci95": float(1.96 * arr.std(ddof=1) / math.sqrt(draws)) if draws > 1 else 0.0,
                }
            )
    return rows


def _subsample_groups(
    pred: PredictionSet,
    prompt_ids: list[int],
    k: int,
    rng: np.random.Generator,
) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for prompt_id in prompt_ids:
        idxs = np.flatnonzero(pred.prompt_id.astype(int) == prompt_id)
        if idxs.size < k:
            continue
        chosen = rng.choice(idxs, size=k, replace=False)
        groups[prompt_id] = sorted(chosen.astype(int).tolist(), key=lambda i: int(pred.rollout_id[i]))
    return groups


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostics(
    predictions_dir: Path,
    output_dir: Path,
    *,
    group_size: int,
    group_sizes: list[int],
    rho_step: float,
    sensitivity_draws: int,
    seed: int,
) -> dict[str, Any]:
    rhos = np.round(np.arange(0.0, 1.0 + rho_step / 2, rho_step), 6)
    val_pred = load_prediction_set(predictions_dir, "val")
    test_pred = load_prediction_set(predictions_dir, "test")
    include_odds = has_binary_rewards(val_pred) and has_binary_rewards(test_pred)
    methods = rho_methods(include_odds)
    val_table = build_token_table(val_pred, group_size=group_size)
    test_table = build_token_table(test_pred, group_size=group_size)
    selected = select_rhos(val_table, rhos, methods)
    result = {
        "config": {
            "group_size": group_size,
            "rho_grid": [float(r) for r in rhos],
            "selection": "rho selected on val split, evaluated on test split",
            "binary_rewards": include_odds,
            "methods": methods,
        },
        "val_selection": selected,
        "closed_form_note": "closed-form rho uses no-intercept least squares: sum(XY)/(sum(X^2)+eps), clipped to [0,1].",
        "val_curves": rho_curves(val_table, rhos, methods),
        "test_curves_descriptive": rho_curves(test_table, rhos, methods),
        "test_summary": summary_at_rhos(test_table, selected, methods),
        "position_summary": position_summary(val_table, test_table, selected, rhos, methods),
        "group_size_sensitivity": group_size_sensitivity(
            val_pred,
            test_pred,
            group_sizes=group_sizes,
            actual_group_size=group_size,
            rhos=rhos,
            draws=sensitivity_draws,
            seed=seed,
            methods=methods,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "diagnostics.json", result)
    write_csv(output_dir / "position_summary.csv", result["position_summary"])
    write_csv(output_dir / "group_size_sensitivity.csv", result["group_size_sensitivity"])
    return result
