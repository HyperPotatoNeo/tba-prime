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
    position_rho: np.ndarray
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


def mixed_methods(include_odds: bool) -> list[str]:
    methods = ["mixed_add", "mixed_add_clipped"]
    if include_odds:
        methods.append("mixed_odds")
    return methods


def deterministic_methods() -> list[str]:
    return ["linear_position"]


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
    prior_alpha: float = 0.5,
    prior_beta: float = 0.5,
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
        "position_rho": [],
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
            p_loo = ((reward_sum - reward) + prior_alpha) / ((group_size - 1) + prior_alpha + prior_beta)
            gen_len = max(int(pred.gen_lengths[idx]), 1)
            pos = pred.positions[sl].astype(np.float64)
            position_rho = np.ones_like(pos) if gen_len <= 1 else pos / (gen_len - 1)
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
            arrays["position_rho"].append(np.clip(position_rho, 0.0, 1.0).astype(np.float64))
            arrays["group_id"].append(np.full(n, kept_group, dtype=np.int32))
            arrays["rollout_id"].append(np.full(n, int(pred.rollout_id[idx]), dtype=np.int32))
        kept_group += 1

    if not arrays["reward"]:
        raise ValueError("No complete groups with non-empty token predictions.")

    return TokenTable(**{k: np.concatenate(v) for k, v in arrays.items()})


def method_prediction(table: TokenTable, method: str, rho: float, alpha: float = 0.0) -> np.ndarray:
    if method == "group_mean":
        return table.group_mean
    if method == "loo":
        return table.loo
    if method == "pure_value":
        return table.value
    if method == "linear":
        return table.loo + rho * (table.value - table.loo)
    if method == "linear_position":
        return table.loo + table.position_rho * (table.value - table.loo)
    if method == "anchored_add":
        return table.loo + rho * (table.value - table.value0)
    if method == "anchored_add_clipped":
        return np.clip(table.loo + rho * (table.value - table.value0), 0.0, 1.0)
    if method == "odds_prior":
        return table.odds_prior
    if method == "anchored_odds":
        return sigmoid(logit(table.odds_prior) + rho * (table.logit - table.logit0))
    if method == "mixed_add":
        return table.loo + alpha * (table.value0 - table.loo) + rho * (table.value - table.value0)
    if method == "mixed_add_clipped":
        return np.clip(table.loo + alpha * (table.value0 - table.loo) + rho * (table.value - table.value0), 0.0, 1.0)
    if method == "mixed_odds":
        prior_logit = logit(table.odds_prior)
        return sigmoid(prior_logit + alpha * (table.logit0 - prior_logit) + rho * (table.logit - table.logit0))
    raise ValueError(f"unknown method {method!r}")


def variance_proxy(
    table: TokenTable,
    method: str,
    rho: float = 0.0,
    mask: np.ndarray | None = None,
    alpha: float = 0.0,
) -> float:
    pred = method_prediction(table, method, rho, alpha)
    err = (table.reward - pred) ** 2
    if mask is not None:
        err = err[mask]
    return float(err.mean()) if err.size else math.nan


def _position_bucket_name(lo: int, hi: int | None, width: int) -> str:
    if hi is None:
        return f"pos_{lo:0{width}d}_plus"
    return f"pos_{lo:0{width}d}_{hi:0{width}d}"


def bucket_masks(table: TokenTable, position_bucket_edges: list[int] | None = None) -> dict[str, np.ndarray]:
    pos = table.position
    frac = table.frac_position
    edges = position_bucket_edges or [0, 512, 1024, 2048, 4096, 6144, 8192]
    if len(edges) < 2 or edges[0] != 0 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("position_bucket_edges must start at 0 and be strictly increasing")
    width = max(4, len(str(edges[-1])))
    masks = {
        "early": frac < 0.25,
        "middle": (frac >= 0.25) & (frac < 0.75),
        "late": frac >= 0.75,
    }
    for lo, hi in zip(edges, edges[1:]):
        masks[_position_bucket_name(lo, hi, width)] = (pos >= lo) & (pos < hi)
    masks[_position_bucket_name(edges[-1], None, width)] = pos >= edges[-1]
    return masks


def rho_curves(table: TokenTable, rhos: np.ndarray, methods: list[str] | None = None) -> dict[str, dict[str, float]]:
    methods = methods or rho_methods(include_odds=True)
    curves: dict[str, dict[str, float]] = {}
    for method in methods:
        curves[method] = {f"{rho:.2f}": variance_proxy(table, method, float(rho)) for rho in rhos}
    return curves


def alpha_rho_curves(
    table: TokenTable,
    alphas: np.ndarray,
    rhos: np.ndarray,
    methods: list[str] | None = None,
    mask: np.ndarray | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    methods = methods or mixed_methods(include_odds=True)
    curves: dict[str, dict[str, dict[str, float]]] = {}
    for method in methods:
        curves[method] = {
            f"{alpha:.2f}": {
                f"{rho:.2f}": variance_proxy(table, method, float(rho), mask=mask, alpha=float(alpha))
                for rho in rhos
            }
            for alpha in alphas
        }
    return curves


def select_rhos(table: TokenTable, rhos: np.ndarray, methods: list[str] | None = None) -> dict[str, float]:
    selected: dict[str, float] = {}
    methods = methods or rho_methods(include_odds=True)
    for method, curve in rho_curves(table, rhos, methods).items():
        selected[method] = float(min(curve.items(), key=lambda kv: kv[1])[0])
    selected["linear_closed_form"] = no_intercept_rho(table.value - table.loo, table.reward - table.loo)
    selected["anchored_add_closed_form"] = no_intercept_rho(table.value - table.value0, table.reward - table.loo)
    return selected


def two_factor_coefficients(
    first: np.ndarray,
    second: np.ndarray,
    target: np.ndarray,
    lo: float = 0.0,
    hi: float = 1.0,
) -> dict[str, float]:
    design = np.stack([first, second], axis=1).astype(np.float64)
    if design.shape[0] == 0:
        return {"alpha": 0.0, "rho": 0.0}
    try:
        alpha, rho = np.linalg.lstsq(design, target.astype(np.float64), rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"alpha": 0.0, "rho": 0.0}
    return {"alpha": float(np.clip(alpha, lo, hi)), "rho": float(np.clip(rho, lo, hi))}


def mixed_add_closed_form(table: TokenTable, mask: np.ndarray | None = None) -> dict[str, float]:
    selected = np.ones_like(table.reward, dtype=bool) if mask is None else mask
    return two_factor_coefficients(
        table.value0[selected] - table.loo[selected],
        table.value[selected] - table.value0[selected],
        table.reward[selected] - table.loo[selected],
    )


def select_mixed_params(
    table: TokenTable,
    alphas: np.ndarray,
    rhos: np.ndarray,
    methods: list[str] | None = None,
    mask: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    selected: dict[str, dict[str, float]] = {}
    methods = methods or mixed_methods(include_odds=True)
    for method in methods:
        best_alpha = 0.0
        best_rho = 0.0
        best_metric = math.inf
        for alpha in alphas:
            for rho in rhos:
                metric = variance_proxy(table, method, float(rho), mask=mask, alpha=float(alpha))
                if metric < best_metric:
                    best_alpha = float(alpha)
                    best_rho = float(rho)
                    best_metric = metric
        selected[method] = {"alpha": best_alpha, "rho": best_rho, "variance": float(best_metric)}
    return selected


def summary_at_rhos(
    table: TokenTable,
    selected_rhos: dict[str, float],
    methods: list[str] | None = None,
    selected_mixed: dict[str, dict[str, float]] | None = None,
    deterministic: list[str] | None = None,
) -> dict[str, Any]:
    methods = methods or rho_methods(include_odds=True)
    deterministic = deterministic or deterministic_methods()
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
    for method in deterministic:
        summary[method] = {
            "schedule": "rho_t=position/max(generated_length-1,1)",
            "variance": variance_proxy(table, method),
        }
    for method, params in (selected_mixed or {}).items():
        alpha = float(params["alpha"])
        rho = float(params["rho"])
        summary[method] = {
            "alpha": alpha,
            "rho": rho,
            "variance": variance_proxy(table, method, rho, alpha=alpha),
        }
        if method == "mixed_add_clipped":
            unclipped = method_prediction(table, "mixed_add", rho, alpha)
            summary[method]["clip_fraction"] = float(((unclipped < 0.0) | (unclipped > 1.0)).mean())
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
    position_bucket_edges: list[int] | None = None,
    methods: list[str] | None = None,
    selected_mixed: dict[str, dict[str, float]] | None = None,
    mixed_grid: np.ndarray | None = None,
    deterministic: list[str] | None = None,
) -> list[dict[str, Any]]:
    methods = methods or rho_methods(include_odds=True)
    deterministic = deterministic or deterministic_methods()
    rows: list[dict[str, Any]] = []
    val_masks = bucket_masks(val_table, position_bucket_edges)
    test_masks = bucket_masks(test_table, position_bucket_edges)
    bucket_tuned_mixed = {"early", "middle", "late"}
    for bucket in val_masks:
        val_tokens = int(val_masks[bucket].sum())
        test_tokens = int(test_masks[bucket].sum())
        if val_tokens == 0 or test_tokens == 0:
            continue
        row: dict[str, Any] = {"bucket": bucket, "val_tokens": val_tokens, "test_tokens": test_tokens}
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
        for method in deterministic:
            row[f"{method}_variance"] = variance_proxy(test_table, method, mask=test_masks[bucket])
        for method, params in (selected_mixed or {}).items():
            alpha_overall = float(params["alpha"])
            rho_overall = float(params["rho"])
            row[f"{method}_alpha_overall"] = alpha_overall
            row[f"{method}_rho_overall"] = rho_overall
            row[f"{method}_variance_overall_params"] = variance_proxy(
                test_table,
                method,
                rho_overall,
                mask=test_masks[bucket],
                alpha=alpha_overall,
            )
            if mixed_grid is not None and bucket in bucket_tuned_mixed:
                bucket_params = select_mixed_params(
                    val_table,
                    mixed_grid,
                    mixed_grid,
                    [method],
                    mask=val_masks[bucket],
                )[method]
                alpha_bucket = float(bucket_params["alpha"])
                rho_bucket = float(bucket_params["rho"])
                row[f"{method}_alpha_bucket"] = alpha_bucket
                row[f"{method}_rho_bucket"] = rho_bucket
                row[f"{method}_variance_bucket_params"] = variance_proxy(
                    test_table,
                    method,
                    rho_bucket,
                    mask=test_masks[bucket],
                    alpha=alpha_bucket,
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
    mixed_grid: np.ndarray,
    draws: int,
    seed: int,
    methods: list[str] | None = None,
    mixed: list[str] | None = None,
    deterministic: list[str] | None = None,
) -> list[dict[str, Any]]:
    methods = methods or rho_methods(include_odds=True)
    mixed = mixed or mixed_methods(include_odds=True)
    deterministic = deterministic or deterministic_methods()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    val_prompts = sorted(set(val_pred.prompt_id.astype(int).tolist()))
    test_prompts = sorted(set(test_pred.prompt_id.astype(int).tolist()))
    for k in group_sizes:
        if k > actual_group_size:
            continue
        draw_metrics: dict[str, list[float]] = {
            method: [] for method in ["loo", "pure_value", *deterministic, *methods, *mixed]
        }
        draw_alpha: dict[str, list[float]] = {method: [] for method in mixed}
        draw_rho: dict[str, list[float]] = {method: [] for method in mixed}
        for draw_idx in range(draws):
            val_groups = _subsample_groups(val_pred, val_prompts, k, rng)
            test_groups = _subsample_groups(test_pred, test_prompts, k, rng)
            val_table = build_token_table(val_pred, group_size=k, group_records=val_groups)
            test_table = build_token_table(test_pred, group_size=k, group_records=test_groups)
            selected = select_rhos(val_table, rhos, methods)
            selected_mixed = select_mixed_params(val_table, mixed_grid, mixed_grid, mixed)
            summary = summary_at_rhos(test_table, selected, methods, selected_mixed, deterministic)
            for method in draw_metrics:
                draw_metrics[method].append(summary[method]["variance"])
                if method in selected_mixed:
                    draw_alpha[method].append(selected_mixed[method]["alpha"])
                    draw_rho[method].append(selected_mixed[method]["rho"])
        for method, values in draw_metrics.items():
            arr = np.asarray(values, dtype=np.float64)
            alpha_values = np.asarray(draw_alpha.get(method, []), dtype=np.float64)
            rho_values = np.asarray(draw_rho.get(method, []), dtype=np.float64)
            rows.append(
                {
                    "group_size": k,
                    "method": method,
                    "draws": draws,
                    "mean_variance": float(arr.mean()),
                    "std_variance": float(arr.std(ddof=1)) if draws > 1 else 0.0,
                    "ci95": float(1.96 * arr.std(ddof=1) / math.sqrt(draws)) if draws > 1 else 0.0,
                    "mean_alpha": float(alpha_values.mean()) if alpha_values.size else None,
                    "mean_rho": float(rho_values.mean()) if rho_values.size else None,
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
    mixed_step: float,
    sensitivity_draws: int,
    seed: int,
    position_bucket_edges: list[int] | None = None,
) -> dict[str, Any]:
    rhos = np.round(np.arange(0.0, 1.0 + rho_step / 2, rho_step), 6)
    mixed_grid = np.round(np.arange(0.0, 1.0 + mixed_step / 2, mixed_step), 6)
    val_pred = load_prediction_set(predictions_dir, "val")
    test_pred = load_prediction_set(predictions_dir, "test")
    include_odds = has_binary_rewards(val_pred) and has_binary_rewards(test_pred)
    methods = rho_methods(include_odds)
    mixed = mixed_methods(include_odds)
    deterministic = deterministic_methods()
    val_table = build_token_table(val_pred, group_size=group_size)
    test_table = build_token_table(test_pred, group_size=group_size)
    selected = select_rhos(val_table, rhos, methods)
    selected_mixed = select_mixed_params(val_table, mixed_grid, mixed_grid, mixed)
    mixed_closed_form = mixed_add_closed_form(val_table)
    result = {
        "config": {
            "group_size": group_size,
            "rho_grid": [float(r) for r in rhos],
            "mixed_grid": [float(v) for v in mixed_grid],
            "selection": "rho selected on val split, evaluated on test split",
            "mixed_selection": "alpha/rho selected jointly on val split, evaluated on test split",
            "binary_rewards": include_odds,
            "methods": methods,
            "mixed_methods": mixed,
            "deterministic_methods": deterministic,
        },
        "val_selection": selected,
        "val_mixed_selection": selected_mixed,
        "val_mixed_closed_form": {"mixed_add": mixed_closed_form},
        "closed_form_note": "closed-form rho uses no-intercept least squares: sum(XY)/(sum(X^2)+eps), clipped to [0,1].",
        "mixed_closed_form_note": "mixed_add closed form uses no-intercept least squares over [V0-LOO, V-V0], then clips alpha/rho to [0,1]. Grid-selected alpha/rho is the reported metric.",
        "val_curves": rho_curves(val_table, rhos, methods),
        "val_mixed_curves": alpha_rho_curves(val_table, mixed_grid, mixed_grid, mixed),
        "test_curves_descriptive": rho_curves(test_table, rhos, methods),
        "test_mixed_curves_descriptive": alpha_rho_curves(test_table, mixed_grid, mixed_grid, mixed),
        "test_summary": summary_at_rhos(test_table, selected, methods, selected_mixed, deterministic),
        "position_summary": position_summary(
            val_table,
            test_table,
            selected,
            rhos,
            position_bucket_edges,
            methods,
            selected_mixed,
            mixed_grid,
            deterministic,
        ),
        "group_size_sensitivity": group_size_sensitivity(
            val_pred,
            test_pred,
            group_sizes=group_sizes,
            actual_group_size=group_size,
            rhos=rhos,
            mixed_grid=mixed_grid,
            draws=sensitivity_draws,
            seed=seed,
            methods=methods,
            mixed=mixed,
            deterministic=deterministic,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "diagnostics.json", result)
    write_csv(output_dir / "position_summary.csv", result["position_summary"])
    write_csv(output_dir / "group_size_sensitivity.csv", result["group_size_sensitivity"])
    return result
