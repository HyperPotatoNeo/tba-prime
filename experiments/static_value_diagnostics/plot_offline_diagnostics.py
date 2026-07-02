from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


POSITION_METHODS = {
    "linear": "linear_variance_overall_rho",
    "mixed_add": "mixed_add_variance_overall_params",
    "mixed_add_clipped": "mixed_add_clipped_variance_overall_params",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot static-policy value-baseline diagnostics.")
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_curves(data: dict, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for method, curve in data["test_curves_descriptive"].items():
        xs = [float(x) for x in curve.keys()]
        ys = [float(y) for y in curve.values()]
        plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=method)
    loo = data["test_summary"]["loo"]["variance"]
    pure = data["test_summary"]["pure_value"]["variance"]
    plt.axhline(loo, linestyle="--", linewidth=1, color="black", label="LOO")
    plt.axhline(pure, linestyle=":", linewidth=1, color="black", label="pure value")
    plt.xlabel("rho")
    plt.ylabel("mean squared advantage proxy")
    plt.title("Descriptive test rho curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "rho_curves.png", dpi=180)
    plt.close()


def plot_summary(data: dict, output_dir: Path) -> None:
    rows = data["test_summary"]
    names = [name for name in rows if name != "group_mean"]
    values = [rows[name]["variance"] for name in names]
    plt.figure(figsize=(9, 4.5))
    plt.bar(names, values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("mean squared advantage proxy")
    plt.title("Admissible test metrics at val-selected coefficients")
    plt.tight_layout()
    plt.savefig(output_dir / "summary.png", dpi=180)
    plt.close()


def plot_mixed_heatmaps(data: dict, output_dir: Path) -> None:
    curves = data.get("val_mixed_curves", {})
    selections = data.get("val_mixed_selection", {})
    for method, alpha_rows in curves.items():
        alpha_keys = sorted(alpha_rows.keys(), key=float)
        if not alpha_keys:
            continue
        rho_keys = sorted(next(iter(alpha_rows.values())).keys(), key=float)
        values = [[float(alpha_rows[alpha][rho]) for rho in rho_keys] for alpha in alpha_keys]
        alpha_values = [float(alpha) for alpha in alpha_keys]
        rho_values = [float(rho) for rho in rho_keys]
        plt.figure(figsize=(7, 5))
        plt.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=[min(rho_values), max(rho_values), min(alpha_values), max(alpha_values)],
        )
        selected = selections.get(method)
        if selected is not None:
            plt.scatter([selected["rho"]], [selected["alpha"]], color="white", marker="x", s=80)
        plt.colorbar(label="validation mean squared advantage proxy")
        plt.xlabel("rho: prefix progress")
        plt.ylabel("alpha: prompt prior correction")
        plt.title(f"Validation alpha/rho grid: {method}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{method}_alpha_rho_heatmap.png", dpi=180)
        plt.close()


def plot_group_size(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    methods = sorted({row["method"] for row in rows})
    plt.figure(figsize=(8, 5))
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda row: int(row["group_size"]))
        xs = [int(row["group_size"]) for row in method_rows]
        ys = [float(row["mean_variance"]) for row in method_rows]
        err = [float(row["ci95"]) for row in method_rows]
        plt.errorbar(xs, ys, yerr=err, marker="o", linewidth=1.5, capsize=3, label=method)
    plt.xlabel("rollouts per prompt")
    plt.ylabel("mean squared advantage proxy")
    plt.title("Group-size sensitivity")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "group_size_sensitivity.png", dpi=180)
    plt.close()


def plot_position_global_delta(rows: list[dict[str, str]], output_dir: Path) -> None:
    absolute_rows = [row for row in rows if row["bucket"].startswith("pos_") and not row["bucket"].endswith("_plus")]
    if not absolute_rows:
        return

    labels = [row["bucket"].removeprefix("pos_").replace("_", "-") for row in absolute_rows]
    xs = list(range(len(labels)))
    plt.figure(figsize=(9, 4.8))
    plt.axhline(0.0, linestyle="--", linewidth=1, color="black", label="LOO")
    for method, column in POSITION_METHODS.items():
        if column not in absolute_rows[0]:
            continue
        ys = []
        for row in absolute_rows:
            loo = float(row["loo_variance"])
            ys.append(100.0 * (float(row[column]) - loo) / loo)
        plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=method)
    plt.xticks(xs, labels, rotation=25, ha="right")
    plt.xlabel("generated-token bucket")
    plt.ylabel("delta vs LOO (%)")
    plt.title("Global-coefficient value baselines by token bucket")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "position_global_delta.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    diagnostics_dir = args.diagnostics_dir
    output_dir = args.output_dir or diagnostics_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_json(diagnostics_dir / "diagnostics.json")
    group_rows = _load_csv(diagnostics_dir / "group_size_sensitivity.csv")
    position_rows = _load_csv(diagnostics_dir / "position_summary.csv")
    plot_curves(data, output_dir)
    plot_summary(data, output_dir)
    plot_mixed_heatmaps(data, output_dir)
    plot_position_global_delta(position_rows, output_dir)
    plot_group_size(group_rows, output_dir)
    if args.wandb_project:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, job_type="plots")
        run.log({path.stem: wandb.Image(str(path)) for path in sorted(output_dir.glob("*.png"))})
        run.finish()
    print(f"wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
