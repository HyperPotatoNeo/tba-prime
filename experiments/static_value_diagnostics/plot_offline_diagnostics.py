from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


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
    names = list(rows.keys())
    values = [rows[name]["variance"] for name in names]
    plt.figure(figsize=(9, 4.5))
    plt.bar(names, values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("mean squared advantage proxy")
    plt.title("Test metrics at val-selected rho")
    plt.tight_layout()
    plt.savefig(output_dir / "summary.png", dpi=180)
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


def main() -> None:
    args = parse_args()
    diagnostics_dir = args.diagnostics_dir
    output_dir = args.output_dir or diagnostics_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_json(diagnostics_dir / "diagnostics.json")
    group_rows = _load_csv(diagnostics_dir / "group_size_sensitivity.csv")
    plot_curves(data, output_dir)
    plot_summary(data, output_dir)
    plot_group_size(group_rows, output_dir)
    if args.wandb_project:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, job_type="plots")
        run.log({path.stem: wandb.Image(str(path)) for path in sorted(output_dir.glob("*.png"))})
        run.finish()
    print(f"wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
