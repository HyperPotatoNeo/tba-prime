from __future__ import annotations

import argparse
from pathlib import Path

from experiments.static_value_diagnostics.diagnostics import run_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute static-policy value-baseline diagnostics.")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--rho-step", type=float, default=0.05)
    parser.add_argument("--sensitivity-draws", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_diagnostics(
        args.predictions_dir,
        args.output_dir,
        group_size=args.group_size,
        group_sizes=args.group_sizes,
        rho_step=args.rho_step,
        sensitivity_draws=args.sensitivity_draws,
        seed=args.seed,
    )
    loo = result["test_summary"]["loo"]["variance"]
    best = min(result["test_summary"].items(), key=lambda kv: kv[1]["variance"])
    print(f"wrote diagnostics to {args.output_dir}")
    print(f"test loo variance={loo:.6g}; best={best[0]} variance={best[1]['variance']:.6g}")
    if args.wandb_project:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args), job_type="diagnostics")
        log_data = {"diagnostics/test_loo_variance": loo}
        for method, row in result["test_summary"].items():
            log_data[f"diagnostics/{method}/variance"] = row["variance"]
            log_data[f"diagnostics/{method}/delta_vs_loo"] = row["delta_vs_loo"]
            if "rho" in row:
                log_data[f"diagnostics/{method}/rho"] = row["rho"]
        run.log(log_data)
        run.finish()


if __name__ == "__main__":
    main()
