"""
Run multiple headless yard simulations using logic from sim_with_scroll.py.

Does not launch the Tkinter UI. Imports SimState and data-loading helpers
from sim_with_scroll and aggregates per-run metrics across repetitions.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sim_with_scroll as sim  # noqa: E402


STAT_FIELDS = [
    "bucket_reshuffles",
    "bucket_placed",
    "bucket_retrieved",
    "bucket_overflow_containers",
    "bucket_overflow_yards",
    "bucket_overflow_penalty",
    "bucket_total_cost",
    "baseline_reshuffles",
    "baseline_placed",
    "baseline_retrieved",
    "baseline_overflow_containers",
    "baseline_overflow_yards",
    "baseline_overflow_penalty",
    "baseline_total_cost",
    "reshuffle_reduction",
    "total_cost_reduction",
]


@dataclass
class ExperimentConfig:
    """Parameters for a batch of headless simulations."""

    runs: int = 10
    seed: int = 13
    import_stacks: int = 8
    export_stacks: int = 8
    new_containers: int = 100
    initial_containers: int = 60
    populate_initial: bool = True
    time_step_hours: int = 1
    output_dir: Path | None = None
    save_results: bool = True
    verbose: bool = True


def run_single_simulation(
    n_import_stacks: int,
    n_export_stacks: int,
    n_containers: int,
    n_initial_containers: int,
    seed: int,
    populate_initial: bool = True,
    time_step_hours: int = 1,
) -> dict[str, Any]:
    containers = sim.load_synthetic_containers(n=n_containers, seed=seed)

    start = min(c.arrival_time for c in containers)
    current_time = start.replace(minute=0, second=0, microsecond=0)

    if populate_initial:
        initial_containers = sim.generate_initial_yard_containers(
            n=n_initial_containers,
            seed=seed + 999,
            base_time=current_time,
        )
    else:
        initial_containers = []

    bucket_sim = sim.SimState(
        "bucket",
        containers,
        n_import_stacks,
        n_export_stacks,
        initial_containers=initial_containers,
    )
    baseline_sim = sim.SimState(
        "baseline",
        containers,
        n_import_stacks,
        n_export_stacks,
        initial_containers=initial_containers,
    )

    while not (bucket_sim.is_complete() and baseline_sim.is_complete()):
        next_time = current_time + timedelta(hours=time_step_hours)
        bucket_sim.process_hour(current_time, next_time)
        baseline_sim.process_hour(current_time, next_time)
        current_time = next_time

    reshuffle_reduction = baseline_sim.total_reshuffles - bucket_sim.total_reshuffles
    total_cost_reduction = baseline_sim.total_cost() - bucket_sim.total_cost()

    return {
        "seed": seed,
        "bucket_reshuffles": bucket_sim.total_reshuffles,
        "bucket_placed": bucket_sim.placed_count,
        "bucket_retrieved": bucket_sim.retrieved_count,
        "bucket_overflow_containers": bucket_sim.yard.overflow_container_count,
        "bucket_overflow_yards": bucket_sim.yard.overflow_yard_count,
        "bucket_overflow_penalty": bucket_sim.overflow_penalty(),
        "bucket_total_cost": bucket_sim.total_cost(),
        "baseline_reshuffles": baseline_sim.total_reshuffles,
        "baseline_placed": baseline_sim.placed_count,
        "baseline_retrieved": baseline_sim.retrieved_count,
        "baseline_overflow_containers": baseline_sim.yard.overflow_container_count,
        "baseline_overflow_yards": baseline_sim.yard.overflow_yard_count,
        "baseline_overflow_penalty": baseline_sim.overflow_penalty(),
        "baseline_total_cost": baseline_sim.total_cost(),
        "reshuffle_reduction": reshuffle_reduction,
        "total_cost_reduction": total_cost_reduction,
    }


def summarize_runs(run_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(run_rows)
    summary_rows = []

    for field in STAT_FIELDS:
        values = df[field].astype(float)
        summary_rows.append(
            {
                "metric": field,
                "mean": values.mean(),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
                "std": values.std(ddof=0),
            }
        )

    return pd.DataFrame(summary_rows)


def print_summary(summary_df: pd.DataFrame, n_runs: int) -> None:
    print(f"\nSummary across {n_runs} runs")
    print("=" * 88)
    print(
        summary_df.to_string(
            index=False,
            formatters={
                "mean": "{:,.3f}".format,
                "median": "{:,.3f}".format,
                "min": "{:,.3f}".format,
                "max": "{:,.3f}".format,
                "std": "{:,.3f}".format,
            },
        )
    )


def start_experiments(
    runs: int = 10,
    seed: int = 13,
    import_stacks: int = 8,
    export_stacks: int = 8,
    new_containers: int = 100,
    initial_containers: int = 60,
    populate_initial: bool = True,
    time_step_hours: int = 1,
    output_dir: Path | str | None = None,
    save_results: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run multiple headless simulations and return per-run + summary results.

    Pass parameters directly when calling from another script or notebook:

        from run_experiments import start_experiments

        results = start_experiments(
            runs=25,
            import_stacks=10,
            export_stacks=10,
            new_containers=150,
            initial_containers=80,
            seed=42,
        )
        print(results["summary"])
    """
    config = ExperimentConfig(
        runs=runs,
        seed=seed,
        import_stacks=import_stacks,
        export_stacks=export_stacks,
        new_containers=new_containers,
        initial_containers=initial_containers,
        populate_initial=populate_initial,
        time_step_hours=time_step_hours,
        output_dir=Path(output_dir) if output_dir is not None else SCRIPT_DIR / "experiment_results",
        save_results=save_results,
        verbose=verbose,
    )

    if config.verbose:
        print("Running headless experiments using sim_with_scroll.py")
        print(
            f"runs={config.runs}, import={config.import_stacks}, "
            f"export={config.export_stacks}, new={config.new_containers}, "
            f"initial={config.initial_containers}, base_seed={config.seed}"
        )

    run_rows: list[dict[str, Any]] = []

    for run_idx in range(config.runs):
        run_seed = config.seed + run_idx

        if config.verbose:
            print(
                f"  run {run_idx + 1}/{config.runs} (seed={run_seed})...",
                end=" ",
                flush=True,
            )

        result = run_single_simulation(
            n_import_stacks=config.import_stacks,
            n_export_stacks=config.export_stacks,
            n_containers=config.new_containers,
            n_initial_containers=config.initial_containers,
            seed=run_seed,
            populate_initial=config.populate_initial,
            time_step_hours=config.time_step_hours,
        )
        result["run"] = run_idx + 1
        run_rows.append(result)

        if config.verbose:
            print(
                "done | "
                f"bucket_cost={result['bucket_total_cost']}, "
                f"baseline_cost={result['baseline_total_cost']}, "
                f"cost_reduction={result['total_cost_reduction']}"
            )

    runs_df = pd.DataFrame(run_rows)
    summary_df = summarize_runs(run_rows)

    runs_path = None
    summary_path = None

    if config.save_results:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_path = config.output_dir / f"experiment_runs_{timestamp}.csv"
        summary_path = config.output_dir / f"experiment_summary_{timestamp}.csv"
        runs_df.to_csv(runs_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        if config.verbose:
            print(f"\nSaved per-run results: {runs_path}")
            print(f"Saved summary stats:    {summary_path}")

    if config.verbose:
        print_summary(summary_df, config.runs)

    return {
        "runs": runs_df,
        "summary": summary_df,
        "runs_path": runs_path,
        "summary_path": summary_path,
        "config": config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated headless yard simulations (no UI)."
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of simulations.")
    parser.add_argument("--seed", type=int, default=13, help="Base random seed.")
    parser.add_argument("--import-stacks", type=int, default=8)
    parser.add_argument("--export-stacks", type=int, default=8)
    parser.add_argument("--new-containers", type=int, default=100)
    parser.add_argument("--initial-containers", type=int, default=60)
    parser.add_argument(
        "--no-initial-yard",
        action="store_true",
        help="Do not pre-populate the yard before arrivals.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "experiment_results",
        help="Where per-run and summary CSVs are written.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write CSV output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_experiments(
        runs=args.runs,
        seed=args.seed,
        import_stacks=args.import_stacks,
        export_stacks=args.export_stacks,
        new_containers=args.new_containers,
        initial_containers=args.initial_containers,
        populate_initial=not args.no_initial_yard,
        output_dir=args.output_dir,
        save_results=not args.no_save,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
