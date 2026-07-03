import random
import pandas as pd
from datetime import timedelta

from rachel_simulation.sim_with_scroll import (
    load_synthetic_containers,
    generate_initial_yard_containers,
    SimState,
)

NUM_RUNS = 1000
RANDOM_SAMPLE_SIZE = 30


def run_single_sim(
    n_containers,
    n_import_stacks,
    n_export_stacks,
    initial_containers,
    seed,
):
    containers = load_synthetic_containers(
        n=n_containers,
        seed=seed,
    )

    start_time = min(c.arrival_time for c in containers)
    start_time = start_time.replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    initial_yard = generate_initial_yard_containers(
        n=initial_containers,
        seed=seed + 999,
        base_time=start_time,
    )

    bucket_sim = SimState(
        "bucket",
        containers,
        n_import_stacks,
        n_export_stacks,
        initial_containers=initial_yard,
    )

    baseline_sim = SimState(
        "baseline",
        containers,
        n_import_stacks,
        n_export_stacks,
        initial_containers=initial_yard,
    )

    current_time = start_time

    while not (
        bucket_sim.is_complete()
        and baseline_sim.is_complete()
    ):
        next_time = current_time + timedelta(hours=1)

        bucket_sim.process_hour(current_time, next_time)
        baseline_sim.process_hour(current_time, next_time)

        current_time = next_time

    bucket_cost = bucket_sim.total_cost()
    baseline_cost = baseline_sim.total_cost()

    return {
        "seed": seed,
        "containers": n_containers,
        "import_stacks": n_import_stacks,
        "export_stacks": n_export_stacks,
        "initial_containers": initial_containers,

        "bucket_reshuffles": bucket_sim.total_reshuffles,
        "baseline_reshuffles": baseline_sim.total_reshuffles,

        "bucket_overflow":
            bucket_sim.yard.overflow_container_count,

        "baseline_overflow":
            baseline_sim.yard.overflow_container_count,

        "bucket_cost": bucket_cost,
        "baseline_cost": baseline_cost,

        "reshuffle_reduction":
            baseline_sim.total_reshuffles
            - bucket_sim.total_reshuffles,

        "cost_reduction":
            baseline_cost
            - bucket_cost,
    }


def choose_parameters(run_id):
    """
    Bias toward runs likely to have fewer reshuffles.
    """

    stack_count = random.choices(
        [8, 9, 10, 11, 12],
        weights=[5, 10, 30, 30, 25],
    )[0]

    n_containers = random.choices(
        [
            random.randint(400, 550),
            random.randint(550, 700),
            random.randint(700, 900),
            random.randint(900, 1000),
        ],
        weights=[25, 40, 25, 10],
    )[0]

    initial_containers = random.randint(20, 80)

    return {
        "n_containers": n_containers,
        "n_import_stacks": stack_count,
        "n_export_stacks": stack_count,
        "initial_containers": initial_containers,
        "seed": 1000 + run_id,
    }


def summarize_metric(
    df,
    baseline_col,
    bucket_col,
    diff_col,
    pct_col,
):
    return {
        "baseline_mean": df[baseline_col].mean(),
        "bucket_mean": df[bucket_col].mean(),

        "baseline_min": df[baseline_col].min(),
        "baseline_max": df[baseline_col].max(),

        "bucket_min": df[bucket_col].min(),
        "bucket_max": df[bucket_col].max(),

        "baseline_q1": df[baseline_col].quantile(0.25),
        "baseline_median": df[baseline_col].median(),
        "baseline_q3": df[baseline_col].quantile(0.75),

        "bucket_q1": df[bucket_col].quantile(0.25),
        "bucket_median": df[bucket_col].median(),
        "bucket_q3": df[bucket_col].quantile(0.75),

        "avg_diff": df[diff_col].mean(),
        "min_diff": df[diff_col].min(),
        "max_diff": df[diff_col].max(),

        "avg_pct_reduction": df[pct_col].mean(),
        "min_pct_reduction": df[pct_col].min(),
        "max_pct_reduction": df[pct_col].max(),
    }


def print_summary(title, stats):
    print(f"\n{'=' * 20}")
    print(title)
    print(f"{'=' * 20}")

    print(
        f"Baseline Mean: {stats['baseline_mean']:.2f}"
    )
    print(
        f"Bucket Mean:   {stats['bucket_mean']:.2f}"
    )

    print(
        f"\nAverage Difference: {stats['avg_diff']:.2f}"
    )
    print(
        f"Min Difference: {stats['min_diff']:.2f}"
    )
    print(
        f"Max Difference: {stats['max_diff']:.2f}"
    )

    print(
        f"\nAverage % Reduction: "
        f"{stats['avg_pct_reduction']:.2f}%"
    )
    print(
        f"Min % Reduction: "
        f"{stats['min_pct_reduction']:.2f}%"
    )
    print(
        f"Max % Reduction: "
        f"{stats['max_pct_reduction']:.2f}%"
    )

    print("\nBaseline Quartiles")
    print(
        f"Min={stats['baseline_min']:.2f} "
        f"Q1={stats['baseline_q1']:.2f} "
        f"Median={stats['baseline_median']:.2f} "
        f"Q3={stats['baseline_q3']:.2f} "
        f"Max={stats['baseline_max']:.2f}"
    )

    print("\nBucket Quartiles")
    print(
        f"Min={stats['bucket_min']:.2f} "
        f"Q1={stats['bucket_q1']:.2f} "
        f"Median={stats['bucket_median']:.2f} "
        f"Q3={stats['bucket_q3']:.2f} "
        f"Max={stats['bucket_max']:.2f}"
    )


def main():

    results = []

    for run in range(NUM_RUNS):

        params = choose_parameters(run)

        print(
            f"[{run + 1}/{NUM_RUNS}] "
            f"containers={params['n_containers']} "
            f"stacks={params['n_import_stacks']}"
        )

        result = run_single_sim(**params)

        results.append(result)

    df = pd.DataFrame(results)

    # -------------------------
    # Percent reductions
    # -------------------------

    df["reshuffle_pct_reduction"] = (
        100
        * (
            df["baseline_reshuffles"]
            - df["bucket_reshuffles"]
        )
        / df["baseline_reshuffles"]
    )

    df["cost_pct_reduction"] = (
        100
        * (
            df["baseline_cost"]
            - df["bucket_cost"]
        )
        / df["baseline_cost"]
    )

    # -------------------------
    # Save all results
    # -------------------------

    df.to_csv(
        "simulation_batch_results.csv",
        index=False,
    )

    # -------------------------
    # Random sample of 30 runs
    # -------------------------

    sample_df = df.sample(
        n=min(RANDOM_SAMPLE_SIZE, len(df)),
        random_state=42,
    )

    sample_df.to_csv(
        "random_30_runs.csv",
        index=False,
    )

    # -------------------------
    # Summary statistics
    # -------------------------

    reshuffle_stats = summarize_metric(
        df,
        "baseline_reshuffles",
        "bucket_reshuffles",
        "reshuffle_reduction",
        "reshuffle_pct_reduction",
    )

    cost_stats = summarize_metric(
        df,
        "baseline_cost",
        "bucket_cost",
        "cost_reduction",
        "cost_pct_reduction",
    )

    # -------------------------
    # Print summaries
    # -------------------------

    print("\n\nSIMULATION COMPLETE")
    print(f"Total Runs: {len(df)}")

    print_summary(
        "RESHUFFLE STATISTICS",
        reshuffle_stats,
    )

    print_summary(
        "COST STATISTICS",
        cost_stats,
    )

    # -------------------------
    # Save boxplot statistics
    # -------------------------

    boxplot_stats = pd.DataFrame({
        "metric": [
            "baseline_reshuffles",
            "bucket_reshuffles",
            "baseline_cost",
            "bucket_cost",
        ],
        "min": [
            df["baseline_reshuffles"].min(),
            df["bucket_reshuffles"].min(),
            df["baseline_cost"].min(),
            df["bucket_cost"].min(),
        ],
        "q1": [
            df["baseline_reshuffles"].quantile(.25),
            df["bucket_reshuffles"].quantile(.25),
            df["baseline_cost"].quantile(.25),
            df["bucket_cost"].quantile(.25),
        ],
        "median": [
            df["baseline_reshuffles"].median(),
            df["bucket_reshuffles"].median(),
            df["baseline_cost"].median(),
            df["bucket_cost"].median(),
        ],
        "q3": [
            df["baseline_reshuffles"].quantile(.75),
            df["bucket_reshuffles"].quantile(.75),
            df["baseline_cost"].quantile(.75),
            df["bucket_cost"].quantile(.75),
        ],
        "max": [
            df["baseline_reshuffles"].max(),
            df["bucket_reshuffles"].max(),
            df["baseline_cost"].max(),
            df["bucket_cost"].max(),
        ],
    })

    boxplot_stats.to_csv(
        "boxplot_statistics.csv",
        index=False,
    )

    # Display random sample

    print("\n")
    print("=" * 100)
    print("RANDOM SAMPLE OF 30 RUNS")
    print("=" * 100)

    print(
        sample_df[
            [
                "containers",
                "import_stacks",
                "baseline_reshuffles",
                "bucket_reshuffles",
                "reshuffle_reduction",
                "reshuffle_pct_reduction",
                "baseline_cost",
                "bucket_cost",
                "cost_reduction",
                "cost_pct_reduction",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    print("\nFiles written:")
    print("  simulation_batch_results.csv")
    print("  random_30_runs.csv")
    print("  boxplot_statistics.csv")


if __name__ == "__main__":
    main()
