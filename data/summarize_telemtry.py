#!/usr/bin/env python3
"""Extract and summarize V3 telemetry parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_JI = DATA_DIR / "V3_jobinstructions_from_telemetry.parquet"
DEFAULT_SH = DATA_DIR / "V3_shifts_from_telemetry.parquet"

# Friendlier names for the pipe-delimited job-instruction columns
JI_RENAME = {
    "che|@|cycle|@|id": "cycle_id",
    "che|@|name": "che_name",
    "che|@|spreader|@|locked|status|ioutput|actual|value": "spreader_locked",
    "che|@|hoist|@|weight|nett|output|actual|#unit#ton|value": "hoist_weight_ton",
    "msg|timestamp_first": "timestamp_first",
    "msg|timestamp_last": "timestamp_last",
    "msg|timestamp_lift": "timestamp_lift",
}


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def summarize_frame(df: pd.DataFrame, name: str) -> None:
    banner(f"{name}: overview")
    print(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    print("\nColumns & dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

    print("\nMissing values:")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    if nulls.empty:
        print("  (none)")
    else:
        for col, n in nulls.items():
            print(f"  {col}: {n:,} ({100 * n / len(df):.1f}%)")

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        print("\nNumeric summary:")
        print(numeric.describe().T.to_string())


def summarize_job_instructions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=JI_RENAME)

    banner("Job instructions: time & categories")
    print(
        f"timestamp_first range: "
        f"{df['timestamp_first'].min()} → {df['timestamp_first'].max()}"
    )
    print(f"Unique CHE names: {df['che_name'].nunique()}")
    print(df["che_name"].value_counts().head(15).to_string())

    print("\ncycle_type counts:")
    print(df["cycle_type"].value_counts().to_string())

    if "spreader_locked" in df.columns:
        print("\nspreader_locked counts:")
        print(df["spreader_locked"].value_counts(dropna=False).to_string())

    # Per-CHE aggregates useful for ops analysis
    agg = (
        df.groupby("che_name", dropna=False)
        .agg(
            cycles=("cycle_id", "count"),
            avg_distance_m=("distance_sum", "mean"),
            avg_drive_s=("drive_time_total_seconds", "mean"),
            avg_wait_count=("waiting_count", "mean"),
            rehandling_pct=("cycle_type", lambda s: (s == "Rehandling").mean()),
        )
        .sort_values("cycles", ascending=False)
        .round(2)
    )

    banner("Job instructions: per-CHE summary")
    print(agg.to_string())

    return agg


def summarize_shifts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    banner("Shifts: overview")
    print(f"Shift Start range: {df['Shift Start'].min()} → {df['Shift Start'].max()}")
    print(f"Shift End range:   {df['Shift End'].min()} → {df['Shift End'].max()}")

    df = df.copy()
    df["duration_hours"] = (df["Shift End"] - df["Shift Start"]).dt.total_seconds() / 3600
    df["n_active_carriers"] = df["Active Straddle Carriers"].apply(len)

    print("\nShift Name counts:")
    print(df["Shift Name"].value_counts().to_string())

    print("\nDuration (hours):")
    print(df["duration_hours"].describe().to_string())

    print("\nActive carriers per shift:")
    print(df["n_active_carriers"].describe().to_string())

    all_carriers = sorted(
        {c for carriers in df["Active Straddle Carriers"] for c in carriers}
    )
    print(f"\nUnique carriers across all shifts ({len(all_carriers)}):")
    print(", ".join(all_carriers))

    shift_summary = df[
        [
            "Shift ID",
            "Shift Name",
            "Shift Start",
            "Shift End",
            "duration_hours",
            "n_active_carriers",
            "Active Straddle Carriers",
        ]
    ].sort_values("Shift Start")

    # Flatten nested carrier positions for export / further analysis
    positions = df.explode("Carrier Positions").dropna(subset=["Carrier Positions"])
    positions = positions.reset_index(drop=True)
    pos_df = pd.json_normalize(positions["Carrier Positions"])
    pos_df["Shift ID"] = positions["Shift ID"].values
    pos_df["Shift Start"] = positions["Shift Start"].values

    banner("Shifts: sample carrier positions (first 10)")
    print(pos_df.head(10).to_string(index=False))

    return shift_summary, pos_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize V3 job-instruction and shift telemetry parquet files."
    )
    parser.add_argument(
        "--job-instructions",
        type=Path,
        default=DEFAULT_JI,
        help="Path to V3_jobinstructions_from_telemetry.parquet",
    )
    parser.add_argument(
        "--shifts",
        type=Path,
        default=DEFAULT_SH,
        help="Path to V3_shifts_from_telemetry.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional directory to write CSV summaries",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of sample rows to print per dataset",
    )
    args = parser.parse_args()

    ji_raw = pd.read_parquet(args.job_instructions)
    sh_raw = pd.read_parquet(args.shifts)

    summarize_frame(ji_raw, "Job instructions (raw)")
    summarize_frame(sh_raw, "Shifts (raw)")

    banner("Job instructions: sample rows")
    print(ji_raw.head(args.sample).to_string())

    banner("Shifts: sample rows")
    print(sh_raw.head(args.sample).to_string())

    che_summary = summarize_job_instructions(ji_raw)
    shift_summary, carrier_positions = summarize_shifts(sh_raw)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

        ji_out = ji_raw.rename(columns=JI_RENAME)
        ji_out.to_csv(args.out_dir / "job_instructions_extracted.csv", index=False)
        che_summary.to_csv(args.out_dir / "job_instructions_by_che.csv")
        shift_summary.to_csv(args.out_dir / "shifts_summary.csv", index=False)
        carrier_positions.to_csv(
            args.out_dir / "shift_carrier_positions.csv", index=False
        )

        meta = {
            "job_instructions": {
                "path": str(args.job_instructions),
                "rows": len(ji_raw),
                "columns": list(ji_raw.columns),
            },
            "shifts": {
                "path": str(args.shifts),
                "rows": len(sh_raw),
                "columns": list(sh_raw.columns),
            },
        }
        (args.out_dir / "summary_meta.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )
        print(f"\nWrote CSV/JSON summaries to {args.out_dir}")


if __name__ == "__main__":
    main()