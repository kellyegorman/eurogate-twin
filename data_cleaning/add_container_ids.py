"""Attach containerId columns to the final pre/post departure model datasets.

The final CSVs drop containerId during encoding. Row alignment is *not* a simple
positional join against the raw snapshots file:

- predeparture (334,633 rows): filter raw snapshots to pre-departure rows, then
  keep rows with elapsed_dwell_hours >= 0.
- postdeparture (145,080 rows): start from the 716,696 post-departure encoded
  rows, then drop duplicates on all columns except snapshot_* fields (same as
  building eurogate_postdeparture.csv from eurogate_train_encoded.csv).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data_cleaning" / "eurogate_all_snapshots.csv"
TRAIN_ENCODED_CSV = PROJECT_ROOT / "data_cleaning" / "eurogate_train_encoded.csv"
FINAL_PRE_CSV = PROJECT_ROOT / "data_final" / "eurogate_predeparture.csv"
FINAL_POST_CSV = PROJECT_ROOT / "data_final" / "eurogate_postdeparture.csv"

SNAPSHOT_COLS = [
    "snapshot_time",
    "snapshot_year",
    "snapshot_date",
    "snapshot_hour",
    "snapshot_minute",
    "snapshot_second",
]


def load_raw_with_snapshot_time() -> pd.DataFrame:
    df_raw = pd.read_csv(RAW_CSV)
    df_raw["arrivalTime"] = pd.to_datetime(df_raw["arrivalTime"], errors="coerce", utc=True)
    df_raw["departureTime"] = pd.to_datetime(df_raw["departureTime"], errors="coerce", utc=True)

    df_raw["snapshot_time"] = (
        df_raw["source_file"]
        .str.extract(r"Containerdaten-(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2})")[0]
        .str.replace("_", " ", regex=False)
        .str.replace(".", ":", regex=False)
    )
    df_raw["snapshot_time"] = pd.to_datetime(df_raw["snapshot_time"], errors="coerce", utc=True)

    final_departure = (
        df_raw.dropna(subset=["departureTime"])
        .groupby("containerId")["departureTime"]
        .min()
        .rename("final_departureTime")
    )
    return df_raw.merge(final_departure, on="containerId", how="left")


def predeparture_container_ids(df_raw: pd.DataFrame) -> pd.Series:
    df_pre_source = df_raw[
        df_raw["departureTime"].isna()
        & df_raw["final_departureTime"].notna()
        & (df_raw["snapshot_time"] < df_raw["final_departureTime"])
    ].copy()

    df_pre_source["elapsed_dwell_hours"] = (
        df_pre_source["snapshot_time"] - df_pre_source["arrivalTime"]
    ).dt.total_seconds() / 3600

    df_pre_source = df_pre_source[df_pre_source["elapsed_dwell_hours"] >= 0].copy()
    return df_pre_source["containerId"].reset_index(drop=True)


def postdeparture_container_ids(df_raw: pd.DataFrame, df_post: pd.DataFrame) -> pd.Series:
    df_raw_post = (
        df_raw[df_raw["departureTime"].notna()]
        .dropna(subset=["gross"])
        .reset_index(drop=True)
    )
    df_train = pd.read_csv(TRAIN_ENCODED_CSV)

    if len(df_raw_post) != len(df_train):
        raise ValueError(
            f"Expected raw post rows ({len(df_raw_post)}) to match train encoded "
            f"rows ({len(df_train)})."
        )

    compare_cols = [c for c in df_post.columns if c not in SNAPSHOT_COLS]
    keep_idx = df_train.drop_duplicates(subset=compare_cols, keep="first").index
    return df_raw_post.loc[keep_idx, "containerId"].reset_index(drop=True)


def add_container_ids(
    df_pre: pd.DataFrame | None = None,
    df_post: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_pre = pd.read_csv(FINAL_PRE_CSV) if df_pre is None else df_pre.copy()
    df_post = pd.read_csv(FINAL_POST_CSV) if df_post is None else df_post.copy()

    df_raw = load_raw_with_snapshot_time()

    pre_ids = predeparture_container_ids(df_raw)
    post_ids = postdeparture_container_ids(df_raw, df_post)

    if len(df_pre) != len(pre_ids):
        raise ValueError(f"Pre row count mismatch: {len(df_pre)} vs {len(pre_ids)}")
    if len(df_post) != len(post_ids):
        raise ValueError(f"Post row count mismatch: {len(df_post)} vs {len(post_ids)}")

    df_pre["containerId"] = pre_ids.to_numpy()
    df_post["containerId"] = post_ids.to_numpy()
    return df_pre, df_post


if __name__ == "__main__":
    pre_with_ids, post_with_ids = add_container_ids()

    pre_out = PROJECT_ROOT / "data_final" / "eurogate_predeparture_with_ids.csv"
    post_out = PROJECT_ROOT / "data_final" / "eurogate_postdeparture_with_ids.csv"

    pre_with_ids.to_csv(pre_out, index=False)
    post_with_ids.to_csv(post_out, index=False)

    print(f"Saved {pre_out} ({len(pre_with_ids):,} rows)")
    print(f"Saved {post_out} ({len(post_with_ids):,} rows)")
