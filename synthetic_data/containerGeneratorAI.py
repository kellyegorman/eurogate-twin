"""
Generate synthetic container rows from smooth_features_added.csv.

The previous Gaussian-copula approach produced valid-looking types (integers,
one-hot groups) but badly distorted column distributions. Rare categories such
as type_DC (~93% in the real data) were spread across many one-hot columns, and
standalone binaries such as iedCode were often thresholded to a single value.

This version bootstraps real rows (samples with replacement), which preserves:
  - integer columns as integers
  - binary / mutually-exclusive one-hot structure
  - marginal distributions and correlations from the source data

Optional small continuous noise can be added after bootstrapping if you want
rows that are not exact copies.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import random

INPUT_CSV = "smooth_features_added.csv"
OUTPUT_CSV = "synthetic_1000_rows.csv"
GENERATOR_STATE_PATH = "synthetic_generator_state.pkl"
N_SAMPLES = 1000
RANDOM_SEED = random.randint(0,1000)

# Fraction of each continuous column's training std used as post-bootstrap noise.
# Set to 0.0 for exact bootstrap copies (best match to describe() output).
CONTINUOUS_NOISE_FRACTION = 0.0

# Keep in sync with containerGenerator.py
EXCLUDED_COLUMNS: List[str] = [
    "numberOfArrivalContainersOnVessel",
    "numberOfRemainOnBoardContainersOnVessel",
    "numberOfDepartureContainersOnVessel",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "is_raining",
    "bad_weather_score",
    "dwell_hours",
    "arrival_hour",
    "arrival_dayofweek",
    "arrival_month",
    "arrival_is_weekend",
    "arrival_hour_sin",
    "arrival_hour_cos",
    "arrival_month_sin",
    "arrival_month_cos",
    "snapshot_time",
    "arrival_year",
    "snapshot_year",
    "snapshot_date",
    "snapshot_hour",
    "snapshot_minute",
    "snapshot_second",
    "containerId",
    "vessel_total_load",
    "vessel_arrival_fraction",
    "vessel_remain_fraction",
    "vessel_departure_fraction",
    "arrival_to_departure_container_ratio",
    "abs_eta_error_hours",
    "late_arrival_flag",
    "very_late_arrival_flag",
    "early_arrival_flag",
    "port_container_count",
    "import_count",
    "export_count",
    "unique_vessels",
    "unique_services",
    "unique_ports_of_loading",
    "import_export_ratio",
    "rolling_7d_port_count",
    "rolling_7d_import_count",
    "rolling_7d_export_count",
    "rolling_14d_port_count",
    "rolling_14d_import_count",
    "rolling_14d_export_count",
    "rolling_30d_port_count",
    "rolling_30d_import_count",
    "rolling_30d_export_count",
    "raw_arrivalPolCode",
    "raw_arrivalVesselName",
    "raw_arrivalServiceName",
    "raw_arrivalServiceCode",
    "dwell_bucket",
    "pol_bucket_mean_smooth",
    "pol_dwell_mean_smooth",
    "pol_dwell_median",
    "pol_dwell_p90",
    "pol_count",
    "vessel_bucket_mean_smooth",
    "vessel_dwell_mean_smooth",
    "vessel_dwell_median",
    "vessel_dwell_p90",
    "vessel_count",
    "service_bucket_mean_smooth",
    "service_dwell_mean_smooth",
    "service_dwell_median",
    "service_dwell_p90",
    "service_count",
    "arrivalLocationX",
    "arrivalLocationY",
    "arrivalLocationZ"
]

# bucket_actual = true dwell-time bucket; bucket = reported bucket (may be misreported)
BUCKET_ACTUAL_PROBS = np.array([0.203400, 0.268643, 0.242107, 0.285850])

# P(reported bucket | actual bucket); rows = actual, columns = reported
BUCKET_MISREPORT_PROBS = np.array(
    [
        [1.0 - 0.1935 - 0.0576 - 0.0057, 0.1935, 0.0576, 0.0057],  # actual 0: 0-36h
        [0.1072, 1.0 - 0.1072 - 0.1146 - 0.0067, 0.1146, 0.0067],  # actual 1: 36-72h
        [0.0250, 0.1047, 1.0 - 0.0250 - 0.1047 - 0.0627, 0.0627],  # actual 2: 72-120h
        [0.0020, 0.0081, 0.0487, 1.0 - 0.0020 - 0.0081 - 0.0487],  # actual 3: 120-336h
    ]
)

ONE_HOT_PREFIXES: Dict[str, str] = {
    "iso": "iso_",
    "service": "service_",
    "line": "line_",
    "arrivalType": "arrivalType_",
    "type": "type_",
}


@dataclass(frozen=True)
class OneHotGroup:
    name: str
    columns: Tuple[str, ...]


@dataclass
class ColumnSpec:
    integer_columns: List[str]
    standalone_binary_columns: List[str]
    continuous_columns: List[str]
    one_hot_groups: List[OneHotGroup]
    one_hot_columns: List[str]
    output_column_order: List[str]


@dataclass
class GeneratorState:
    """Cached preprocessed source data used for fast bootstrap sampling."""

    training: pd.DataFrame
    spec: ColumnSpec
    source_path: str
    source_mtime: float


def _is_binary_series(series: pd.Series) -> bool:
    values = pd.unique(series.dropna())
    if len(values) == 0:
        return False
    allowed = {0, 1, 0.0, 1.0, np.int64(0), np.int64(1)}
    return set(values).issubset(allowed)


def _is_integer_series(series: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(series):
        return True

    non_null = series.dropna()
    if non_null.empty:
        return False

    return np.allclose(non_null.to_numpy(), np.round(non_null.to_numpy()), rtol=0, atol=1e-9)


def _discover_one_hot_groups(columns: Sequence[str]) -> List[OneHotGroup]:
    groups: List[OneHotGroup] = []
    for group_name, prefix in ONE_HOT_PREFIXES.items():
        group_columns = tuple(col for col in columns if col.startswith(prefix))
        if group_columns:
            groups.append(OneHotGroup(name=group_name, columns=group_columns))
    return groups


def classify_columns(frame: pd.DataFrame) -> ColumnSpec:
    one_hot_groups = _discover_one_hot_groups(frame.columns)
    one_hot_columns = [col for group in one_hot_groups for col in group.columns]
    one_hot_column_set = set(one_hot_columns)

    integer_columns: List[str] = []
    standalone_binary_columns: List[str] = []
    continuous_columns: List[str] = []

    for column in frame.columns:
        if column in one_hot_column_set:
            continue

        series = frame[column]
        if _is_binary_series(series):
            standalone_binary_columns.append(column)
        elif _is_integer_series(series):
            integer_columns.append(column)
        else:
            continuous_columns.append(column)

    return ColumnSpec(
        integer_columns=integer_columns,
        standalone_binary_columns=standalone_binary_columns,
        continuous_columns=continuous_columns,
        one_hot_groups=one_hot_groups,
        one_hot_columns=one_hot_columns,
        output_column_order=list(frame.columns),
    )


def load_training_frame(csv_path: str = INPUT_CSV) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
    return frame.drop(columns=EXCLUDED_COLUMNS, errors="ignore")


def build_generator_state(csv_path: str = INPUT_CSV) -> GeneratorState:
    source = Path(csv_path)
    training = load_training_frame(csv_path)
    return GeneratorState(
        training=training,
        spec=classify_columns(training),
        source_path=str(source.resolve()),
        source_mtime=source.stat().st_mtime,
    )


def save_generator_state(
    state: GeneratorState,
    path: str | Path = GENERATOR_STATE_PATH,
) -> None:
    with open(path, "wb") as outfile:
        pickle.dump(state, outfile)


def load_generator_state(
    path: str | Path = GENERATOR_STATE_PATH,
    csv_path: str = INPUT_CSV,
    rebuild_if_stale: bool = True,
) -> GeneratorState:
    cache_path = Path(path)
    source = Path(csv_path).resolve()

    if cache_path.exists():
        with open(cache_path, "rb") as infile:
            state: GeneratorState = pickle.load(infile)

        is_current = (
            state.source_path == str(source)
            and state.source_mtime == source.stat().st_mtime
        )
        if is_current or not rebuild_if_stale:
            return state

    state = build_generator_state(csv_path)
    save_generator_state(state, cache_path)
    return state


def _apply_optional_continuous_noise(
    synthetic: pd.DataFrame,
    training: pd.DataFrame,
    continuous_columns: Sequence[str],
    noise_fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if noise_fraction <= 0:
        return synthetic

    result = synthetic.copy()
    for column in continuous_columns:
        std = float(training[column].std(ddof=0))
        if std == 0:
            continue

        noise = rng.normal(0.0, std * noise_fraction, size=len(result))
        lower = float(training[column].min())
        upper = float(training[column].max())
        result[column] = (result[column].to_numpy(dtype=float) + noise).clip(lower, upper)

    return result


def _enforce_column_types(synthetic: pd.DataFrame, spec: ColumnSpec) -> pd.DataFrame:
    result = synthetic.copy()

    for column in spec.integer_columns:
        result[column] = result[column].round().astype(int)

    for column in spec.standalone_binary_columns + spec.one_hot_columns:
        result[column] = result[column].round().astype(int)

    for column in spec.continuous_columns:
        result[column] = result[column].astype(float)

    return result[spec.output_column_order]


def _add_bucket_columns(
    synthetic: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    result = synthetic.copy()
    n_rows = len(result)

    bucket_actual = rng.choice(4, size=n_rows, p=BUCKET_ACTUAL_PROBS).astype(int)
    bucket = np.empty(n_rows, dtype=int)

    for actual in range(4):
        row_mask = bucket_actual == actual
        row_count = int(row_mask.sum())
        if row_count > 0:
            bucket[row_mask] = rng.choice(
                4,
                size=row_count,
                p=BUCKET_MISREPORT_PROBS[actual],
            )

    result["bucket"] = bucket
    result["bucket_actual"] = bucket_actual
    return result


def sample_from_state(
    state: GeneratorState,
    n_samples: int,
    random_seed: int = RANDOM_SEED,
    noise_fraction: float = CONTINUOUS_NOISE_FRACTION,
) -> pd.DataFrame:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(random_seed)
    sample_indices = rng.choice(len(state.training), size=n_samples, replace=True)
    synthetic = state.training.iloc[sample_indices].reset_index(drop=True)

    synthetic = _apply_optional_continuous_noise(
        synthetic,
        state.training,
        state.spec.continuous_columns,
        noise_fraction,
        rng,
    )
    synthetic = _enforce_column_types(synthetic, state.spec)
    return _add_bucket_columns(synthetic, rng)


def generate_synthetic_data(
    n_samples: int = N_SAMPLES,
    csv_path: str = INPUT_CSV,
    random_seed: int = RANDOM_SEED,
    noise_fraction: float = CONTINUOUS_NOISE_FRACTION,
    state_path: str | Path = GENERATOR_STATE_PATH,
    use_cached_state: bool = True,
) -> pd.DataFrame:
    if use_cached_state:
        state = load_generator_state(path=state_path, csv_path=csv_path)
    else:
        state = build_generator_state(csv_path)

    return sample_from_state(
        state,
        n_samples=n_samples,
        random_seed=random_seed,
        noise_fraction=noise_fraction,
    )


def fit_and_save_generator(
    csv_path: str = INPUT_CSV,
    state_path: str | Path = GENERATOR_STATE_PATH,
) -> GeneratorState:
    state = build_generator_state(csv_path)
    save_generator_state(state, state_path)
    return state


def main() -> None:
    fit_and_save_generator(csv_path=INPUT_CSV, state_path=GENERATOR_STATE_PATH)
    synthetic = generate_synthetic_data(
        n_samples=N_SAMPLES,
        csv_path=INPUT_CSV,
        state_path=GENERATOR_STATE_PATH,
        use_cached_state=True,
    )
    synthetic.to_csv(OUTPUT_CSV, index=False, encoding="latin-1")
    print(f"Saved generator state -> {GENERATOR_STATE_PATH}")
    print(f"Generated {len(synthetic)} rows -> {OUTPUT_CSV}")

def generateData(samples) -> None:
    fit_and_save_generator(csv_path=INPUT_CSV, state_path=GENERATOR_STATE_PATH)
    synthetic = generate_synthetic_data(
        n_samples=samples,
        csv_path=INPUT_CSV,
        state_path=GENERATOR_STATE_PATH,
        use_cached_state=True,
    )
    return synthetic


if __name__ == "__main__":
    main()
