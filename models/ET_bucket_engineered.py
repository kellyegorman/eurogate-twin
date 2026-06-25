import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score
)

all_snaps = "data_final/eurogate_postdeparture_with_ids.csv"
daily_snaps = "eurogate_daily"
out_engineered_path = "data_final/postdeparture_engineered_congestion_final.csv"

result_path = "feature_importance/final_extra_trees_results.csv"
importance_path = "feature_importance/final_extra_trees_feature_importance.csv"
predict_path = "feature_importance/final_extra_trees_predictions.csv"

df = pd.read_csv(all_snaps)
df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce", utc=True)
df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.date

# cols for daily snaps to be used
usecols = [
    "containerId",
    "iedCode",
    "arrivalPolCode",
    "arrivalVesselName",
    "arrivalServiceName",
    "arrivalServiceCode",
    "arrivalTime",
    "arrivalVoyageEta",
    "departureTime",
    "released",
    "numberOfArrivalContainersOnVessel",
    "numberOfRemainOnBoardContainersOnVessel",
    "numberOfDepartureContainersOnVessel",
]

raw_parts = []

for file in sorted(Path(daily_snaps).glob("Containerdaten-*.csv")):
    try:
        temp = pd.read_csv(
            file,
            usecols=lambda c: c in usecols,
            low_memory=False,
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        temp = pd.read_csv(
            file,
            usecols=lambda c: c in usecols,
            low_memory=False,
            encoding="latin1"
        )

    stamp = file.stem.replace("Containerdaten-", "")
    snapshot_time = pd.to_datetime(
        stamp.replace("_", " ").replace(".", ":"),
        errors="coerce",
        utc=True
    )

    temp["snapshot_time"] = snapshot_time
    temp["snapshot_date"] = snapshot_time.date()

    raw_parts.append(temp)

df_raw = pd.concat(raw_parts, ignore_index=True)

print("Raw rows loaded:", len(df_raw))

df_raw["arrivalTime"] = pd.to_datetime(df_raw["arrivalTime"], errors="coerce", utc=True)
df_raw["arrivalVoyageEta"] = pd.to_datetime(df_raw["arrivalVoyageEta"], errors="coerce", utc=True)
df_raw["departureTime"] = pd.to_datetime(df_raw["departureTime"], errors="coerce", utc=True)

for c in [
    "numberOfArrivalContainersOnVessel",
    "numberOfRemainOnBoardContainersOnVessel",
    "numberOfDepartureContainersOnVessel",
]:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

# post busyness congestion daily features
df_raw["is_import"] = (df_raw["iedCode"] == "IMPORT").astype(int)
df_raw["is_export"] = (df_raw["iedCode"] == "EXPORT").astype(int)

daily_stats = (
    df_raw
    .groupby("snapshot_date")
    .agg(
        port_container_count=("containerId", "nunique"),
        import_count=("is_import", "sum"),
        export_count=("is_export", "sum"),
        unique_vessels=("arrivalVesselName", "nunique"),
        unique_services=("arrivalServiceName", "nunique"),
        unique_ports_of_loading=("arrivalPolCode", "nunique"),
    )
    .reset_index()
)

daily_stats["import_export_ratio"] = (
    daily_stats["import_count"] /
    daily_stats["export_count"].replace(0, np.nan)
)

daily_stats = daily_stats.sort_values("snapshot_date")

for window in [7, 14, 30]:
    daily_stats[f"rolling_{window}d_port_count"] = (
        daily_stats["port_container_count"]
        .rolling(window=window, min_periods=1)
        .mean()
    )

    daily_stats[f"rolling_{window}d_import_count"] = (
        daily_stats["import_count"]
        .rolling(window=window, min_periods=1)
        .mean()
    )

    daily_stats[f"rolling_{window}d_export_count"] = (
        daily_stats["export_count"]
        .rolling(window=window, min_periods=1)
        .mean()
    )

# RAW CATEGORICAL ATTRIBUTES PER CONTAINER
container_attrs = (
    df_raw
    .sort_values("snapshot_time")
    .groupby("containerId")
    .agg(
        raw_arrivalPolCode=("arrivalPolCode", "first"),
        raw_arrivalVesselName=("arrivalVesselName", "first"),
        raw_arrivalServiceName=("arrivalServiceName", "first"),
        raw_arrivalServiceCode=("arrivalServiceCode", "first"),
    )
    .reset_index()
)

# vessel workload features
df["vessel_total_load"] = (
    df["numberOfArrivalContainersOnVessel"]
    + df["numberOfRemainOnBoardContainersOnVessel"]
    + df["numberOfDepartureContainersOnVessel"]
)

df["vessel_arrival_fraction"] = (
    df["numberOfArrivalContainersOnVessel"]
    / df["vessel_total_load"].replace(0, np.nan)
)

df["vessel_remain_fraction"] = (
    df["numberOfRemainOnBoardContainersOnVessel"]
    / df["vessel_total_load"].replace(0, np.nan)
)

df["vessel_departure_fraction"] = (
    df["numberOfDepartureContainersOnVessel"]
    / df["vessel_total_load"].replace(0, np.nan)
)

df["arrival_to_departure_container_ratio"] = (
    df["numberOfArrivalContainersOnVessel"]
    / df["numberOfDepartureContainersOnVessel"].replace(0, np.nan)
)

# ETA FEATURES
df["abs_eta_error_hours"] = df["eta_error_hours"].abs()
df["late_arrival_flag"] = (df["eta_error_hours"] > 24).astype(int)
df["very_late_arrival_flag"] = (df["eta_error_hours"] > 72).astype(int)
df["early_arrival_flag"] = (df["eta_error_hours"] < -24).astype(int)

# MERGE ENGINEERED FEATURES
df = df.merge(daily_stats, on="snapshot_date", how="left")
df = df.merge(container_attrs, on="containerId", how="left")

df = df.replace([np.inf, -np.inf], np.nan)
df.to_csv(out_engineered_path, index=False)

print("Saved engineered dataset:", out_engineered_path)
print("New shape:", df.shape)

# modeling dataset, 4 buckets, <=336 hrs/ 2 wks
df_model = df[df["dwell_hours"] <= 336].copy()

bins = [0, 36, 72, 120, 336]

df_model["dwell_bucket"] = pd.cut(
    df_model["dwell_hours"],
    bins=bins,
    labels=[0, 1, 2, 3],
    include_lowest=True
).astype(int)

print("\nBucket distribution:")
print(df_model["dwell_bucket"].value_counts(normalize=True).sort_index())

train_df, test_df = train_test_split(
    df_model,
    test_size=0.20,
    random_state=42,
    stratify=df_model["dwell_bucket"]
)

# train-only historical category features
def add_train_only_target_encoding(
    train_df,
    test_df,
    col,
    target_col="dwell_bucket",
    dwell_col="dwell_hours",
    prefix=None,
    drop_count_features=False
):
    if prefix is None:
        prefix = col

    global_bucket_mean = train_df[target_col].mean()
    global_dwell_mean = train_df[dwell_col].mean()

    stats = (
        train_df
        .groupby(col)
        .agg(
            bucket_mean=(target_col, "mean"),
            dwell_mean=(dwell_col, "mean"),
            dwell_median=(dwell_col, "median"),
            dwell_p90=(dwell_col, lambda x: x.quantile(0.90)),
            category_count=(target_col, "size")
        )
        .reset_index()
    )

    smoothing = 20

    stats[f"{prefix}_bucket_mean_smooth"] = (
        (stats["bucket_mean"] * stats["category_count"] + global_bucket_mean * smoothing)
        / (stats["category_count"] + smoothing)
    )

    stats[f"{prefix}_dwell_mean_smooth"] = (
        (stats["dwell_mean"] * stats["category_count"] + global_dwell_mean * smoothing)
        / (stats["category_count"] + smoothing)
    )

    stats = stats.rename(columns={
        "dwell_median": f"{prefix}_dwell_median",
        "dwell_p90": f"{prefix}_dwell_p90",
        "category_count": f"{prefix}_count"
    })

    keep_cols = [
        col,
        f"{prefix}_bucket_mean_smooth",
        f"{prefix}_dwell_mean_smooth",
        f"{prefix}_dwell_median",
        f"{prefix}_dwell_p90",
        f"{prefix}_count"
    ]

    train_df = train_df.merge(stats[keep_cols], on=col, how="left")
    test_df = test_df.merge(stats[keep_cols], on=col, how="left")

    for c in keep_cols[1:]:
        if "bucket" in c:
            fill_value = global_bucket_mean
        elif "count" in c:
            fill_value = 0
        else:
            fill_value = global_dwell_mean

        train_df[c] = train_df[c].fillna(fill_value)
        test_df[c] = test_df[c].fillna(fill_value)

    if drop_count_features:
        count_col = f"{prefix}_count"
        train_df = train_df.drop(columns=[count_col], errors="ignore")
        test_df = test_df.drop(columns=[count_col], errors="ignore")

    return train_df, test_df


DROP_COUNT_FEATURES = False

for col, prefix in [
    ("raw_arrivalPolCode", "pol"),
    ("raw_arrivalVesselName", "vessel"),
    ("raw_arrivalServiceName", "service"),
]:
    train_df, test_df = add_train_only_target_encoding(
        train_df,
        test_df,
        col=col,
        prefix=prefix,
        drop_count_features=DROP_COUNT_FEATURES
    )

# feature matrix
drop_cols = [
    "dwell_hours",
    "dwell_bucket",
    "dwell_bucket_candidate",
    "dwell_bucket_3",
    "containerId",
    "snapshot_time",
    "snapshot_date",
    "snapshot_year",
    "snapshot_hour",
    "snapshot_minute",
    "snapshot_second",
    "arrival_year",
    "arrivalLocationX",
    "arrivalLocationY",
    "arrivalLocationZ",
    "raw_arrivalPolCode",
    "raw_arrivalVesselName",
    "raw_arrivalServiceName",
    "raw_arrivalServiceCode",
    "occupied_slot_count",
    "occupied_bay_count",
    "avg_stack_height",
    "max_stack_height",
    "containers_per_occupied_slot",
    "containers_per_occupied_bay",
]

X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

y_train = train_df["dwell_bucket"]
y_test = test_df["dwell_bucket"]

train_df.to_csv("data_final/train_with_smoothed_features.csv", index=False)
test_df.to_csv("data_final/test_with_smoothed_features.csv", index=False)

print("Saved train/test with smoothed features.")

X_train = X_train.drop(columns=X_train.select_dtypes(exclude=[np.number]).columns)
X_test = X_test.drop(columns=X_test.select_dtypes(exclude=[np.number]).columns)

X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

medians = X_train.median(numeric_only=True)

X_train = X_train.fillna(medians).astype(np.float32)
X_test = X_test.fillna(medians).astype(np.float32)

# DROP VERY LOW-IMPORTANCE FEATURES
# DROP_LOW_IMPORTANCE = [
    # "type_FR",
    # "line_64",
    # "service_Diverse (absolute Ausnahme NUR Hauptschiffe)",
    # "line_161",
    # "early_arrival_flag",
    # "line_322",
    # "service_RFS FEEDER IE2",
    # "line_196",
    # "line_27",
    # "iso_22T6",
    # "type_AC",
    # "iso_4EG1",
    # "iso_45R5",
    # "service_PLX POLAND",
    # "service_Norwegendienst Westküste",
    # "type_DO",
    # "iso_22GB",
    # "iso_L5G1",
    # "service_X-PRESS FEEDER",
    # "iso_22R1",
    # "iso_22U1",
# ]

# X_train = X_train.drop(columns=[c for c in DROP_LOW_IMPORTANCE if c in X_train.columns])
# X_test = X_test.drop(columns=[c for c in DROP_LOW_IMPORTANCE if c in X_test.columns])

print("\nFinal feature count:", X_train.shape[1])

# use extra trees classifier model because it is best performing (over RT)
model = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=40,
    min_samples_leaf=2,
    min_samples_split=5,
    max_features="sqrt",
    class_weight={0: 1.0, 1: 1.1, 2: 1.3, 3: 1.6},
    random_state=42,
    n_jobs=-1
)

print("\nTraining Extra Trees...")

model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)

# evaluate
acc = accuracy_score(y_test, pred)
bal_acc = balanced_accuracy_score(y_test, pred)
macro_recall = recall_score(y_test, pred, average="macro")
weighted_recall = recall_score(y_test, pred, average="weighted")
per_class_recall = recall_score(y_test, pred, average=None)

under_rate = (pred < y_test).mean()
over_rate = (pred > y_test).mean()
severe_under_rate = (pred <= y_test - 2).mean()

print("extra trees model!!")
print(f"Accuracy:              {acc:.5f}")
print(f"Balanced Accuracy:     {bal_acc:.5f}")
print(f"Macro Recall:          {macro_recall:.5f}")
print(f"Weighted Recall:       {weighted_recall:.5f}")
print(f"Under rate:            {under_rate:.5f}")
print(f"Over rate:             {over_rate:.5f}")
print(f"Severe under rate:     {severe_under_rate:.5f}")

print("\nPer-bucket Recall:")
for bucket, r in enumerate(per_class_recall):
    print(f"Bucket {bucket}: {r:.5f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    pred,
    target_names=[
        "0-36h",
        "36-72h",
        "72-120h",
        "120-336h"
    ],
    zero_division=0
))

# save the results in a csv
results_df = pd.DataFrame([{
    "model": "Extra Trees",
    "accuracy": acc,
    "balanced_accuracy": bal_acc,
    "macro_recall": macro_recall,
    "weighted_recall": weighted_recall,
    "under_rate": under_rate,
    "over_rate": over_rate,
    "severe_under_rate": severe_under_rate,
    "bucket_0_recall": per_class_recall[0],
    "bucket_1_recall": per_class_recall[1],
    "bucket_2_recall": per_class_recall[2],
    "bucket_3_recall": per_class_recall[3],
    "feature_count": X_train.shape[1]
}])

results_df.to_csv(result_path, index=False)

predictions_df = pd.DataFrame({
    "actual_bucket": y_test.values,
    "predicted_bucket": pred,
    "prob_bucket_0": prob[:, 0],
    "prob_bucket_1": prob[:, 1],
    "prob_bucket_2": prob[:, 2],
    "prob_bucket_3": prob[:, 3],
    "bucket_error": pred - y_test.values
})

predictions_df.to_csv(predict_path, index=False)

importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

importance_df.to_csv(importance_path, index=False)

print("\nSaved:")
print(out_engineered_path)
print(result_path)
print(predict_path)
print(importance_path)