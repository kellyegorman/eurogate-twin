import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# LOAD DATA
df = pd.read_csv("data_final/eurogate_postdeparture.csv")

# TARGET
TARGET = "dwell_hours"

# FEATURES
drop_cols = [
    "dwell_hours",      # target
    "snapshot_time",    # datetime string
    "snapshot_date"     # date string
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[TARGET]

# CLEAN
X = X.replace([np.inf, -np.inf], np.nan)

# Remove any remaining string columns
object_cols = X.select_dtypes(include=["object"]).columns.tolist()

if len(object_cols) > 0:
    print("Dropping object columns:")
    print(object_cols)
    X = X.drop(columns=object_cols)

X = X.fillna(X.median(numeric_only=True))

# LOG TRANSFORM TARGET
y_log = np.log1p(y)

# TRAIN TEST SPLIT
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X,
    y_log,
    test_size=0.20,
    random_state=42
)

y_test = np.expm1(y_test_log)

# EVALUATION FUNCTION
def evaluate_model(name, model):

    model.fit(X_train, y_train_log)

    pred_log = model.predict(X_test)

    pred = np.expm1(pred_log)
    pred = np.maximum(pred, 0)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"MAE:  {mae:.2f} hours")
    print(f"RMSE: {rmse:.2f} hours")
    print(f"R²:   {r2:.4f}")

    print(f"Within 1 hour : {(np.abs(y_test-pred)<=1).mean():.2%}")
    print(f"Within 6 hours : {(np.abs(y_test-pred)<=6).mean():.2%}")
    print(f"Within 12 hours: {(np.abs(y_test-pred)<=12).mean():.2%}")
    print(f"Within 24 hours: {(np.abs(y_test-pred)<=24).mean():.2%}")

    return model, pred

# BASELINE
baseline = DummyRegressor(strategy="median")

baseline_model, baseline_pred = evaluate_model(
    "Baseline",
    baseline
)

# RANDOM FOREST
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

rf_model, rf_pred = evaluate_model(
    "Random Forest",
    rf
)

# FEATURE IMPORTANCE
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nTop 30 Features")
print(importance.head(30))

importance.to_csv(
    "feature_importance/total_dwell_time_feature_importance.csv",
    index=False
)

# SAVE PREDICTIONS
results = pd.DataFrame({
    "actual_dwell_hours": y_test,
    "predicted_dwell_hours": rf_pred,
    "error_hours": rf_pred - y_test
})

results.to_csv(
    "feature_importance/total_dwell_time_predictions.csv",
    index=False
)

print("\nSaved:")
# print("data/dwell_time_feature_importance.csv")
# print("data/dwell_time_predictions.csv")