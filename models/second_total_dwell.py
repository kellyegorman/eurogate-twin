import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("data_final/eurogate_postdeparture.csv")

TARGET = "dwell_hours"

# Drop leakage / suspicious / non-model columns
drop_cols = [
    TARGET,

    # raw timestamps / file-snapshot artifacts
    "snapshot_time",
    "snapshot_date",
    "snapshot_year",
    "snapshot_hour",
    "snapshot_minute",
    "snapshot_second",

    # usually constant or not useful
    "arrival_year",

    # pandas index leftovers
    "Unnamed: 0",
    "Unnamed: 0.1",
    "Unnamed: 0.2",

    # if present, direct leakage
    "departureTime",
    "remaining_dwell_hours",
    "elapsed_dwell_hours",
]

df = df.dropna(subset=[TARGET]).copy()

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[TARGET]

# Clean features
X = X.replace([np.inf, -np.inf], np.nan)

object_cols = X.select_dtypes(include=["object"]).columns.tolist()
print("Dropping object columns:", object_cols)
X = X.drop(columns=object_cols)

bool_cols = X.select_dtypes(include=["bool"]).columns
X[bool_cols] = X[bool_cols].astype(int)

X = X.fillna(X.median(numeric_only=True))

# Optional: remove extreme top 1% dwell outliers
upper = y.quantile(0.99)
mask = y <= upper

X = X.loc[mask].copy()
y = y.loc[mask].copy()

# Log target helps with skewed dwell time
y_log = np.log1p(y)

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X,
    y_log,
    test_size=0.2,
    random_state=42
)

y_test = np.expm1(y_test_log)

# -----------------------------
# Evaluation helper
# -----------------------------
def evaluate_model(name, model):
    model.fit(X_train, y_train_log)

    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)
    pred = np.maximum(pred, 0)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    within_6 = (np.abs(y_test - pred) <= 6).mean()
    within_12 = (np.abs(y_test - pred) <= 12).mean()
    within_24 = (np.abs(y_test - pred) <= 24).mean()

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print(f"MAE:  {mae:.2f} hours")
    print(f"RMSE: {rmse:.2f} hours")
    print(f"R²:   {r2:.4f}")
    print(f"Within 6 hours:  {within_6:.2%}")
    print(f"Within 12 hours: {within_12:.2%}")
    print(f"Within 24 hours: {within_24:.2%}")

    return {
        "model_name": name,
        "model": model,
        "pred": pred,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

# -----------------------------
# Models to test
# -----------------------------
models = [
    ("Baseline Median", DummyRegressor(strategy="median")),
    ("Ridge Regression", Ridge(alpha=1.0)),
    ("Random Forest", RandomForestRegressor(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )),
    ("Extra Trees", ExtraTreesRegressor(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )),
    ("Hist Gradient Boosting", HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=42
    )),
]

results = []

for name, model in models:
    results.append(evaluate_model(name, model))

# -----------------------------
# Summary table
# -----------------------------
summary = pd.DataFrame([
    {
        "model": r["model_name"],
        "MAE": r["mae"],
        "RMSE": r["rmse"],
        "R2": r["r2"]
    }
    for r in results
]).sort_values("MAE")

print("\nModel comparison:")
print(summary)

summary.to_csv("/Users/kellyg/eurogate-twin-1/feature_importance/total_dwell_model_comparison.csv", index=False)

# -----------------------------
# Feature importance for best tree model
# -----------------------------
best = min(results, key=lambda r: r["mae"])
best_model = best["model"]

if hasattr(best_model, "feature_importances_"):
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop 30 Features:")
    print(importance.head(30))

    importance.to_csv("/Users/kellyg/eurogate-twin-1/feature_importance/total_dwell_feature_importance_updated.csv", index=False)

# -----------------------------
# Save predictions from best model
# -----------------------------
predictions = pd.DataFrame({
    "actual_dwell_hours": y_test,
    "predicted_dwell_hours": best["pred"],
    "error_hours": best["pred"] - y_test
})

predictions.to_csv("/Users/kellyg/eurogate-twin-1/feature_importance/total_dwell_predictions_updated.csv", index=False)

print("\nSaved:")
print("/Users/kellyg/eurogate-twin-1/feature_importance/total_dwell_model_comparison.csv")
print("/Users/kellyg/eurogate-twin-1/feature_importance/total_dwell_predictions_updated.csv")