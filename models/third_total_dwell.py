import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional models
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


df = pd.read_csv("data_final/eurogate_postdeparture.csv")

TARGET = "dwell_hours"

# DROP LEAKAGE / NON-MODEL COLUMNS
drop_cols = [
    TARGET,
    "snapshot_time",
    "snapshot_date",
    "snapshot_year",
    "snapshot_hour",
    "snapshot_minute",
    "snapshot_second",
    "arrival_year",
    "Unnamed: 0",
    "Unnamed: 0.1",
    "Unnamed: 0.2",
    "departureTime",
    "remaining_dwell_hours",
    "elapsed_dwell_hours",
]

df = df.dropna(subset=[TARGET]).copy()

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[TARGET]

# CLEAN FEATURES
X = X.replace([np.inf, -np.inf], np.nan)

object_cols = X.select_dtypes(include=["object"]).columns.tolist()
print("Dropping object columns:", object_cols)
X = X.drop(columns=object_cols)

bool_cols = X.select_dtypes(include=["bool"]).columns
X[bool_cols] = X[bool_cols].astype(int)

X = X.fillna(X.median(numeric_only=True))

# Remove extreme top 1% dwell outliers
upper = y.quantile(0.99)
mask = y <= upper

X = X.loc[mask].copy()
y = y.loc[mask].copy()

# Log target
y_log = np.log1p(y)

# TRAIN/TEST SPLIT
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X,
    y_log,
    test_size=0.2,
    random_state=100
)

y_test = np.expm1(y_test_log)

# EVALUATION FUNCTION
def evaluate_model(name, model, X_train, X_test, y_train_log, y_test_log):
    model.fit(X_train, y_train_log)

    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)
    pred = np.maximum(pred, 0)

    y_test = np.expm1(y_test_log)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"MAE:  {mae:.2f} hours")
    print(f"RMSE: {rmse:.2f} hours")
    print(f"R²:   {r2:.4f}")
    print(f"Within 6 hours:  {(np.abs(y_test - pred) <= 6).mean():.2%}")
    print(f"Within 12 hours: {(np.abs(y_test - pred) <= 12).mean():.2%}")
    print(f"Within 24 hours: {(np.abs(y_test - pred) <= 24).mean():.2%}")

    return {
        "model_name": name,
        "model": model,
        "pred": pred,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# STEP 1: INITIAL EXTRA TREES FOR FEATURE IMPORTANCE
base_et = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

base_result = evaluate_model(
    "Initial Extra Trees",
    base_et,
    X_train,
    X_test,
    y_train_log,
    y_test_log
)

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": base_et.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(
    "/Users/kellyg/eurogate-twin-1/feature_importance/initial_extra_trees_importance.csv",
    index=False
)

print("\nTop 30 initial features:")
print(importance.head(30))

# STEP 2: REMOVE DEAD FEATURES
IMPORTANCE_THRESHOLD = 0.001

keep_features = importance.loc[
    importance["importance"] >= IMPORTANCE_THRESHOLD,
    "feature"
].tolist()

print(f"\nOriginal feature count: {X.shape[1]}")
print(f"Kept feature count: {len(keep_features)}")
print(f"Dropped feature count: {X.shape[1] - len(keep_features)}")

X_reduced = X[keep_features].copy()

X_train_r, X_test_r, y_train_log_r, y_test_log_r = train_test_split(
    X_reduced,
    y_log,
    test_size=0.2,
    random_state=42
)

# STEP 3: TEST MORE MODELS ON REDUCED FEATURE SET
models = [
    ("Baseline Median", DummyRegressor(strategy="median")),

    ("Random Forest", RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )),

    ("Extra Trees", ExtraTreesRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )),

    ("Hist Gradient Boosting", HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.04,
        max_leaf_nodes=63,
        l2_regularization=0.05,
        random_state=42
    )),
]

if HAS_XGB:
    models.append((
        "XGBoost",
        XGBRegressor(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    ))

if HAS_LGBM:
    models.append((
        "LightGBM",
        LGBMRegressor(
            n_estimators=800,
            learning_rate=0.04,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
    ))

results = []

for name, model in models:
    results.append(
        evaluate_model(
            name,
            model,
            X_train_r,
            X_test_r,
            y_train_log_r,
            y_test_log_r
        )
    )

summary = pd.DataFrame([
    {
        "model": r["model_name"],
        "MAE": r["mae"],
        "RMSE": r["rmse"],
        "R2": r["r2"]
    }
    for r in results
]).sort_values("MAE")

print("\nModel comparison on reduced feature set:")
print(summary)

summary.to_csv(
    "/Users/kellyg/eurogate-twin-1/feature_importance/reduced_feature_model_comparison.csv",
    index=False
)

# STEP 4: FINETUNE BEST MODEL FAMILY: EXTRA TREES
param_dist = {
    "n_estimators": [300, 500, 800, 1000],
    "max_depth": [None, 20, 30, 40],
    "min_samples_leaf": [1, 2, 3, 5],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", 0.5, 0.75, 1.0],
    "bootstrap": [False, True]
}

et_search = RandomizedSearchCV(
    estimator=ExtraTreesRegressor(
        random_state=42,
        n_jobs=-1
    ),
    param_distributions=param_dist,
    n_iter=25,
    scoring="neg_mean_absolute_error",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

et_search.fit(X_train_r, y_train_log_r)

print("\nBest Extra Trees parameters:")
print(et_search.best_params_)

best_et = et_search.best_estimator_

best_et_result = evaluate_model(
    "Tuned Extra Trees",
    best_et,
    X_train_r,
    X_test_r,
    y_train_log_r,
    y_test_log_r
)

# =====================================================
# STEP 5: SAVE FINAL RESULTS
# =====================================================

final_importance = pd.DataFrame({
    "feature": keep_features,
    "importance": best_et.feature_importances_
}).sort_values("importance", ascending=False)

final_importance.to_csv(
    "/Users/kellyg/eurogate-twin-1/feature_importance/tuned_extra_trees_feature_importance.csv",
    index=False
)

final_predictions = pd.DataFrame({
    "actual_dwell_hours": np.expm1(y_test_log_r),
    "predicted_dwell_hours": best_et_result["pred"],
    "error_hours": best_et_result["pred"] - np.expm1(y_test_log_r)
})

final_predictions.to_csv(
    "/Users/kellyg/eurogate-twin-1/feature_importance/tuned_extra_trees_predictions.csv",
    index=False
)

final_summary = pd.concat([
    summary,
    pd.DataFrame([{
        "model": "Tuned Extra Trees",
        "MAE": best_et_result["mae"],
        "RMSE": best_et_result["rmse"],
        "R2": best_et_result["r2"]
    }])
]).sort_values("MAE")

final_summary.to_csv(
    "/Users/kellyg/eurogate-twin-1/feature_importance/final_total_dwell_model_comparison.csv",
    index=False
)

print("\nSaved:")
print("/Users/kellyg/eurogate-twin-1/feature_importance/initial_extra_trees_importance.csv")
print("/Users/kellyg/eurogate-twin-1/feature_importance/reduced_feature_model_comparison.csv")
print("/Users/kellyg/eurogate-twin-1/feature_importance/tuned_extra_trees_feature_importance.csv")
print("/Users/kellyg/eurogate-twin-1/feature_importance/tuned_extra_trees_predictions.csv")
print("/Users/kellyg/eurogate-twin-1/feature_importance/final_total_dwell_model_comparison.csv")