import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/eurogate_train_encoded.csv")

df = df.drop(columns=[c for c in ["Unnamed: 0", "Unnamed: 0.1"] if c in df.columns])
df = df.dropna(subset=["dwell_hours"])
object_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("Dropping object columns:", object_cols)
df = df.drop(columns=object_cols)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median(numeric_only=True))
X = df.drop(columns=["dwell_hours"])
y = df["dwell_hours"]
y_log = np.log1p(y)
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X,
    y_log,
    test_size=0.2,
    random_state=42
)
y_test = np.expm1(y_test_log)

def evaluate_model(name, model):
    model.fit(X_train, y_train_log)

    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)

    pred = np.maximum(pred, 0)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    within_1h = (np.abs(y_test - pred) <= 1).mean()
    within_6h = (np.abs(y_test - pred) <= 6).mean()
    within_12h = (np.abs(y_test - pred) <= 12).mean()
    within_24h = (np.abs(y_test - pred) <= 24).mean()

    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE: {mae:.2f} hours")
    print(f"RMSE: {rmse:.2f} hours")
    print(f"R²: {r2:.4f}")
    print(f"Within 1 hour: {within_1h:.2%}")
    print(f"Within 6 hours: {within_6h:.2%}")
    print(f"Within 12 hours: {within_12h:.2%}")
    print(f"Within 24 hours: {within_24h:.2%}")

    return model, pred

baseline = DummyRegressor(strategy="median")
baseline_model, baseline_pred = evaluate_model("Baseline: Predict Median", baseline)

ridge = Ridge(alpha=1.0)
ridge_model, ridge_pred = evaluate_model("Ridge Regression", ridge)

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model, rf_pred = evaluate_model("Random Forest", rf)

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 25 Random Forest Features:")
print(importance.head(25))

importance.to_csv("data/feature_importance_random_forest.csv", index=False)
print("\nSaved feature importance to data/feature_importance_random_forest.csv")