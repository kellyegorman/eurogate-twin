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

df = pd.read_csv("data/eurogate_train_encoded.csv")

drop_cols = [
    "dwell_hours",
    "snapshot_time",
    "snapshot_date",
    "arrival_year",
    "snapshot_year"
]

df = df.drop(
    columns=[c for c in drop_cols if c in df.columns]
)

df = df.dropna(subset=["remaining_dwell_hours"])

object_cols = df.select_dtypes(include="object").columns

if len(object_cols):
    print("Dropping:", list(object_cols))
    df = df.drop(columns=object_cols)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median(numeric_only=True))

X = df.drop(columns=["remaining_dwell_hours"])

y = df["remaining_dwell_hours"]

# log-transform target
y_log = np.log1p(y)

# Train/Test split
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X,
    y_log,
    test_size=0.20,
    random_state=42
)

y_test = np.expm1(y_test_log)

# Evaluation function
def evaluate(name, model):

    model.fit(X_train, y_train_log)

    pred_log = model.predict(X_test)

    pred = np.expm1(pred_log)

    pred = np.maximum(pred, 0)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print("\n" + "="*50)
    print(name)
    print("="*50)

    print(f"MAE:  {mae:.2f} hours")
    print(f"RMSE: {rmse:.2f} hours")
    print(f"R²:   {r2:.4f}")

    print(
        f"Within 1 hr : {(abs(pred-y_test)<=1).mean():.2%}"
    )

    print(
        f"Within 6 hr : {(abs(pred-y_test)<=6).mean():.2%}"
    )

    print(
        f"Within 12 hr: {(abs(pred-y_test)<=12).mean():.2%}"
    )

    print(
        f"Within 24 hr: {(abs(pred-y_test)<=24).mean():.2%}"
    )

    return model

# Baseline
baseline = DummyRegressor(strategy="median")

evaluate(
    "Baseline",
    baseline
)

# Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=25,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

rf_model = evaluate(
    "Random Forest",
    rf
)

# Feature Importance
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nTop 30 Features:")
print(importance.head(30))

importance.to_csv(
    "feature_importance_remaining_dwell.csv",
    index=False
)

# print quality metrics for top 3 features only
top_features = importance.head(3)["feature"].tolist()