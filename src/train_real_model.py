from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path("data/real_world/real_traffic_5min.csv")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").copy()

vehicles = ["car", "motorbike", "bus", "truck"]
groups = ["intersection", "camera", "date", "Direction"]

# Create lag features separately for each camera and direction.
feature_frames = []

for _, g in df.groupby(groups, sort=False):
    g = g.sort_values("timestamp").copy()

    for col in vehicles + ["total_vehicles"]:
        g[f"{col}_lag1"] = g[col].shift(1)
        g[f"{col}_lag2"] = g[col].shift(2)

    g["target"] = g["total_vehicles"].shift(-1)
    g["next_timestamp"] = g["timestamp"].shift(-1)

    # Do not train across missing five-minute intervals.
    valid_next = (
        g["next_timestamp"] - g["timestamp"]
    ) == pd.Timedelta(minutes=5)

    g = g[valid_next].copy()
    feature_frames.append(g)

data = pd.concat(feature_frames, ignore_index=True)
data = data.dropna()

data["hour"] = data["timestamp"].dt.hour
data["minute"] = data["timestamp"].dt.minute
data["day_of_week"] = data["timestamp"].dt.dayofweek

numeric = (
    vehicles + ["total_vehicles"] +
    [f"{c}_lag{i}" for c in vehicles + ["total_vehicles"] for i in (1, 2)] +
    ["hour", "minute", "day_of_week"]
)
categorical = ["intersection", "camera", "Direction"]

features = numeric + categorical
X = data[features]
y = data["target"]

# Chronological split: earliest 80% for training, latest 20% for testing.
cutoff = data["timestamp"].quantile(0.8)
train_mask = data["timestamp"] < cutoff
test_mask = ~train_mask

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

if len(X_train) < 100 or len(X_test) < 20:
    raise RuntimeError("Not enough usable data for a reliable train/test split.")

preprocessor = ColumnTransformer([
    ("categories", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Test period begins:", cutoff)

model.fit(X_train, y_train)
pred = np.maximum(0, model.predict(X_test))

# Persistence baseline: next interval is predicted to equal current interval.
baseline = X_test["total_vehicles"].to_numpy()

metrics = {
    "model": "RandomForestRegressor",
    "target": "next_5_minute_total_vehicles",
    "train_records": int(len(X_train)),
    "test_records": int(len(X_test)),
    "test_start": str(cutoff),
    "mae": float(mean_absolute_error(y_test, pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    "r2": float(r2_score(y_test, pred)),
    "baseline_mae": float(mean_absolute_error(y_test, baseline)),
    "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline))),
    "baseline_r2": float(r2_score(y_test, baseline))
}

joblib.dump(model, MODEL_DIR / "real_traffic_forecaster.joblib")

with open(RESULT_DIR / "real_model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

predictions = data.loc[test_mask, groups + ["timestamp"]].copy()
predictions["actual_next_5min"] = y_test.to_numpy()
predictions["predicted_next_5min"] = pred
predictions.to_csv(RESULT_DIR / "real_model_predictions.csv", index=False)

print("\nMODEL EVALUATION")
for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nSaved model: models/real_traffic_forecaster.joblib")
print("Saved metrics: results/real_model_metrics.json")
print("Saved predictions: results/real_model_predictions.csv")
