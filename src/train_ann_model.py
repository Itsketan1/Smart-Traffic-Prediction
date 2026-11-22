import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

tf.keras.utils.set_random_seed(42)

DATA = Path("data/real_world/real_traffic_5min.csv")
vehicles = ["car", "motorbike", "bus", "truck"]
groups = ["intersection", "camera", "date", "Direction"]

df = pd.read_csv(DATA)
df["timestamp"] = pd.to_datetime(df["timestamp"])

frames = []
for _, g in df.groupby(groups, sort=False):
    g = g.sort_values("timestamp").copy()

    for col in vehicles + ["total_vehicles"]:
        g[f"{col}_lag1"] = g[col].shift(1)
        g[f"{col}_lag2"] = g[col].shift(2)

    g["target"] = g["total_vehicles"].shift(-1)
    next_time = g["timestamp"].shift(-1)
    g = g[(next_time - g["timestamp"]) == pd.Timedelta(minutes=5)]
    frames.append(g)

data = pd.concat(frames, ignore_index=True).dropna()
data = data.sort_values("timestamp").reset_index(drop=True)

data["hour"] = data["timestamp"].dt.hour
data["minute"] = data["timestamp"].dt.minute
data["day_of_week"] = data["timestamp"].dt.day_of_week

numeric = (
    vehicles + ["total_vehicles"] +
    [f"{c}_lag{i}" for c in vehicles + ["total_vehicles"] for i in (1, 2)] +
    ["hour", "minute", "day_of_week"]
)
categorical = ["intersection", "camera", "Direction"]
features = numeric + categorical

cutoff = data["timestamp"].quantile(0.8)
train = data[data["timestamp"] < cutoff].copy()
test = data[data["timestamp"] >= cutoff].copy()

# Chronological validation split from training data.
validation_cutoff = train["timestamp"].quantile(0.85)
fit = train[train["timestamp"] < validation_cutoff]
val = train[train["timestamp"] >= validation_cutoff]

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric),
    ("categorical", OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    ), categorical)
])

X_fit = preprocessor.fit_transform(fit[features]).astype("float32")
X_val = preprocessor.transform(val[features]).astype("float32")
X_test = preprocessor.transform(test[features]).astype("float32")

# Log-transform the target to handle skewed traffic volumes.
y_fit = np.log1p(fit["target"].to_numpy(dtype="float32"))
y_val = np.log1p(val["target"].to_numpy(dtype="float32"))
y_test = test["target"].to_numpy(dtype="float32")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_fit.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.Huber(),
    metrics=["mae"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=5, factor=0.5
    )
]

print("Training:", len(X_fit))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

history = model.fit(
    X_fit, y_fit,
    validation_data=(X_val, y_val),
    epochs=120,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

pred = np.maximum(0, np.expm1(model.predict(X_test, verbose=0).ravel()))

metrics = {
    "model": "Improved ANN - Log Target and Huber Loss",
    "mae": float(mean_absolute_error(y_test, pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    "r2": float(r2_score(y_test, pred)),
    "train_records": len(train),
    "test_records": len(test),
    "test_start": str(cutoff)
}

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

model.save("models/ann_traffic_forecaster.keras")
joblib.dump(preprocessor, "models/ann_preprocessor.joblib")
joblib.dump(features, "models/ann_feature_columns.joblib")
joblib.dump(history.history, "models/ann_training_history.joblib")

with open("results/ann_model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

predictions = test[groups + ["timestamp"]].copy()
predictions["actual_next_5min"] = y_test
predictions["ann_prediction"] = pred
predictions.to_csv("results/ann_model_predictions.csv", index=False)

rf = json.load(open("results/real_model_metrics.json"))

print("\nIMPROVED ANN EVALUATION")
for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nFINAL MODEL COMPARISON")
print(f"ANN MAE: {metrics['mae']:.2f} | RF MAE: {rf['mae']:.2f}")
print(f"ANN RMSE: {metrics['rmse']:.2f} | RF RMSE: {rf['rmse']:.2f}")
print(f"ANN R2: {metrics['r2']:.3f} | RF R2: {rf['r2']:.3f}")
