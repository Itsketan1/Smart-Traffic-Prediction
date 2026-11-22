
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Traffic Prediction",
    page_icon="🚦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data/real_world/real_traffic_5min.csv"
ANN_PATH = BASE_DIR / "models/ann_traffic_forecaster.keras"
PREPROCESSOR_PATH = BASE_DIR / "models/ann_preprocessor.joblib"
FEATURES_PATH = BASE_DIR / "models/ann_feature_columns.joblib"
HISTORY_PATH = BASE_DIR / "models/ann_training_history.joblib"
ANN_METRICS_PATH = BASE_DIR / "results/ann_model_metrics.json"
RF_METRICS_PATH = BASE_DIR / "results/real_model_metrics.json"


# --------------------------------------------------
# Load models and data
# --------------------------------------------------
@st.cache_resource
def load_ann():
    model = tf.keras.models.load_model(ANN_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, preprocessor, feature_columns


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    # Support common timestamp column names
    timestamp_candidates = [
        "timestamp", "Time", "datetime", "date_time"
    ]
    timestamp_col = next(
        (col for col in timestamp_candidates if col in data.columns),
        None
    )

    if timestamp_col is None:
        raise ValueError(
            "No timestamp column found. Check the processed CSV column names."
        )

    data["timestamp"] = pd.to_datetime(
        data[timestamp_col], errors="coerce"
    )
    data = data.dropna(subset=["timestamp"]).copy()

    # Normalize direction column
    if "Direction" not in data.columns and "direction" in data.columns:
        data["Direction"] = data["direction"]

    required = [
        "intersection", "camera", "Direction",
        "car", "motorbike", "bus", "truck"
    ]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    for col in ["car", "motorbike", "bus", "truck"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    data["total_vehicles"] = data[
        ["car", "motorbike", "bus", "truck"]
    ].sum(axis=1)

    data = data.sort_values("timestamp").copy()
    data["date"] = data["timestamp"].dt.date
    data["hour"] = data["timestamp"].dt.hour
    data["minute"] = data["timestamp"].dt.minute
    data["day_of_week"] = data["timestamp"].dt.dayofweek

    # Match the lag feature construction used during training
    group_cols = ["intersection", "camera", "date", "Direction"]

    for col in ["car", "motorbike", "bus", "truck", "total_vehicles"]:
        data[f"{col}_lag1"] = (
            data.groupby(group_cols)[col].shift(1)
        )
        data[f"{col}_lag2"] = (
            data.groupby(group_cols)[col].shift(2)
        )

    return data


def show_error(message):
    st.error(message)
    st.stop()


# --------------------------------------------------
# Load project files
# --------------------------------------------------
st.title("🚦 Smart Traffic Prediction")
st.caption(
    "Artificial Neural Network (ANN) based traffic-volume forecasting"
)

if not DATA_PATH.exists():
    show_error(f"Dataset not found: {DATA_PATH}")

if not ANN_PATH.exists():
    show_error(f"ANN model not found: {ANN_PATH}")

try:
    data = load_data()
    ann_model, preprocessor, feature_columns = load_ann()
except Exception as error:
    show_error(f"Could not load project files: {error}")

if not ANN_METRICS_PATH.exists():
    show_error("ANN metrics file is missing.")

with open(ANN_METRICS_PATH, "r") as file:
    ann_metrics = json.load(file)

rf_metrics = None
if RF_METRICS_PATH.exists():
    with open(RF_METRICS_PATH, "r") as file:
        rf_metrics = json.load(file)


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Traffic Data Filters")

intersections = sorted(data["intersection"].dropna().unique())
selected_intersection = st.sidebar.selectbox(
    "Select intersection", intersections
)

intersection_data = data[
    data["intersection"] == selected_intersection
]

cameras = sorted(intersection_data["camera"].dropna().unique())
selected_camera = st.sidebar.selectbox("Select camera", cameras)

camera_data = intersection_data[
    intersection_data["camera"] == selected_camera
]

directions = sorted(camera_data["Direction"].dropna().unique())
selected_direction = st.sidebar.selectbox(
    "Select direction", directions
)

stream = camera_data[
    camera_data["Direction"] == selected_direction
].copy()

available_dates = sorted(stream["date"].dropna().unique())

if not available_dates:
    show_error("No records are available for this selection.")

selected_date = st.sidebar.selectbox(
    "Select date",
    available_dates,
    index=len(available_dates) - 1
)

stream = stream[stream["date"] == selected_date].copy()
stream = stream.sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------
# Selected traffic data
# --------------------------------------------------
st.subheader("Traffic Overview")

latest = stream.iloc[-1]

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Cars", f"{latest['car']:.0f}")
metric2.metric("Motorbikes", f"{latest['motorbike']:.0f}")
metric3.metric("Buses", f"{latest['bus']:.0f}")
metric4.metric("Trucks", f"{latest['truck']:.0f}")

st.metric(
    "Latest recorded vehicle volume",
    f"{latest['total_vehicles']:.0f} vehicles"
)

st.caption(
    f"Selected: {selected_intersection} | Camera: {selected_camera} | "
    f"Direction: {selected_direction} | Date: {selected_date}"
)

st.subheader("Traffic Volume Over Time")

chart_data = stream.set_index("timestamp")[
    ["car", "motorbike", "bus", "truck", "total_vehicles"]
]
st.line_chart(chart_data)


# --------------------------------------------------
# ANN prediction
# --------------------------------------------------
st.subheader("🧠 ANN Traffic Forecast")

st.write(
    "The ANN uses current traffic counts, previous observations, "
    "time information, and camera/intersection details to forecast "
    "the next five-minute traffic volume."
)

if len(stream) < 3:
    st.warning(
        "At least three records are required to generate a prediction "
        "because the model uses two previous observations."
    )
else:
    prediction_row = stream.iloc[-1].copy()

    if prediction_row[feature_columns].isnull().any():
        st.warning(
            "The latest record does not have enough previous observations. "
            "Select another date or stream."
        )
    else:
        input_df = pd.DataFrame(
            [prediction_row[feature_columns].to_dict()],
            columns=feature_columns
        )

        try:
            transformed = preprocessor.transform(input_df)
            log_prediction = ann_model.predict(
                transformed, verbose=0
            ).reshape(-1)[0]

            predicted_vehicles = max(0, float(np.expm1(log_prediction)))

            st.success("ANN prediction generated successfully.")

            prediction_col, info_col = st.columns(2)

            with prediction_col:
                st.metric(
                    "Predicted vehicles in next 5 minutes",
                    f"{predicted_vehicles:.1f}"
                )

            with info_col:
                st.metric(
                    "Latest observed 5-minute volume",
                    f"{latest['total_vehicles']:.1f}"
                )

            st.caption(
                "Forecast is based on the latest available five-minute "
                "record and the preceding observations."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# Model performance comparison
# --------------------------------------------------
st.divider()
st.subheader("📊 Model Performance")

if rf_metrics:
    comparison = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "ANN": [
            ann_metrics["mae"],
            ann_metrics["rmse"],
            ann_metrics["r2"]
        ],
        "Random Forest": [
            rf_metrics["mae"],
            rf_metrics["rmse"],
            rf_metrics["r2"]
        ]
    })

    st.dataframe(
        comparison.style.format({
            "ANN": "{:.3f}",
            "Random Forest": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    chart_comparison = comparison.set_index("Metric")
    st.bar_chart(chart_comparison[["ANN", "Random Forest"]])

    st.caption(
        "Lower MAE and RMSE are better. Higher R² is better. "
        "Both models were evaluated using the same chronological "
        "train/test split."
    )
else:
    st.info("Random Forest comparison metrics are not available.")


# --------------------------------------------------
# ANN training history
# --------------------------------------------------
st.divider()
st.subheader("📉 ANN Training History")

if HISTORY_PATH.exists():
    try:
        history = joblib.load(HISTORY_PATH)

        if hasattr(history, "history"):
            history = history.history

        if isinstance(history, dict):
            loss_key = "loss"
            val_loss_key = "val_loss"

            if loss_key in history:
                loss_df = pd.DataFrame({
                    "Training Loss": history[loss_key]
                })

                if val_loss_key in history:
                    loss_df["Validation Loss"] = history[val_loss_key]

                loss_df.index = range(1, len(loss_df) + 1)
                loss_df.index.name = "Epoch"

                st.line_chart(loss_df)
            else:
                st.info("Training loss data was not found in the history file.")
        else:
            st.info("Training history format is not recognized.")

    except Exception as error:
        st.warning(f"Could not display training history: {error}")
else:
    st.info("ANN training history file is not available.")


# --------------------------------------------------
# Model details and disclaimer
# --------------------------------------------------
with st.expander("About the ANN model"):
    st.write("""
    **Model architecture**

    - Input: 21 traffic and time-related features
    - Dense layer: 64 neurons, ReLU activation
    - Dropout: 0.15
    - Dense layer: 32 neurons, ReLU activation
    - Dense layer: 16 neurons, ReLU activation
    - Output: 1 neuron, linear activation

    **Training approach**

    - Log-transformed prediction target
    - Huber loss function
    - Adam optimizer
    - Early stopping and learning-rate reduction
    - Chronological training and testing split
    """)

with st.expander("Dataset and project limitations"):
    st.write("""
    This project uses historical traffic-count data collected at
    three Pune intersections during January 11–18, 2023.

    The dashboard forecasts traffic volume from historical observations.
    It does not connect to live CCTV cameras, live traffic sensors,
    or real-time traffic feeds.

    Predictions are experimental estimates and should not be used
    as the sole basis for real-world traffic-control decisions.
    """)

st.caption(
    "Smart Traffic Prediction | ANN-based academic forecasting project"
)
