# Smart Traffic Prediction Using Artificial Neural Networks

An academic machine learning project that forecasts short-term traffic volume using historical traffic-count data from Pune, India.

The project uses an Artificial Neural Network (ANN) as its primary forecasting model and compares its performance with a Random Forest regression model. A Streamlit dashboard provides an interactive interface for exploring traffic observations and generating forecasts.

## 1. Project Objectives

- Analyze historical traffic volume at multiple intersections.
- Study vehicle counts by vehicle category and direction.
- Develop an ANN model to forecast traffic volume for the next five-minute interval.
- Compare ANN performance with a Random Forest benchmark.
- Visualize traffic patterns, model performance, and training history through Streamlit.

## 2. Dataset

The project uses the Heterogeneous Traffic Count Dataset, Pune (January 2023).

Dataset characteristics:

- Location: Pune, Maharashtra, India
- Collection period: January 11–18, 2023
- Intersections: AlankarChowk, JehangirChowk, RTOChowk
- Camera streams: 8
- Directions: UP, DOWN, LEFT, RIGHT
- Vehicle categories: Car, Motorbike, Bus, Truck
- Original CSV files processed: 115
- Unique one-second readings: approximately 2.76 million
- Processed five-minute records: 9,528

The preprocessing pipeline combines observations, handles repeated readings, calculates vehicle-count increments, and aggregates the observations into five-minute intervals.

Processed dataset:
`data/real_world/real_traffic_5min.csv`

## 3. Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib
- Streamlit

## 4. Project Structure

```text
Smart-Traffic-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── real_world/
│   │   └── real_traffic_5min.csv
│   ├── traffic_dataset.csv
│   └── processed_data.npz
│
├── models/
│   ├── ann_traffic_forecaster.keras
│   ├── ann_preprocessor.joblib
│   ├── ann_feature_columns.joblib
│   ├── ann_training_history.joblib
│   └── real_traffic_forecaster.joblib
│
├── results/
│   ├── ann_model_metrics.json
│   ├── ann_model_predictions.csv
│   ├── real_model_metrics.json
│   └── real_model_predictions.csv
│
├── screenshots/
│
└── src/
    ├── preprocess.py
    ├── preprocess_real.py
    ├── train_ann_model.py
    ├── train_real_model.py
    ├── train_model.py
    ├── evaluate_model.py
    └── visualize_results.py
```

Additional experimental model files are retained in the repository.

## 5. Data Preprocessing

The real-world preprocessing pipeline performs the following operations:

1. Reads the traffic-count CSV files.
2. Combines readings from the available camera streams.
3. Handles repeated frame readings using median aggregation.
4. Calculates vehicle-count increments from cumulative counters.
5. Aggregates the observations into five-minute intervals.
6. Creates time-related and historical lag features.
7. Prepares the dataset for chronological model training and testing.

## 6. ANN Model

The ANN is the primary forecasting model.

### Input features

The model uses 21 features, including:

- Current car, motorbike, bus, and truck counts.
- Total vehicle count.
- Previous two observations (lag features).
- Hour, minute, and day of the week.
- Intersection, camera, and traffic direction.

### Architecture

| Layer | Configuration |
|---|---|
| Input | 21 features after preprocessing |
| Dense | 64 neurons, ReLU |
| Dropout | 0.15 |
| Dense | 32 neurons, ReLU |
| Dense | 16 neurons, ReLU |
| Output | 1 neuron, linear activation |

### Training configuration

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Huber loss
- Target transformation: Log1p
- Early stopping
- Reduce learning rate on plateau
- Chronological training and testing split

The ANN was trained using 7,004 records and evaluated on 1,756 test records.

## 7. Model Evaluation

The ANN was evaluated against a Random Forest regression benchmark using the same chronological test period.

### Results

| Metric | ANN | Random Forest |
|---|---:|---:|
| MAE | 30.94 | 30.06 |
| RMSE | 199.30 | 133.14 |
| R² | 0.108 | 0.602 |

### Interpretation

- MAE measures the average absolute prediction error.
- RMSE gives greater weight to large prediction errors.
- R² measures the proportion of target variance explained by the model.

The Random Forest model performed better on all three reported metrics in the current evaluation.

The ANN remains the primary model developed and demonstrated in this project. The comparison is included to provide an honest assessment of its performance.

## 8. Streamlit Dashboard

The interactive dashboard provides:

- Intersection selection.
- Camera selection.
- Traffic direction selection.
- Date selection.
- Vehicle-count overview.
- Historical traffic-volume visualization.
- ANN next-five-minute traffic forecast.
- ANN versus Random Forest performance comparison.
- ANN training and validation loss visualization.
- Model architecture and dataset limitations.

Run the dashboard using:

```bash
streamlit run app.py
```

## 9. Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Smart-Traffic-Prediction.git
cd Smart-Traffic-Prediction
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / GitHub Codespaces:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Ensure the processed dataset and trained model files are available at their expected paths.

## 10. Limitations

- The dataset contains historical observations from January 2023.
- The application does not connect to live CCTV cameras or real-time traffic sensors.
- Predictions are estimates based on historical observations.
- Performance may vary across intersections, camera streams, and traffic conditions.
- The current ANN has higher RMSE and lower R² than the Random Forest benchmark.

## 11. Future Scope

- Integrate live traffic sensor or CCTV-derived data.
- Collect larger datasets across different seasons and times.
- Experiment with LSTM and other time-series forecasting architectures.
- Improve prediction accuracy through feature engineering and hyperparameter tuning.
- Explore deployment on a cloud platform.
- Extend the system to additional intersections.

## 12. Academic Project Disclaimer

This project is developed for academic learning and demonstration. It is a historical-data-based traffic forecasting prototype and is not a live traffic monitoring or traffic-control system.
