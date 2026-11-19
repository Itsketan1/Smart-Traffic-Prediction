
import numpy as np
import pandas as pd
import os

# Reproducibility
np.random.seed(42)

# Number of records
n = 5000

# Generate input features
hour = np.random.randint(0, 24, n)
day_of_week = np.random.randint(0, 7, n)

vehicle_count = []

for h, day in zip(hour, day_of_week):

    # Base traffic
    base = 25

    # Morning and evening rush hours
    if 7 <= h <= 10:
        base += 65
    elif 16 <= h <= 20:
        base += 75

    # Afternoon traffic
    elif 11 <= h <= 15:
        base += 25

    # Weekend traffic variation
    if day >= 5:
        base += 5

    # Add realistic random variation
    count = np.random.normal(base, 15)

    vehicle_count.append(max(0, int(count)))

vehicle_count = np.array(vehicle_count)

# Generate congestion score
rush_hour = np.isin(hour, [7, 8, 9, 10, 16, 17, 18, 19, 20])

congestion_score = (
    vehicle_count
    + rush_hour * 15
    + np.random.normal(0, 5, n)
)

# Assign traffic categories
traffic_level = []

for score in congestion_score:

    if score < 60:
        traffic_level.append("Low")

    elif score < 110:
        traffic_level.append("Medium")

    else:
        traffic_level.append("High")

# Create DataFrame
df = pd.DataFrame({
    "hour": hour,
    "day_of_week": day_of_week,
    "vehicle_count": vehicle_count,
    "traffic_level": traffic_level
})

# Save dataset
os.makedirs("data", exist_ok=True)

df.to_csv("data/traffic_dataset.csv", index=False)

print("Dataset generated successfully!")
print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nTraffic category distribution:")
print(df["traffic_level"].value_counts())

print("\nDataset saved at data/traffic_dataset.csv")
