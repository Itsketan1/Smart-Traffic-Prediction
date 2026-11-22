from pathlib import Path
import re
import pandas as pd

ROOT = Path("data/real_world/traffic")
OUTPUT = Path("data/real_world/real_traffic_5min.csv")
VEHICLES = ["car", "motorbike", "bus", "truck"]

records = []
errors = []

files = sorted(ROOT.rglob("*.csv"))
print(f"Found {len(files)} CSV files")

for file in files:
    try:
        match = re.search(r"_(\d{1,2})(?:_one)?$", file.stem)
        if not match:
            continue

        day = int(match.group(1))
        if not 1 <= day <= 31:
            continue

        camera = file.stem.split("_")[0]
        intersection = file.parents[1].name
        date = pd.Timestamp(year=2023, month=1, day=day)

        df = pd.read_csv(file, usecols=["Time", "Direction"] + VEHICLES)
        df = df.dropna(subset=["Time", "Direction"])

        for col in VEHICLES:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=VEHICLES)

        # Reduce repeated video frames to one reading per second/direction
        sec = df.groupby(["Time", "Direction"], as_index=False)[VEHICLES].median()

        sec["timestamp"] = pd.to_datetime(
            date.strftime("%Y-%m-%d") + " " + sec["Time"],
            errors="coerce"
        )
        sec = sec.dropna(subset=["timestamp"])

        # Calculate vehicle increments independently within each CSV file.
        # The first reading is a counter baseline, not a new vehicle count.
        sec = sec.sort_values(["Direction", "timestamp"]).copy()

        for col in VEHICLES:
            change = sec.groupby("Direction")[col].diff()
            sec[col] = change.where(change >= 0, sec[col]).fillna(0)

        sec["intersection"] = intersection
        sec["camera"] = camera
        sec["date"] = date.date()

        records.append(
            sec[["intersection", "camera", "date",
                 "timestamp", "Direction"] + VEHICLES]
        )

        print("Read:", file.relative_to(ROOT))

    except Exception as e:
        errors.append((str(file), str(e)))
        print("Skipped:", file.name, e)

if not records:
    raise RuntimeError("No usable CSV records found.")

print("\nCombining camera recordings...")
all_sec = pd.concat(records, ignore_index=True)

keys = ["intersection", "camera", "date", "timestamp", "Direction"]

# Merge readings sharing the same second across recording segments
all_sec = (
    all_sec.groupby(keys, as_index=False)[VEHICLES]
           .median()
           .sort_values(keys)
)

print("Unique one-second readings:", len(all_sec))

# Counter differences have already been calculated within each CSV file.
group_keys = ["intersection", "camera", "date", "Direction"]

all_sec["total_vehicles"] = all_sec[VEHICLES].sum(axis=1)

# Aggregate into 5-minute traffic volumes
all_sec = all_sec.set_index("timestamp")

five = (
    all_sec.groupby(group_keys)[VEHICLES + ["total_vehicles"]]
           .resample("5min")
           .sum()
           .reset_index()
)

five = five.sort_values(
    ["intersection", "camera", "date", "Direction", "timestamp"]
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
five.to_csv(OUTPUT, index=False)

print("\nPREPROCESSING COMPLETE")
print("Output:", OUTPUT)
print("5-minute records:", len(five))
print("Intersections:", five["intersection"].nunique())
print("Cameras:", five["camera"].nunique())
print("Directions:", sorted(five["Direction"].unique()))
print("Date range:", five["date"].min(), "to", five["date"].max())

print("\nTraffic volume summary:")
print(five[VEHICLES + ["total_vehicles"]].describe().round(2))

if errors:
    print("\nFiles with errors:", len(errors))
    for name, error in errors[:10]:
        print(name, error)
