import os
import pandas as pd

csv_file = "2025-02-19_16-44-43_metr-la_pred-15min_his-60min_centralized_analysis-neighbours-and-no-neighbours/val_metric/delta.csv"

print(f"{__file__}")

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

path = os.path.join(base_dir, 'logs', csv_file)
print(f"Processing: {path}")

df = pd.read_csv(path, header=None)  # Assuming no headers
df = df.T
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Flatten the DataFrame and get the top 10 highest values with their indices
stacked = df.stack()
top_10 = stacked.nlargest(10)

# Output results in the requested format
for (sensor_id, datastep), value in top_10.items():
    print(f"SensorId: {sensor_id} | {datastep}: {value}")