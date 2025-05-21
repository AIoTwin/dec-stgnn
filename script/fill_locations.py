import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
csv_file = os.path.join(base_dir, "data/pems-bay", "locations.csv")

# Load the CSV file
df = pd.read_csv(csv_file)

# Create a new index starting from 0
df.insert(0, 'index', range(len(df)))

# Save the updated DataFrame back to the same CSV file
df.to_csv(csv_file, index=False)

print(f"Index has been updated and saved back to {csv_file}")