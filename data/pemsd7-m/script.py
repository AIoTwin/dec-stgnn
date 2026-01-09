import pandas as pd

input_file = "locations-raw.csv"
output_file = "locations.csv"

df = pd.read_csv(input_file)

# Rename the relevant columns
df_renamed = df.rename(columns={
    'ID': 'sensor_id',
    'Latitude': 'latitude',
    'Longitude': 'longitude'
})

# Keep only the desired columns
df_final = df_renamed[['index', 'sensor_id', 'latitude', 'longitude']]

# Save to new CSV
df_final.to_csv(output_file, index=False)

print(f"Converted CSV saved to: {output_file}")