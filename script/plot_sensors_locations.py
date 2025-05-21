import os
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
dataset = "metr-la"
experiment = "experiment_1"
csv_file_path = os.path.join(base_dir, f"locations/{dataset}", "locations.csv")
plot_location = os.path.join(base_dir, f"locations/{dataset}", f"sensors_locations_{dataset}_{experiment}.png")

# List of sensor IDs to process
# sensor_ids = [174, 160, 187, 12, 163, 61, 159, 0, 37, 164, 193, 127, 137, 113] # highest average prediction delta
sensor_ids = [174, 0, 114, 37, 39, 22, 71, 70] # highest error count between -6 and -3 (they have spikes in this area)
# Sensor location data
data = pd.read_csv(csv_file_path)

# Plot all sensors
for idx, sensor in data.iterrows():
    if idx in sensor_ids:
        # Plot the specified sensors in red
        plt.scatter(sensor['longitude'], sensor['latitude'], c='red', marker='o', s=2, label='Selected Sensors' if 'Selected Sensors' not in plt.gca().get_legend_handles_labels()[1] else None)
        plt.text(sensor['longitude'] + 0.002, sensor['latitude'] + 0.002, str(idx), fontsize=3, ha='right', va='bottom', color='red')
    else:
        # Plot all other sensors in blue
        plt.scatter(sensor['longitude'], sensor['latitude'], c='blue', marker='o', s=2, label='Other Sensors' if 'Other Sensors' not in plt.gca().get_legend_handles_labels()[1] else None)

# Add labels and title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Selected Sensor Locations')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.savefig(plot_location, dpi=300, bbox_inches='tight')

print(f"Selected Sensor IDs: {sensor_ids}")
print(f"Plot saved to: {plot_location}")
