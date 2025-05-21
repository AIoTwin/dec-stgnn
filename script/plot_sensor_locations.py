import os
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from matplotlib.patches import Polygon
import utility
import numpy as np
import scipy.sparse as sp

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
dataset = "pems-bay"
experiment = "experiment_1"
csv_file_path = os.path.join(base_dir, f"locations/{dataset}", "locations.csv")
plot_location = os.path.join(base_dir, f"locations/{dataset}", f"sensor_locations_{dataset}_{experiment}.png")

adj_file_path = os.path.join(base_dir, f"data/{dataset}", "adj.npz")
adj = sp.load_npz(adj_file_path)
adj = adj.tocsc()

sensor_id = 234
sensor_neighbours = adj.getrow(sensor_id).indices
weights = adj.getrow(sensor_id).data
print(f"Number of neighbours: {len(sensor_neighbours)}")
neighbor_weights = list(zip(sensor_neighbours, weights))
# Sort by weight in descending order and take the top 5
sorted_neighbor_weights = sorted(neighbor_weights, key=lambda x: x[1], reverse=True)[:12]
# Print sensor, neighbors, and their weights
for neighbor, weight in sorted_neighbor_weights:
    print(f"Neighbor {neighbor} with weight {weight}")

# Sensor location
data = pd.read_csv(csv_file_path)

# Plot all sensors
for idx, sensor in data.iterrows():
    if idx == sensor_id:
        # Plot the main sensor in a distinct color
        plt.scatter(sensor['longitude'], sensor['latitude'], c='red', marker='o', s=2, label='Sensor ID')
        plt.text(sensor['longitude'] + 0.002, sensor['latitude'] + 0.002, str(idx), fontsize=3, ha='right', va='bottom', color='red')
    elif idx in sensor_neighbours:
        # Plot neighbors in a different color
        plt.scatter(sensor['longitude'], sensor['latitude'], c='orange', marker='o', s=2, label='Neighbours' if 'Neighbours' not in plt.gca().get_legend_handles_labels()[1] else None)
        plt.text(sensor['longitude'] + 0.002, sensor['latitude'] + 0.002, str(idx), fontsize=3, ha='right', va='bottom', color='black')
    else:
        # Plot all other sensors in blue
        plt.scatter(sensor['longitude'], sensor['latitude'], c='blue', marker='o', s=2, label='Other Sensors' if 'Other Sensors' not in plt.gca().get_legend_handles_labels()[1] else None)

# Add labels and title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Sensor Locations and Cloudlets')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.savefig(plot_location, dpi=300, bbox_inches='tight')

print(f"Plot saved to: {plot_location}")