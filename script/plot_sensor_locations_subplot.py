import os
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from matplotlib.patches import Polygon
import utility
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
datasets = [
    {"name": "metr-la", "title": "METR-LA"},
    {"name": "pems-bay", "title": "PeMS-BAY"}
]
datasets = [
    {"name": "pems-bay", "title": "PeMS-BAY"},
    {"name": "pemsd7-m", "title": "PeMSD-7M"}
]

experiment = "experiment_1"
plot_location = os.path.join(base_dir, f"logs/plots", f"locations_metr-la_pems-bay_{experiment}_subplot.png")

# Create the subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

for i, dataset_info in enumerate(datasets):
    dataset = dataset_info["name"]
    title = dataset_info["title"]
    csv_file_path = os.path.join(base_dir, f"locations/{dataset}", "locations.csv")
    cloudlet_info_path = os.path.join(base_dir, f"locations/{dataset}", "locations.json")

    # Sensor location
    data = pd.read_csv(csv_file_path)

    # Cloudlet location and range
    cloudlet_data_json = utility.load_json_file(cloudlet_info_path)
    cloudlets, radius_km = utility.get_cloudlet_location_info_from_json(experiment, cloudlet_data_json)

    def is_within_radius(lat1, lon1, lat2, lon2, radius_km):
        return geodesic((lat1, lon1), (lat2, lon2)).km <= radius_km

    def calculate_distance(lat1, lon1, lat2, lon2):
        return geodesic((lat1, lon1), (lat2, lon2)).km

    def create_circle(lat, lon, radius_km):
        points = []
        for bearing in range(0, 361, 1):  # 0 to 360 degrees
            point = geodesic(kilometers=radius_km).destination((lat, lon), bearing)
            points.append((point.longitude, point.latitude))
        return points

    # Assign sensors to cloudlets
    cloudlet_sensors = {name: [] for name in cloudlets}
    assigned_sensors = set()

    for idx, sensor in data.iterrows():
        sensor_loc = (sensor['latitude'], sensor['longitude'])
        closest_cloudlet = None
        min_distance = float('inf')

        for name, loc in cloudlets.items():
            if is_within_radius(sensor_loc[0], sensor_loc[1], loc['lat'], loc['lon'], radius_km):
                distance = calculate_distance(sensor_loc[0], sensor_loc[1], loc['lat'], loc['lon'])
                if distance < min_distance:
                    min_distance = distance
                    closest_cloudlet = name

        if closest_cloudlet is not None:
            cloudlet_sensors[closest_cloudlet].append(idx)
            assigned_sensors.add(idx)

    # Plot sensors for each cloudlet with different colors
    ax = axes[i]
    for name, loc in cloudlets.items():
        sensor_indices = cloudlet_sensors[name]
        sensor_data = data.iloc[sensor_indices]
        ax.scatter(sensor_data['longitude'], sensor_data['latitude'], c=loc['color'], marker='o', label=name)

        # Plot cloudlet location and circle
        ax.scatter(loc['lon'], loc['lat'], c=loc['color'], marker='x')

        # Plot filled circle
        circle_points = create_circle(loc['lat'], loc['lon'], radius_km)
        circle = Polygon(circle_points, closed=True, color=loc['color'], alpha=0.3)
        ax.add_patch(circle)

    # Find unassigned sensors
    unassigned_sensors = list(set(data.index) - assigned_sensors)

    # Plot unassigned sensors
    if unassigned_sensors:
        unassigned_sensor_data = data.loc[unassigned_sensors]
        ax.scatter(unassigned_sensor_data['longitude'], unassigned_sensor_data['latitude'], c='black', marker='*', label='Unassigned Sensors')

    # Determine the range of x and y axes
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Generate tick values with 0.05 granularity
    x_ticks = np.arange(round(x_min, 2), round(x_max + 0.05, 2), 0.05)
    y_ticks = np.arange(round(y_min, 2), round(y_max + 0.05, 2), 0.05)

    ax.set_aspect('equal', adjustable='box')

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Set labels, title, and legend
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Adjust layout and save the plot
plt.subplots_adjust(hspace=0.3)  # Reduce vertical space between plots
plt.tight_layout()
plt.savefig(plot_location, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_location}")