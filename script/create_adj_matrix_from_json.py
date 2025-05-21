import os
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix, save_npz
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
file_path = "adj_matrix_direct_connection.json"
dataset = "pems-bay"
json_file_path = os.path.join(base_dir, f"data/{dataset}", file_path)

# Load JSON
with open(json_file_path, "r") as f:
    data = json.load(f)["Sheet1"]

# Parse the sensor neighbours
for entry in data:
    entry["sensor_neighbours"] = list(map(int, entry["sensor_neighbours"].split(",")))
    entry["sensor_id"] = int(entry["sensor_id"])

# Number of sensors
num_sensors = len(data)
print(f"num_sensors: {num_sensors}")

# Initialize adjacency matrix
adj_matrix = np.zeros((num_sensors, num_sensors), dtype=int)

# Check bidirectional connections
bidirectional = True
for entry in data:
    sensor_id = entry["sensor_id"]
    neighbours = entry["sensor_neighbours"]

    for neighbour in neighbours:
        try:
            # Add the connection to the adjacency matrix
            adj_matrix[sensor_id, neighbour] = 1

            # Check bidirectionality
            neighbour_entry = next((e for e in data if e["sensor_id"] == neighbour), None)
            if neighbour_entry and sensor_id not in neighbour_entry["sensor_neighbours"]:
                print(f"Unidirectional connection found: {sensor_id} -> {neighbour}")
                bidirectional = False
        except Exception as e:
            print(f"Error with sensor_id={sensor_id}, neighbour={neighbour}: {e}")
            raise

if bidirectional:
    print("All connections are bidirectional.")
else:
    print("Not all connections are bidirectional.")
    exit()

# Add diagonal values (self-loops)
np.fill_diagonal(adj_matrix, 1)

# Convert to CSC format
adj_csc = csc_matrix(adj_matrix)

# Save the adjacency matrix to .npz
output_path = os.path.join(base_dir, f"data/{dataset}", "adj_direct_0_1.npz")
save_npz(output_path, adj_csc)

print(f"Adjacency matrix saved to {output_path}")