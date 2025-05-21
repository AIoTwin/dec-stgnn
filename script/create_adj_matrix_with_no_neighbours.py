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

with open(json_file_path, "r") as f:
    data = json.load(f)["Sheet1"]

# Number of sensors
num_sensors = len(data)
print(f"num_sensors: {num_sensors}")

# Initialize adjacency matrix
adj_matrix = np.zeros((num_sensors, num_sensors), dtype=int)
np.fill_diagonal(adj_matrix, 1)

# Convert to CSC format
adj_csc = csc_matrix(adj_matrix)

# Save the adjacency matrix to .npz
output_path = os.path.join(base_dir, f"data/{dataset}", "adj_direct_no_neighbours.npz")
save_npz(output_path, adj_csc)

print(f"Adjacency matrix saved to {output_path}")