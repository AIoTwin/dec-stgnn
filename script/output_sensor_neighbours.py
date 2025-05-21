import os
import numpy as np
import scipy.sparse as sp

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
dataset = "metr-la"
idx = 33

# Load the adjacency matrix from the .npz file
adj_file_path = os.path.join(base_dir, f"data/{dataset}", "adj.npz")
adj = sp.load_npz(adj_file_path)
adj = adj.tocsc()

num_sensors = adj.shape[0]  # Number of sensors (rows in the adjacency matrix)

for sensor_idx in range(num_sensors):
    if idx == sensor_idx:
        # Find neighbors (non-zero column indices in the row)
        neighbors = adj.getrow(sensor_idx).indices  # Efficiently fetch non-zero indices
        weights = adj.getrow(sensor_idx).data
        # print(f"Sensor {sensor_idx}: Neighbors -> {list(neighbors)} with weight {weights}")
        print(f"Sensor {sensor_idx}: Neighbors -> {list(neighbors)}")

        neighbor_weights = list(zip(neighbors, weights))
        # Sort by weight in descending order and take the top 5
        sorted_neighbor_weights = sorted(neighbor_weights, key=lambda x: x[1], reverse=True)[:5]
        # Print sensor, neighbors, and their weights
        for neighbor, weight in sorted_neighbor_weights:
            print(f"Neighbor {neighbor} with weight {weight}")

# # Assuming the adjacency matrix is stored with key 'adj'
# adj_matrix = data['adj']  # Replace 'adj' with the actual key if it's different

# # Convert the adjacency matrix to a dense format (if it's sparse)
# if hasattr(adj_matrix, "todense"):  # Check if it's in sparse format
#     adj_matrix = adj_matrix.todense()

# # Loop through each sensor and print its neighbors
# num_sensors = adj_matrix.shape[0]  # Number of sensors (rows/columns)
# for sensor_idx in range(num_sensors):
#     # Find neighbors (indices with non-zero values in the row)
#     neighbors = np.where(adj_matrix[sensor_idx] > 0)[1]  # Use [0] if adj_matrix is dense
#     print(f"Sensor {sensor_idx}: Neighbors -> {list(neighbors)}")