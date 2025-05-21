import pandas as pd
import numpy as np
import os

# Define your dataset name
dataset_name = 'metr-la'

dataset_path = './data'
dataset_path = os.path.join(dataset_path, dataset_name)
vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

# Define output npz file path
npz_file = 'vel.npz'

# Convert DataFrame to a NumPy array
data_array = vel.values

# Reshape the 2D array into a 3D array of shape (time_sequence, num_nodes, 1)
time_sequence, num_nodes = data_array.shape
data_array_3d = data_array.reshape(time_sequence, num_nodes, 1)

# Duplicate the values along the last dimension to create a (time_sequence, num_nodes, 2) array
data_array_3d_dup = np.repeat(data_array_3d, 2, axis=2)

# Save the 3D array to a .npz file
np.savez(os.path.join(dataset_path, npz_file), data=data_array_3d_dup)

print(f"Data saved successfully to {npz_file} with shape: {data_array_3d_dup.shape}")