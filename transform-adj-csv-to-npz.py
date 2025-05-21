import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import lil_matrix
from scipy.sparse import save_npz
import os

# Median distance is 367.15 for pemsd4, max is 2712.1
def preprocess_data(edges_path, o=None, k=3000):
    # Read the edges CSV file
    edges_df = pd.read_csv(os.path.join(edges_path, 'edges.csv'))
    
    # Extract unique node IDs from 'from' and 'to' columns to determine num_nodes
    node_ids = np.unique(edges_df[['from', 'to']].values)
    num_nodes = node_ids.max() + 1  # Assuming node IDs start from 0

    # Extract 'from', 'to', and 'cost' columns from the dataframe
    nodes_from = edges_df['from'].values
    nodes_to = edges_df['to'].values
    distances = edges_df['cost'].values

    # Calculate standard deviation (o) based on edge distances if not provided
    if o is None:
        o = np.std(distances)
    print(f"standard deviation: {o}")
    # Initialize a sparse matrix for the weighted adjacency matrix
    adj_matrix = lil_matrix((num_nodes, num_nodes))
    # adj_matrix = np.zeros([num_nodes, num_nodes])

    # Compute the weighted adjacency matrix based on the given formula
    for i in range(len(nodes_from)):
        u = nodes_from[i]
        v = nodes_to[i]
        dist_uv = distances[i]

        if dist_uv <= k:
            weight_uv = np.exp(-(dist_uv**2) / (o**2))
            adj_matrix[u, v] = weight_uv
    # Set diagonal elements to 1
    # adj_matrix.setdiag(1)

    adj_matrix = adj_matrix.tocsc()
    print(f"adj_matrix: {adj_matrix}")

    return adj_matrix

def save_adjacency_matrix(adj_matrix, output_file):
    # Save the adjacency matrix as a sparse matrix in .npz format
    save_npz(output_file, adj_matrix)

if __name__ == "__main__":
    edges_path = './data'
    npz_file = 'adj.npz'

    edges_path = os.path.join(edges_path, 'pemsd4')

    # Preprocess the data and generate the adjacency matrix
    adj_matrix = preprocess_data(edges_path)

    # Save the adjacency matrix to a .npz file
    save_npz(os.path.join(edges_path, npz_file), adj_matrix)

    print(f"Weighted adjacency matrix saved.")