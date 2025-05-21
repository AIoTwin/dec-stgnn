import os
import utility
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix, k_hop_subgraph
import pandas as pd
import math
import numpy as np

dataset = "metr-la"
experiment = "experiment_1"
stblock_num = 2

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

csv_file_path = os.path.join(base_dir, f"locations/{dataset}", "locations.csv")
cloudlet_info_path = os.path.join(base_dir, f"locations/{dataset}", "locations.json")

cloudlet_data_json = utility.load_json_file(cloudlet_info_path)
cloudlets, radius_km = utility.get_cloudlet_location_info_from_json(experiment, cloudlet_data_json)

print(f"cloudlets: {cloudlets}")
print(f"radius_km: {radius_km}")

dataset_path = os.path.join(base_dir, "data", dataset)
adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz')) # dimensions: (207, 207)
adj = adj.tocsc() # convert adj from compressed sparse row (CSR) to compressed sparse column format (CSC)

# instead of using hardcoded values for num of nodes, we can read from .csv file to get num of nodes
if dataset == 'metr-la':
    n_vertex = 207
elif dataset == 'pems-bay':
    n_vertex = 325
elif dataset == 'pemsd7-m':
    n_vertex = 228
elif dataset == 'pemsd4':
    n_vertex = 307

print(f"n_vertex: {n_vertex}")

edge_index, _ = from_scipy_sparse_matrix(adj)

locations_data = pd.read_csv(os.path.join(dataset_path, 'locations.csv'))

cloudlet_nodes_list = [[] for _ in range(len(cloudlets))]

for idx, sensor in locations_data.iterrows():
    sensor_loc = (sensor['latitude'], sensor['longitude'])
    closest_cloudlet = None
    min_distance = float('inf')

    for name, loc in cloudlets.items():
        if utility.is_within_radius(sensor_loc[0], sensor_loc[1], loc['lat'], loc['lon'], radius_km):
            distance = utility.calculate_distance(sensor_loc[0], sensor_loc[1], loc['lat'], loc['lon'])
            if distance < min_distance:
                min_distance = distance
                closest_cloudlet = loc['id']

    if closest_cloudlet is not None:
        cloudlet_nodes_list[closest_cloudlet].append(idx)

for i in range(len(cloudlet_nodes_list)):
    cloudlet_nodes_list[i].sort()

cln_id = 0
for cln_nodes in cloudlet_nodes_list:
    print(f"cloudlet {cln_id}: {cln_nodes}")
    cln_id = cln_id + 1

cln_nodes_subgraph_list = []
cln_edge_index_list = []
cln_node_map_list = []

for cln_nodes in cloudlet_nodes_list:
    cln_nodes_subgraph, cln_edge_index, cln_node_map, _ = k_hop_subgraph(cln_nodes, stblock_num, edge_index, relabel_nodes=True, num_nodes=adj.shape[0])
    cln_nodes_subgraph_list.append(cln_nodes_subgraph)
    cln_edge_index_list.append(cln_edge_index)
    cln_node_map_list.append(cln_node_map)

data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
# recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
# using dataset split rate as train: val: test = 70: 15: 15
val_and_test_rate = 0.15

len_val = int(math.floor(data_col * val_and_test_rate))
len_test = int(math.floor(data_col * val_and_test_rate))
len_train = int(data_col - len_val - len_test)

print(f"len_val: {len_val}")
print(f"len_test: {len_test}")
print(f"len_train: {len_train}")

vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

train = vel[: len_train] # if default (len_train = 23991), get 23991 from dataset for training
val = vel[len_train: len_train + len_val] # if default(23991 & 5140), get validation dataset from 23991-29131
test = vel[len_train + len_val:] # if default, get test dataset from 29131 to the final value

# Create datasets for each cloudlet
cln_train_datasets = []
cln_val_datasets = []
cln_test_datasets = []

for cloudlet_nodes in cloudlet_nodes_list:
    node_indices = cloudlet_nodes

    # Extract data for the current cloudlet's nodes
    cloudlet_train_data = train.values[...,node_indices]
    cloudlet_val_data = val.values[...,node_indices]
    cloudlet_test_data = test.values[...,node_indices]

    # Append the extracted data to the lists of cloudlet datasets
    cln_train_datasets.append(cloudlet_train_data)
    cln_val_datasets.append(cloudlet_val_data)
    cln_test_datasets.append(cloudlet_test_data)

cln_id = 0
for cln_train, cln_val, cln_test in zip(cln_train_datasets, cln_val_datasets, cln_test_datasets):
    print(f"cloudlet {cln_id} test dataset size: {np.array(cln_test).size}")
    cln_id = cln_id + 1