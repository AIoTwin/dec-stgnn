import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import torch
import math
import os
import csv
from geopy.distance import geodesic
import pandas as pd
import json
import random

def is_within_radius(lat1, lon1, lat2, lon2, radius_km):
    return geodesic((lat1, lon1), (lat2, lon2)).km <= radius_km

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

def permute_4d_y_pred_to_3d(model, x, edge_index):
    y_pred = model(x, edge_index)
    batch_size, _, _, num_nodes = y_pred.shape # (batch_size, num_node_features, num_of_things_to_predict, num_nodes)
    y_pred = y_pred.permute(0, 3, 1, 2).contiguous().view(batch_size, num_nodes, -1) # [batch_size, num_nodes, num_node_features]

    return y_pred

def zscore_preprocess_3d_data(zscore, train, val, test, use_fit_transform = False):
    # dimensions for train, val and test
    num_time_sequence_train, num_nodes_train, num_node_features_train = train.shape
    num_time_sequence_val, num_nodes_val, num_node_features_val = val.shape
    num_time_sequence_test, num_nodes_test, num_node_features_test = test.shape

    # Shape train, val, and test from 3D to 2D (time_sequence * num_nodes, num_node_features)
    train_shaped = train.reshape(num_time_sequence_train * num_nodes_train, num_node_features_train)
    val_shaped = val.reshape(num_time_sequence_val * num_nodes_val, num_node_features_val)
    test_shaped = test.reshape(num_time_sequence_test * num_nodes_test, num_node_features_test)

    if use_fit_transform:
        train_shaped = zscore.fit_transform(train_shaped)
    else:
        train_shaped = zscore.transform(train_shaped) 
    val_shaped = zscore.transform(val_shaped)
    test_shaped = zscore.transform(test_shaped)

    # Reshape train, val and test from 2D back to 3D
    train = train_shaped.reshape(num_time_sequence_train, num_nodes_train, num_node_features_train)
    val = val_shaped.reshape(num_time_sequence_val, num_nodes_val, num_node_features_val)
    test = test_shaped.reshape(num_time_sequence_test, num_nodes_test, num_node_features_test)

    return train, val, test

def zscore_preprocess_2d_data(zscore, train, val, test, use_fit_transform = False):
    # dimensions for train, val and test
    num_time_sequence_train, num_nodes_train = train.shape
    num_time_sequence_val, num_nodes_val = val.shape
    num_time_sequence_test, num_nodes_test = test.shape

    # Shape train, val, and test from 2D (time_sequence, num_nodes) to 2D (time_sequence * num_nodes, 1)
    train_shaped = train.reshape(-1, 1)
    val_shaped = val.reshape(-1, 1)
    test_shaped = test.reshape(-1, 1)

    if use_fit_transform:
        train_shaped = zscore.fit_transform(train_shaped)
    else:
        train_shaped = zscore.transform(train_shaped) 
    val_shaped = zscore.transform(val_shaped)
    test_shaped = zscore.transform(test_shaped)

    # Reshape train, val and test from 2D (time_sequence * num_nodes, 1) back to 2D (time_sequence, num_nodes)
    train = train_shaped.reshape(num_time_sequence_train, num_nodes_train)
    val = val_shaped.reshape(num_time_sequence_val, num_nodes_val)
    test = test_shaped.reshape(num_time_sequence_test, num_nodes_test)

    return train, val, test

def zscore_preprocess_2d_data_1_node(zscore, train, val, test, use_fit_transform = False):
    # dimensions for train, val and test
    num_time_sequence_train = train.shape[0]
    num_time_sequence_val = val.shape[0]
    num_time_sequence_test = test.shape[0]

    # Shape train, val, and test from 2D (time_sequence, num_nodes) to 2D (time_sequence * num_nodes, 1)
    train_shaped = train.reshape(-1, 1)
    val_shaped = val.reshape(-1, 1)
    test_shaped = test.reshape(-1, 1)

    if use_fit_transform:
        train_shaped = zscore.fit_transform(train_shaped)
    else:
        train_shaped = zscore.transform(train_shaped) 
    val_shaped = zscore.transform(val_shaped)
    test_shaped = zscore.transform(test_shaped)

    # Reshape train, val and test from 2D (time_sequence * num_nodes, 1) back to 2D (time_sequence, num_nodes)
    train = train_shaped.reshape(num_time_sequence_train, 1)
    val = val_shaped.reshape(num_time_sequence_val, 1)
    test = test_shaped.reshape(num_time_sequence_test, 1)

    return train, val, test

def save_variance_logs(logs_folder, cloudlet_id, epoch, variance):
    # Define the CSV file path
    run_folder = os.path.join(logs_folder, 'var')
    csv_file_path = os.path.join(run_folder, f'{cloudlet_id}.csv')

    # Create the directory path for logs if it doesn't exist
    os.makedirs(run_folder, exist_ok=True)

    # Write the headers if the CSV file doesn't exist
    write_headers = not os.path.exists(csv_file_path)

    # Open the CSV file in append mode ('a')
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['Epoch', 'Variance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()

        writer.writerow({
            'Epoch': epoch,
            'Variance': variance
        })

def save_val_logs(logs_folder, cloudlet_id, epoch, lr, train_loss, val_loss, gpu_occupy):    
    # Define the CSV file path
    run_folder = os.path.join(logs_folder, 'val')
    csv_file_path = os.path.join(run_folder, f'{cloudlet_id}.csv')

    # Create the directory path for logs if it doesn't exist
    os.makedirs(run_folder, exist_ok=True)

    # Write the headers if the CSV file doesn't exist
    write_headers = not os.path.exists(csv_file_path)
    
    # Open the CSV file in append mode ('a')
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['Epoch', 'LR', 'Train Loss', 'Val Loss', 'GPU Occupancy (MiB)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()

        writer.writerow({
            'Epoch': epoch,
            'LR': lr,
            'Train Loss': train_loss,
            'Val Loss': val_loss,
            'GPU Occupancy (MiB)': gpu_occupy
        })

def save_test_logs(logs_folder, file_name, test_loss, mae, rmse, wmape, best_epoch = -1):
    # Define the CSV file path
    run_folder = os.path.join(logs_folder, 'test')
    csv_file_path = os.path.join(run_folder, f'{file_name}.csv')

    # Create the directory path for logs if it doesn't exist
    os.makedirs(run_folder, exist_ok=True)

    # Write the headers if the CSV file doesn't exist
    write_headers = not os.path.exists(csv_file_path)

    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['Test Loss', 'MAE', 'RMSE', 'WMAPE']
        if (best_epoch > 0):
            fieldnames = ['Epoch', 'Test Loss', 'MAE', 'RMSE', 'WMAPE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()
        
        if (best_epoch > 0):
            writer.writerow({
                'Epoch': best_epoch,
                'Test Loss': test_loss,
                'MAE': mae,
                'RMSE': rmse,
                'WMAPE': wmape
            })
            return None
            
        writer.writerow({
            'Test Loss': test_loss,
            'MAE': mae,
            'RMSE': rmse,
            'WMAPE': wmape
        })

def save_val_metric_logs(logs_folder, cloudlet_id, epoch, mae, rmse, wmape):
    # Define the CSV file path
    run_folder = os.path.join(logs_folder, 'val_metric')
    csv_file_path = os.path.join(run_folder, f'{cloudlet_id}.csv')

    # Create the directory path for logs if it doesn't exist
    os.makedirs(run_folder, exist_ok=True)

    # Write the headers if the CSV file doesn't exist
    write_headers = not os.path.exists(csv_file_path)

    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['Epoch', 'MAE', 'RMSE', 'WMAPE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()

        writer.writerow({
            'Epoch': epoch,
            'MAE': mae,
            'RMSE': rmse,
            'WMAPE': wmape
        })

def save_val_metric_logs_edge_score(logs_folder, cloudletId, percentage, mae, rmse, wmape):
    run_folder = os.path.join(logs_folder, f"{cloudletId}")
    csv_file_path = os.path.join(run_folder, f'cloudlet_{cloudletId}_{percentage}-percent-masked.csv')

    os.makedirs(run_folder, exist_ok=True)

    write_headers = not os.path.exists(csv_file_path)

    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['MAE', 'RMSE', 'WMAPE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()

        writer.writerow({
            'MAE': mae,
            'RMSE': rmse,
            'WMAPE': wmape
        })

def save_d_analysis_logs(logs_folder, file_name, d):
    # Define the CSV file path
    run_folder = os.path.join(logs_folder, 'val_metric')
    csv_file_path = os.path.join(run_folder, f'{file_name}.csv')

    # Create the directory path for logs if it doesn't exist
    os.makedirs(run_folder, exist_ok=True)

    np.savetxt(csv_file_path, d, delimiter=",")

def save_y_pred_dict_to_csv(logs_folder, file_name, y_pred_dict):
    run_folder = os.path.join(logs_folder)
    csv_file_path = os.path.join(run_folder, f'{file_name}.csv')

    # Create directory if it doesn't exist
    os.makedirs(run_folder, exist_ok=True)

    # Merge all batches for each node
    merged_data = {}
    for node_id, batches in y_pred_dict.items():
        merged_data[node_id] = np.concatenate(batches)  # Flatten all batches into one array

    # Convert to DataFrame (align columns by longest sequence)
    df = pd.DataFrame.from_dict(merged_data, orient='index').transpose()

    # Save to CSV with node IDs as headers
    df.to_csv(csv_file_path, index=False)

def save_modified_y_pred_to_csv(logs_folder, modified_y_pred):
    run_folder = os.path.join(logs_folder, 'edge_analysis')
    os.makedirs(run_folder, exist_ok=True)

    for selected_node, removed_edge_data in modified_y_pred.items():
        file_path = os.path.join(run_folder, f"y_pred_modified_node_{selected_node}.csv")

        # Open the CSV file for writing
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header row: edge_{edgeId}_removed, edge_{edgeId}_removed, ...
            header = [f'edge_{edge_idx}_removed' for edge_idx in removed_edge_data.keys()]
            writer.writerow(header)
            
            # Flatten all batches for each edge into a single array
            flattened_predictions = {}
            for edge_idx, batches in removed_edge_data.items():
                flattened_predictions[edge_idx] = np.concatenate(batches)  # Flatten all batches
            
            # Determine the number of predictions (should be 64: 32 * 2)
            num_predictions = len(next(iter(flattened_predictions.values())))
            
            # Write each prediction as a row
            for i in range(num_predictions):
                row = []
                for edge_idx in flattened_predictions.keys():
                    row.append(flattened_predictions[edge_idx][i])  # Append the prediction for the current edge
                writer.writerow(row)

def save_modified_y_pred_to_csv_edge_weight(logs_folder, modified_y_pred):
    run_folder = os.path.join(logs_folder, 'edge_analysis')
    os.makedirs(run_folder, exist_ok=True)

    for selected_node, removed_edge_data in modified_y_pred.items():
        file_path = os.path.join(run_folder, f"y_pred_modified_node_{selected_node}.csv")

        # Open the CSV file for writing
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header row: edge_{edgeId}_removed, edge_{edgeId}_removed, ...
            header = [f'{edge_idx}' for edge_idx in removed_edge_data.keys()]
            writer.writerow(header)
            
            # Flatten all batches for each edge into a single array
            flattened_predictions = {}
            for edge_idx, batches in removed_edge_data.items():
                flattened_predictions[edge_idx] = np.concatenate(batches)  # Flatten all batches
            
            # Determine the number of predictions (should be 64: 32 * 2)
            num_predictions = len(next(iter(flattened_predictions.values())))
            
            # Write each prediction as a row
            for i in range(num_predictions):
                row = []
                for edge_idx in flattened_predictions.keys():
                    row.append(flattened_predictions[edge_idx][i])  # Append the prediction for the current edge
                writer.writerow(row)

def save_total_trainable_parameters_transfer_size(logs_folder, cloudlet_id, epoch, total_size):
    # Define the CSV file path
    run_folder = os.path.join(logs_folder, 'communication_size')
    csv_file_path = os.path.join(run_folder, f'{cloudlet_id}_total_trainable_parameters_transfer_size.csv')

    # Create the directory path for logs if it doesn't exist
    os.makedirs(run_folder, exist_ok=True)

    # Write the headers if the CSV file doesn't exist
    write_headers = not os.path.exists(csv_file_path)

    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['Epoch', 'Bytes', 'Kilobytes', 'Megabytes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()

        writer.writerow({
            'Epoch': epoch,
            'Bytes': total_size,
            'Kilobytes': total_size/1024,
            'Megabytes': (total_size/1024)/1024
        })

def save_total_transfer_size_node_features(logs_folder, cloudlet_id, epoch, total_size):
    # Define the CSV file path
    run_folder = os.path.join(logs_folder, 'communication_size')
    csv_file_path = os.path.join(run_folder, f'{cloudlet_id}_total_transfer_size_node_features.csv')

    # Create the directory path for logs if it doesn't exist
    os.makedirs(run_folder, exist_ok=True)

    # Write the headers if the CSV file doesn't exist
    write_headers = not os.path.exists(csv_file_path)

    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['Epoch', 'Bytes', 'Kilobytes', 'Megabytes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()

        writer.writerow({
            'Epoch': epoch,
            'Bytes': total_size,
            'Kilobytes': total_size/1024,
            'Megabytes': (total_size/1024)/1024
        })

def save_edge_scores_logs(logs_folder, file_name, edge_scores, edge_positions):
    # Define the CSV file path
    run_folder = os.path.join(logs_folder, 'edge_scores')
    csv_file_path = os.path.join(run_folder, f'{file_name}.csv')

    # Create the directory path for logs if it doesn't exist
    os.makedirs(run_folder, exist_ok=True)

    edge_scores_np = edge_scores.cpu().numpy()
    edge_positions_np = edge_positions.reshape(1, -1)  # Reshape to match scores

    header = ",".join([str(pos) for pos in edge_positions_np[0]])
    data = ",".join([f"{score:.4f}" for score in edge_scores_np])

    with open(csv_file_path, 'a', newline='') as f:
        f.write(header + "\n")
        f.write(data + "\n")

def save_edge_counts_logs(logs_folder, file_name, edge_counts, edge_positions):
    # Define the CSV file path
    run_folder = os.path.join(logs_folder, 'edge_counts')
    csv_file_path = os.path.join(run_folder, f'{file_name}.csv')

    # Create the directory path for logs if it doesn't exist
    os.makedirs(run_folder, exist_ok=True)

    edge_counts_np = edge_counts.cpu().numpy()
    edge_positions_np = edge_positions.reshape(1, -1)  # Reshape to match counts

    header = ",".join([str(pos) for pos in edge_positions_np[0]])
    data = ",".join([str(count) for count in edge_counts_np])
    
    with open(csv_file_path, 'a', newline='') as f:
        f.write(header + "\n")
        f.write(data + "\n")

def calc_edge_index(adj):
    # Extract the non-zero elements (edges) from the adjacency matrix
    rows, cols = adj.nonzero()

    # Create the edge index tensor with shape [2, num_edges]
    edge_index = torch.tensor([rows, cols], dtype=torch.long)

    return edge_index

# Fix bug that could lead to cloudlets with 0 nodes (if <0.5 then use greater amount, if >0.5 use lower amount)
def partition_nodes_to_cloudlets(adj, cloudlet_num):
    n_vertex = adj.shape[0]
    nodes_per_cloudlet = math.ceil(n_vertex / cloudlet_num)
    #nodes_per_cloudlet = math.floor(n_vertex / cloudlet_num)
    print(f"nodes_per_cloudlet: {nodes_per_cloudlet}")
    cloudlet_nodes_list = [[] for _ in range(cloudlet_num)]
    node_ids = {x for x in range(n_vertex)}

    def add_node_neighbours_to_cloudlet(node_id, cloudlet_id):
        # keep track of neighbour ids that have been added to cloudlet
        neighbours_added = set()

        # if cloudlet is already full
        if (len(cloudlet_nodes_list[cloudlet_id]) >= nodes_per_cloudlet):
            return

        # Try to add as much node's neighbours as possible
        for neighbor, connected in enumerate(adj[node_id].toarray()[0]):
            # if cloudlet is full while trying to add node's neighbours
            if (len(cloudlet_nodes_list[cloudlet_id]) >= nodes_per_cloudlet):
                return
            
            # if nodes are neighbours and if node hasn't been taken by another cloudlet already
            if connected and neighbor in node_ids:
                node_ids.remove(neighbor)
                cloudlet_nodes_list[cloudlet_id].append(neighbor)
                neighbours_added.add(neighbor)

        # once we're done with node_id neighbours, check if cloudlet is full now
        # if yes, then go back
        # if no, then go to each neighbour neighbours that has been added to the cloudlet
        if (len(cloudlet_nodes_list[cloudlet_id]) >= nodes_per_cloudlet):
            return

        for neighbour_id in neighbours_added:
            add_node_neighbours_to_cloudlet(neighbour_id, cloudlet_id)

    for cloudlet_id in range(cloudlet_num):
        while node_ids:
            node_id = node_ids.pop()

            cloudlet_nodes_list[cloudlet_id].append(node_id)
            add_node_neighbours_to_cloudlet(node_id, cloudlet_id)

            if (len(cloudlet_nodes_list[cloudlet_id]) >= nodes_per_cloudlet):
                break

    print(f"Number of cloudlets: {len(cloudlet_nodes_list)}")
    print(f"cloudlet_nodes_list: {cloudlet_nodes_list}")
    return cloudlet_nodes_list

def partition_nodes_to_cloudlets_by_range_proximity(cloudlets, radius_km, dataset_name):
    # load locations from .csv file
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    locations_data = pd.read_csv(os.path.join(dataset_path, 'locations.csv'))

    cloudlet_nodes_list = [[] for _ in range(len(cloudlets))]

    for idx, sensor in locations_data.iterrows():
        sensor_loc = (sensor['latitude'], sensor['longitude'])
        closest_cloudlet = None
        min_distance = float('inf')

        for name, loc in cloudlets.items():
            if is_within_radius(sensor_loc[0], sensor_loc[1], loc['lat'], loc['lon'], radius_km):
                distance = calculate_distance(sensor_loc[0], sensor_loc[1], loc['lat'], loc['lon'])
                if distance < min_distance:
                    min_distance = distance
                    closest_cloudlet = loc['id']

        if closest_cloudlet is not None:
            cloudlet_nodes_list[closest_cloudlet].append(idx)

    print(f"cloudlet_nodes_list: {cloudlet_nodes_list}")
    return cloudlet_nodes_list

def partition_nodes_to_cloudlets_by_range_sequential(cloudlets, radius_km, dataset_name):
    # load locations from .csv file
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    locations_data = pd.read_csv(os.path.join(dataset_path, 'locations.csv'))

    cloudlet_nodes_list = [[] for _ in range(len(cloudlets))]

    for name, loc in cloudlets.items():
        for idx, sensor in locations_data.iterrows():
            if idx not in cloudlet_nodes_list:
                sensor_loc = (sensor['latitude'], sensor['longitude'])
                if is_within_radius(sensor_loc[0], sensor_loc[1], loc['lat'], loc['lon'], radius_km):
                    cloudlet_nodes_list[loc['id']].append(idx)

    print(f"cloudlet_nodes_list: {cloudlet_nodes_list}")
    return cloudlet_nodes_list

def create_cln_adj_matrix_by_distance(cloudlets, radius_km):
    # Find neighbors for each cloudlet
    cloudlet_neighbors = {name: [] for name in cloudlets}
    cloudlet_names = list(cloudlets.keys())
    for i, name1 in enumerate(cloudlet_names):
        for j, name2 in enumerate(cloudlet_names):
            if i != j:
                loc1 = cloudlets[name1]
                loc2 = cloudlets[name2]
                if is_within_radius(loc1['lat'], loc1['lon'], loc2['lat'], loc2['lon'], radius_km):
                    cloudlet_neighbors[name1].append(name2)

    # Create adjacency matrix
    num_cloudlets = len(cloudlets)
    adj_matrix = torch.zeros((num_cloudlets, num_cloudlets), dtype=torch.int)

    for i, name1 in enumerate(cloudlet_names):
        for j, name2 in enumerate(cloudlet_names):
            if name2 in cloudlet_neighbors[name1]:
                adj_matrix[i, j] = 1

    print(f"adj_matrix: {adj_matrix}")
    return adj_matrix

def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0] # number of nodes, if default (metr-la): 207

    # check if matrix mostly contains 0s (sparse) or actual data
    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    # Create an identity matrix (207, 207)
    id = sp.identity(n_vertex, format='csc')

    # Symmetrizing an adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    #adj = 0.5 * (dir_adj + dir_adj.transpose())
    
    # if symmertric renormalized adjacency or random walk renormalized adjacency
    # or symmetric renormalized laplacian or random walk renormalized laplacian
    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id
    
    # if symmetric normalized adjacency or symmetric renormalized adjacency
    # or symmetric normalized laplacian or symmetric renormalized laplacian
    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
        or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso

def calc_chebynet_gso(gso):
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')
    # If you encounter a NotImplementedError, please update your scipy version to 1.10.1 or later.
    eigval_max = norm(gso, 2)

    # If the gso is symmetric or random walk normalized Laplacian,
    # then the maximum eigenvalue is smaller than or equals to 2.
    if eigval_max >= 2:
        gso = gso - id
    else:
        gso = 2 * gso / eigval_max - id

    return gso

# Never used ???
def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')

def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_pyg_model_master(model, loss, data_iter, edge_index):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x, edge_index).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_3d_pyg_model_master(model, loss, data_iter, edge_index):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = permute_4d_y_pred_to_3d(model, x, edge_index)
        
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_pyg_model(model, loss, data_iter, cln_edge_index, cln_node_map):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x, cln_edge_index).view(len(x), -1)
            y = y[...,cln_node_map]
            y_pred = y_pred[...,cln_node_map]
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_3d_pyg_model(model, loss, data_iter, cln_edge_index, cln_node_map):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = permute_4d_y_pred_to_3d(model, x, cln_edge_index)
            y = y[:,cln_node_map,:]
            y_pred = y_pred[:,cln_node_map,:]
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        #MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        #return MAE, MAPE, RMSE
        return MAE, RMSE, WMAPE

def evaluate_pyg_metric_master(model, data_iter, scaler, edge_index):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x, edge_index).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        #MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        #return MAE, MAPE, RMSE
        return MAE, RMSE, WMAPE

def evaluate_pyg_metric_analysis(model, data_iter, scaler, edge_index):
    model.eval()
    all_d = []
    all_y_pred = []
    with torch.no_grad():
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy())
            threshold = 1e-5
            if np.all(np.abs(y) < threshold):
                print(f"All values are below the threshold")
                continue
            y_pred = scaler.inverse_transform(model(x, edge_index).view(len(x), -1).cpu().numpy())
            d = np.abs(y - y_pred)
            all_d.append(d)
            all_y_pred.append(y_pred)

        full_d = np.concatenate(all_d, axis=0)
        full_y_pred = np.concatenate(all_y_pred, axis=0)
        return full_d, full_y_pred

def evaluate_cloudlet_pyg_metric_analysis(model, data_iter, scaler, edge_index, cln_node_map):
    model.eval()
    all_d = []
    all_y_pred = []
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x, edge_index)
            y = y[...,cln_node_map]
            y_pred = y_pred[...,cln_node_map]

            y = scaler.inverse_transform(y.cpu().numpy())
            threshold = 1e-5
            if np.all(np.abs(y) < threshold):
                print(f"All values are below the threshold")
                continue
            y_pred = scaler.inverse_transform(y_pred.view(len(x), -1).cpu().numpy())
            d = np.abs(y - y_pred)
            all_d.append(d)
            all_y_pred.append(y_pred)

        full_d = np.concatenate(all_d, axis=0)
        full_y_pred = np.concatenate(all_y_pred, axis=0)
        return full_d, full_y_pred

def evaluate_pyg_metric_single_node(model, data_iter, scaler, edge_index, n_nodes):
    model.eval()
    with torch.no_grad():
        mae = [[] for _ in range(n_nodes)]
        sum_y = [[] for _ in range(n_nodes)]
        mape = [[] for _ in range(n_nodes)]
        mse = [[] for _ in range(n_nodes)]

        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy())
            y_pred = scaler.inverse_transform(model(x, edge_index).view(len(x), -1).cpu().numpy())

            d = np.abs(y - y_pred)
            for i in range(n_nodes):
                mae[i] += d[:, i].tolist()
                sum_y[i] += y[:, i].tolist()
                mape[i] += (d[:, i] / y[:, i]).tolist()
                mse[i] += (d[:, i] ** 2).tolist()
        
        mae_per_node = [np.mean(errors) for errors in mae]
        sum_y_per_node = [np.sum(values) for values in sum_y]
        rmse_per_node = [np.sqrt(np.mean(errors)) for errors in mse]
        
        MAE = np.mean(mae_per_node)
        RMSE = np.sqrt(np.mean([val ** 2 for val in rmse_per_node]))

        total_mae = np.sum([np.sum(errors) for errors in mae])
        total_sum_y = np.sum(sum_y_per_node)
        WMAPE = total_mae / total_sum_y

        wmape_per_node = [
            np.sum(mae[i]) / sum_y_per_node[i] if sum_y_per_node[i] != 0 else 0
            for i in range(n_nodes)
        ]

        return MAE, RMSE, WMAPE, mae_per_node, rmse_per_node, wmape_per_node

def evaluate_partial_edge_influence(model, data_iter, scaler, edge_index, n_nodes, num_nodes=5, num_batches=-1):
    model.eval()
    with torch.no_grad():
        # Select random nodes
        selected_nodes = random.sample(range(n_nodes), num_nodes)
        
        # Initialize storage
        original_y_pred = {node: [] for node in selected_nodes}
        modified_y_pred = {node: {} for node in selected_nodes}  # Dict of removed edge index -> predictions

        # Process limited batches
        batch_counter = 0
        for x, y in data_iter:
            if num_batches != -1 and batch_counter >= num_batches:
                break
            
            y_pred = scaler.inverse_transform(model(x, edge_index).view(len(x), -1).cpu().numpy())
            
            # Store original predictions for selected nodes
            for node in selected_nodes:
                original_y_pred[node].append(y_pred[:, node].tolist())
            
            # Iterate over selected nodes and their edges
            for node in selected_nodes:
                node_edges = (edge_index[0] == node).nonzero(as_tuple=True)[0].tolist()
                
                for edge_idx in node_edges:
                    # Create modified edge_index by removing one edge
                    modified_edge_index = torch.clone(edge_index)
                    modified_edge_index = torch.cat((modified_edge_index[:, :edge_idx], modified_edge_index[:, edge_idx+1:]), dim=1)
                    
                    # Get prediction with modified edge_index
                    mod_y_pred = scaler.inverse_transform(model(x, modified_edge_index).view(len(x), -1).cpu().numpy())
                    
                    # Store modified predictions
                    if edge_idx not in modified_y_pred[node]:
                        modified_y_pred[node][edge_idx] = []
                    modified_y_pred[node][edge_idx].append(mod_y_pred[:, node].tolist())
            
            batch_counter += 1
    
    return original_y_pred, modified_y_pred

def evaluate_partial_edge_influence_with_edge_weight(model, data_iter, scaler, edge_index, edge_weight, n_nodes, num_nodes=5, num_batches=-1, weight_bins=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]):
    model.eval()
    with torch.no_grad():
        # Select random nodes
        selected_nodes = random.sample(range(n_nodes), num_nodes)
        
        # Initialize storage
        results = {
            node: {
                'original': [],
                'modified': {f"{high:.1f}-{low:.1f}": [] 
                           for high, low in zip(weight_bins[:-1], weight_bins[1:])}
            } for node in selected_nodes
        }

        # Create weight groups
        weight_groups = []
        for high, low in zip(weight_bins[:-1], weight_bins[1:]):
            group_edges = ((edge_weight <= high) & (edge_weight > low)).nonzero().flatten()
            weight_groups.append((f"{high:.1f}-{low:.1f}", group_edges))

        # Process limited batches
        batch_counter = 0
        for x, y in data_iter:
            if num_batches != -1 and batch_counter >= num_batches:
                break
            
            y_pred = scaler.inverse_transform(model(x, edge_index).view(len(x), -1).cpu().numpy())
            
            # Store original predictions for selected nodes
            for node in selected_nodes:
                results[node]['original'].append(y_pred[:, node].tolist())
            
            # Get predictions with weight groups removed
                for group_name, edge_indices in weight_groups:
                    # Remove all edges in this weight group
                    mask = torch.ones(edge_index.size(1), dtype=torch.bool)
                    mask[edge_indices] = False
                    modified_edge_index = edge_index[:, mask]
                    
                    # Get modified predictions
                    mod_y_pred = scaler.inverse_transform(
                        model(x, modified_edge_index).view(len(x), -1).cpu().numpy())
                    results[node]['modified'][group_name].append(mod_y_pred[:, node].tolist())
            
            batch_counter += 1
    
    original_y_pred = {
        node: results[node]['original'] 
        for node in results
    }

    modified_y_pred = {
        node: {
            f"weight_group_{group}": preds 
            for group, preds in results[node]['modified'].items()
        } 
        for node in results
    }

    return original_y_pred, modified_y_pred

def evaluate_3d_pyg_metric_master(model, data_iter, scaler, edge_index, node_feature = 0):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y = calculate_inverse_transform(scaler, y, None, node_feature)

            y_pred = permute_4d_y_pred_to_3d(model, x, edge_index)
            y_pred = calculate_inverse_transform(scaler, y_pred, None, node_feature)

            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        #MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        #return MAE, MAPE, RMSE
        return MAE, RMSE, WMAPE

def evaluate_pyg_metric(model, data_iter, scaler, cln_edge_index, cln_node_map):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y_pred = model(x, cln_edge_index)
            y = y[...,cln_node_map]
            y_pred = y_pred[...,cln_node_map]

            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(y_pred.view(len(x), -1).cpu().numpy()).reshape(-1)

            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        #MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        #return MAE, MAPE, RMSE
        return MAE, RMSE, WMAPE

def evaluate_3d_pyg_metric(model, data_iter, scaler, cln_edge_index, cln_node_map, node_feature = 0):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y = calculate_inverse_transform(scaler, y, cln_node_map, node_feature)

            y_pred = permute_4d_y_pred_to_3d(model, x, cln_edge_index)
            y_pred = calculate_inverse_transform(scaler, y_pred, cln_node_map, node_feature)

            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        #MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        #return MAE, MAPE, RMSE
        return MAE, RMSE, WMAPE

def validate_pyg_metric(model, data_iter, scaler, cln_edge_index, cln_node_map):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y_pred = model(x, cln_edge_index)
            y = y[...,cln_node_map]
            y_pred = y_pred[...,cln_node_map]

            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(y_pred.view(len(x), -1).cpu().numpy()).reshape(-1)

            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        return MAE, RMSE, WMAPE, MAPE

def validate_pyg_metric_master(model, data_iter, scaler, edge_index):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x, edge_index).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        return MAE, RMSE, WMAPE, MAPE

def calculate_inverse_transform(scaler, y, node_map = None, node_feature = 0):
    if node_map is not None and len(node_map) > 0:
        y = y[:,node_map,:]
    batch_size, num_nodes, num_node_features = y.shape
    y = y.reshape(batch_size * num_nodes, num_node_features)
    y = scaler.inverse_transform(y.cpu().numpy())
    y = y[:,node_feature:node_feature+1]
    #.reshape(-1)

    return y

def collect_node_features(node_map, x, y):
    num_batch_size = y.shape[0]
    num_all_node_ids = y.shape[1]
    num_mapped_nodes = len(node_map)
    num_unmapped_nodes = num_all_node_ids - num_mapped_nodes
    if len(y.shape) == 3:
        num_node_features = y.shape[2]
    elif len(y.shape) == 2:
        num_node_features = 1
    else:
        assert False, f'Cannot work with data that has {len(y.shape)} dimension'
    transfer_size_in_bytes = (num_batch_size * num_unmapped_nodes * num_node_features) * 2
    return transfer_size_in_bytes

def collect_node_features_centralized(x, y):
    num_batch_size = y.shape[0]
    num_all_node_ids = y.shape[1]
    if len(y.shape) == 3:
        num_node_features = y.shape[2]
    elif len(y.shape) == 2:
        num_node_features = 1
    else:
        assert False, f'Cannot work with data that has {len(y.shape)} dimension'
    transfer_size_in_bytes = num_batch_size * num_all_node_ids * num_node_features
    return transfer_size_in_bytes
# Create a random adjacency matrix for cloudlets, i.e. which cloudlet will communicate with another cloudlet
def create_random_cln_ajd_matrix(cln_num):
    if cln_num <= 0:
        assert False, f'Number of cloudlets cannot be 0 or less; cln_num: {cln_num}'
    
    # Create an upper triangular matrix with random 0s and 1s
    upper_tri = np.triu(np.random.randint(2, size=(cln_num, cln_num)), k=1)

    # Create a matrix by adding the upper triangular to its transpose
    full_matrix = upper_tri + upper_tri.T

    # Ensure diagonal values are 0
    np.fill_diagonal(full_matrix, 0)
    
    return full_matrix

def compute_parameter_variance(model_params):
    param_stack = torch.stack([torch.cat([p.flatten() for p in params]) for params in model_params])
    param_mean = param_stack.mean(dim=0) # If dim=1, it throws an error
    
    variances = []
    for params in param_stack:
        param_variance = (params - param_mean) ** 2
        variances.append(param_variance.mean())
    
    return variances

def compute_parameter_variance_master(model_params):
    variances = {}
    for name, param in model_params:
        if param.requires_grad:
            variance = torch.var(param.data)
            variances[name] = variance.item()
    return variances

def log_variance_info(param_variances):
    for i, variance in enumerate(param_variances):
        variance_mean = variance.mean().item()
        print(f"Cloudlet {i} Parameter Variance - Mean: {variance_mean}")

def load_json_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def get_cloudlet_location_info_from_json(experiment_name, cloudlet_info_json):
    cloudlet_info = cloudlet_info_json.get(experiment_name)
    if cloudlet_info:
        cloudlets = cloudlet_info["cloudlets"]
        radius_km = cloudlet_info["radius_km"]
        return cloudlets, radius_km
    else:
        raise ValueError(f"Experiment '{experiment_name}' not found in the json file.")