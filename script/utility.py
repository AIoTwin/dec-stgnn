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
from scipy.signal import find_peaks

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
    if test_shaped.shape[0] > 0:
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

def save_val_new_metric_logs(
        logs_folder,
        cloudlet_id,
        epoch,
        big_error_count,
        big_error_rate,
        sudden_event_count,
        sudden_event_hits,
        SUDDEN_EVENT_RATE,
        jam_event_count,
        jam_event_hits,
        JAM_EVENT_RATE,
        rec_event_count,
        rec_event_hits,
        REC_EVENT_RATE
    ):
    # Define the CSV file path
    run_folder = os.path.join(logs_folder, 'new_val_metric')
    csv_file_path = os.path.join(run_folder, f'{cloudlet_id}.csv')

    # Create the directory path for logs if it doesn't exist
    os.makedirs(run_folder, exist_ok=True)

    # Write the headers if the CSV file doesn't exist
    write_headers = not os.path.exists(csv_file_path)

    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = [
            'Epoch',
            'Big Error Count',
            'Big Error Rate',
            'Total sudden change in speed count',
            'Total correct preidction sudden change in speed count',
            'Total sudden change in speed rate',
            'Traffic jam count',
            'Correct traffic jam prediction count',
            'Traffic jam rate',
            'Traffic jam recovery count',
            'Correct traffic jam recovery prediction count',
            'Traffic jam recovery rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()

        writer.writerow({
            'Epoch': epoch,
            'Big Error Count': big_error_count,
            'Big Error Rate': big_error_rate,
            'Total sudden change in speed count': sudden_event_count,
            'Total correct preidction sudden change in speed count': sudden_event_hits,
            'Total sudden change in speed rate': SUDDEN_EVENT_RATE,
            'Traffic jam count': jam_event_count,
            'Correct traffic jam prediction count': jam_event_hits,
            'Traffic jam rate': JAM_EVENT_RATE,
            'Traffic jam recovery count': rec_event_count,
            'Correct traffic jam recovery prediction count': rec_event_hits,
            'Traffic jam recovery rate': REC_EVENT_RATE
        })

def save_val_alpha_propagation_metric_logs(
        logs_folder,
        cloudlet_id,
        epoch,
        precision,
        recall,
        f1,
        iou,
        accuracy,
        gt_cong_rate,
        total_points,
        total_gt_cong,
        total_pred_cong
    ):
    # Define the CSV file path
    run_folder = os.path.join(logs_folder, 'alpha_propagation_metric')
    csv_file_path = os.path.join(run_folder, f'{cloudlet_id}.csv')

    # Create the directory path for logs if it doesn't exist
    os.makedirs(run_folder, exist_ok=True)

    # Write the headers if the CSV file doesn't exist
    write_headers = not os.path.exists(csv_file_path)

    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = [
            'Epoch',
            'recall',
            'precision',
            'f1',
            'iou',
            'accuracy',
            'gt_cong_rate',
            'total_points',
            'total_gt_cong',
            'total_pred_cong']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()

        writer.writerow({
            'Epoch': epoch,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'iou': iou,
            'accuracy': accuracy,
            'gt_cong_rate': gt_cong_rate,
            'total_points': total_points,
            'total_gt_cong': total_gt_cong,
            'total_pred_cong': total_pred_cong
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

def save_node_scores_logs(logs_folder, file_name, node_scores, node_positions):
    # Define the CSV file path
    run_folder = os.path.join(logs_folder, 'node_scores')
    csv_file_path = os.path.join(run_folder, f'{file_name}.csv')

    # Create the directory path for logs if it doesn't exist
    os.makedirs(run_folder, exist_ok=True)

    node_scores_np = node_scores.cpu().numpy()
    node_positions_np = node_positions.reshape(1, -1)  # Reshape to match scores

    header = ",".join([str(pos) for pos in node_positions_np[0]])
    data = ",".join([f"{score:.4f}" for score in node_scores_np])

    with open(csv_file_path, 'a', newline='') as f:
        f.write(header + "\n")
        f.write(data + "\n")

def save_node_counts_logs(logs_folder, file_name, node_counts, node_positions):
    run_folder = os.path.join(logs_folder, 'node_counts')
    csv_file_path = os.path.join(run_folder, f'{file_name}.csv')

    os.makedirs(run_folder, exist_ok=True)

    node_counts_np = node_counts.cpu().numpy()
    node_positions_np = node_positions.reshape(1, -1)

    header = ",".join([str(pos) for pos in node_positions_np[0]])
    data = ",".join([str(count) for count in node_counts_np])

    with open(csv_file_path, 'a', newline='') as f:
        f.write(header + "\n")
        f.write(data + "\n")

def save_ineligible_nodes_logs(logs_folder, cln_id, ineligible_nodes):
    save_dir = os.path.join(logs_folder, 'other', f'cloudlet_{cln_id}')
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, 'ineligible_nodes.csv')

    with open(file_path, 'a') as f:
        if ineligible_nodes:
            f.write(','.join(map(str, sorted(ineligible_nodes))) + '\n')
        else:
            f.write('\n')

def save_nodes_removed_by_distribution_logs(logs_folder, cln_id, removed_nodes_by_distribution):
    save_dir = os.path.join(logs_folder, 'other', f'cloudlet_{cln_id}')
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, 'removed_nodes_by_distribution.csv')

    with open(file_path, 'a') as f:
        if removed_nodes_by_distribution:
            f.write(','.join(map(str, sorted(removed_nodes_by_distribution))) + '\n')
        else:
            f.write('\n')

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
            x = x.to('cuda:0')
            edge_index = edge_index.to('cuda:0')
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

def evaluate_cloudlet_pyg_new_metric_analysis(
        model,
        data_iter,
        scaler,
        edge_index,
        cln_node_map,
        big_err_threshold=20.0,   # large-error threshold (abs diff)
        change_window=12,         # how many recent points to consider for sudden change
        change_delta=20.0,        # minimum change to qualify as jam/recovery
        change_tolerance=10.0,    # correctness band at the event endpoint
        cooldown=None,            # steps to skip after detecting an event (default: change_window//2)
    ):
    """
    Computes:
      - MAE, RMSE, WMAPE
      - BIG_ERR_COUNT, BIG_ERR_RATE
      - SUDDEN_EVENT_COUNT, SUDDEN_EVENT_HITS, SUDDEN_EVENT_RATE
      - JAM_EVENT_COUNT, JAM_EVENT_HITS, JAM_EVENT_RATE
      - REC_EVENT_COUNT, REC_EVENT_HITS, REC_EVENT_RATE

    Sudden-change event at time t (per node) if within the last `change_window` steps there
    exists k < t such that:
        Jam: GT[t] <= GT[k] - change_delta
        Rec: GT[t] >= GT[k] + change_delta
    Correct if |PRED[t] - GT[t]| <= change_tolerance.
    """
    if cooldown is None:
        cooldown = max(1, change_window // 2)

    model.eval()
    with torch.no_grad():
        mae, sum_y, mse = [], [], []
        big_err_count = 0
        total_preds = 0

        # Sudden-change aggregates
        sudden_event_count = 0
        sudden_event_hits = 0

        jam_event_count = 0
        jam_event_hits = 0

        rec_event_count = 0
        rec_event_hits = 0

        for x, y in data_iter:
            x = x.to('cuda:0')
            edge_index = edge_index.to('cuda:0')
            y_pred = model(x, edge_index)
            y = y[...,cln_node_map]
            y_pred = y_pred[...,cln_node_map]

            # inverse-scale ground truth and preds to mile/h
            y_np = scaler.inverse_transform(y.cpu().numpy())

            x = x.to('cuda:0')
            edge_index = edge_index.to('cuda:0')  # avoid shadowing arg name
            y_pred_np = scaler.inverse_transform(
                y_pred.view(len(x), -1).cpu().numpy()
            )

            d = np.abs(y_np - y_pred_np)

            # accumulate base metrics
            mae.extend(d.ravel().tolist())
            sum_y.extend(y_np.ravel().tolist())
            mse.extend((d ** 2).ravel().tolist())

            # accumulate large-error stats
            big_err_count += int(np.sum(d >= big_err_threshold))
            total_preds += d.size

            # --- sudden-change events
            # y_np and y_pred_np are [T_batch, N_num_nodes]
            T, N = y_np.shape
            # Per-node cooldown counters
            cool = np.zeros(N, dtype=int)

            for t in range(1, T):
                # decrease cooldowns
                cool = np.maximum(0, cool - 1)

                # define window start
                w_start = max(0, t - change_window)
                # values in the lookback window (excluding t)
                # shape: [W, N]
                past = y_np[w_start:t, :]
                # print(f"past: {past}")
                if past.size == 0:
                    continue

                # current value at t for all nodes
                cur = y_np[t, :]           # shape [N]
                pred_cur = y_pred_np[t, :] # shape [N]
                # print(f"cur: {cur}")
                # print(f"pred_cur: {pred_cur}")

                # For each node, find the reference k in the window that maximizes |cur - past_k|
                # We do both directions separately to classify jam vs recovery.
                # Jam detection: need a sufficiently higher past value that drops to cur.
                #   jam_margin = past - cur  (positive means drop from past to cur)
                jam_margin = past - cur[None, :]
                # print(f"jam_margin: {jam_margin}")
                # max over time axis -> best candidate drop in the window
                jam_best = np.max(jam_margin, axis=0)  # shape [N]
                # Recovery detection: rise from past to cur
                rec_margin = cur[None, :] - past
                # print(f"rec_margin: {rec_margin}")
                rec_best = np.max(rec_margin, axis=0)  # shape [N]

                # Boolean masks for events at time t
                jam_mask = (jam_best >= change_delta) & (cool == 0)
                rec_mask = (rec_best >= change_delta) & (cool == 0)

                # print(f"jam_mask: {jam_mask}")
                # print(f"rec_mask: {rec_mask}")

                if not (jam_mask.any() or rec_mask.any()):
                    continue

                # Evaluate correctness at endpoint t:
                # correct if |pred_cur - cur| <= change_tolerance
                abs_err = np.abs(pred_cur - cur)
                # print(f"abs_err: {abs_err}")

                # Count jams
                if jam_mask.any():
                    idx = np.where(jam_mask)[0]
                    hits = np.sum(abs_err[idx] <= change_tolerance)
                    jam_event_count += idx.size
                    jam_event_hits  += int(hits)
                    # apply cooldown for those nodes
                    cool[idx] = np.maximum(cool[idx], cooldown)

                # Count recoveries
                if rec_mask.any():
                    idx = np.where(rec_mask)[0]
                    hits = np.sum(abs_err[idx] <= change_tolerance)
                    rec_event_count += idx.size
                    rec_event_hits  += int(hits)
                    # apply cooldown for those nodes
                    cool[idx] = np.maximum(cool[idx], cooldown)

            # accumulate totals
            sudden_event_count += jam_event_count + rec_event_count - sudden_event_count
            sudden_event_hits  += jam_event_hits + rec_event_hits   - sudden_event_hits

        MAE = float(np.mean(mae)) if len(mae) > 0 else 0.0
        RMSE = float(np.sqrt(np.mean(mse))) if len(mse) > 0 else 0.0
        denom_sum_y = float(np.sum(sum_y))
        WMAPE = float(np.sum(mae) / denom_sum_y) if denom_sum_y != 0 else 0.0
        BIG_ERR_RATE = big_err_count / total_preds if total_preds > 0 else 0.0
        # Rates
        SUDDEN_EVENT_RATE = (
            sudden_event_hits / sudden_event_count if sudden_event_count > 0 else 0.0
        )
        JAM_EVENT_RATE = (
            jam_event_hits / jam_event_count if jam_event_count > 0 else 0.0
        )
        REC_EVENT_RATE = (
            rec_event_hits / rec_event_count if rec_event_count > 0 else 0.0
        )

        return (
            MAE, RMSE, WMAPE,
            big_err_count, BIG_ERR_RATE,
            sudden_event_count, sudden_event_hits, SUDDEN_EVENT_RATE,
            jam_event_count, jam_event_hits, JAM_EVENT_RATE,
            rec_event_count, rec_event_hits, REC_EVENT_RATE,
        )

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

def evaluate_cloudlet_pyg_new_metric_for_node_score(
        model,
        data_iter,
        scaler,
        edge_index,
        cln_node_map,
        change_window=12,         # how many recent points to consider for sudden change
        change_delta=20.0,        # minimum change to qualify as jam/recovery
        change_tolerance=10.0,    # correctness band at the event endpoint
        cooldown=None,            # steps to skip after detecting an event (default: change_window//2)
    ):
    """
    Computes:
      - SUDDEN_EVENT_RATE

    Sudden-change event at time t (per node) if within the last `change_window` steps there
    exists k < t such that:
        Jam: GT[t] <= GT[k] - change_delta
        Rec: GT[t] >= GT[k] + change_delta
    Correct if |PRED[t] - GT[t]| <= change_tolerance.
    """
    if cooldown is None:
        cooldown = max(1, change_window // 2)

    model.eval()
    with torch.no_grad():
        # Sudden-change aggregates
        sudden_event_count = 0
        sudden_event_hits = 0

        jam_event_count = 0
        jam_event_hits = 0

        rec_event_count = 0
        rec_event_hits = 0

        for x, y in data_iter:
            x = x.to('cuda:0')
            edge_index = edge_index.to('cuda:0')
            y_pred = model(x, edge_index)
            y = y[...,cln_node_map]
            y_pred = y_pred[...,cln_node_map]

            # inverse-scale ground truth and preds to mile/h
            y_np = scaler.inverse_transform(y.cpu().numpy())

            x = x.to('cuda:0')
            edge_index = edge_index.to('cuda:0')  # avoid shadowing arg name
            y_pred_np = scaler.inverse_transform(
                y_pred.view(len(x), -1).cpu().numpy()
            )

            # --- sudden-change events
            # y_np and y_pred_np are [T_batch, N_num_nodes]
            T, N = y_np.shape
            # Per-node cooldown counters
            cool = np.zeros(N, dtype=int)

            for t in range(1, T):
                # decrease cooldowns
                cool = np.maximum(0, cool - 1)

                # define window start
                w_start = max(0, t - change_window)
                # values in the lookback window (excluding t)
                # shape: [W, N]
                past = y_np[w_start:t, :]
                if past.size == 0:
                    continue

                # current value at t for all nodes
                cur = y_np[t, :]           # shape [N]
                pred_cur = y_pred_np[t, :] # shape [N]
                # For each node, find the reference k in the window that maximizes |cur - past_k|
                # We do both directions separately to classify jam vs recovery.
                # Jam detection: need a sufficiently higher past value that drops to cur.
                #   jam_margin = past - cur  (positive means drop from past to cur)
                jam_margin = past - cur[None, :]
                # max over time axis -> best candidate drop in the window
                jam_best = np.max(jam_margin, axis=0)  # shape [N]
                # Recovery detection: rise from past to cur
                rec_margin = cur[None, :] - past
                rec_best = np.max(rec_margin, axis=0)  # shape [N]

                # Boolean masks for events at time t
                jam_mask = (jam_best >= change_delta) & (cool == 0)
                rec_mask = (rec_best >= change_delta) & (cool == 0)

                if not (jam_mask.any() or rec_mask.any()):
                    continue

                # Evaluate correctness at endpoint t:
                # correct if |pred_cur - cur| <= change_tolerance
                abs_err = np.abs(pred_cur - cur)

                # Count jams
                if jam_mask.any():
                    idx = np.where(jam_mask)[0]
                    hits = np.sum(abs_err[idx] <= change_tolerance)
                    jam_event_count += idx.size
                    jam_event_hits  += int(hits)
                    # apply cooldown for those nodes
                    cool[idx] = np.maximum(cool[idx], cooldown)

                # Count recoveries
                if rec_mask.any():
                    idx = np.where(rec_mask)[0]
                    hits = np.sum(abs_err[idx] <= change_tolerance)
                    rec_event_count += idx.size
                    rec_event_hits  += int(hits)
                    # apply cooldown for those nodes
                    cool[idx] = np.maximum(cool[idx], cooldown)

            # accumulate totals
            sudden_event_count += jam_event_count + rec_event_count - sudden_event_count
            sudden_event_hits  += jam_event_hits + rec_event_hits   - sudden_event_hits

        # Rates
        SUDDEN_EVENT_RATE = (
            sudden_event_hits / sudden_event_count if sudden_event_count > 0 else 0.0
        )

        return SUDDEN_EVENT_RATE

def evaluate_cloudlet_pyg_metric_analysis(model, data_iter, scaler, edge_index, cln_node_map):
    model.eval()
    all_d = []
    all_y_pred = []
    with torch.no_grad():
        for x, y in data_iter:
            x = x.to('cuda:0')
            edge_index = edge_index.to('cuda:0')
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

def evaluate_cloudlet_pyg_new_metric_analysis_with_alpha_propagation(
        model,
        data_iter,
        scaler,
        edge_index,
        cln_node_map,
        big_err_threshold=20.0,   # large-error threshold (abs diff)
        change_window=12,         # how many recent points to consider for sudden change
        change_delta=20.0,        # minimum change to qualify as jam/recovery
        change_tolerance=10.0,    # correctness band at the event endpoint
        cooldown=None,            # steps to skip after detecting an event (default: change_window//2)
        alpha=0.5,                # alpha for congestion rule
    ):
    """
    Computes:
      - MAE, RMSE, WMAPE
      - BIG_ERR_COUNT, BIG_ERR_RATE
      - SUDDEN_EVENT_COUNT, SUDDEN_EVENT_HITS, SUDDEN_EVENT_RATE
      - JAM_EVENT_COUNT, JAM_EVENT_HITS, JAM_EVENT_RATE
      - REC_EVENT_COUNT, REC_EVENT_HITS, REC_EVENT_RATE
      - (NEW) α-based congestion metrics:
          precision, recall (== CONGESTION_HIT_RATE), f1, iou, accuracy, GT congestion rate
    """
    if cooldown is None:
        cooldown = max(1, change_window // 2)

    # Pre-build graph neighbors for the *subset* cln_node_map
    # If no edges survive the filter, nbrs becomes all-empty -> corridor fallback will be used inside the detector.
    if isinstance(cln_node_map, torch.Tensor):
        cln_list = cln_node_map.detach().cpu().numpy().tolist()
    else:
        cln_list = list(cln_node_map)
    nbrs = None
    try:
        nbrs = _build_subgraph_neighbors(edge_index, cln_list, num_nodes_subset=len(cln_list))
        # If literally everyone has <2 neighbors, it's effectively empty for our spatial rule
        if all(len(n) < 2 for n in nbrs):
            nbrs = None
    except Exception:
        nbrs = None

    model.eval()
    with torch.no_grad():
        mae, sum_y, mse = [], [], []
        big_err_count = 0
        total_preds = 0

        # Sudden-change aggregates
        sudden_event_count = 0
        sudden_event_hits = 0

        jam_event_count = 0
        jam_event_hits = 0

        rec_event_count = 0
        rec_event_hits = 0

        # α-based congestion aggregates
        TP = FP = TN = FN = 0
        total_points = 0
        total_gt_cong = 0
        total_pred_cong = 0

        vhat_running = None  # we compute medians per batch; keep last if you prefer stability

        for x, y in data_iter:
            x = x.to('cuda:0')
            ei = edge_index.to('cuda:0') if isinstance(edge_index, torch.Tensor) else edge_index
            y_pred = model(x, ei)

            y      = y[..., cln_node_map]
            y_pred = y_pred[..., cln_node_map]

            # inverse-scale to mph (or your unit)
            y_np = scaler.inverse_transform(y.cpu().numpy())
            y_pred_np = scaler.inverse_transform(
                y_pred.view(len(x), -1).cpu().numpy()
            )

            # ---- base errors
            d = np.abs(y_np - y_pred_np)
            mae.extend(d.ravel().tolist())
            sum_y.extend(y_np.ravel().tolist())
            mse.extend((d ** 2).ravel().tolist())
            big_err_count += int(np.sum(d >= big_err_threshold))
            total_preds   += d.size

            # ---- SCSR events
            T, N = y_np.shape
            cool = np.zeros(N, dtype=int)
            for t in range(1, T):
                cool = np.maximum(0, cool - 1)
                w_start = max(0, t - change_window)
                past = y_np[w_start:t, :]
                if past.size == 0:
                    continue
                cur = y_np[t, :]
                pred_cur = y_pred_np[t, :]

                jam_best = np.max(past - cur[None, :], axis=0)
                rec_best = np.max(cur[None, :] - past, axis=0)

                jam_mask = (jam_best >= change_delta) & (cool == 0)
                rec_mask = (rec_best >= change_delta) & (cool == 0)
                if not (jam_mask.any() or rec_mask.any()):
                    continue

                abs_err = np.abs(pred_cur - cur)

                if jam_mask.any():
                    idx = np.where(jam_mask)[0]
                    hits = np.sum(abs_err[idx] <= change_tolerance)
                    jam_event_count += idx.size
                    jam_event_hits  += int(hits)
                    cool[idx] = np.maximum(cool[idx], cooldown)

                if rec_mask.any():
                    idx = np.where(rec_mask)[0]
                    hits = np.sum(abs_err[idx] <= change_tolerance)
                    rec_event_count += idx.size
                    rec_event_hits  += int(hits)
                    cool[idx] = np.maximum(cool[idx], cooldown)

            # ---- α-based congestion (GT vs Pred)
            gt_mask, vhat_running = _detect_congestion_alpha_propagation_batch(
                y_np, alpha=alpha, vhat=None, nbrs=nbrs
            )
            pr_mask, _ = _detect_congestion_alpha_propagation_batch(
                y_pred_np, alpha=alpha, vhat=vhat_running, nbrs=nbrs
            )

            gt = gt_mask.reshape(-1)
            pr = pr_mask.reshape(-1)

            TP += int(np.sum(gt & pr))
            TN += int(np.sum(~gt & ~pr))
            FP += int(np.sum(~gt & pr))
            FN += int(np.sum(gt & ~pr))
            total_points   += gt.size
            total_gt_cong  += int(np.sum(gt))
            total_pred_cong+= int(np.sum(pr))

        # ---- compose metrics
        MAE = float(np.mean(mae)) if len(mae) > 0 else 0.0
        RMSE = float(np.sqrt(np.mean(mse))) if len(mse) > 0 else 0.0
        denom_sum_y = float(np.sum(sum_y))
        WMAPE = float(np.sum(mae) / denom_sum_y) if denom_sum_y != 0 else 0.0
        BIG_ERR_RATE = big_err_count / total_preds if total_preds > 0 else 0.0

        SUDDEN_EVENT_RATE = ( (jam_event_hits + rec_event_hits) / (jam_event_count + rec_event_count)
                              if (jam_event_count + rec_event_count) > 0 else 0.0 )
        JAM_EVENT_RATE = (jam_event_hits / jam_event_count) if jam_event_count > 0 else 0.0
        REC_EVENT_RATE = (rec_event_hits / rec_event_count) if rec_event_count > 0 else 0.0

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # <-- CONGESTION_HIT_RATE (analogue to SCSR hit-rate)
        f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
        iou       = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
        accuracy  = (TP + TN) / total_points if total_points > 0 else 0.0
        gt_cong_rate = total_gt_cong / total_points if total_points > 0 else 0.0

        return (
            # base
            MAE, RMSE, WMAPE,
            big_err_count, BIG_ERR_RATE,
            # SCSR
            jam_event_count, jam_event_hits, JAM_EVENT_RATE,
            rec_event_count, rec_event_hits, REC_EVENT_RATE,
            (jam_event_count + rec_event_count),
            (jam_event_hits  + rec_event_hits),
            SUDDEN_EVENT_RATE,
            # α-based congestion
            precision, recall, f1, iou, accuracy, gt_cong_rate,
            total_points, total_gt_cong, total_pred_cong
        )

def _build_subgraph_neighbors(edge_index, cln_node_map, num_nodes_subset):
    """
    Build undirected neighbor lists for the subset defined by cln_node_map.
    Returns: list of lists 'nbrs' where nbrs[i] = list of neighbor indices in [0, num_nodes_subset)
    """
    # map original node id -> local subset index
    local_idx = {int(orig): i for i, orig in enumerate(cln_node_map.tolist() if hasattr(cln_node_map, 'tolist') else cln_node_map)}
    nbrs = [[] for _ in range(num_nodes_subset)]

    if isinstance(edge_index, torch.Tensor):
        ei = edge_index.detach().cpu().numpy()
    else:
        ei = np.asarray(edge_index)
    assert ei.shape[0] == 2, "edge_index must be shape [2, E]"

    for u, w in ei.T:
        if u in local_idx and w in local_idx:
            iu = local_idx[u]; iw = local_idx[w]
            if iu != iw:
                nbrs[iu].append(iw)
                nbrs[iw].append(iu)
    return nbrs

def _detect_congestion_alpha_propagation_batch(
    v,                      # [T, N] numpy
    alpha=0.5,
    vhat=None,              # if None, compute from this batch
    nbrs=None,              # list-of-lists neighbors per node (graph-aware). If None, fall back to corridor.
):
    """
    Implements the paper's rule:
      base: v[t,i] < alpha * vhat[i]
      spatial: at time t, (at least two spatial neighbors are base-congested). If deg(i)==2, require both.
      temporal: at time t, both t-1 and t+1 are base-congested for the same sensor.
    Returns: (mask[T,N], vhat[N])
    """
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        v = v[:, None]
    elif v.ndim > 2:
        v = v.reshape(v.shape[0], -1)
    T, N = v.shape

    if vhat is None:
        # per-sensor median over this batch
        vhat = np.nanmedian(v, axis=0)

    thr = alpha * vhat
    thr_full = np.broadcast_to(thr[None, :], (T, N))

    base = (v < thr_full)

    # spatial continuity
    spatial = np.zeros_like(base, dtype=bool)
    if nbrs is not None:
        # graph-aware
        for i in range(N):
            deg = len(nbrs[i])
            if deg < 2:
                continue
            cong_count = np.sum(base[:, nbrs[i]], axis=1)  # [T]
            if deg == 2:
                spatial[:, i] = (cong_count == 2)
            else:
                spatial[:, i] = (cong_count >= 2)
    else:
        # corridor fallback: immediate left/right neighbors
        if N >= 3:
            left  = base[:, :-2]
            right = base[:,  2:]
            spatial[:, 1:-1] = left & right

    # temporal continuity
    temporal = np.zeros_like(base, dtype=bool)
    if T >= 3:
        temporal[1:-1, :] = base[:-2, :] & base[2:, :]

    congested = base | spatial | temporal
    return congested, vhat