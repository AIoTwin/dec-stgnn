import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

def load_adj(dataset_name):
    adj, n_vertex = base_load_adj(dataset_name, 'adj.npz')
    return adj, n_vertex

def load_adj_direct_0_1(dataset_name):
    adj, n_vertex = base_load_adj(dataset_name, 'adj_direct_0_1.npz')
    return adj, n_vertex

def load_adj_direct_no_neighbours(dataset_name):
    adj, n_vertex = base_load_adj(dataset_name, 'adj_direct_no_neighbours.npz')
    return adj, n_vertex

def base_load_adj(dataset_name, adj_file_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = sp.load_npz(os.path.join(dataset_path, adj_file_name)) # dimensions: (207, 207)
    adj = adj.tocsc() # convert adj from compressed sparse row (CSR) to compressed sparse column format (CSC)

    # instead of using hardcoded values for num of nodes, we can read from .csv file to get num of nodes
    if dataset_name == 'metr-la':
        n_vertex = 207
    elif dataset_name == 'pems-bay':
        n_vertex = 325
    elif dataset_name == 'pemsd7-m':
        n_vertex = 228
    elif dataset_name == 'pemsd4':
        n_vertex = 307
    return adj, n_vertex

def load_data(dataset_name, len_train, len_val):
    # load dataset from .csv file
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

    train = vel[: len_train] # if metr_la (len_train = 23991), get 23991 from dataset for training
    val = vel[len_train: len_train + len_val] # if metr_la(23991 & 5140), get validation dataset from 23991-29131
    test = vel[len_train + len_val:] # if metr_la, get test dataset from 29131 to the final value

    return train, val, test

def load_3d_data(dataset_name, len_train, len_val):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = np.load(os.path.join(dataset_path, 'vel.npz'))
    vel = vel['data']

    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]

    return train, val, test

def data_transform(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1] # number of nodes (if default: 207)
    len_record = len(data) # if default: train - 23991, if val or test - 5140
    num = len_record - n_his - n_pred # number of sequences, if default (for training): 23991 - 12 - 3 = 23976

    if num <= 0:
        return None, None
    
    # Init a NumPy array with 0s, representing input data tensor with 4-dimensions
    # if default (for training): [23976 x 1 x 12 x 207] - is 1 number of node features(???)
    x = np.zeros([num, 1, n_his, n_vertex])
    # Init a NumPy array with 0s, representing target data tensor with 2-dimensions
    # if default (for training): [23976 x 207]
    y = np.zeros([num, n_vertex])
    
    # loop over each sequence and change values of 1st dimension for both x and y tensor
    for i in range(num):
        head = i # define start of each sequence
        tail = i + n_his # define end of each sequence
        # data[head: tail] - get data from head to tail (0-12, 1-13, etc.)
        # Example (1st iteration): head = 0, tail = 12, so extract data from 0:12
        # Reshape that 2D matrix into a 3D one with dimensins [1, 12, 207]
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        # Change values of 1st dimensions
        y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

def data_3d_transform(data, n_his, n_pred, device):
    num_time_sequence, num_nodes, num_node_features = data.shape
    num = num_time_sequence - n_his - n_pred

    # if default (for training): [23976 x 1 x 12 x 207]
    x = np.zeros([num, num_node_features, n_his, num_nodes])
    # if default (for training): [23976 x 207 x 1]
    y = np.zeros([num, num_nodes, num_node_features])
    
    # loop over each sequence and change values of 1st dimension for both x and y tensor
    for i in range(num):
        head = i # define start of each sequence
        tail = i + n_his # define end of each sequence
        # data[head: tail] - get data from head to tail (0-12, 1-13, etc.)
        # Example (1st iteration): head = 0, tail = 12, so extract data from 0:12
        # Reshape that 2D matrix into a 3D one with dimensins [1, 12, 207]
        x[i, :, :, :] = data[head: tail].reshape(num_node_features, n_his, num_nodes)
        # Change values of 1st dimensions
        y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

def create_cloudlet_datasets(cloudlet_nodes_list, dataset_name, len_train, len_val):
    # Load the data using the load_data function
    train, val, test = load_data(dataset_name, len_train, len_val)

    # Create datasets for each cloudlet
    cloudlet_train_datasets = []
    cloudlet_val_datasets = []
    cloudlet_test_datasets = []

    for cloudlet_nodes in cloudlet_nodes_list:
        node_indices = cloudlet_nodes.tolist()

        # Extract data for the current cloudlet's nodes
        cloudlet_train_data = train.values[...,node_indices]
        cloudlet_val_data = val.values[...,node_indices]
        cloudlet_test_data = test.values[...,node_indices]

        # Append the extracted data to the lists of cloudlet datasets
        cloudlet_train_datasets.append(cloudlet_train_data)
        cloudlet_val_datasets.append(cloudlet_val_data)
        cloudlet_test_datasets.append(cloudlet_test_data)

    return cloudlet_train_datasets, cloudlet_val_datasets, cloudlet_test_datasets

def create_cloudlet_3d_datasets(cloudlet_nodes_list, dataset_name, len_train, len_val):
    # Load the 3D dataset
    train, val, test = load_3d_data(dataset_name, len_train, len_val)

    # Create datasets for each cloudlet
    cloudlet_train_datasets = []
    cloudlet_val_datasets = []
    cloudlet_test_datasets = []

    for cloudlet_nodes in cloudlet_nodes_list:
        node_indices = cloudlet_nodes.tolist()

        # Extract data for the current cloudlet's nodes
        cloudlet_train_data = train[:, node_indices]
        cloudlet_val_data = val[:, node_indices]
        cloudlet_test_data = test[:, node_indices]

        # Append the extracted data to the lists of cloudlet datasets
        cloudlet_train_datasets.append(cloudlet_train_data)
        cloudlet_val_datasets.append(cloudlet_val_data)
        cloudlet_test_datasets.append(cloudlet_test_data)

    return cloudlet_train_datasets, cloudlet_val_datasets, cloudlet_test_datasets

def get_cloudlet_adj_matrices(adj_matrix, cloudlet_nodes_list):
    # Initialize an empty list to store adjacency matrices for each cloudlet
    cloudlet_adj_matrices = []

    # Iterate over each cloudlet in the list of cloudlets
    for cloudlet_nodes in cloudlet_nodes_list:
        # Extract the indices corresponding to nodes in the cloudlet
        node_indices = np.array(cloudlet_nodes)
        
        # Create a mask to select rows and columns corresponding to the cloudlet nodes
        mask = np.in1d(adj_matrix.row, node_indices) & np.in1d(adj_matrix.col, node_indices)

        # Extract the data, rows, and columns based on the mask
        data = adj_matrix.data[mask]
        rows = adj_matrix.row[mask]
        cols = adj_matrix.col[mask]

        # Create a new COO matrix for the cloudlet by applying the mask
        cloudlet_adj = sp.coo_matrix((data, (rows, cols)), shape=(len(cloudlet_nodes), len(cloudlet_nodes)))

        # Convert the cloudlet adjacency matrix to CSR format for efficient processing
        cloudlet_adj = cloudlet_adj.tocsr()

        # Append the cloudlet adjacency matrix to the list
        cloudlet_adj_matrices.append(cloudlet_adj)

    return cloudlet_adj_matrices