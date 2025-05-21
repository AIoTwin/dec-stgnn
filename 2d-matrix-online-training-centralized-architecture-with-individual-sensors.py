import logging
import os
import argparse
import math
import random
import tqdm
import numpy as np
import pandas as pd
import copy
import datetime
from sklearn import preprocessing
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.utils.data import Subset, ConcatDataset

from script import dataloader, utility, earlystopping
from model import models
from torch_geometric.utils import from_scipy_sparse_matrix, k_hop_subgraph
from collections import defaultdict

def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for an multi-GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # :m:n - specify workspace size and number of partitions
    # m - size of cuBLAS workspace in MB (4096 MB = 4 GB) -> by configuring its size, we can optimize memory usage and performance based on the specific requirements to our app and hardware
    # n - number of partitions
    # used for temporary storage during matrix operations
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED']=str(seed) # Control hash algoritham
    random.seed(seed) # fixed seed for rng
    np.random.seed(seed) # fixed seed for rng
    torch.manual_seed(seed) # fixed seed for rng on the CPU
    torch.cuda.manual_seed(seed) # fixed seed for rng on the current GPU 
    torch.cuda.manual_seed_all(seed) # fixed seed for rng on all GPUs
    
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True # enable deterministic cuDNN behaviour, ensuring that the convolution algorithm's output is reproducible given the same input and configuration
    # torch.use_deterministic_algorithms(True) # ensure that operations produce deterministic outputs for reproducibility

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN') # create ArgumentParser object
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='pems-bay', choices=['metr-la', 'pems-bay']) # selected dataset
    parser.add_argument('--n_his', type=int, default=12) # history window size
    parser.add_argument('--n_pred', type=int, default=12, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5) # never used???
    parser.add_argument('--Kt', type=int, default=3) # temporal kernel size (Kt)
    parser.add_argument('--stblock_num', type=int, default=2) # number of ST blocks
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu']) # select activation function (GLU, or GTU) there's also relu and silu
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2]) # spatial kernel size used in convolutional operations
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv']) # Select either Chebyshev Polynomial Approximation or Generalization of Graph Convolution
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj']) # GSO (Graph Shift Operator) - used in utility.py
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True') # Add bias to trainable parameters or not (True - weight + bias, False - weight)
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.00001, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', choices=['rmsprop', 'adam', 'adamw'], help='optimizer, default as adam') # optimization algorithm
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--cloudlet_num', type=int, default=7, help='the number of cloudlets for semi-dec training, default as 10')
    parser.add_argument('--cloudlet_location_data', type=str, default="experiment_1", help='Experiment name from cloudlet location json file')
    parser.add_argument('--enable_es', type=bool, default=False, help='Disable or enable early stopping')
    parser.add_argument('--enable_seed', type=bool, default=False, help='Disable or enable fixed seed')
    parser.add_argument('--end_of_initial_data_index', type=int, default=26481, help='End of initial data for training dataset (model will train using datapoints 0-end_of_initial_data_index from all sensors) (METR-LA: 16491) (PeMS-BAY: )')
    parser.add_argument('--data_per_step', type=int, default=250, help='How much data will be taken from entire training dataset each "epoch" (each time model will be trained) DEFAULT: 250')
    parser.add_argument('--adj_matrix_type', type=str, default="original", help="original | no_neighbours")
    args = parser.parse_args()

    # For stable experiment results
    if (args.enable_seed == True):
        print(f"Enabling seed")
        set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Calculate kernel size output (Ko) based on:
    # history window size (n_his)
    # temporal kernel size (Kt)
    # number of ST block
    # With default values: Ko = 12 - (3 - 1) * 2 * 2 = 12 - 2 * 2 * 2 = 12 - 8 = 4
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size for each stblock_num and output layer
    blocks = [] # if default (stblock_num = 2): [[1], [64, 16, 64], [64, 16, 64], [128, 128], [1]] 
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    
    return args, device, blocks

def data_preparate(args, device):
    # load adjacency matrix and number of nodes
    if (args.adj_matrix_type == "original"):
        print(f"Using original adjacency matrix")
        adj, n_vertex = dataloader.load_adj(args.dataset)
    elif (args.adj_matrix_type == "no_neighbours"):
        print(f"Using no neighbours adjacency matrix")
        adj, n_vertex = dataloader.load_adj_direct_no_neighbours(args.dataset)
    else:
        print(f"Wrong adj_matrix_type selected. Expected values are original | no_neighbours. You put: {args.adj_matrix_type}")
        exit(1)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    # load dataset from specific file path
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)
    print(f"Total amount of data: {data_col}")
    print(f"Data for train: {len_train}")
    print(f"Data for validation: {len_val}")
    print(f"Data for test: {len_test}")
    
    # adj_list = defaultdict(list)
    # for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
    #     adj_list[src].append(dst)
    # adj_list = dict(adj_list)
    # sorted_adj_list = {node: adj_list[node] for node in sorted(adj_list)}
    # for node, neighbours in sorted_adj_list.items():
    #     print(f"Node {node} has neighbors: {neighbours}")

    if ((len_train - args.end_of_initial_data_index) % args.data_per_step == 0):
        print("End index and data step correctly selected!")
    else:
        print(f"End index and data step WRONGLY selected: {(len_train - args.end_of_initial_data_index) % args.data_per_step}")
        exit(1)

    # data is standardized using Z-score normalization
    zscore = preprocessing.StandardScaler()

    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)

    node_val_list = []
    node_train_list = []
    node_test_list = []
    for node_id in range (n_vertex):
        node_val = val.values[...,node_id]
        node_train = train.values[...,node_id]
        node_test = test.values[...,node_id]

        node_val_list.append(node_val)
        node_train_list.append(node_train)
        node_test_list.append(node_test)

    nodes_edge_index_list = []

    for node_id in range (n_vertex):
        _, node_edge_index, _, _ = k_hop_subgraph(node_id, args.stblock_num, edge_index, relabel_nodes=True, num_nodes=adj.shape[0])
        nodes_edge_index_list.append(node_edge_index)

    train, val, test = utility.zscore_preprocess_2d_data(zscore, train.values, val.values, test.values, use_fit_transform=True)

    node_vals = []
    node_trains = []
    node_tests = []

    # use zscaler on all training param, and when cloudlets use transforms, dont use fit
    for node_train, node_val, node_test in zip(node_train_list, node_val_list, node_test_list):
        node_train, node_val, node_test = utility.zscore_preprocess_2d_data_1_node(zscore, node_train, node_val, node_test)

        node_vals.append(node_val)
        node_trains.append(node_train)
        node_tests.append(node_test)

    # transform data into input-output pairs suitable for training
    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
    x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)

    # Create TensorDataset object for validation, and testing sets
    # Create DataLoaders objects for iterating over the dataset in batches during validating, and testing
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    # For individual nodes
    node_x_trains = []
    node_y_trains = []
    for df in node_trains:
        x, y = dataloader.data_transform(df, args.n_his, args.n_pred, device)
        node_x_trains.append(x)
        node_y_trains.append(y)

    node_x_vals = []
    node_y_vals = []
    for df in node_vals:
        x, y = dataloader.data_transform(df, args.n_his, args.n_pred, device)
        node_x_vals.append(x)
        node_y_vals.append(y)

    node_x_tests = []
    node_y_tests = []
    for df in node_tests:
        x, y = dataloader.data_transform(df, args.n_his, args.n_pred, device)
        node_x_tests.append(x)
        node_y_tests.append(y)

    node_train_datas = []
    for (x, y) in zip(node_x_trains, node_y_trains):
        node_train_data = utils.data.TensorDataset(x, y)
        node_train_datas.append(node_train_data)
    node_train_iters = []
    for td in node_train_datas:
        node_train_iter = utils.data.DataLoader(dataset=td, batch_size=args.batch_size, shuffle=True)
        node_train_iters.append(node_train_iter)

    node_val_datas = []
    for (x, y) in zip(node_x_vals, node_y_vals):
        node_val_data = utils.data.TensorDataset(x, y)
        node_val_datas.append(node_val_data)
    node_val_iters = []
    for td in node_val_datas:
        node_val_iter = utils.data.DataLoader(dataset=td, batch_size=args.batch_size, shuffle=True)
        node_val_iters.append(node_val_iter)

    node_test_datas = []
    for (x, y) in zip(node_x_tests, node_y_tests):
        node_test_data = utils.data.TensorDataset(x, y)
        node_test_datas.append(node_test_data)
    node_test_iters = []
    for td in node_test_datas:
        node_test_iter = utils.data.DataLoader(dataset=td, batch_size=args.batch_size, shuffle=True)
        node_test_iters.append(node_test_iter)

    return edge_index, zscore, len_train, train, x_train, y_train, val_iter, test_iter, node_x_trains, node_y_trains, node_val_iters, node_test_iters, nodes_edge_index_list, n_vertex

def prepare_model(args, blocks):
    loss = nn.MSELoss() # init Mean Squared Error (MSE) loss function
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience) # init early stopping object

    # RUN IN PyTorch Geometric!!!!!!!!
    model = models.STGCNGraphConvPyG(args, blocks).to(device)

    # define optimization algorithm
    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler

def train(loss, args, optimizer, scheduler, es, model, len_train, train_dataset, x_train, y_train, val_iter, edge_index, writer, logs_folder, zscore, node_x_trains, node_y_trains, node_val_iters, node_edge_indices, n_vertex):
    num_epochs = int((len_train - args.end_of_initial_data_index) / args.data_per_step)

    current_train = train_dataset[:args.end_of_initial_data_index]
    inital_x_train = x_train[:args.end_of_initial_data_index]
    inital_y_train = y_train[:args.end_of_initial_data_index]
    train_data = utils.data.TensorDataset(inital_x_train, inital_y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

    adj_list = defaultdict(list)
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj_list[src].append(dst)
    adj_list = dict(adj_list)
    sorted_adj_list = {node: adj_list[node] for node in sorted(adj_list)}

    for epoch in range(1, num_epochs + 1):
        l_sum, n = 0.0, 0
        model.train()
        for batch_idx, (x, y) in enumerate(tqdm.tqdm(train_iter)):
            x.requires_grad = True  # Enable gradient tracking for node features

            y_pred = model(x, edge_index).view(len(x), -1)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()

            # Compute gradient norms per node
            grad_norms = torch.norm(x.grad, dim=1)  # Compute L2 (Euclidean) norm of gradients (shape: (batch_size, n_his, n_vertex))
            batch_size, num_node_features, n_his, num_nodes = x.shape

            neighbour_influence = {}
            nodes_influence = torch.mean(grad_norms, dim=(0,1))
            for tgt in sorted_adj_list: # iterate over keys (nodes only)
                neighbour_influence[tgt] = {}
                
                for neighbour in sorted_adj_list[tgt]: # iterate over each neighbour
                    # 1.) (for line below) compute mean once outside the 2nd for loop (by combining the 1st 2 dimensions)
                    # influence = grad_norms[:, :, neighbour].mean().item() # Averaging across batch & time
                    neighbour_influence[tgt][neighbour] = nodes_influence[neighbour]

                # print(f"Epoch {epoch}: Node {tgt} neighbor contributions: {neighbour_influence[tgt]}")
            
            # for node, contributions in node_contributions.items():
            #     print(f"Epoch {epoch}: Node {node} neighbor contributions: {dict(contributions)}")

            # for i in range(edge_index.shape[1]):  # Loop through edges
            #     src, tgt = edge_index[:, i]  # Get source and target nodes
            #     src_node, tgt_node = src.item(), tgt.item()
            #     # Ensure indices are within valid range
            #     if tgt_node >= num_nodes or src_node >= num_nodes:
            #         print(f"Warning: src_node={src_node} or tgt_node={tgt_node} is out of bounds (num_nodes={num_nodes})")
            #         continue
            #     batch_src_idx = src_node % batch_size
            #     if batch_src_idx >= batch_size:
            #         print(f"Warning: src.item()={src_node} is out of batch bounds (size={batch_size})")
            #         continue
            #     # Compute contribution of src_node to tgt_node at each time step
            #     for t in range(n_his):  # Loop over time steps
            #         grad_value = grad_norms[batch_src_idx, t, src_node].item()  # Correct indexing
            #         neighbor_contributions[tgt_node][src_node].append(grad_value)

            # neighbor_contributions = defaultdict(lambda: defaultdict(list))  # node -> {neighbor -> [contributions]}
            # for i in range(edge_index.shape[1]):  # Loop through edges
            #     src, tgt = edge_index[:, i]  # Get source and target nodes
            #     src_node, tgt_node = src.item(), tgt.item()
            #     # Ensure indices are within valid range
            #     if tgt_node >= num_nodes or src_node >= num_nodes:
            #         print(f"Warning: src_node={src_node} or tgt_node={tgt_node} is out of bounds (num_nodes={num_nodes})")
            #         continue
            #     batch_src_idx = src_node % batch_size
            #     if batch_src_idx >= batch_size:
            #         print(f"Warning: src.item()={src_node} is out of batch bounds (size={batch_size})")
            #         continue
            #     # Compute contribution of src_node to tgt_node at each time step
            #     for t in range(n_his):  # Loop over time steps
            #         grad_value = grad_norms[batch_src_idx, t, src_node].item()  # Correct indexing
            #         neighbor_contributions[tgt_node][src_node].append(grad_value)

            optimizer.step()

            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        # utility.save_gradient_neighbour_contribution(logs_folder, "neighbour_contribution", neighbour_influence)
        scheduler.step()
        model.eval()

        if epoch < num_epochs:
            # Create validation dataset
            new_x_val = x_train[
                args.end_of_initial_data_index + (args.data_per_step * (epoch - 1)):
                args.end_of_initial_data_index + (args.data_per_step * (epoch))
                ]
            new_y_val = y_train[
                args.end_of_initial_data_index + (args.data_per_step * (epoch - 1)):
                args.end_of_initial_data_index + (args.data_per_step * (epoch))
            ]
            new_val_data = utils.data.TensorDataset(new_x_val, new_y_val)
            new_val_iter = utils.data.DataLoader(dataset=new_val_data, batch_size=args.batch_size, shuffle=False)

            # Calculate validation loss
            val_loss = val(model, new_val_iter, edge_index)
        elif epoch == num_epochs:
            val_loss = val(model, val_iter, edge_index)

        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        # Get number of parameters
        num_of_parameters = sum(p.numel() for p in model.parameters())
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB | Num of parameters: {:d}'.\
            format(epoch, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc, num_of_parameters))
        # Save logs to CSV and tensorboard
        writer.add_scalar(f"Loss/train", l_sum / n, epoch)
        writer.add_scalar(f"Loss/val", val_loss, epoch)
        utility.save_val_logs(logs_folder, "central", epoch, optimizer.param_groups[0]['lr'], l_sum / n, val_loss.item(), gpu_mem_alloc)

        if epoch < num_epochs:
            # val_MAE, val_RMSE, val_WMAPE = utility.evaluate_pyg_metric_master(model, new_val_iter, zscore, edge_index)
            val_MAE, val_RMSE, val_WMAPE, val_mae_per_node, val_rmse_per_node, val_wmape_per_node = utility.evaluate_pyg_metric_single_node(model, new_val_iter, zscore, edge_index, n_vertex)
        elif epoch == num_epochs:
            val_MAE, val_RMSE, val_WMAPE, val_mae_per_node, val_rmse_per_node, val_wmape_per_node = utility.evaluate_pyg_metric_single_node(model, val_iter, zscore, edge_index, n_vertex)
            # val_MAE, val_RMSE, val_WMAPE = utility.evaluate_pyg_metric_master(model, val_iter, zscore, edge_index)
        utility.save_val_metric_logs(logs_folder, f"central", epoch, val_MAE, val_RMSE, val_WMAPE)
        print(f'Epoch: {epoch:03d} | MAE {val_MAE:.6f} | RMSE {val_RMSE:.6f} | WMAPE {val_WMAPE:.8f}')

        for node_idx, (val_mae_node, val_rmse_node, val_wmape_node) in enumerate(zip(val_mae_per_node, val_rmse_per_node, val_wmape_per_node)):
            utility.save_val_metric_logs(logs_folder, node_idx, epoch, val_mae_node, val_rmse_node, val_wmape_node)

        # Create new train dataset
        current_train = train_dataset[:args.end_of_initial_data_index + (args.data_per_step * (epoch - 1))]
        new_train_datastep = train_dataset[
            args.end_of_initial_data_index + (args.data_per_step * (epoch - 1)):
            args.end_of_initial_data_index + (args.data_per_step * (epoch))
        ]

        random_sample_size = (args.batch_size - 1) * args.data_per_step
        # Randomly sample indices from current_train
        if len(current_train) > random_sample_size:
            random_indices = np.random.choice(current_train.shape[0], random_sample_size, replace=False)
        else:
            # If current_train has fewer than the required samples, take all of it
            random_indices = current_train
        new_train_datastep = [train_dataset[idx] for idx in range(len(new_train_datastep))]

        new_x_train = x_train[
            args.end_of_initial_data_index + (args.data_per_step * (epoch - 1)):
            args.end_of_initial_data_index + (args.data_per_step * (epoch))
        ]
        sampled_x_train = x_train[random_indices, :]
        new_x_train = torch.cat((sampled_x_train, new_x_train), dim=0)
        new_y_train = y_train[
            args.end_of_initial_data_index + (args.data_per_step * (epoch - 1)):
            args.end_of_initial_data_index + (args.data_per_step * (epoch))
        ]
        sampled_y_train = y_train[random_indices, :]
        new_y_train = torch.cat((sampled_y_train, new_y_train), dim=0)

        train_data = utils.data.TensorDataset(new_x_train, new_y_train)
        train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

@torch.no_grad()
def val(model, val_iter, edge_index):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x, edge_index).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)

@torch.no_grad() 
def test(zscore, loss, model, test_iter, args, edge_index, logs_folder):
    model.eval()
    test_MSE = utility.evaluate_pyg_model_master(model, loss, test_iter, edge_index)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_pyg_metric_master(model, test_iter, zscore, edge_index)
    utility.save_test_logs(logs_folder, f"central", test_MSE, test_MAE, test_RMSE, test_WMAPE)
    print(f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

if __name__ == "__main__":
    # Logging
    #logger = logging.getLogger('stgcn')
    #logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    writer = SummaryWriter()

    args, device, blocks = get_parameters()

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"Start date & time: {current_datetime}")
    logs_folder = os.path.join('logs', f"{current_datetime}_{args.dataset}_pred-{args.n_pred * 5}min_his-{args.n_his * 5}min_centralized_online-training_individual-sensors")

    edge_index, zscore, len_train, train_dataset, x_train, y_train, val_iter, test_iter, node_x_trains, node_y_trains, node_val_iters, node_test_iters, nodes_edge_index_list, n_vertex = data_preparate(args, device)

    for i in range(len(nodes_edge_index_list)):
        nodes_edge_index_list[i] = nodes_edge_index_list[i].to(device)

    edge_index = edge_index.to(device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks)

    train(loss, args, optimizer, scheduler, es, model, len_train, train_dataset, x_train, y_train, val_iter, edge_index, writer, logs_folder, zscore, node_x_trains, node_y_trains, node_val_iters, nodes_edge_index_list, n_vertex)
    test(zscore, loss, model, test_iter, args, edge_index, logs_folder)

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"End date & time: {current_datetime}")