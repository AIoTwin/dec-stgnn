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

from script import dataloader, utility, earlystopping
from model import models
from actors import cloudlet, master_server
from torch_geometric.utils import from_scipy_sparse_matrix, k_hop_subgraph

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
    # if you change stblock_num to greater number, you have to change n_his to a bigger number, otherwise it will throw an error
    parser = argparse.ArgumentParser(description='STGCN') # create ArgumentParser object
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay']) # selected dataset
    parser.add_argument('--n_his', type=int, default=12) # history window size
    parser.add_argument('--n_pred', type=int, default=12, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5) # never used???
    parser.add_argument('--Kt', type=int, default=3) # temporal kernel size (Kt)
    parser.add_argument('--stblock_num', type=int, default=2) # number of ST blocks
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu']) # select activation function (GLU, or GTU)
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2]) # spatial kernel size used in convolutional operations
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv']) # Select either Chebyshev Polynomial Approximation or Generalization of Graph Convolution
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj']) # GSO (Graph Shift Operator) - used in utility.py
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True') # Add bias to trainable parameters or not (True - weight + bias, False - weight)
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.00001, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam') # optimization algorithm
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--cloudlet_num', type=int, default=7, help='the number of cloudlets for semi-dec training, default as 10')
    parser.add_argument('--cloudlet_location_data', type=str, default="experiment_1", help='Experiment name from cloudlet location json file')
    parser.add_argument('--enable_es', type=bool, default=False, help='Disable or enable early stopping')
    parser.add_argument('--enable_seed', type=bool, default=False, help='Disable or enable fixed seed')
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
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
    logging.info(f"Ko value: {Ko}")

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
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
    root_dir = os.path.dirname(os.path.abspath(__file__))
    cloudlet_info_path = os.path.join(root_dir, f"locations/{args.dataset}", "locations.json")

    cloudlet_data_json = utility.load_json_file(cloudlet_info_path)
    cloudlets, radius_km = utility.get_cloudlet_location_info_from_json(args.cloudlet_location_data, cloudlet_data_json)

    if (len(cloudlets) != args.cloudlet_num):
        assert False, f'Number of defined cloudlets doesn\'t match the number of cloudlets in json file -> {args.cloudlet_num}\{len(cloudlets)}'

    # load adjacency matrix and number of nodes
    adj, _ = dataloader.load_adj(args.dataset)

    # Get edge_index from adj matrix
    edge_index, _ = from_scipy_sparse_matrix(adj)

    cln_nodes_list = utility.partition_nodes_to_cloudlets_by_range_proximity(cloudlets, radius_km, args.dataset)

    for i in range(len(cln_nodes_list)):
        cln_nodes_list[i].sort()

    cln_nodes_subgraph_list = []
    cln_edge_index_list = []
    cln_node_map_list = []

    for cln_nodes in cln_nodes_list:
        cln_nodes_subgraph, cln_edge_index, cln_node_map, _ = k_hop_subgraph(cln_nodes, args.stblock_num, edge_index, relabel_nodes=True, num_nodes=adj.shape[0])
        cln_nodes_subgraph_list.append(cln_nodes_subgraph)
        cln_edge_index_list.append(cln_edge_index)
        cln_node_map_list.append(cln_node_map)

    # load dataset from specific file path
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    dataset = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))  # If default (metr-la): [34271 rows x 207 columns]
    data_col = dataset.shape[0]

    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)
    
    # 1st is for cloudlets, 2nd is for master aggregator server
    cln_train_datasets, cln_val_datasets, cln_test_datasets = dataloader.create_cloudlet_datasets(cln_nodes_subgraph_list, args.dataset, len_train, len_val)
    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)

    # This is for cloudlets,...
    cln_train_list = []
    cln_val_list = []
    cln_test_list = []

    # Shared zscore standard scaler
    master_zscore = preprocessing.StandardScaler()

    train, val, test = utility.zscore_preprocess_2d_data(master_zscore, train.values, val.values, test.values, use_fit_transform=True)

    # make cloudlets dont have separate zscore, let them share 1
    # use zscaler on all training param, and when cloudlets use transforms, dont use fit
    for (cln_train, cln_val, cln_test) in zip(cln_train_datasets, cln_val_datasets, cln_test_datasets):
        cln_train, cln_val, cln_test = utility.zscore_preprocess_2d_data(master_zscore, cln_train, cln_val, cln_test)

        cln_train_list.append(cln_train)
        cln_val_list.append(cln_val)
        cln_test_list.append(cln_test)   

    # Test and val dataset for master (we don't need to train master model, only test aggregated cloudlet models and validate for early stopping)
    x_val_master, y_test_master = dataloader.data_transform(val, args.n_his, args.n_pred, device)
    val_data_master = utils.data.TensorDataset(x_val_master, y_test_master)
    val_iter_master = utils.data.DataLoader(dataset=val_data_master, batch_size=args.batch_size, shuffle=False)

    x_test_master, y_test_master = dataloader.data_transform(test, args.n_his, args.n_pred, device)
    test_data_master = utils.data.TensorDataset(x_test_master, y_test_master)
    test_iter_master = utils.data.DataLoader(dataset=test_data_master, batch_size=args.batch_size, shuffle=False)
    
    # transform scaled data into input-output pairs suitable for training
    x_train = []
    y_train = []
    for df in cln_train_list:
        x, y = dataloader.data_transform(df, args.n_his, args.n_pred, device)
        x_train.append(x)
        y_train.append(y)

    x_val = []
    y_val = []
    for df in cln_val_list:
        x, y = dataloader.data_transform(df, args.n_his, args.n_pred, device)
        x_val.append(x)
        y_val.append(y)

    x_test = []
    y_test = []
    for df in cln_test_list:
        x, y = dataloader.data_transform(df, args.n_his, args.n_pred, device)
        x_test.append(x)
        y_test.append(y)

    train_datas = []
    for (x, y) in zip(x_train, y_train):
        train_data = utils.data.TensorDataset(x, y)
        train_datas.append(train_data)
    train_iters = []
    for td in train_datas:
        train_iter = utils.data.DataLoader(dataset=td, batch_size=args.batch_size, shuffle=True)
        train_iters.append(train_iter)
    
    val_datas = []
    for (x, y) in zip (x_val, y_val):
        val_data = utils.data.TensorDataset(x, y)
        val_datas.append(val_data)
    val_iters = []
    for vd in val_datas:
        val_iter = utils.data.DataLoader(dataset=vd, batch_size=args.batch_size, shuffle=False)
        val_iters.append(val_iter)

    test_datas = []
    for (x, y) in zip (x_test, y_test):
        test_data = utils.data.TensorDataset(x, y)
        test_datas.append(test_data)
    test_iters = []
    for td in test_datas:
        test_iter = utils.data.DataLoader(dataset=td, batch_size=args.batch_size, shuffle=False)
        test_iters.append(test_iter)

    return train_iters, val_iters, test_iters, cln_edge_index_list, cln_node_map_list, master_zscore, test_iter_master, val_iter_master, edge_index

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

def setup_loss_optimizer_and_scheduler(model):
    loss = nn.MSELoss()

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

    return loss, optimizer, scheduler

# Function to calculate FLOPs for a batch
def calculate_flops_for_epoch(model, cln_models):
    # Total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    
    # FLOPs for initialization (copy operation)
    flops_init = total_params
    
    # FLOPs for accumulation (adding parameters)
    flops_accumulation = total_params * (len(cln_models) - 1)
    
    # FLOPs for averaging (dividing parameters)
    flops_averaging = total_params
    
    # Total FLOPs for average_model
    total_flops = flops_init + flops_accumulation + flops_averaging
    
    print(f"FLOPs for central aggregator: {total_flops:,}")

if __name__ == "__main__":
    args, device, blocks = get_parameters()

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logs_folder = os.path.join('logs', f"{current_datetime}_{args.dataset}_pred-{args.n_pred * 5}min_his-{args.n_his * 5}min_semi-dec-fl-distance")

    train_iters, val_iters, test_iters, cln_edge_index_list, cln_node_map_list, master_zscore, test_iter_master, val_iter_master, edge_index_master = data_preparate(args, device)
    
    edge_index_master = edge_index_master.to(device)
    for i in range(len(cln_edge_index_list)):
        cln_edge_index_list[i] = cln_edge_index_list[i].to(device)
    for i in range(len(cln_node_map_list)):
        cln_node_map_list[i] = cln_node_map_list[i].to(device)
    
    # Master model
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks)
    # cloudlet models
    cloudlet_models = [copy.deepcopy(model) for _ in range(args.cloudlet_num)]

    calculate_flops_for_epoch(model, cloudlet_models)