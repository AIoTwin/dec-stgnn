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
    parser = argparse.ArgumentParser(description='STGCN') # create ArgumentParser object
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='pems-bay', choices=['metr-la', 'pems-bay']) # selected dataset
    parser.add_argument('--n_his', type=int, default=12) # history window size
    parser.add_argument('--n_pred', type=int, default=3, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5) # never used???
    parser.add_argument('--Kt', type=int, default=3) # temporal kernel size (Kt)
    parser.add_argument('--stblock_num', type=int, default=2) # number of ST blocks
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu']) # select activation function (GLU, or GTU) there's also relu and silu
    parser.add_argument('--Ks', type=int, default=2, choices=[3, 2]) # spatial kernel size used in convolutional operations
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
    root_dir = os.path.dirname(os.path.abspath(__file__))
    cloudlet_info_path = os.path.join(root_dir, f"locations/{args.dataset}", "locations.json")

    cloudlet_data_json = utility.load_json_file(cloudlet_info_path)
    cloudlets, radius_km = utility.get_cloudlet_location_info_from_json(args.cloudlet_location_data, cloudlet_data_json)

    if (len(cloudlets) != args.cloudlet_num):
        assert False, f'Number of defined cloudlets doesn\'t match the number of cloudlets in json file -> {args.cloudlet_num}\{len(cloudlets)}'

    # load adjacency matrix and number of nodes
    # adj, _ = dataloader.load_adj(args.dataset)
    adj, _ = dataloader.load_adj_direct_no_neighbours(args.dataset)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    cln_nodes_list = utility.partition_nodes_to_cloudlets_by_range_proximity(cloudlets, radius_km, args.dataset)

    for i in range(len(cln_nodes_list)):
        cln_nodes_list[i].sort()

    cln_nodes_subgraph_list = []
    cln_edge_index_list = []
    cln_node_map_list = []

    for cln_nodes in cln_nodes_list:
        cln_nodes_subgraph, cln_edge_index, cln_node_map, _ = k_hop_subgraph(cln_nodes, args.stblock_num * (args.Ks - 1), edge_index, relabel_nodes=True, num_nodes=adj.shape[0])
        cln_nodes_subgraph_list.append(cln_nodes_subgraph)
        cln_edge_index_list.append(cln_edge_index)
        cln_node_map_list.append(cln_node_map)

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

    # data is standardized using Z-score normalization
    zscore = preprocessing.StandardScaler()

    cln_train_datasets, cln_val_datasets, cln_test_datasets = dataloader.create_cloudlet_datasets(cln_nodes_subgraph_list, args.dataset, len_train, len_val)

    cln_val_list = []
    cln_test_list = []

    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
    train, val, test = utility.zscore_preprocess_2d_data(zscore, train.values, val.values, test.values, use_fit_transform=True)

    # use zscaler on all training param, and when cloudlets use transforms, dont use fit
    for (cln_train, cln_val, cln_test) in zip(cln_train_datasets, cln_val_datasets, cln_test_datasets):
        _, cln_val, cln_test = utility.zscore_preprocess_2d_data(zscore, cln_train, cln_val, cln_test)

        cln_val_list.append(cln_val)
        cln_test_list.append(cln_test)

    # transform data into input-output pairs suitable for training
    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
    x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)

    # Create TensorDataset object for training, validation, and testing sets
    # Create DataLoaders objects for iterating over the dataset in batches during training, validating, and testing
    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    cln_x_vals = []
    cln_y_vals = []
    for df in cln_val_list:
        x, y = dataloader.data_transform(df, args.n_his, args.n_pred, device)
        cln_x_vals.append(x)
        cln_y_vals.append(y)

    cln_x_tests = []
    cln_y_tests = []
    for df in cln_test_list:
        x, y = dataloader.data_transform(df, args.n_his, args.n_pred, device)
        cln_x_tests.append(x)
        cln_y_tests.append(y)

    cln_val_datas = []
    for (x, y) in zip (cln_x_vals, cln_y_vals):
        cln_val_data = utils.data.TensorDataset(x, y)
        cln_val_datas.append(cln_val_data)
    cln_val_iters = []
    for vd in cln_val_datas:
        cln_val_iter = utils.data.DataLoader(dataset=vd, batch_size=args.batch_size, shuffle=False)
        cln_val_iters.append(cln_val_iter)

    cln_test_datas = []
    for (x, y) in zip (cln_x_tests, cln_y_tests):
        cln_test_data = utils.data.TensorDataset(x, y)
        cln_test_datas.append(cln_test_data)
    cln_test_iters = []
    for td in cln_test_datas:
        cln_test_iter = utils.data.DataLoader(dataset=td, batch_size=args.batch_size, shuffle=False)
        cln_test_iters.append(cln_test_iter)

    return edge_index, zscore, train_iter, val_iter, test_iter, cln_node_map_list, cln_val_iters, cln_test_iters, cln_edge_index_list

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

def train(loss, args, optimizer, scheduler, es, model, train_iter, val_iter, edge_index, writer, logs_folder, zscore, best_model, cln_val_iters, cln_node_maps, cln_edge_indices):
    best_WMAPE = 100
    best_epoch = 0
    for epoch in range(args.epochs):
        cln_id = 0
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train() # train the model
        for x, y in tqdm.tqdm(train_iter):
            y_pred = model(x, edge_index).view(len(x), -1)  # [batch_size, num_nodes]
            l = loss(y_pred, y) # compute loss between predicted output (y_pred) and actual target (y)
            optimizer.zero_grad() # clear the gradients of all optimized tensors
            l.backward() # compute the gradients of the loss with respect to the model parameters
            optimizer.step() # update model parameters based on computed gradients and optimization algorithm
            l_sum += l.item() * y.shape[0] # accumulate batch loss
            n += y.shape[0] # accumulate number of instances processed
        scheduler.step() # adjust learning rate parameter
        val_loss = val(model, val_iter, edge_index) # compute validation loss
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        # Get number of parameters
        num_of_parameters = sum(p.numel() for p in model.parameters())
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB | Num of parameters: {:d}'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc, num_of_parameters))
        # Save logs to CSV and tensorboard
        writer.add_scalar(f"Loss/train", l_sum / n, epoch+1)
        writer.add_scalar(f"Loss/val", val_loss, epoch+1)
        utility.save_val_logs(logs_folder, "central", epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss.item(), gpu_mem_alloc)

        # log validation metrics
        model.eval()
        val_MAE, val_RMSE, val_WMAPE = utility.evaluate_pyg_metric_master(model, val_iter, zscore, edge_index)
        utility.save_val_metric_logs(logs_folder, f"central", epoch+1, val_MAE, val_RMSE, val_WMAPE)
        print(f'Epoch: {epoch+1:03d} | MAE {val_MAE:.6f} | RMSE {val_RMSE:.6f} | WMAPE {val_WMAPE:.8f}')

        # Log cloudlet metrics
        for cln_val_iter, cln_node_map, cln_edge_index in zip(cln_val_iters, cln_node_maps, cln_edge_indices):
            cln_val_MAE, cln_val_RMSE, cln_val_WMAPE, _ = utility.validate_pyg_metric(model, cln_val_iter, zscore, cln_edge_index, cln_node_map)
            utility.save_val_metric_logs(logs_folder, cln_id, epoch+1, cln_val_MAE, cln_val_RMSE, cln_val_WMAPE)
            print(f'CLN: {cln_id:03d} | Epoch: {epoch+1:03d} | MAE {cln_val_MAE:.6f} | RMSE {cln_val_RMSE:.6f} | WMAPE {cln_val_WMAPE:.8f}')
            cln_id = cln_id + 1

        # log variance
        param_variances = utility.compute_parameter_variance_master(model.named_parameters())
        utility.save_variance_logs(logs_folder, f"central", epoch+1, param_variances)

        # Keep best model for testing
        if (val_WMAPE < best_WMAPE):
            best_WMAPE = val_WMAPE
            best_epoch = epoch
            for cln_model_param_1, cln_model_param_2 in zip(best_model.parameters(), model.parameters()):
                cln_model_param_1.data.copy_(cln_model_param_2.data)

        if (args.enable_es == True):
            # check for early stopping
            if es.step(val_loss):
                print('Early stopping.')
                break

    return best_epoch

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
def test(zscore, loss, model, test_iter, args, edge_index, logs_folder, best_epoch):
    model.eval()
    test_MSE = utility.evaluate_pyg_model_master(model, loss, test_iter, edge_index)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_pyg_metric_master(model, test_iter, zscore, edge_index)
    utility.save_test_logs(logs_folder, f"central", test_MSE, test_MAE, test_RMSE, test_WMAPE, best_epoch+1)
    print(f'Epoch {best_epoch+1:03d} | Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

if __name__ == "__main__":
    # Logging
    #logger = logging.getLogger('stgcn')
    #logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    writer = SummaryWriter()

    args, device, blocks = get_parameters()

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"Start date & time: {current_datetime}")
    logs_folder = os.path.join('logs', f"{current_datetime}_{args.dataset}_pred-{args.n_pred * 5}min_his-{args.n_his * 5}min_centralized")

    edge_index, zscore, train_iter, val_iter, test_iter, cln_node_map_list, cln_val_iters, cln_test_iters, cln_edge_index_list = data_preparate(args, device)

    for i in range(len(cln_edge_index_list)):
        cln_edge_index_list[i] = cln_edge_index_list[i].to(device)
    for i in range(len(cln_node_map_list)):
        cln_node_map_list[i] = cln_node_map_list[i].to(device)

    edge_index = edge_index.to(device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks)

    best_model = copy.deepcopy(model)

    best_epoch = train(loss, args, optimizer, scheduler, es, model, train_iter, val_iter, edge_index, writer, logs_folder, zscore, best_model, cln_val_iters, cln_node_map_list, cln_edge_index_list)
    test(zscore, loss, best_model, test_iter, args, edge_index, logs_folder, best_epoch)

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"End date & time: {current_datetime}")