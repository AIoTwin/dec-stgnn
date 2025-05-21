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
    # adj, _ = dataloader.load_adj(args.dataset)
    adj_no_neighbours, _ = dataloader.load_adj_direct_no_neighbours(args.dataset)
    adj_original, n_vertex = dataloader.load_adj(args.dataset)

    edge_index_no_neighbours, _ = from_scipy_sparse_matrix(adj_no_neighbours)
    edge_index_original, _ = from_scipy_sparse_matrix(adj_original)

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
    print(f"len_train: {len_train}")
    print(f"len_val: {len_val}")
    print(f"len_test: {len_test}")

    # data is standardized using Z-score normalization
    zscore = preprocessing.StandardScaler()

    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
    train, val, test = utility.zscore_preprocess_2d_data(zscore, train.values, val.values, test.values, use_fit_transform=True)

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

    def aaaa(data_iter):
        for batch_idx, (x, y) in enumerate(data_iter):
            y_actual = zscore.inverse_transform(y.cpu().numpy()) #y shape -> [batch_size, n_vertex]
            threshold = 1e-5
            if np.all(np.abs(y_actual) < threshold):
                print(f"Batch {batch_idx}: ALL values are close to zero (below threshold {threshold})")
            else:
                print(f"Batch {batch_idx}: NOT all values are close to zero")
            # if (np.any(y_actual == 0)):
            #     print(f"There are zeros")
            #     continue
            # else:
            #     #print(f"There are no zeros")
            #     continue
    
    # zero_counts = count_zeros_in_dataset(train_iter, zscore)
    # zero_counts = count_zeros_in_dataset(val_iter, zscore)
    # aaaa(train_iter)
    aaaa(val_iter)
    # print("Zero count per timestep and sensor:\n", zero_counts)

    return edge_index_original, edge_index_no_neighbours, n_vertex, zscore, train_iter, val_iter, test_iter

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

def setup_loss_optimizer_and_scheduler(args, model):
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

def train(args, loss_original, optimizer_original, scheduler_original, model_original, loss_no_neighbours, optimizer_no_neighbours, scheduler_no_neighbours, model_no_neighbours, train_iter, val_iter, edge_index_original, edge_index_no_neighbours, writer, logs_folder, zscore):
    for epoch in range(args.epochs):
        l_sum_original, n_original = 0.0, 0
        l_sum_no_neighbours, n_no_neighbours = 0.0, 0
        model_original.train() # train the model
        for x, y in tqdm.tqdm(train_iter):
            y_pred_original = model_original(x, edge_index_original).view(len(x), -1)  # [batch_size, num_nodes]
            l_original = loss_original(y_pred_original, y) # compute loss between predicted output (y_pred) and actual target (y)
            optimizer_original.zero_grad() # clear the gradients of all optimized tensors
            l_original.backward() # compute the gradients of the loss with respect to the model parameters
            optimizer_original.step() # update model parameters based on computed gradients and optimization algorithm
            l_sum_original += l_original.item() * y.shape[0] # accumulate batch loss
            n_original += y.shape[0] # accumulate number of instances processed
        scheduler_original.step() # adjust learning rate parameter
        val_loss_original = val(model_original, val_iter, edge_index_original) # compute validation loss
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        # Get number of parameters
        num_of_parameters = sum(p.numel() for p in model_original.parameters())
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB | Num of parameters: {:d}'.\
            format(epoch+1, optimizer_original.param_groups[0]['lr'], l_sum_original / n_original, val_loss_original, gpu_mem_alloc, num_of_parameters))

        # log validation metrics
        model_original.eval()
        d_original, y_pred_original = utility.evaluate_pyg_metric_analysis(model_original, val_iter, zscore, edge_index_original)
        if (args.epochs == epoch+1):
            utility.save_d_analysis_logs(logs_folder, f"d_original", d_original)
            utility.save_d_analysis_logs(logs_folder, f"y_pred_original", y_pred_original)

        model_no_neighbours.train()
        for x, y in tqdm.tqdm(train_iter):
            y_pred_no_neighbours = model_no_neighbours(x, edge_index_no_neighbours).view(len(x), -1)  # [batch_size, num_nodes]
            l_no_neighbours = loss_no_neighbours(y_pred_no_neighbours, y) # compute loss between predicted output (y_pred) and actual target (y)
            optimizer_no_neighbours.zero_grad() # clear the gradients of all optimized tensors
            l_no_neighbours.backward() # compute the gradients of the loss with respect to the model parameters
            optimizer_no_neighbours.step() # update model parameters based on computed gradients and optimization algorithm
            l_sum_no_neighbours += l_no_neighbours.item() * y.shape[0] # accumulate batch loss
            n_no_neighbours += y.shape[0] # accumulate number of instances processed
        scheduler_no_neighbours.step() # adjust learning rate parameter
        val_loss_no_neighbours = val(model_no_neighbours, val_iter, edge_index_no_neighbours) # compute validation loss
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        # Get number of parameters
        num_of_parameters = sum(p.numel() for p in model_no_neighbours.parameters())
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB | Num of parameters: {:d}'.\
            format(epoch+1, optimizer_no_neighbours.param_groups[0]['lr'], l_sum_no_neighbours / n_no_neighbours, val_loss_no_neighbours, gpu_mem_alloc, num_of_parameters))

        # log validation metrics
        model_no_neighbours.eval()
        d_no_neighbours, y_pred_no_neighbours = utility.evaluate_pyg_metric_analysis(model_no_neighbours, val_iter, zscore, edge_index_no_neighbours)
        if (args.epochs == epoch+1):
            utility.save_d_analysis_logs(logs_folder, f"d_no_neighbours", d_no_neighbours)
            utility.save_d_analysis_logs(logs_folder, f"y_pred_no_neighbours", y_pred_no_neighbours)

        if (args.epochs == epoch+1):
            d_delta = d_original - d_no_neighbours
            utility.save_d_analysis_logs(logs_folder, f"delta", d_delta)

@torch.no_grad()
def val(model, val_iter, edge_index):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x, edge_index).view(len(x), -1)
        l = loss_original(y_pred, y)
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
    logs_folder = os.path.join('logs', f"{current_datetime}_{args.dataset}_pred-{args.n_pred * 5}min_his-{args.n_his * 5}min_centralized_analysis-neighbours-and-no-neighbours")

    edge_index_original, edge_index_no_neighbours, n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)

    edge_index_original = edge_index_original.to(device)
    edge_index_no_neighbours = edge_index_no_neighbours.to(device)
    loss_original, es_original, model_original, optimizer_original, scheduler_original = prepare_model(args, blocks)
    model_no_neighbours = copy.deepcopy(model_original)
    loss_no_neighbours, optimizer_no_neighbours, scheduler_no_neighbours = setup_loss_optimizer_and_scheduler(args, model_no_neighbours)

    best_epoch = train(args, loss_original, optimizer_original, scheduler_original, model_original, loss_no_neighbours, optimizer_no_neighbours, scheduler_no_neighbours, model_no_neighbours, train_iter, val_iter, edge_index_original, edge_index_no_neighbours, writer, logs_folder, zscore)

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"End date & time: {current_datetime}")