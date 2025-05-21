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
    # cuddn benchmark
    # True - cuDNN runs a set of algorithms to find the best performance for the given input tensors and hardware configuration
    # During the first iteration of a network, cuDNN might spend extra time benchmarking different algorithms to select the fastest one, which can lead to inconsistent runtimes between iterations
    # False - cuDNN uses a pre-selected algorithm for each operation, based on the hardware and input sizes
    # This can lead to more consistent runtimes between iterations, which is important for reproducibility, especially when training deep learning models
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True # enable deterministic cuDNN behaviour, ensuring that the convolution algorithm's output is reproducible given the same input and configuration
    # torch.use_deterministic_algorithms(True) # ensure that operations produce deterministic outputs for reproducibility

def get_parameters():
    # if you change stblock_num to greater number, you have to change n_his to a bigger number, otherwise it will throw an error
    parser = argparse.ArgumentParser(description='STGCN') # create ArgumentParser object
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='pemsd4', choices=['pemsd4']) # selected dataset
    parser.add_argument('--n_his', type=int, default=12) # history window size
    parser.add_argument('--n_pred', type=int, default=3, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5) # never used???
    parser.add_argument('--Kt', type=int, default=3) # temporal kernel size (Kt)
    parser.add_argument('--stblock_num', type=int, default=2) # number of ST blocks
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu']) # select activation function (GLU, or GTU)
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2]) # spatial kernel size used in convolutional operations
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv']) # Select either Chebyshev Polynomial Approximation or Generalization of Graph Convolution
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj']) # GSO (Graph Shift Operator) - used in utility.py
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True') # Add bias to trainable parameters or not (True - weight + bias, False - weight)
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam') # optimization algorithm
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--cloudlet_num', type=int, default=5, help='the number of cloudlets for semi-dec training, default as 10')
    parser.add_argument('--node_features_to_evaluate', type=list, default=[0], help='list of node features you want to evaluate')
    args = parser.parse_args()

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    return args, device

def data_preparate(args, device):
    # load adjacency matrix and number of nodes
    adj, _ = dataloader.load_adj(args.dataset)

    # Get edge_index from adj matrix
    edge_index, _ = from_scipy_sparse_matrix(adj)

    cln_nodes_list = utility.partition_nodes_to_cloudlets(adj, args.cloudlet_num)

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
    dataset = np.load(os.path.join(dataset_path, 'vel.npz'))
    dataset = dataset['data'] # If default (metr-la): [34271 rows x 207 columns x 1 node_features]
    data_col = dataset.shape[0]
    num_node_features = dataset.shape[2]

    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)
    
    # 1st is for cloudlets, 2nd is for master aggregator server
    cln_train_datasets, cln_val_datasets, cln_test_datasets = dataloader.create_cloudlet_3d_datasets(cln_nodes_subgraph_list, args.dataset, len_train, len_val)
    train, val, test = dataloader.load_3d_data(args.dataset, len_train, len_val)

    # This is for cloudlets,...
    cln_train_list = []
    cln_val_list = []
    cln_test_list = []

    # Shared zscore standard scaler
    master_zscore = preprocessing.StandardScaler()

    train, val, test = utility.zscore_preprocess_3d_data(master_zscore, train, val, test, use_fit_transform=True)

    # make cloudlets dont have separate zscore, let them share 1
    # use zscaler on all training param, and when cloudlets use transforms, dont use fit
    for (cln_train, cln_val, cln_test) in zip(cln_train_datasets, cln_val_datasets, cln_test_datasets):
        cln_train, cln_val, cln_test = utility.zscore_preprocess_3d_data(master_zscore, cln_train, cln_val, cln_test)

        cln_train_list.append(cln_train)
        cln_val_list.append(cln_val)
        cln_test_list.append(cln_test)   

    # Test and val dataset for master (we don't need to train master model, only test aggregated cloudlet models and validate for early stopping)
    x_val_master, y_test_master = dataloader.data_3d_transform(val, args.n_his, args.n_pred, device)
    val_data_master = utils.data.TensorDataset(x_val_master, y_test_master)
    val_iter_master = utils.data.DataLoader(dataset=val_data_master, batch_size=args.batch_size, shuffle=False)

    x_test_master, y_test_master = dataloader.data_3d_transform(test, args.n_his, args.n_pred, device)
    test_data_master = utils.data.TensorDataset(x_test_master, y_test_master)
    test_iter_master = utils.data.DataLoader(dataset=test_data_master, batch_size=args.batch_size, shuffle=False)
    
    # transform scaled data into input-output pairs suitable for training
    x_train = []
    y_train = []
    for df in cln_train_list:
        x, y = dataloader.data_3d_transform(df, args.n_his, args.n_pred, device)
        x_train.append(x)
        y_train.append(y)

    x_val = []
    y_val = []
    for df in cln_val_list:
        x, y = dataloader.data_3d_transform(df, args.n_his, args.n_pred, device)
        x_val.append(x)
        y_val.append(y)

    x_test = []
    y_test = []
    for df in cln_test_list:
        x, y = dataloader.data_3d_transform(df, args.n_his, args.n_pred, device)
        x_test.append(x)
        y_test.append(y)

    train_datas = []
    for (x, y) in zip(x_train, y_train):
        train_data = utils.data.TensorDataset(x, y)
        train_datas.append(train_data)
    train_iters = []
    for td in train_datas:
        train_iter = utils.data.DataLoader(dataset=td, batch_size=args.batch_size, shuffle=False)
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

    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
    blocks = []
    blocks.append([num_node_features])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([num_node_features])

    return blocks, train_iters, val_iters, test_iters, cln_edge_index_list, cln_node_map_list, master_zscore, test_iter_master, val_iter_master, edge_index

# use this to create a "master" model, and then args.cloudlet_num models
# instead of for loop args.cloudlet_num times, find a function that copies a model for args.cloudlet_num
# MAKE SURE ALL MODELS ARE EXACTLY THE SAME!!!!
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

def cloudlet_train(cln_losses, args, cln_optimizers, cln_schedulers, es, master_model, cloudlet_models, train_iters, val_iters, cln_edge_index_list, cln_node_map_list, writer, logs_folder, edge_index_master, master_loss, val_iter_master):
    total_data_sent = 0
    for epoch in range(args.epochs):
        cloudlet_id = 1
        total_num_of_parameters = 0
        for (cloudlet_model, train_iter, val_iter, cln_edge_index, cln_node_map, cln_loss, cln_optimizer, cln_scheduler) in zip(cloudlet_models, train_iters, val_iters, cln_edge_index_list, cln_node_map_list, cln_losses, cln_optimizers, cln_schedulers):
            l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
            cloudlet_model.train() # train the model
            for x, y in tqdm.tqdm(train_iter):
                y_pred = utility.permute_4d_y_pred_to_3d(cloudlet_model, x, cln_edge_index)

                y = y[:,cln_node_map,:]
                y_pred = y_pred[:,cln_node_map,:]

                l = cln_loss(y_pred, y) # compute loss between predicted output (y_pred) and actual target (y)
                cln_optimizer.zero_grad() # clear the gradients of all optimized tensors
                l.backward() # compute the gradients of the loss with respect to the model parameters
                cln_optimizer.step() # update model parameters based on computed gradients and optimization algorithm
                l_sum += l.item() * y.shape[0] # accumulate batch loss
                n += y.shape[0] # accumulate number of instances processed
            cln_scheduler.step() # adjust learning rate parameter
            val_loss = val(cloudlet_model, val_iter, cln_edge_index, cln_node_map, cln_loss) # compute validation loss
            # GPU memory usage
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            # Get number of parameter for cloudlet
            cln_num_of_parameters = sum(p.numel() for p in cloudlet_model.parameters())
            total_num_of_parameters += cln_num_of_parameters
            print('CLN: {:03d} | Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB | Num of parameters: {:d}'.\
                format(cloudlet_id, epoch+1, cln_optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc, cln_num_of_parameters))
            # Save logs to CSV and tensorboard
            writer.add_scalar(f"Loss/train_cln-{cloudlet_id}", l_sum / n, epoch+1)
            writer.add_scalar(f"Loss/val_cln-{cloudlet_id}", val_loss, epoch+1)
            utility.save_val_logs(logs_folder, cloudlet_id, epoch+1, cln_optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc)
            cloudlet_id += 1

        # average all clodulet_models into a master_model
        master_model = average_model(cloudlet_models, master_model)
        # copy new master_model to all cloudlet_models
        cloudlet_models = copy_model(cloudlet_models, master_model)

        # Each cloudlet sends it's trainable parameters to the master server...
        # ...and master server sends new trainable parameters to each cloudlet
        total_size_in_bytes_per_epoch = (total_num_of_parameters * args.cloudlet_num) * 2
        total_data_sent += total_size_in_bytes_per_epoch
        print(f"Epoch: {epoch+1:03d} | Size in bytes: {total_size_in_bytes_per_epoch} | in kilobytes: {total_size_in_bytes_per_epoch/1024} | in megabytes: {(total_size_in_bytes_per_epoch/1024)/1024}")
        utility.save_total_trainable_parameters_transfer_size(logs_folder, "master_server", epoch, total_data_sent)
        master_val_loss = master_val(master_model, val_iter_master, edge_index_master, master_loss)

        # check for early stopping
        if es.step(master_val_loss):
            print('Early stopping.')
            break

    # utility.save_total_trainable_parameters_transfer_size(logs_folder, total_data_sent)
    print(f"Total data sent in bytes: {total_data_sent} | in kilobytes: {total_data_sent/1024} | in megabytes: {(total_data_sent/1024)/1024}")

def copy_model(cloudlet_models, master_model):
    # Copy the new master_model to all cloudlet_models
    for cloudlet_model in cloudlet_models:
        cloudlet_model.load_state_dict(master_model.state_dict())

    return cloudlet_models

@torch.no_grad()
def average_model(cloudlet_models, master_model):
    # Initialize master model's parameters
    for master_model_param, cloudlet_model_param in zip(master_model.parameters(), cloudlet_models[0].parameters()):
        master_model_param.copy_(cloudlet_model_param.data.clone())
    # Accumulate parameters from all cloudlet models
    for cloudlet_model in cloudlet_models[1:]:
        for master_model_param, cloudlet_model_param in zip(master_model.parameters(), cloudlet_model.parameters()):
            master_model_param.data.add_(cloudlet_model_param.data)
    # Average accumulated parameters
    for master_model_param in master_model.parameters():
        master_model_param.data.div_(len(cloudlet_models))

    return master_model

# disable gradient calculation within a block of code
# i.e. don't calculate gradients during validation or testing process
@torch.no_grad()
def val(model, val_iter, cln_edge_index, cln_node_map, loss):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = utility.permute_4d_y_pred_to_3d(model, x, cln_edge_index)
        
        y = y[:,cln_node_map,:]
        y_pred = y_pred[:,cln_node_map,:]

        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)

# disable gradient calculation within a block of code
# i.e. don't calculate gradients during validation or testing process
@torch.no_grad()
def master_val(model, val_iter, edge_index, loss):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = utility.permute_4d_y_pred_to_3d(model, x, edge_index)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)

@torch.no_grad() 
def test(zscore, loss, model, test_iter, args, cln_edge_index, cln_node_map, cln_id, logs_folder):
    model.eval()
    test_MSE = utility.evaluate_3d_pyg_model(model, loss, test_iter, cln_edge_index, cln_node_map)
    for node_feature_to_evaluate in args.node_features_to_evaluate:
        test_MAE, test_RMSE, test_WMAPE = utility.evaluate_3d_pyg_metric(model, test_iter, zscore, cln_edge_index, cln_node_map, node_feature_to_evaluate)
        utility.save_test_logs(logs_folder, f"cln-{cln_id}_node-feature-{node_feature_to_evaluate}", test_MSE, test_MAE, test_RMSE, test_WMAPE)
        print(f'CLN: {cln_id:03d} | Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

@torch.no_grad() 
def test_master(zscore, loss, model, test_iter, args, edge_index, logs_folder):
    model.eval()
    test_MSE = utility.evaluate_3d_pyg_model_master(model, loss, test_iter, edge_index)
    for node_feature_to_evaluate in args.node_features_to_evaluate:
        test_MAE, test_RMSE, test_WMAPE = utility.evaluate_3d_pyg_metric_master(model, test_iter, zscore, edge_index, node_feature_to_evaluate)
        utility.save_test_logs(logs_folder, f"master-server_node-feature-{node_feature_to_evaluate}", test_MSE, test_MAE, test_RMSE, test_WMAPE)
        print(f'Master model | Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

if __name__ == "__main__":
    # Logging
    #logger = logging.getLogger('stgcn')
    #logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    writer = SummaryWriter()

    args, device = get_parameters()

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logs_folder = os.path.join('logs', f"{current_datetime}_{args.dataset}_{args.n_pred * 5}min_semi-dec-fl")

    blocks, train_iters, val_iters, test_iters, cln_edge_index_list, cln_node_map_list, master_zscore, test_iter_master, val_iter_master, edge_index_master = data_preparate(args, device)
    edge_index_master = edge_index_master.to(device)
    for i in range(len(cln_edge_index_list)):
        cln_edge_index_list[i] = cln_edge_index_list[i].to(device)

    for i in range(len(cln_node_map_list)):
        cln_node_map_list[i] = cln_node_map_list[i].to(device)
    # Master model
    master_loss, es, model, optimizer, scheduler = prepare_model(args, blocks)
    # cloudlet models
    cloudlet_models = [copy.deepcopy(model) for _ in range(args.cloudlet_num)]
    cln_losses = []
    cln_optimizers = []
    cln_schedulers = []
    for cln_model in cloudlet_models:
        cln_loss, cln_optimizer, cln_scheduler = setup_loss_optimizer_and_scheduler(cln_model)
        cln_losses.append(cln_loss)
        cln_optimizers.append(cln_optimizer)
        cln_schedulers.append(cln_scheduler)

    cloudlet_train(cln_losses, args, cln_optimizers, cln_schedulers, es, model, cloudlet_models, train_iters, val_iters, cln_edge_index_list, cln_node_map_list, writer, logs_folder, edge_index_master, master_loss, val_iter_master)    
    test_master(master_zscore, master_loss, model, test_iter_master, args, edge_index_master, logs_folder)
    # test each cloudlet
    i = 1
    for (cloudlet_model, cln_loss, test_iter, cln_edge_index, cln_node_map) in zip(cloudlet_models, cln_losses, test_iters, cln_edge_index_list, cln_node_map_list):
        test(master_zscore, cln_loss, cloudlet_model, test_iter, args, cln_edge_index, cln_node_map, i, logs_folder)
        i+=1

    writer.flush()
