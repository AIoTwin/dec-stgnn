import logging
import os
import argparse
import math
import random
import numpy as np
import pandas as pd
import copy
import datetime
import ray
from sklearn import preprocessing
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import dataloader, utility, earlystopping
from model import models
from actors import cloudlet_ray
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
    parser.add_argument('--dataset', type=str, default='pems-bay', choices=['metr-la', 'pems-bay']) # selected dataset
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
        # device = torch.device('cuda:0')
        device = torch.device('cpu')
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

    logging.info(f"Block value: {blocks}")

    return args, device, blocks

def data_preparate(args, device):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    cloudlet_info_path = os.path.join(root_dir, f"locations/{args.dataset}", "locations.json")

    cloudlet_data_json = utility.load_json_file(cloudlet_info_path)
    cloudlets, radius_km = utility.get_cloudlet_location_info_from_json(args.cloudlet_location_data, cloudlet_data_json)

    if (len(cloudlets) != args.cloudlet_num):
        assert False, f'Number of defined cloudlets doesn\'t match the number of cloudlets in json file -> {args.cloudlet_num}\{len(cloudlets)}'

    cln_adj_matrix = utility.create_cln_adj_matrix_by_distance(cloudlets, radius_km)

    # load adjacency matrix and number of nodes
    if (args.adj_matrix_type == "original"):
        print(f"Using original adjacency matrix")
        adj, _ = dataloader.load_adj(args.dataset)
    elif (args.adj_matrix_type == "no_neighbours"):
        print(f"Using no neighbours adjacency matrix")
        adj, _ = dataloader.load_adj_direct_no_neighbours(args.dataset)
    else:
        print(f"Wrong adj_matrix_type selected. Expected values are original | no_neighbours. You put: {args.adj_matrix_type}")
        exit(1)

    # Get edge_index from adj matrix
    edge_index, _ = from_scipy_sparse_matrix(adj)

    cln_nodes_list = utility.partition_nodes_to_cloudlets_by_range_proximity(cloudlets, radius_km, args.dataset)

    for i in range(len(cln_nodes_list)):
        cln_nodes_list[i].sort()

    cln_nodes_subgraph_list = []
    cln_edge_index_list = []
    cln_node_map_list = []

    cross_cloudlet_nodes_list = []

    cross_cloudlet_edge_index_list = []
    # cross_cloudlet_edges_map_list = []
    cross_cloudlet_edge_mask_list = []

    for cln_nodes in cln_nodes_list:
        cln_nodes_subgraph, cln_edge_index, cln_node_map, _ = k_hop_subgraph(cln_nodes, args.stblock_num * (args.Ks - 1), edge_index, relabel_nodes=True, num_nodes=adj.shape[0])
        cln_nodes_subgraph_list.append(cln_nodes_subgraph)
        cln_edge_index_list.append(cln_edge_index)
        cln_node_map_list.append(cln_node_map)

        cln_nodes_tensor = torch.tensor(cln_nodes, device=edge_index.device)
        src, dst = cln_edge_index
        #cross_cloudlet_edge_mask = ~torch.isin(dst, cln_nodes_tensor)
        cross_cloudlet_edge_mask = ~(torch.isin(dst, cln_nodes_tensor) & torch.isin(src, cln_nodes_tensor))

        cross_cloudlet_edge_index = cln_edge_index[:, cross_cloudlet_edge_mask]

        cross_cloudlet_edge_index_list.append(cross_cloudlet_edge_index)
        cross_cloudlet_edge_mask_list.append(cross_cloudlet_edge_mask)

        all_node_indices = torch.arange(cln_nodes_subgraph.size(0), device=cln_nodes_subgraph.device)
        cross_cloudlet_node_indices = all_node_indices[~torch.isin(all_node_indices, cln_node_map)]
        cross_cloudlet_nodes = cln_nodes_subgraph[cross_cloudlet_node_indices]

        cross_cloudlet_nodes_list.append(cross_cloudlet_nodes)

        # print(f"cross_cloudlet_nodes.size(0): {cross_cloudlet_nodes.size(0)}")
        # print(f"cln_nodes.size(0): {len(cln_nodes)}")
        # print(f"cln_nodes_subgraph.size(0): {cln_nodes_subgraph.size(0)}")
        # print(f"cross_cloudlet_nodes: {cross_cloudlet_nodes}")
        # print(f"cln_nodes: {cln_nodes}")
        # print(f"cln_nodes_subgraph: {cln_nodes_subgraph}")

        # print(f"cln_edge_index.size(1): {cln_edge_index.size(1)}")
        # print(f"cross_cloudlet_edge_index.size(1): {cross_cloudlet_edge_index.size(1)}")

        # print(f"cross_cloudlet_edge_mask.size(1): {cross_cloudlet_edge_mask.size(0)}")
        # print(f"cln_nodes_subgraph: {cln_nodes_subgraph}")
        # print(f"cln_edge_index: {cln_edge_index}")
        # print(f"cln_node_map: {cln_node_map}")
    
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

    if ((len_train - args.end_of_initial_data_index) % args.data_per_step == 0):
        print("End index and data step correctly selected!")
        print(f"Number of epochs: {(len_train - args.end_of_initial_data_index) / args.data_per_step}")
    else:
        print(f"End index and data step WRONGLY selected: {(len_train - args.end_of_initial_data_index) % args.data_per_step}")
        exit(1)
    
    # 1st is for cloudlets, 2nd is for master aggregator server
    cln_train_datasets, cln_val_datasets, cln_test_datasets = dataloader.create_cloudlet_datasets(cln_nodes_subgraph_list, args.dataset, len_train, len_val)
    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)

    # This is for cloudlets,...
    cln_train_list = []
    cln_val_list = []
    cln_test_list = []

    # Shared zscore standard scaler
    master_zscore = preprocessing.StandardScaler()

    _, _, _ = utility.zscore_preprocess_2d_data(master_zscore, train.values, val.values, test.values, use_fit_transform=True)

    # make cloudlets dont have separate zscore, let them share 1
    # use zscaler on all training param, and when cloudlets use transforms, dont use fit
    for (cln_train, cln_val, cln_test) in zip(cln_train_datasets, cln_val_datasets, cln_test_datasets):
        cln_train, cln_val, cln_test = utility.zscore_preprocess_2d_data(master_zscore, cln_train, cln_val, cln_test)

        cln_train_list.append(cln_train)
        cln_val_list.append(cln_val)
        cln_test_list.append(cln_test)   
    
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

    return adj.shape[0], cln_nodes_list, cln_adj_matrix, cln_train_list, cln_val_list, len_train, train_iters, x_train, y_train, val_iters, test_iters, cln_edge_index_list, cross_cloudlet_edge_index_list, cross_cloudlet_edge_mask_list, cln_node_map_list, master_zscore

# use this to create a "master" model, and then args.cloudlet_num models
# instead of for loop args.cloudlet_num times, find a function that copies a model for args.cloudlet_num
# MAKE SURE ALL MODELS ARE EXACTLY THE SAME!!!!
def prepare_model(args, blocks):
    loss = nn.MSELoss() # init Mean Squared Error (MSE) loss function
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience) # init early stopping object

    # RUN IN PyTorch Geometric!!!!!!!!
    model = models.STGCNGraphConvPyG(args, blocks).to('cuda:0')

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

def cloudlet_train(args, cln_actors, cln_models, zscore, len_train, num_nodes, device):
    num_epochs = int((len_train - args.end_of_initial_data_index) / args.data_per_step)

    for epoch in range(1, num_epochs + 1):
        for cln_actor in cln_actors:
            ray.get(cln_actor.choose_cross_cloudlet_edges_by_distribution.remote(args.stblock_num, args.Ks, num_nodes, device, args.n_his, args.n_pred, args.batch_size))
        
        for i in range(args.cloudlet_num):
            cln_models[i] = cln_actors[i].online_train_no_master.remote(epoch, zscore, num_epochs)
        
        cln_done = []
        for cln_done_singal in cln_done:
            ray.get(cln_done_singal)

        # TODO Calcualte edge scores
        cln_done = []
        for cln_actor in cln_actors:
            # masked_cross_cloudlet_edges = ray.get(cln_actor.choose_random_cross_cloudlet_edges_after_distribution_2.remote())
            masked_cross_cloudlet_edges = ray.get(cln_actor.choose_random_cross_cloudlet_edges_after_distribution.remote())
            # masked_cross_cloudlet_edges = ray.get(cln_actor.choose_random_cross_cloudlet_edges.remote())
            # masked_cln_edge_index = ray.get(cln_actor.remove_edges.remote(masked_cross_cloudlet_edges))
            masked_cln_edge_index = ray.get(cln_actor.remove_edges_distribution.remote(masked_cross_cloudlet_edges))
            cln_done.append(cln_actor.delta_error_between_original_and_masked_for_online_training.remote(epoch, zscore, num_epochs, masked_cln_edge_index, masked_cross_cloudlet_edges))

        for cln_done_singal in cln_done:
            ray.get(cln_done_singal)

        # Average models from neighbours and that's the new model for each cloudlet
        cln_done = []
        for cln_actor in cln_actors:
            cln_done.append(cln_actor.average_neighbours.remote(cln_models))

        for cln_done_singal in cln_done:
            ray.get(cln_done_singal)

    for i in range(args.cloudlet_num):
        ray.get(cln_actors[i].log_edge_score_results.remote())

if __name__ == "__main__":
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init()
    # Logging
    #logger = logging.getLogger('stgcn')
    #logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    writer = SummaryWriter()

    args, device, blocks = get_parameters()

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"Start date & time: {current_datetime}")
    logs_folder = os.path.join('logs', f"{current_datetime}_{args.dataset}_pred-{args.n_pred * 5}min_his-{args.n_his * 5}min_ray-semi-dec-server-free-fl-distance_online-training_with-distribution-adaptive-edge-score")

    num_nodes, cln_nodes_list, cln_adj_matrix, cln_train_list, cln_val_list, len_train, train_iters, x_trains, y_trains, val_iters, test_iters, cln_edge_index_list, cross_cloudlet_edge_index_list, cross_cloudlet_edge_mask_list, cln_node_map_list, master_zscore = data_preparate(args, device)

    original_cln_edge_index_list = cln_edge_index_list
    for i in range(len(original_cln_edge_index_list)):
        original_cln_edge_index_list[i] = original_cln_edge_index_list[i].to(device)
    # for i in range(len(cln_edge_index_list)):
    #     cln_edge_index_list[i] = cln_edge_index_list[i].to(device)
    # for i in range(len(cln_node_map_list)):
    #     cln_node_map_list[i] = cln_node_map_list[i].to(device)
    
    # Master model
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks)
    # cloudlet models
    cloudlet_models = [copy.deepcopy(model) for _ in range(args.cloudlet_num)]

    cln_actors = []
    cln_id = 0
    for cln_nodes, cln_model, cln_edge_index, original_cln_edge_index, cln_node_map, cln_train_iter, cln_x_train, cln_y_train, cln_val_iter, cln_test_iter, cln_train_dataset, cln_val_dataset, cross_cloudlet_edge_index, cross_cloudlet_edge_mask in zip(cln_nodes_list, cloudlet_models, cln_edge_index_list, original_cln_edge_index_list, cln_node_map_list, train_iters, x_trains, y_trains, val_iters, test_iters, cln_train_list, cln_val_list, cross_cloudlet_edge_index_list, cross_cloudlet_edge_mask_list):
        cln_loss, cln_optimizer, cln_scheduler = setup_loss_optimizer_and_scheduler(cln_model)
        cln_actor = cloudlet_ray.Cloudlet.remote(model,
                                                 cln_model,
                                                 cln_loss,
                                                 cln_optimizer,
                                                 cln_scheduler,
                                                 cln_id,
                                                 cln_edge_index,
                                                 cln_node_map,
                                                 cln_train_iter,
                                                 cln_val_iter,
                                                 cln_test_iter,
                                                 cln_adj_matrix,
                                                 logs_folder,
                                                 train_dataset=cln_train_dataset,
                                                 x_train=cln_x_train,
                                                 y_train=cln_y_train,
                                                 end_of_initial_data_index=args.end_of_initial_data_index,
                                                 data_per_step=args.data_per_step,
                                                 batch_size=args.batch_size,
                                                 cross_cloudlet_edge_index=cross_cloudlet_edge_index,
                                                 cross_cloudlet_edge_mask=cross_cloudlet_edge_mask,
                                                 local_cln_nodes = cln_nodes,
                                                 val_dataset = cln_val_dataset,
                                                 original_edge_index = original_cln_edge_index)
        ray.get(cln_actor.initialize_writer.remote())
        ray.get(cln_actor.init_loss_optimizer_scheduler.remote(args))
        cln_actors.append(cln_actor)
        cln_id +=1

    cloudlet_train(args, cln_actors, cloudlet_models, master_zscore, len_train, num_nodes, device)

    # master_actor.test_2d_data_using_best_model(master_zscore, args, best_epoch)

    # for cln_actor in cln_actors:
    #     cln_actor.test_2d_data_using_best_model(master_zscore, args, best_epoch)

    for cln_actor in cln_actors:
        ray.get(cln_actor.flush_writer.remote())

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"End date & time: {current_datetime}")
