import torch
import ray
import copy
from actors import network_actor as na
from actors import cloudlet_base as cb
from script import utility, dataloader
from messages import messages
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch.utils as utils
from collections import deque, defaultdict
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import k_hop_subgraph

@ray.remote(num_gpus=0.14)
class Cloudlet(na.NetworkActor, cb.CloudletBase):
    def __init__(self, server, model, loss, optimizer, scheduler, cln_id, edge_index, node_map, train_iter, val_iter, test_iter, cln_adj_matrix, logs_folder, train_dataset = [], x_train = [], y_train = [], val_dataset = [], end_of_initial_data_index = 0, data_per_step = 0, batch_size = 32, cross_cloudlet_edge_index = None, cross_cloudlet_edge_mask = None, local_cln_nodes = None):
        super().__init__(server, model, loss, optimizer, scheduler, cln_id, edge_index, node_map, train_iter, x_train, y_train, val_iter, test_iter, logs_folder, end_of_initial_data_index, data_per_step, train_dataset, batch_size, cross_cloudlet_edge_index, cross_cloudlet_edge_mask, local_cln_nodes, val_dataset)
        super().init_wandb()

        # get cloudlet neighbour indices from cln_adj_matrix
        self.neighbour_cln_indices = []
        if len(cln_adj_matrix) != 0:
            cln_neighbour_indices = torch.nonzero(cln_adj_matrix[self.cln_id,:] == 1).squeeze().tolist()
            if (isinstance(cln_neighbour_indices, int)):
                cln_neighbour_indices = [cln_neighbour_indices]
            self.neighbour_cln_indices = cln_neighbour_indices

        # Define buffer for gossip learning
        self.models_gossip_buffer = deque(maxlen=2)
        self.models_gossip_buffer.append(model)
        self.models_gossip_buffer.append(model)

        self.best_epoch = 0
        self.best_WMAPE = 100

        self.edge_scores = torch.zeros(cross_cloudlet_edge_index.shape[1], dtype=torch.float32)
        self.edge_counts = torch.zeros(cross_cloudlet_edge_index.shape[1], dtype=torch.int32)

    # Because of ray, you have to init SummaryWriter separately due to thread locking
    def initialize_writer(self):
        self.writer = SummaryWriter()

    def init_loss_optimizer_scheduler(self, args):
        self.loss = nn.MSELoss()

        # define optimization algorithm
        if args.opt == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
        elif args.opt == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
        elif args.opt == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
        else:
            raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)

    def flush_writer(self):
        if self.writer:
            self.writer.flush()

    def finish_wandb(self):
        super().finish_wandb()

    def train(self, epoch, master_actor):
        self.__train(epoch)

        # using NetworkActor, send model to master model
        self.send_to(master_actor, messages.CloudletModelMessage(self.model, self.cln_id))

    def train_no_master(self, epoch):
        self.__train(epoch)

        return self.model

    def online_train_no_master(self, epoch, zscore, num_epochs):
        self.__train_online(epoch, zscore, num_epochs)

        return self.model

    def check_best_model_using_metrics(self, epoch, zscore):
        _, _, val_WMAPE = utility.evaluate_pyg_metric(self.model, self.val_iter, zscore, self.edge_index, self.node_map)

        # Keep best model for testing
        if (val_WMAPE < self.best_WMAPE):
            self.best_WMAPE = val_WMAPE
            self.best_epoch = epoch
            super().store_new_best_model()

    def test_2d_data_using_best_model(self, zscore, args):
        super().test_2d_data_using_best_model(zscore, args, self.best_epoch)
    
    def calculate_val_loss(self):
        return self.val()

    def get_cln_id(self):
        return self.cln_id
    
    def get_model_params(self):
        return self.model.parameters()

    def on_receive(self, message):
        if isinstance(message, messages.MasterModelMessage):
            for cln_model_param_1, cln_model_param_2 in zip(self.model.parameters(), message.master_model.parameters()):
                cln_model_param_1.data.copy_(cln_model_param_2.data)
        else:
            assert False, 'unrecognized message type'

    def __train(self, epoch):
        super().train(epoch)

    def __train_online(self, epoch, zscore, num_epochs):
        train_iter = self.create_train_iter_for_online(epoch, self.x_train, self.y_train)
        super().online_train(epoch, train_iter, zscore, num_epochs)

    def average_neighbours(self, cln_models):
        # Get only neighbour models
        cln_neighbour_models = [ray.get(cln_models[i]) for i in self.neighbour_cln_indices]

        # Accumulate parameters from all neighbour cloudlet models
        for cln_neighbour_model in cln_neighbour_models:
            for cln_model_param, cln_neighbour_model_param in zip(self.model.parameters(), cln_neighbour_model.parameters()):
                cln_model_param.data.add_(cln_neighbour_model_param.data)

        # Average accumulated parameters
        for cln_model_param in self.model.parameters():
            cln_model_param.data.div_(len(cln_neighbour_models) + 1)

        return None

    def average_models_gossip_buffer(self):
        if (len(self.models_gossip_buffer) < self.models_gossip_buffer.maxlen):
            return None

        models = list(self.models_gossip_buffer)

        for cln_model_param_1, cln_model_param_2 in zip(self.model.parameters(), models[0].parameters()):
            cln_model_param_1.data.copy_(cln_model_param_2.data)
    
        for cln_model_param_1, cln_model_param_2 in zip(self.model.parameters(), models[1].parameters()):
            cln_model_param_1.data.add_(cln_model_param_2.data)

        # Average accumulated parameters
        for cln_model_param in self.model.parameters():
            cln_model_param.data.div_(self.models_gossip_buffer.maxlen)
        
        return None

    def send_model_to_cloudlet_gossip(self, cln_actors, cln_models):
        model = ray.get(cln_models[self.cln_id])

        # Get only neighbour actors
        cln_neighbour_actors = [cln_actors[i] for i in self.neighbour_cln_indices]

        # Randomly select one neighbor actor
        selected_actor = random.choice(cln_neighbour_actors)

        # Call the store_model_to_gossip_buffer function on the selected actor
        selected_actor.store_model_to_gossip_buffer.remote(model)

        return None

    def store_model_to_gossip_buffer(self, model):
        self.models_gossip_buffer.append(model)
        return None
    
    def log_transfer_neighbour_model_parameters(self, cln_models, epoch):
        # Get only neighbour models
        cln_neighbour_models = [ray.get(cln_models[i]) for i in self.neighbour_cln_indices]
        total_num_of_parameters = 0
        for cln_neighbour_model in cln_neighbour_models:
            cln_num_of_parameters = sum(p.numel() for p in cln_neighbour_model.parameters())
            total_num_of_parameters += cln_num_of_parameters

        utility.save_total_trainable_parameters_transfer_size(self.logs_folder, self.cln_id, epoch+1, total_num_of_parameters)

    def log_transfer_neighbour_model_parameters_gossip_learning(self, epoch):
        cln_num_of_parameters = sum(p.numel() for p in self.model.parameters())
        utility.save_total_trainable_parameters_transfer_size(self.logs_folder, self.cln_id, epoch+1, cln_num_of_parameters)

    def log_variance_info(self, var_mean, epoch):
        utility.save_variance_logs(self.logs_folder, self.cln_id, epoch, var_mean)

    def delta_error_between_original_and_masked_for_online_training(self, epoch, zscore, num_epochs, masked_edge_index, masked_cross_cloudlet_edges):
        d_err = super().delta_error_between_original_and_masked_for_online_training(epoch, zscore, num_epochs, masked_edge_index)

        # Clamp delta_err to get values ONLY between -10 and 10
        delta_err_clamped = torch.clamp(torch.tensor(d_err, device=self.edge_scores.device), -10, 10)

        self.edge_scores[masked_cross_cloudlet_edges] += delta_err_clamped
        self.edge_counts[masked_cross_cloudlet_edges] += 1

    def log_edge_score_results(self):
        file_name = f"cloudlet_{self.cln_id}"  # Assuming you have a `self.index` for cloudlet ID
        edge_positions = torch.nonzero(self.cross_cloudlet_edge_mask).flatten().cpu().numpy()
        utility.save_edge_scores_logs(self.logs_folder, file_name, self.edge_scores, edge_positions)
        utility.save_edge_counts_logs(self.logs_folder, file_name, self.edge_counts, edge_positions)

    def exploit_edge_scores(self, zscore):
        # Create a mask for non-zero edge_counts to avoid division by zero
        non_zero_mask = self.edge_counts != 0
        edge_scores = torch.full_like(self.edge_scores, 0)

        # Only divide where edge_counts is not zero
        edge_scores[non_zero_mask] = self.edge_scores[non_zero_mask] / self.edge_counts[non_zero_mask].float()

        num_edges = self.cross_cloudlet_edge_index.shape[1]

        for p in np.arange(0.05, 1.05, 0.05):  # 5% to 100%
            num_select = max(1, int(p * num_edges))  # Ensure we select at least 1 edge

            # Get the indices of the edges with lowest scores
            _, lowest_indices = torch.topk(edge_scores, k=num_select, largest=False)

            # Create the mask for cross cloudlet edges (True means edge is selected for removal)
            masked_cross_cloudlet_edges = torch.zeros(self.cross_cloudlet_edge_index.shape[1], dtype=torch.bool)
            masked_cross_cloudlet_edges[lowest_indices] = True

            # Create the final edge mask for all edges (including non-cross-cloudlet edges)
            cloudlet_edge_mask = torch.ones(self.edge_index.shape[1], dtype=torch.bool)
            cloudlet_edge_mask[self.cross_cloudlet_edge_mask] = ~masked_cross_cloudlet_edges
        
            # Create the masked edge index
            masked_cln_edge_index = self.edge_index[:, cloudlet_edge_mask]

            val_MAE, val_RMSE, val_WMAPE = utility.evaluate_pyg_metric_master(self.model, self.val_iter, zscore, masked_cln_edge_index)

            percent_str = f"{int(p*100):02d}"
            utility.save_val_metric_logs_edge_score(self.logs_folder, self.cln_id, percent_str, val_MAE, val_RMSE, val_WMAPE)

        val_MAE, val_RMSE, val_WMAPE = utility.evaluate_pyg_metric_master(self.model, self.val_iter, zscore, self.edge_index)
        utility.save_val_metric_logs_edge_score(self.logs_folder, self.cln_id, "original", val_MAE, val_RMSE, val_WMAPE)
        return
    
    def remove_cross_cloudlet_edges_using_edge_score(self, stblock_num, Ks, num_nodes, device, n_his, n_pred):
         # Create a mask for non-zero edge_counts to avoid division by zero
        non_zero_mask = self.edge_counts != 0
        edge_scores = torch.full_like(self.edge_scores, 0)

        # Only divide where edge_counts is not zero
        edge_scores[non_zero_mask] = self.edge_scores[non_zero_mask] / self.edge_counts[non_zero_mask].float()

        num_edges = self.cross_cloudlet_edge_index.shape[1]

        # For now, remove x% of the cross cloudlet edges (later implement to remove them by certain criteria)
        num_select = max(1, int(0.95 * num_edges))  # Ensure we select at least 1 edge

        # Get the indices of the edges with lowest scores
        _, lowest_indices = torch.topk(edge_scores, k=num_select, largest=False)

        # Create the mask for cross cloudlet edges (True means edge is selected for removal)
        masked_cross_cloudlet_edges = torch.zeros(self.cross_cloudlet_edge_index.shape[1], dtype=torch.bool)
        masked_cross_cloudlet_edges[lowest_indices] = True

        # TODO: USE ACTUAL edge_index, NOT CLOUDLET ONE!!!!
        # Create the final edge mask for all edges (including non-cross-cloudlet edges)
        cloudlet_edge_mask = torch.ones(self.edge_index.shape[1], dtype=torch.bool)
        cloudlet_edge_mask[self.cross_cloudlet_edge_mask] = ~masked_cross_cloudlet_edges
    
        # Create the masked edge index
        masked_cln_edge_index = self.edge_index[:, cloudlet_edge_mask]

        cln_nodes_subgraph, cln_edge_index, cln_node_map, _ = k_hop_subgraph(self.local_cln_nodes, stblock_num * (Ks - 1), masked_cln_edge_index, relabel_nodes=True, num_nodes=num_nodes)
        cln_edge_index = cln_edge_index.to(device=self.edge_index.device)
        cln_node_map = cln_node_map.to(device=device)

        self.edge_index = cln_edge_index
        self.node_map = cln_node_map

        new_train_dataset = self.train_dataset[:, cln_nodes_subgraph.cpu().numpy()]

        x_train, y_train = dataloader.data_transform(new_train_dataset, n_his, n_pred, device)

        self.x_train = x_train
        self.y_train = y_train

    def remove_cross_cloudlet_edges_using_grouped_edge_score(self, stblock_num, Ks, num_nodes, device, n_his, n_pred, batch_size):
        # 1. Identify cross-cloudlet nodes (those NOT in local_cln_nodes)
        node_mask = torch.ones(num_nodes, dtype=torch.bool)
        node_mask[self.local_cln_nodes] = False
        cross_cln_nodes = torch.nonzero(node_mask).squeeze()

        # 2. Create a mask for non-zero edge_counts to avoid division by zero
        edge_scores = torch.zeros_like(self.edge_scores, dtype=torch.float)
        non_zero_mask = self.edge_counts != 0
        edge_scores[non_zero_mask] = self.edge_scores[non_zero_mask] # / self.edge_counts[non_zero_mask].float()

         # 3. Initialize dictionaries to accumulate total score and count per cross-cloudlet node
        node_score_sum = defaultdict(float)
        node_score_count = defaultdict(int)

        cross_edges = self.cross_cloudlet_edge_index  # shape [2, num_cross_edges]

        for idx in range(cross_edges.shape[1]):
            u, v = cross_edges[0, idx].item(), cross_edges[1, idx].item()
            score = edge_scores[idx].item()
            count = self.edge_counts[idx].item()

            # If either u or v is a cross-cloudlet node, accumulate their score and count
            if u in cross_cln_nodes:
                node_score_sum[u] += score
                node_score_count[u] += count
            if v in cross_cln_nodes and v != u:
                node_score_sum[v] += score
                node_score_count[v] += count

        # 4. Compute average score per cross-cloudlet node
        node_avg_score = {}
        for node in node_score_sum:
            if node_score_count[node] > 0:
                node_avg_score[node] = node_score_sum[node] / node_score_count[node]
            else:
                node_avg_score[node] = float('inf')  # Avoid division by zero

        # 5. Sort nodes by avg score (lowest to highest -> the higher the score is, the more important the edge is)
        sorted_nodes = sorted(node_avg_score.items(), key=lambda x: x[1])

        # 6. Remove edges connected to worst X% of cross-cloudlet nodes
        num_to_remove = max(1, int(0.95 * len(sorted_nodes)))
        nodes_to_remove = set([node for node, _ in sorted_nodes[:num_to_remove]])

        # 7. Build a mask for cross-cloudlet edges to remove
        masked_cross_edges = torch.zeros(cross_edges.shape[1], dtype=torch.bool)
        for idx in range(cross_edges.shape[1]):
            u, v = cross_edges[0, idx].item(), cross_edges[1, idx].item()
            if u in nodes_to_remove or v in nodes_to_remove:
                masked_cross_edges[idx] = True

        # 8. Apply the mask to the full edge_index
        cloudlet_edge_mask = torch.ones(self.edge_index.shape[1], dtype=torch.bool)
        cloudlet_edge_mask[self.cross_cloudlet_edge_mask] = ~masked_cross_edges

        masked_cln_edge_index = self.edge_index[:, cloudlet_edge_mask]

        # 9. Generate subgraph and remap
        cln_nodes_subgraph, cln_edge_index, cln_node_map, _ = k_hop_subgraph(
            self.local_cln_nodes, stblock_num * (Ks - 1), masked_cln_edge_index,
            relabel_nodes=True, num_nodes=num_nodes
        )

        cln_edge_index = cln_edge_index.to(device=self.edge_index.device)
        cln_node_map = cln_node_map.to(device=device)

        self.edge_index = cln_edge_index
        self.node_map = cln_node_map

        # 10. Recreate training data with new node set
        new_train_dataset = self.train_dataset[:, cln_nodes_subgraph.cpu().numpy()]
        x_train, y_train = dataloader.data_transform(new_train_dataset, n_his, n_pred, device)

        self.x_train = x_train
        self.y_train = y_train

        # 11. Recreate validation data with new node set
        new_val_dataset = self.val_dataset[:, cln_nodes_subgraph.cpu().numpy()]
        x_val, y_val = dataloader.data_transform(new_val_dataset, n_his, n_pred, device)
        val_data = utils.data.TensorDataset(x_val, y_val)
        val_iter = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

        self.val_iter = val_iter