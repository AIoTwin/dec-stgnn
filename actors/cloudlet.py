import torch
import torch.utils as utils
from actors import network_actor as na
from actors import cloudlet_base as cb
from script import utility
from messages import messages
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import k_hop_subgraph
from script import utility, dataloader

class Cloudlet(na.NetworkActor, cb.CloudletBase):
    def __init__(self, server, model, loss, optimizer, scheduler, cln_id, edge_index, node_map, train_iter, val_iter, test_iter, logs_folder, cln_adj_matrix = [], train_dataset = [], val_dataset = [], x_train = [], y_train = [], end_of_initial_data_index = 0, data_per_step = 0, batch_size = 32, cross_cloudlet_edge_index = None, cross_cloudlet_edge_mask = None, local_cln_nodes = None, original_edge_index = None, cross_cloudlet_nodes = None):
        super().__init__(server, model, loss, optimizer, scheduler, cln_id, edge_index, node_map, train_iter, x_train, y_train, val_iter, test_iter, logs_folder, end_of_initial_data_index, data_per_step, train_dataset, batch_size, cross_cloudlet_edge_index, cross_cloudlet_edge_mask, local_cln_nodes, val_dataset, original_edge_index, cross_cloudlet_nodes)
        super().init_wandb()

        self.averaged_model = model

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
        
        self.model_backup = model

        self.node_scores = torch.zeros(cross_cloudlet_nodes.size(0), dtype=torch.float32)
        self.node_counts = torch.zeros(cross_cloudlet_nodes.size(0), dtype=torch.int32)

    # Because of ray, you have to init SummaryWriter separately due to thread locking
    def initialize_writer(self):
        self.writer = SummaryWriter()

    def flush_writer(self):
        if self.writer:
            self.writer.flush()

    def finish_wandb(self):
        super().finish_wandb()

    def train(self, epoch, master_actor):
        self.__train(epoch)

        # using NetworkActor, send model to master model
        self.send_to(master_actor, messages.CloudletModelMessage(self.model, self.cln_id))

    def online_train(self, epoch, master_actor, zscore, num_epochs):
        self.__train_online(epoch, zscore, num_epochs)

        # using NetworkActor, send model to master model
        self.send_to(master_actor, messages.CloudletModelMessage(self.model, self.cln_id))

    def train_no_master(self, epoch):
        self.__train(epoch)

    def online_train_no_master(self, epoch, zscore, num_epochs):
        self.__train_online(epoch, zscore, num_epochs)

    def on_receive(self, message):
        if isinstance(message, messages.MasterModelMessage):
            for cln_model_param_1, cln_model_param_2 in zip(self.model.parameters(), message.master_model.parameters()):
                cln_model_param_1.data.copy_(cln_model_param_2.data)
            # self.model.load_state_dict(message.master_model.state_dict())
        else:
            assert False, 'unrecognized message type'

    def get_cln_models_from_neighbours(self, cln_adj_matrix, cln_actors):
        # get cloudlet neighbours from cln_adj_matrix
        cln_neighbour_indices = torch.nonzero(cln_adj_matrix[self.cln_id,:] == 1).squeeze().tolist()
        #cln_neighbour_actors = cln_actors[cln_neighbour_indices]
        if (isinstance(cln_neighbour_indices, int)):
            cln_neighbour_indices = [cln_neighbour_indices]
        cln_neighbour_actors = [cln_actors[i] for i in cln_neighbour_indices]
        cln_neighbour_models = []

        if isinstance(cln_neighbour_actors, list) == False:
            cln_neighbour_actors = [cln_neighbour_actors]
        
        for cln_neighbour_actor in cln_neighbour_actors:
            # send a request to other cloudlets to fetch their model
            cln_neighbour_model = cln_neighbour_actor.get_cln_model()
            cln_neighbour_models.append(cln_neighbour_model)

        return cln_neighbour_models

    def get_cln_model(self):
        return self.model
    
    def get_val_loss_from_average_model(self):
        return self.val()
    
    def get_cln_id(self):
        return self.cln_id
    
    def get_model_params(self):
        return self.model.parameters()

    @torch.no_grad()
    def average_model(self, cln_neighbour_models):
        # Accumulate parameters from all neighbour cloudlet models
        for cln_neighbour_model in cln_neighbour_models:
            for cln_averaged_model_param, cln_neighbour_model_param in zip(self.averaged_model.parameters(), cln_neighbour_model.parameters()):
                cln_averaged_model_param.data.add_(cln_neighbour_model_param.data)
        # Average accumulated parameters
        for cln_averaged_model_param in self.averaged_model.parameters():
            cln_averaged_model_param.data.div_(len(cln_neighbour_models) + 1)

    @torch.no_grad()
    def average_models_gossip_buffer(self):
        if (len(self.models_gossip_buffer) < self.models_gossip_buffer.maxlen):
            return None

        models = list(self.models_gossip_buffer)
        self.model = self.model_backup
        for model in models:
            for cln_model_param, cln_gossip_model_param in zip (self.model.parameters(), model.parameters()):
                cln_model_param.data.add_(cln_gossip_model_param.data)

        # Average accumulated parameters
        for cln_model_param in self.model.parameters():
            cln_model_param.data.div_(self.models_gossip_buffer.maxlen)

        return None
    
    def send_model_to_cloudlet_gossip(self, cln_actors):
        # Get only neighbour actors
        cln_neighbour_actors = [cln_actors[i] for i in self.neighbour_cln_indices]

        # Randomly select one neighbor actor
        selected_actor = random.choice(cln_neighbour_actors)

        # Call the store_model_to_gossip_buffer function on the selected actor
        selected_actor.store_model_to_gossip_buffer(self.model)
    
    def store_model_to_gossip_buffer(self, model):
        self.models_gossip_buffer.append(model)

    def copy_averaged_model_to_current(self):
        for current_model, averaged_model in zip(self.model.parameters(), self.averaged_model.parameters()):
            current_model.data.copy_(averaged_model.data)
        # self.model.load_state_dict(self.averaged_model.state_dict())

    def log_transfer_neighbour_model_parameters_gossip_learning(self, epoch):
        cln_num_of_parameters = sum(p.numel() for p in self.model.parameters())
        utility.save_total_trainable_parameters_transfer_size(self.logs_folder, self.cln_id, epoch+1, cln_num_of_parameters)

    def __train(self, epoch):
        super().train(epoch)

        # Copy trained model
        self.averaged_model.load_state_dict(self.model.state_dict())

    def __train_online(self, epoch, zscore, num_epochs):
        train_iter = self.create_train_iter_for_online(epoch, self.x_train, self.y_train)
        super().online_train(epoch, train_iter, zscore, num_epochs)

        # Copy trained model
        self.averaged_model.load_state_dict(self.model.state_dict())

    def store_new_best_model(self):
        super().store_new_best_model()

    def log_variance_info(self, var_mean, epoch):
        utility.save_variance_logs(self.logs_folder, self.cln_id, epoch, var_mean)

    def log_val_metrics(self, epoch, zscore):
        super().log_val_metrics(epoch, zscore)

    def log_var_info(self, epoch):
        super().log_var_info(epoch)

    def choose_cross_cloudlet_nodes_by_distribution(self, stblock_num, Ks, num_nodes, device, n_his, n_pred, batch_size, epoch, scaler, percentage = 0.1):
        if self.cross_cloudlet_edge_index.size(1) == 0:
            return None
        
        # Detect vehicle speed peaks for local nodes - if detected, don't remove it's neighbours (cross cloudlet nodes)
        ineligible_nodes = self.detect_vehicle_speed_peaks(epoch, scaler, self.original_y_train)
        utility.save_ineligible_nodes_logs(self.logs_folder, self.cln_id, ineligible_nodes)

        # === 1. Compute average scores per node (avoid division by zero) ===
        avg_scores = torch.zeros_like(self.node_scores)
        non_zero_mask = self.node_counts != 0
        avg_scores[non_zero_mask] = self.node_scores[non_zero_mask] / self.node_counts[non_zero_mask].float()

        # === 2. Create score map for valid (non-zero, non-NaN) nodes ===
        valid_mask = non_zero_mask & ~torch.isnan(avg_scores) & (avg_scores > 0)
        valid_nodes = self.original_cross_cloudlet_nodes[valid_mask]
        valid_scores = avg_scores[valid_mask]

        # === Filter out ineligible nodes ===
        ineligible_tensor = torch.tensor(list(ineligible_nodes), device=valid_nodes.device)
        eligible_mask = ~torch.isin(valid_nodes, ineligible_tensor)
        valid_nodes = valid_nodes[eligible_mask]
        valid_scores = valid_scores[eligible_mask]

        if valid_nodes.numel() == 0:
            print(f"{self.cln_id}: No valid scored nodes found.")
            utility.save_nodes_removed_by_distribution_logs(self.logs_folder, self.cln_id, set())
            return
        else:
            print(f"{self.cln_id} - valid_nodes: {valid_nodes}")
        
        # === 3. Normalize scores into a probability distribution ===
        total_score = valid_scores.sum().item()
        node_probs = valid_scores / total_score

        # === 4. Sample N% of nodes based on distribution ===
        target_count = int(percentage * self.original_cross_cloudlet_nodes.size(0))
        num_to_select = max(1, min(target_count, valid_nodes.size(0)))  # Don't over-select
        selected_indices = torch.multinomial(node_probs, num_samples=num_to_select, replacement=False)
        selected_nodes = set(valid_nodes[selected_indices].tolist())
        # num_to_select = max(1, int(percentage * valid_nodes.size(0)))
        # selected_indices = torch.multinomial(node_probs, num_samples=num_to_select, replacement=False)
        # selected_nodes = set(valid_nodes[selected_indices].tolist())
        utility.save_nodes_removed_by_distribution_logs(self.logs_folder, self.cln_id, selected_nodes)
        print(f"{self.cln_id}: Selected cross-cloudlet nodes for node removal: {selected_nodes}")

        # === 5. Mask out cross-cloudlet edges connected to selected nodes ===
        masked_cross_cloudlet_edges = torch.zeros(
            self.original_cross_cloudlet_edge_index.size(1),
            dtype=torch.bool,
            device=self.original_cross_cloudlet_edge_index.device
        )

        src, dst = self.original_cross_cloudlet_edge_index
        for idx in range(src.size(0)):
            u, v = src[idx].item(), dst[idx].item()
            if u in selected_nodes or v in selected_nodes:
                masked_cross_cloudlet_edges[idx] = True

        # === 6. Create new edge index by masking out selected edges ===
        cloudlet_edge_mask = torch.ones(self.original_edge_index.size(1), dtype=torch.bool, device=self.original_edge_index.device)
        cloudlet_edge_mask[self.original_cross_cloudlet_edge_mask] = ~masked_cross_cloudlet_edges
        masked_cln_edge_index = self.original_edge_index[:, cloudlet_edge_mask]

        cln_nodes_subgraph, cln_edge_index, cln_node_map, _ = k_hop_subgraph(
            self.local_cln_nodes,
            stblock_num * (Ks - 1),
            masked_cln_edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes
        )
        cln_edge_index = cln_edge_index.to(device=device)
        cln_node_map = cln_node_map.to(device=device)

        self.edge_index = cln_edge_index
        self.node_map = cln_node_map

        # Set new cross cloudlet nodes
        all_node_indices = torch.arange(cln_nodes_subgraph.size(0), device=cln_nodes_subgraph.device)
        cross_cloudlet_node_indices = all_node_indices[~torch.isin(all_node_indices, cln_node_map)]
        self.cross_cloudlet_nodes = cln_nodes_subgraph[cross_cloudlet_node_indices]

        # === 8. Recalculate cross-cloudlet edge index/mask ===
        cln_nodes_tensor = torch.tensor(self.local_cln_nodes, device=device)
        src, dst = cln_edge_index
        cross_cloudlet_edge_mask = ~(torch.isin(dst, cln_nodes_tensor) & torch.isin(src, cln_nodes_tensor))
        cross_cloudlet_edge_index = cln_edge_index[:, cross_cloudlet_edge_mask]

        self.cross_cloudlet_edge_index = cross_cloudlet_edge_index
        self.cross_cloudlet_edge_mask = cross_cloudlet_edge_mask

        # === 9. Update train/val datasets ===
        cln_nodes_subgraph_cpu = cln_nodes_subgraph.cpu().numpy()

        new_train_dataset = self.train_dataset[:, cln_nodes_subgraph_cpu]
        x_train, y_train = dataloader.data_transform(new_train_dataset, n_his, n_pred, device)
        self.x_train = x_train
        self.y_train = y_train

        new_val_dataset = self.val_dataset[:, cln_nodes_subgraph_cpu]
        x_val, y_val = dataloader.data_transform(new_val_dataset, n_his, n_pred, device)
        val_data = utils.data.TensorDataset(x_val, y_val)
        self.val_iter = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

    def choose_cross_cloudlet_nodes_by_distribution_with_adaptive_percentage(self, stblock_num, Ks, num_nodes, device, n_his, n_pred, batch_size, epoch, scaler, percentage = None):
        # If there are no cross cloudlet edges, return None
        if self.cross_cloudlet_edge_index.size(1) == 0:
            return None
        
        # Detect vehicle speed peaks for local nodes - if detected, don't remove it's neighbours (cross cloudlet nodes)
        ineligible_nodes = self.detect_vehicle_speed_peaks(epoch, scaler, self.original_y_train)
        utility.save_ineligible_nodes_logs(self.logs_folder, self.cln_id, ineligible_nodes)

        # === 1. Compute average scores per node (avoid division by zero) ===
        avg_scores = torch.zeros_like(self.node_scores)
        non_zero_mask = self.node_counts != 0
        avg_scores[non_zero_mask] = self.node_scores[non_zero_mask] / self.node_counts[non_zero_mask].float()

        # === 2. Create score map for valid (non-zero, non-NaN) nodes ===
        valid_mask = non_zero_mask & ~torch.isnan(avg_scores) & (avg_scores > 0)
        valid_nodes = self.original_cross_cloudlet_nodes[valid_mask]
        valid_scores = avg_scores[valid_mask]

        # === Filter out ineligible nodes ===
        ineligible_tensor = torch.tensor(list(ineligible_nodes), device=valid_nodes.device)
        eligible_mask = ~torch.isin(valid_nodes, ineligible_tensor)
        valid_nodes = valid_nodes[eligible_mask]
        valid_scores = valid_scores[eligible_mask]

        if valid_nodes.numel() == 0:
            print(f"{self.cln_id}: No valid scored nodes found.")
            utility.save_nodes_removed_by_distribution_logs(self.logs_folder, self.cln_id, set())
            return
        # else:
        #     print(f"{self.cln_id} - valid_nodes: {valid_nodes}")
        
        # === 3. Normalize scores into a probability distribution ===
        total_score = valid_scores.sum().item()
        node_probs = valid_scores / total_score

        # === 4. Sample N% of nodes based on distribution ===
        n_pool = int(self.original_cross_cloudlet_nodes.size(0))
        if percentage is None:
            perc = self.comm_ctrl.current_fraction(n_pool)   # <-- controller decides
        else:
            perc = percentage
        target_count = int(perc * n_pool)
        num_to_select = max(1, min(target_count, valid_nodes.size(0)))  # Don't over-select
        selected_indices = torch.multinomial(node_probs, num_samples=num_to_select, replacement=False)
        selected_nodes = set(valid_nodes[selected_indices].tolist())
        # num_to_select = max(1, int(percentage * valid_nodes.size(0)))
        # selected_indices = torch.multinomial(node_probs, num_samples=num_to_select, replacement=False)
        # selected_nodes = set(valid_nodes[selected_indices].tolist())
        utility.save_nodes_removed_by_distribution_logs(self.logs_folder, self.cln_id, selected_nodes)
        print(f"{self.cln_id}: Removing ~{perc:.2f} of pool (n={n_pool}) → {len(selected_nodes)} nodes")

        # === 5. Mask out cross-cloudlet edges connected to selected nodes ===
        masked_cross_cloudlet_edges = torch.zeros(
            self.original_cross_cloudlet_edge_index.size(1),
            dtype=torch.bool,
            device=self.original_cross_cloudlet_edge_index.device
        )

        src, dst = self.original_cross_cloudlet_edge_index
        for idx in range(src.size(0)):
            u, v = src[idx].item(), dst[idx].item()
            if u in selected_nodes or v in selected_nodes:
                masked_cross_cloudlet_edges[idx] = True

        # === 6. Create new edge index by masking out selected edges ===
        cloudlet_edge_mask = torch.ones(self.original_edge_index.size(1), dtype=torch.bool, device=self.original_edge_index.device)
        cloudlet_edge_mask[self.original_cross_cloudlet_edge_mask] = ~masked_cross_cloudlet_edges
        masked_cln_edge_index = self.original_edge_index[:, cloudlet_edge_mask]

        cln_nodes_subgraph, cln_edge_index, cln_node_map, _ = k_hop_subgraph(
            self.local_cln_nodes,
            stblock_num * (Ks - 1),
            masked_cln_edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes
        )
        cln_edge_index = cln_edge_index.to(device=device)
        cln_node_map = cln_node_map.to(device=device)

        self.edge_index = cln_edge_index
        self.node_map = cln_node_map

        # Set new cross cloudlet nodes
        all_node_indices = torch.arange(cln_nodes_subgraph.size(0), device=cln_nodes_subgraph.device)
        cross_cloudlet_node_indices = all_node_indices[~torch.isin(all_node_indices, cln_node_map)]
        self.cross_cloudlet_nodes = cln_nodes_subgraph[cross_cloudlet_node_indices]

        # === 8. Recalculate cross-cloudlet edge index/mask ===
        cln_nodes_tensor = torch.tensor(self.local_cln_nodes, device=device)
        src, dst = cln_edge_index
        cross_cloudlet_edge_mask = ~(torch.isin(dst, cln_nodes_tensor) & torch.isin(src, cln_nodes_tensor))
        cross_cloudlet_edge_index = cln_edge_index[:, cross_cloudlet_edge_mask]

        self.cross_cloudlet_edge_index = cross_cloudlet_edge_index
        self.cross_cloudlet_edge_mask = cross_cloudlet_edge_mask

        # === 9. Update train/val datasets ===
        cln_nodes_subgraph_cpu = cln_nodes_subgraph.cpu().numpy()

        new_train_dataset = self.train_dataset[:, cln_nodes_subgraph_cpu]
        x_train, y_train = dataloader.data_transform(new_train_dataset, n_his, n_pred, device)
        self.x_train = x_train
        self.y_train = y_train

        new_val_dataset = self.val_dataset[:, cln_nodes_subgraph_cpu]
        x_val, y_val = dataloader.data_transform(new_val_dataset, n_his, n_pred, device)
        val_data = utils.data.TensorDataset(x_val, y_val)
        self.val_iter = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

    def delta_error_between_original_and_masked_for_node_score_online_training(self, epoch, zscore, num_epochs, masked_edge_index, masked_cross_cloudlet_nodes, masked_val_iter, masked_cln_node_map):
        d_err = super().delta_error_between_original_and_masked_for_online_training(epoch, zscore, num_epochs, masked_val_iter, masked_edge_index, masked_cln_node_map)

        # Clamp delta_err to get values ONLY between -10 and 10
        delta_err_clamped = torch.clamp(torch.tensor(d_err, device=self.node_scores.device), -10, 10)

        self.node_scores[masked_cross_cloudlet_nodes] += delta_err_clamped
        self.node_counts[masked_cross_cloudlet_nodes] += 1

    def log_node_score_results(self):
        file_name = f"cloudlet_{self.cln_id}"  # Assuming you have a `self.index` for cloudlet ID
        node_positions = self.original_cross_cloudlet_nodes.cpu().numpy()
        utility.save_node_scores_logs(self.logs_folder, file_name, self.node_scores, node_positions)
        utility.save_node_counts_logs(self.logs_folder, file_name, self.node_counts, node_positions)