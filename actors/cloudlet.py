import torch
from actors import network_actor as na
from actors import cloudlet_base as cb
from script import utility
from messages import messages
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter

class Cloudlet(na.NetworkActor, cb.CloudletBase):
    def __init__(self, server, model, loss, optimizer, scheduler, cln_id, edge_index, node_map, train_iter, val_iter, test_iter, logs_folder, cln_adj_matrix = [], train_dataset = [], val_dataset = [], x_train = [], y_train = [], end_of_initial_data_index = 0, data_per_step = 0, batch_size = 32, cross_cloudlet_edges = None, cross_cloudlet_edges_map = None, local_cln_nodes = None):
        super().__init__(server, model, loss, optimizer, scheduler, cln_id, edge_index, node_map, train_iter, x_train, y_train, val_iter, test_iter, logs_folder, end_of_initial_data_index, data_per_step, train_dataset, batch_size, cross_cloudlet_edges, cross_cloudlet_edges_map, local_cln_nodes, val_dataset)
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
        self.model_backup = model

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

    def online_train(self, epoch, master_actor):
        self.__train_online(epoch)

        # using NetworkActor, send model to master model
        self.send_to(master_actor, messages.CloudletModelMessage(self.model, self.cln_id))

    def train_no_master(self, epoch):
        self.__train(epoch)

    def online_train_no_master(self, epoch):
        self.__train_online(epoch)

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

    def __train_online(self, epoch):
        train_iter = self.create_train_iter_for_online(epoch, self.x_train, self.y_train)
        super().online_train(epoch, train_iter)

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