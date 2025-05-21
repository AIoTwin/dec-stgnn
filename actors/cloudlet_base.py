import tqdm
import torch
import torch.utils as utils
import numpy as np
import random
from script import utility
from collections import defaultdict
import wandb

class CloudletBase:
    def __init__(self, server, model, loss, optimizer, scheduler, cln_id, edge_index, node_map, train_iter, x_train, y_train, val_iter, test_iter, logs_folder, end_of_initial_data_index, data_per_step, train_dataset, batch_size, cross_cloudlet_edge_index, cross_cloudlet_edge_mask, local_cln_nodes, val_dataset):
        self.server = server
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cln_id = cln_id
        self.edge_index = edge_index
        self.original_edge_index = edge_index # Does not change
        self.node_map = node_map
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.original_val_iter = val_iter # Does not change
        self.test_iter = test_iter
        self.writer = None
        self.logs_folder = logs_folder
        self.total_num_of_parameters = 0
        self.best_model = model
        self.x_train = x_train
        self.y_train = y_train
        self.original_x_train = x_train # Does not change
        self.original_y_train = y_train # Does not change
        self.end_of_initial_data_index = end_of_initial_data_index
        self.data_per_step = data_per_step
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.cross_cloudlet_edge_index = cross_cloudlet_edge_index
        self.cross_cloudlet_edge_mask = cross_cloudlet_edge_mask
        self.local_cln_nodes = local_cln_nodes
        self.val_dataset = val_dataset

    def init_wandb(self):
        wandb.init(
            mode="offline",
            project="stgcn-distributed"
        )

    def finish_wandb(self):
        wandb.finish()

    @torch.no_grad()
    def val_2d_data_metric(self, epoch, zscore):
        self.model.eval()
        val_MAE, val_RMSE, val_WMAPE, val_MAPE = utility.validate_pyg_metric(self.model, self.val_iter, zscore, self.edge_index, self.node_map)
        utility.save_val_metric_logs(self.logs_folder, self.cln_id, epoch+1, val_MAE, val_RMSE, val_WMAPE)
        print(f'CLN: {self.cln_id:03d} | Epoch: {epoch+1:03d} | MAE {val_MAE:.6f} | RMSE {val_RMSE:.6f} | WMAPE {val_WMAPE:.8f} | MAPE {val_MAPE:.6f}')

    @torch.no_grad()
    def val_2d_data_metric_online(self, epoch, zscore, val_iter):
        self.model.eval()
        val_MAE, val_RMSE, val_WMAPE, val_MAPE = utility.validate_pyg_metric(self.model, val_iter, zscore, self.edge_index, self.node_map)
        utility.save_val_metric_logs(self.logs_folder, self.cln_id, epoch, val_MAE, val_RMSE, val_WMAPE)
        print(f'CLN: {self.cln_id:03d} | Epoch: {epoch+1:03d} | MAE {val_MAE:.6f} | RMSE {val_RMSE:.6f} | WMAPE {val_WMAPE:.8f} | MAPE {val_MAPE:.6f}')

    @torch.no_grad()
    def test_3d_data(self, zscore, args):
        self.model.eval()
        test_MSE = utility.evaluate_3d_pyg_model(self.model, self.loss, self.test_iter, self.edge_index, self.node_map)
        for node_feature_to_evaluate in args.node_features_to_evaluate:
            test_MAE, test_RMSE, test_WMAPE = utility.evaluate_3d_pyg_metric(self.model, self.test_iter, zscore, self.edge_index, self.node_map, node_feature_to_evaluate)
            utility.save_test_logs(self.logs_folder, f"cln-{self.cln_id}_node-feature-{node_feature_to_evaluate}", test_MSE, test_MAE, test_RMSE, test_WMAPE)
            print(f'CLN: {self.cln_id:03d} | Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

    @torch.no_grad()
    def test_2d_data(self, zscore, args):
        self.model.eval()
        test_MSE = utility.evaluate_pyg_model(self.model, self.loss, self.test_iter, self.edge_index, self.node_map)
        test_MAE, test_RMSE, test_WMAPE = utility.evaluate_pyg_metric(self.model, self.test_iter, zscore, self.edge_index, self.node_map)
        utility.save_test_logs(self.logs_folder, f"cln-{self.cln_id}", test_MSE, test_MAE, test_RMSE, test_WMAPE)
        print(f'CLN: {self.cln_id:03d} | Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

    @torch.no_grad()
    def test_2d_data_using_best_model(self, zscore, args, best_epoch):
        self.best_model.eval()
        test_MSE = utility.evaluate_pyg_model(self.best_model, self.loss, self.test_iter, self.edge_index, self.node_map)
        test_MAE, test_RMSE, test_WMAPE = utility.evaluate_pyg_metric(self.best_model, self.test_iter, zscore, self.edge_index, self.node_map)
        utility.save_test_logs(self.logs_folder, f"cln-{self.cln_id}", test_MSE, test_MAE, test_RMSE, test_WMAPE, best_epoch+1)
        print(f'CLN: {self.cln_id:03d} | Epoch: {best_epoch+1:03d} | Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

    # disable gradient calculation within a block of code
    # i.e. don't calculate gradients during validation or testing process
    @torch.no_grad()
    def val(self):
        self.model.eval()
        l_sum, n = 0.0, 0
        for x, y in self.val_iter:
            if len(y.shape) == 3:
                y_pred = utility.permute_4d_y_pred_to_3d(self.model, x, self.edge_index)
                y = y[:,self.node_map,:]
                y_pred = y_pred[:,self.node_map,:]
            elif len(y.shape) == 2:
                y_pred = self.model(x, self.edge_index).view(len(x), -1)
                y = y[...,self.node_map]
                y_pred = y_pred[...,self.node_map]
            else:
                assert False, f'Cannot work with data that has {len(y.shape)} dimension'
            l = self.loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return torch.tensor(l_sum / n)
    
    # disable gradient calculation within a block of code
    # i.e. don't calculate gradients during validation or testing process
    @torch.no_grad()
    def val_online_train(self, val_iter):
        self.model.eval()
        l_sum, n = 0.0, 0
        for x, y in val_iter:
            if len(y.shape) == 3:
                y_pred = utility.permute_4d_y_pred_to_3d(self.model, x, self.edge_index)
                y = y[:,self.node_map,:]
                y_pred = y_pred[:,self.node_map,:]
            elif len(y.shape) == 2:
                y_pred = self.model(x, self.edge_index).view(len(x), -1)
                y = y[...,self.node_map]
                y_pred = y_pred[...,self.node_map]
            else:
                assert False, f'Cannot work with data that has {len(y.shape)} dimension'
            l = self.loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return torch.tensor(l_sum / n)

    def train(self, epoch):
        print(f"Start training...")
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        self.model.train() # train the model
        total_transfer_size_of_node_features = 0
        for x, y in tqdm.tqdm(self.train_iter):
            transfer_size_of_node_features = utility.collect_node_features(self.node_map, x, y)
            total_transfer_size_of_node_features += transfer_size_of_node_features
            if len(y.shape) == 3:
                y_pred = utility.permute_4d_y_pred_to_3d(self.model, x, self.edge_index)
                y = y[:,self.node_map,:]
                y_pred = y_pred[:,self.node_map,:]
            elif len(y.shape) == 2:
                y_pred = self.model(x, self.edge_index).view(len(x), -1)  # [batch_size, num_nodes]     
                y = y[...,self.node_map]
                y_pred = y_pred[...,self.node_map]
            else:
                assert False, f'Cannot work with data that has {len(y.shape)} dimension'
            l = self.loss(y_pred, y) # compute loss between predicted output (y_pred) and actual target (y)
            self.optimizer.zero_grad() # clear the gradients of all optimized tensors
            l.backward() # compute the gradients of the loss with respect to the model parameters
            self.optimizer.step() # update model parameters based on computed gradients and optimization algorithm
            l_sum += l.item() * y.shape[0] # accumulate batch loss
            n += y.shape[0] # accumulate number of instances processed
        self.scheduler.step() # adjust learning rate parameter
        val_loss = self.val() # compute validation loss
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        # Get number of parameter for cloudlet
        cln_num_of_parameters = sum(p.numel() for p in self.model.parameters())
        self.total_num_of_parameters += cln_num_of_parameters
        # total_num_of_parameters += cln_num_of_parameters
        print('CLN: {:03d} | Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB | Num of parameters: {:d}'.\
            format(self.cln_id, epoch+1, self.optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc, cln_num_of_parameters))
        print(f"Total transfer size of node features | in bytes: {total_transfer_size_of_node_features} | in kilobytes: {total_transfer_size_of_node_features/1024} | in megabytes: {(total_transfer_size_of_node_features/1024)/1024}")
        # Save logs to CSV and tensorboard
        self.writer.add_scalar(f"Loss/train_cln-{self.cln_id}", l_sum / n, epoch+1)
        self.writer.add_scalar(f"Loss/val_cln-{self.cln_id}", val_loss, epoch+1)
        wandb.log({f"Loss/train_cln-{self.cln_id}": l_sum / n, f"Loss/val_cln-{self.cln_id}": val_loss})
        utility.save_val_logs(self.logs_folder, self.cln_id, epoch+1, self.optimizer.param_groups[0]['lr'], l_sum / n, val_loss.item(), gpu_mem_alloc)
        utility.save_total_transfer_size_node_features(self.logs_folder, self.cln_id, epoch+1, total_transfer_size_of_node_features)

    def online_train(self, epoch, train_iter, zscore, num_epochs = 30):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        self.model.train() # train the model
        total_transfer_size_of_node_features = 0
        for x, y in tqdm.tqdm(train_iter):
            transfer_size_of_node_features = utility.collect_node_features(self.node_map, x, y)
            total_transfer_size_of_node_features += transfer_size_of_node_features
            if len(y.shape) == 3:
                y_pred = utility.permute_4d_y_pred_to_3d(self.model, x, self.edge_index)
                y = y[:,self.node_map,:]
                y_pred = y_pred[:,self.node_map,:]
            elif len(y.shape) == 2:
                y_pred = self.model(x, self.edge_index).view(len(x), -1)  # [batch_size, num_nodes]     
                y = y[...,self.node_map]
                y_pred = y_pred[...,self.node_map]
            else:
                assert False, f'Cannot work with data that has {len(y.shape)} dimension'
            l = self.loss(y_pred, y) # compute loss between predicted output (y_pred) and actual target (y)
            self.optimizer.zero_grad() # clear the gradients of all optimized tensors
            l.backward() # compute the gradients of the loss with respect to the model parameters
            self.optimizer.step() # update model parameters based on computed gradients and optimization algorithm
            l_sum += l.item() * y.shape[0] # accumulate batch loss
            n += y.shape[0] # accumulate number of instances processed
        self.scheduler.step() # adjust learning rate parameter
        self.model.eval()
        if epoch < num_epochs:
            # Create masked validation dataset
            new_x_val = self.x_train[
                self.end_of_initial_data_index + (self.data_per_step * (epoch - 1)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch))
                ]
            new_y_val = self.y_train[
                self.end_of_initial_data_index + (self.data_per_step * (epoch - 1)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch))
            ]
            new_val_data = utils.data.TensorDataset(new_x_val, new_y_val)
            new_val_iter = utils.data.DataLoader(dataset=new_val_data, batch_size=self.batch_size, shuffle=False)

            # Create original validation dataset
            original_x_val = self.original_x_train[
                self.end_of_initial_data_index + (self.data_per_step * (epoch - 1)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch))
                ]
            original_y_val = self.original_y_train[
                self.end_of_initial_data_index + (self.data_per_step * (epoch - 1)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch))
            ]
            original_val_data = utils.data.TensorDataset(original_x_val, original_y_val)
            original_val_iter = utils.data.DataLoader(dataset=original_val_data, batch_size=self.batch_size, shuffle=False)

            # Calculate validation loss using new val iter
            val_loss = self.val_online_train(new_val_iter)

            # Calculate masked val metrics
            val_MAE, val_RMSE, val_WMAPE = utility.evaluate_pyg_metric_master(self.model, new_val_iter, zscore, self.edge_index)
            utility.save_val_metric_logs(self.logs_folder, f"masked_{self.cln_id}", epoch, val_MAE, val_RMSE, val_WMAPE)
            # Calculate original val metrics (to compare to masked)
            val_MAE, val_RMSE, val_WMAPE = utility.evaluate_pyg_metric_master(self.model, original_val_iter, zscore, self.original_edge_index)
            utility.save_val_metric_logs(self.logs_folder, f"original_{self.cln_id}", epoch, val_MAE, val_RMSE, val_WMAPE)
        elif epoch == num_epochs:
            # Calculate validation loss using standard val iter
            val_loss = self.val()

            # Calculate masked val metrics
            val_MAE, val_RMSE, val_WMAPE = utility.evaluate_pyg_metric_master(self.model, self.val_iter, zscore, self.edge_index)
            utility.save_val_metric_logs(self.logs_folder, f"masked_{self.cln_id}", epoch, val_MAE, val_RMSE, val_WMAPE)
            # Calculate original val metrics
            val_MAE, val_RMSE, val_WMAPE = utility.evaluate_pyg_metric_master(self.model, self.original_val_iter, zscore, self.original_edge_index)
            utility.save_val_metric_logs(self.logs_folder, f"original_{self.cln_id}", epoch, val_MAE, val_RMSE, val_WMAPE)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        # Get number of parameter for cloudlet
        cln_num_of_parameters = sum(p.numel() for p in self.model.parameters())
        self.total_num_of_parameters += cln_num_of_parameters
        # total_num_of_parameters += cln_num_of_parameters
        print('CLN: {:03d} | Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB | Num of parameters: {:d}'.\
            format(self.cln_id, epoch, self.optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc, cln_num_of_parameters))
        print(f"Total transfer size of node features | in bytes: {total_transfer_size_of_node_features} | in kilobytes: {total_transfer_size_of_node_features/1024} | in megabytes: {(total_transfer_size_of_node_features/1024)/1024}")
        # Save logs to CSV and tensorboard
        self.writer.add_scalar(f"Loss/train_cln-{self.cln_id}", l_sum / n, epoch)
        self.writer.add_scalar(f"Loss/val_cln-{self.cln_id}", val_loss, epoch)
        wandb.log({f"Loss/train_cln-{self.cln_id}": l_sum / n, f"Loss/val_cln-{self.cln_id}": val_loss})
        utility.save_val_logs(self.logs_folder, self.cln_id, epoch, self.optimizer.param_groups[0]['lr'], l_sum / n, val_loss.item(), gpu_mem_alloc)
        utility.save_total_transfer_size_node_features(self.logs_folder, self.cln_id, epoch, total_transfer_size_of_node_features)

    def store_new_best_model(self):
        for cln_model_param_1, cln_model_param_2 in zip(self.best_model.parameters(), self.model.parameters()):
            cln_model_param_1.data.copy_(cln_model_param_2.data)

    def log_val_metrics(self, epoch, zscore):
        self.model.eval()
        val_MAE, val_RMSE, val_WMAPE = utility.evaluate_pyg_metric_master(self.model, self.val_iter, zscore, self.edge_index)
        utility.save_val_metric_logs(self.logs_folder, self.cln_id, epoch+1, val_MAE, val_RMSE, val_WMAPE)
        print(f'Cloudlet: {self.cln_id:02d} | Epoch: {epoch+1:03d} | MAE {val_MAE:.6f} | RMSE {val_RMSE:.6f} | WMAPE {val_WMAPE:.8f}')

    def log_var_info(self, epoch):
        param_variances = utility.compute_parameter_variance_master(self.model.named_parameters())
        utility.save_variance_logs(self.logs_folder, self.cln_id, epoch+1, param_variances)

    def log_validation_metric_for_online_training(self, epoch, zscore, num_epochs):
        if (epoch < num_epochs):
            new_x_val = self.x_train[
                self.end_of_initial_data_index + (self.data_per_step * (epoch - 1)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch))
                ]
            new_y_val = self.y_train[
                self.end_of_initial_data_index + (self.data_per_step * (epoch - 1)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch))
                ]

            new_val_data = utils.data.TensorDataset(new_x_val, new_y_val)
            new_val_iter = utils.data.DataLoader(dataset=new_val_data, batch_size=self.batch_size, shuffle=False)

            self.val_2d_data_metric_online(epoch, zscore, new_val_iter)
        else:
            self.val_2d_data_metric(epoch-1, zscore)

    def create_train_iter_for_online(self, epoch, x_train, y_train):
        current_train = self.train_dataset[:self.end_of_initial_data_index + (self.data_per_step * (epoch - 1))]
        if epoch == 1:
            inital_x_train = x_train[:self.end_of_initial_data_index]
            inital_y_train = y_train[:self.end_of_initial_data_index]
            train_data = utils.data.TensorDataset(inital_x_train, inital_y_train)
            train_iter = utils.data.DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)

            return train_iter
        else:
            new_train_datastep = self.train_dataset[
                self.end_of_initial_data_index + (self.data_per_step * (epoch - 1)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch))
            ]

            random_sample_size = (self.batch_size - 1) * self.data_per_step
            # Randomly sample indices from current_train
            if len(current_train) > random_sample_size:
                random_indices = np.random.choice(current_train.shape[0], random_sample_size, replace=False)
            else:
                # If current_train has fewer than the required samples, take all of it
                random_indices = current_train
            new_train_datastep = [self.train_dataset[idx] for idx in range(len(new_train_datastep))]

            new_x_train = x_train[
                self.end_of_initial_data_index + (self.data_per_step * (epoch - 1)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch))
            ]
            sampled_x_train = x_train[random_indices, :]
            new_x_train = torch.cat((sampled_x_train, new_x_train), dim=0)
            new_y_train = y_train[
                self.end_of_initial_data_index + (self.data_per_step * (epoch - 1)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch))
            ]
            sampled_y_train = y_train[random_indices, :]
            new_y_train = torch.cat((sampled_y_train, new_y_train), dim=0)

            train_data = utils.data.TensorDataset(new_x_train, new_y_train)
            train_iter = utils.data.DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)

            return train_iter
        
    def choose_random_cross_cloudlet_edges(self):
        if self.cross_cloudlet_edge_index.size(1) == 0:  # No cross-cloudlet edges
            masked_cross_cloudlet_edges = torch.empty((2, 0), dtype=torch.long, device=self.cross_cloudlet_edges.device)
            return masked_cross_cloudlet_edges
        
        masked_cross_cloudlet_edges = torch.rand(self.cross_cloudlet_edge_index.shape[1]) < 0.5 # select random 50% edges

        return masked_cross_cloudlet_edges
    
    def remove_edges(self, masked_cross_cloudlet_edges):
        if masked_cross_cloudlet_edges.numel() == 0:
            return self.edge_index  # Nothing to remove

        cloudlet_edge_mask = torch.ones(self.edge_index.shape[1], dtype=torch.bool)
        cloudlet_edge_mask[self.cross_cloudlet_edge_mask] = ~masked_cross_cloudlet_edges # Select edges which are NOT REMOVED!!!
        masked_cln_edge_index = self.edge_index[:,cloudlet_edge_mask]
        
        return masked_cln_edge_index
    
    def delta_error_between_original_and_masked_for_online_training(self, epoch, zscore, num_epochs, masked_edge_index):
        if (epoch < num_epochs):
            new_x_val = self.x_train[
                self.end_of_initial_data_index + (self.data_per_step * (epoch - 1)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch))
                ]
            new_y_val = self.y_train[
                self.end_of_initial_data_index + (self.data_per_step * (epoch - 1)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch))
                ]

            new_val_data = utils.data.TensorDataset(new_x_val, new_y_val)
            new_val_iter = utils.data.DataLoader(dataset=new_val_data, batch_size=self.batch_size, shuffle=False)

            self.model.eval()
            d_original, _ = utility.evaluate_cloudlet_pyg_metric_analysis(self.model, new_val_iter, zscore, self.edge_index, self.node_map)
            d_masked, _ = utility.evaluate_cloudlet_pyg_metric_analysis(self.model, new_val_iter, zscore, masked_edge_index, self.node_map)

            d_err = (d_masked - d_original).mean()
        else:
            self.model.eval()
            d_original, _ = utility.evaluate_cloudlet_pyg_metric_analysis(self.model, self.val_iter, zscore, self.edge_index, self.node_map)
            d_masked, _ = utility.evaluate_cloudlet_pyg_metric_analysis(self.model, self.val_iter, zscore, masked_edge_index, self.node_map)

            d_err = (d_masked - d_original).mean()

        return d_err
        
    def choose_cross_cloudlet_edges_by_distribution(self, num_nodes, percentage = 0.1):
        if self.cross_cloudlet_edge_index.size(1) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.cross_cloudlet_edge_index.device)
    
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
        total_score = 0.0
        for node in node_score_sum:
            if node_score_count[node] > 0:
                avg = node_score_sum[node] / node_score_count[node]
                node_avg_score[node] = avg
                total_score += avg
            else:
                node_avg_score[node] = float('inf')  # Avoid division by zero

        # === 5. Separate inf nodes from normal ones ===
        inf_nodes = [n for n, s in node_avg_score.items() if s == float('inf')]
        scored_nodes = {n: s for n, s in node_avg_score.items() if s != float('inf')}

        # === 6. Create probability distribution from scored nodes ===
        if total_score > 0:
            node_probs = {n: s / total_score for n, s in scored_nodes.items()}
        else:
            node_probs = {}

        # === 7. Sample N% of scored nodes ===
        selected_nodes = set()
        if node_probs:
            num_to_select = max(1, int(percentage * len(scored_nodes)))  # ensure at least one
            selected_nodes.update(random.choices(
                list(node_probs.keys()),
                weights=list(node_probs.values()),
                k=num_to_select
            ))

        # Always include nodes with inf scores
        selected_nodes.update(inf_nodes)

        # === 8. Select all edges where u or v in selected_nodes ===
        selected_mask = torch.zeros(self.cross_cloudlet_edge_index.size(1), dtype=torch.bool, device=self.cross_cloudlet_edge_index.device)

        print(f"selected_mask: {selected_mask}")

        # for idx in range(cross_edges.shape[1]):
        #     u, v = cross_edges[0, idx].item(), cross_edges[1, idx].item()
        #     if u in selected_nodes or v in selected_nodes:
        #         selected_mask[idx] = True

        # return selected_mask