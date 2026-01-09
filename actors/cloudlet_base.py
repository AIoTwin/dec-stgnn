import tqdm
import torch
import torch.utils as utils
import numpy as np
import random
from script import utility
from actors import adaptive_comm_controller
from scipy.signal import find_peaks
from script import utility, dataloader
import wandb
from torch_geometric.utils import k_hop_subgraph

class CloudletBase:
    def __init__(self, server, model, loss, optimizer, scheduler, cln_id, edge_index, node_map, train_iter, x_train, y_train, val_iter, test_iter, logs_folder, end_of_initial_data_index, data_per_step, train_dataset, batch_size, cross_cloudlet_edge_index, cross_cloudlet_edge_mask, local_cln_nodes, val_dataset, original_edge_index, cross_cloudlet_nodes):
        self.server = server
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cln_id = cln_id
        self.edge_index = edge_index
        self.original_edge_index = original_edge_index # Does not change
        self.node_map = node_map
        self.original_node_map = node_map # Does not change
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
        self.original_cross_cloudlet_edge_index = cross_cloudlet_edge_index # Does not change
        self.cross_cloudlet_edge_mask = cross_cloudlet_edge_mask
        self.original_cross_cloudlet_edge_mask = cross_cloudlet_edge_mask # Does not change
        self.local_cln_nodes = local_cln_nodes
        self.val_dataset = val_dataset
        self.cross_cloudlet_nodes = cross_cloudlet_nodes
        self.original_cross_cloudlet_nodes = cross_cloudlet_nodes # Does not change

        self.comm_ctrl = adaptive_comm_controller.AdaptiveCommController()

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
                x = x.to('cuda:0')
                edge_index = self.edge_index.to('cuda:0')
                y_pred = self.model(x, edge_index).view(len(x), -1)
                y = y[...,self.node_map]
                y_pred = y_pred[...,self.node_map]
            else:
                assert False, f'Cannot work with data that has {len(y.shape)} dimension'
            y = y.to('cuda:0')
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
                x = x.to('cuda:0')
                edge_index = self.edge_index.to('cuda:0')
                y_pred = self.model(x, edge_index).view(len(x), -1)
                y = y[...,self.node_map]
                y_pred = y_pred[...,self.node_map]
            else:
                assert False, f'Cannot work with data that has {len(y.shape)} dimension'
            y = y.to('cuda:0')
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
        edge_index = self.edge_index.to('cuda:0')
        for x, y in tqdm.tqdm(train_iter):
            transfer_size_of_node_features = utility.collect_node_features(self.node_map, x, y)
            total_transfer_size_of_node_features += transfer_size_of_node_features
            if len(y.shape) == 3:
                y_pred = utility.permute_4d_y_pred_to_3d(self.model, x, self.edge_index)
                y = y[:,self.node_map,:]
                y_pred = y_pred[:,self.node_map,:]
            elif len(y.shape) == 2:
                x = x.to('cuda:0')
                y_pred = self.model(x, edge_index).view(len(x), -1)  # [batch_size, num_nodes]     
                y = y[...,self.node_map]
                y_pred = y_pred[...,self.node_map]
            else:
                assert False, f'Cannot work with data that has {len(y.shape)} dimension'
            y = y.to('cuda:0')
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
                self.end_of_initial_data_index + (self.data_per_step * (epoch)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch + 1))
                ]
            new_y_val = self.y_train[
                self.end_of_initial_data_index + (self.data_per_step * (epoch)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch + 1))
            ]
            new_val_data = utils.data.TensorDataset(new_x_val, new_y_val)
            new_val_iter = utils.data.DataLoader(dataset=new_val_data, batch_size=self.batch_size, shuffle=False)

            # Create original validation dataset
            original_x_val = self.original_x_train[
                self.end_of_initial_data_index + (self.data_per_step * (epoch)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch + 1))
                ]
            original_y_val = self.original_y_train[
                self.end_of_initial_data_index + (self.data_per_step * (epoch)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch + 1))
            ]
            original_val_data = utils.data.TensorDataset(original_x_val, original_y_val)
            original_val_iter = utils.data.DataLoader(dataset=original_val_data, batch_size=self.batch_size, shuffle=False)

            # Calculate validation loss using new val iter
            val_loss = self.val_online_train(new_val_iter)

            # Calculate masked val metrics
            # val_MAE, val_RMSE, val_WMAPE = utility.evaluate_pyg_metric_master(self.model, new_val_iter, zscore, self.edge_index)
            # val_MAE, val_RMSE, val_WMAPE, big_error_count, big_error_rate, sudden_event_count, sudden_event_hits, SUDDEN_EVENT_RATE, jam_event_count, jam_event_hits, JAM_EVENT_RATE, rec_event_count, rec_event_hits, REC_EVENT_RATE = utility.evaluate_cloudlet_pyg_new_metric_analysis(
            #     self.model, new_val_iter, zscore, self.edge_index, self.node_map
            # )
            val_MAE, val_RMSE, val_WMAPE, big_error_count, big_error_rate, sudden_event_count, sudden_event_hits, SUDDEN_EVENT_RATE, jam_event_count, jam_event_hits, JAM_EVENT_RATE, rec_event_count, rec_event_hits, REC_EVENT_RATE, precision, recall, f1, iou, accuracy, gt_cong_rate, total_points, total_gt_cong, total_pred_cong = utility.evaluate_cloudlet_pyg_new_metric_analysis_with_alpha_propagation(
                self.model, new_val_iter, zscore, self.edge_index, self.node_map
            )
            utility.save_val_metric_logs(self.logs_folder, f"masked_{self.cln_id}", epoch, val_MAE, val_RMSE, val_WMAPE)
            utility.save_val_new_metric_logs(self.logs_folder, f"masked_{self.cln_id}", epoch, big_error_count, big_error_rate, sudden_event_count, sudden_event_hits, SUDDEN_EVENT_RATE, jam_event_count, jam_event_hits, JAM_EVENT_RATE, rec_event_count, rec_event_hits, REC_EVENT_RATE)
            utility.save_val_alpha_propagation_metric_logs(self.logs_folder, f"masked_{self.cln_id}", epoch, precision, recall, f1, iou, accuracy, gt_cong_rate, total_points, total_gt_cong, total_pred_cong)

            # Update controller with masked metric only
            n_pool = int(self.original_cross_cloudlet_nodes.size(0))
            self.comm_ctrl.update_after_validation(epoch=epoch,
                                                rate=float(SUDDEN_EVENT_RATE),
                                                n_pool=n_pool,
                                                logs_folder=self.logs_folder,
                                                cln_id=self.cln_id)

            # Calculate original val metrics (to compare to masked)
            val_MAE, val_RMSE, val_WMAPE = utility.evaluate_pyg_metric_master(self.model, original_val_iter, zscore, self.original_edge_index)
            utility.save_val_metric_logs(self.logs_folder, f"original_{self.cln_id}", epoch, val_MAE, val_RMSE, val_WMAPE)
        elif epoch == num_epochs:
            # Calculate validation loss using standard val iter
            val_loss = self.val()

            # Calculate masked val metrics
            # val_MAE, val_RMSE, val_WMAPE = utility.evaluate_pyg_metric_master(self.model, self.val_iter, zscore, self.edge_index)
            # val_MAE, val_RMSE, val_WMAPE, big_error_count, big_error_rate, sudden_event_count, sudden_event_hits, SUDDEN_EVENT_RATE, jam_event_count, jam_event_hits, JAM_EVENT_RATE, rec_event_count, rec_event_hits, REC_EVENT_RATE = utility.evaluate_cloudlet_pyg_new_metric_analysis(
            #     self.model, self.val_iter, zscore, self.edge_index, self.node_map
            # )
            val_MAE, val_RMSE, val_WMAPE, big_error_count, big_error_rate, sudden_event_count, sudden_event_hits, SUDDEN_EVENT_RATE, jam_event_count, jam_event_hits, JAM_EVENT_RATE, rec_event_count, rec_event_hits, REC_EVENT_RATE, precision, recall, f1, iou, accuracy, gt_cong_rate, total_points, total_gt_cong, total_pred_cong = utility.evaluate_cloudlet_pyg_new_metric_analysis_with_alpha_propagation(
                self.model, self.val_iter, zscore, self.edge_index, self.node_map
            )
            utility.save_val_metric_logs(self.logs_folder, f"masked_{self.cln_id}", epoch, val_MAE, val_RMSE, val_WMAPE)
            utility.save_val_new_metric_logs(self.logs_folder, f"masked_{self.cln_id}", epoch, big_error_count, big_error_rate, sudden_event_count, sudden_event_hits, SUDDEN_EVENT_RATE, jam_event_count, jam_event_hits, JAM_EVENT_RATE, rec_event_count, rec_event_hits, REC_EVENT_RATE)
            utility.save_val_alpha_propagation_metric_logs(self.logs_folder, f"masked_{self.cln_id}", epoch, precision, recall, f1, iou, accuracy, gt_cong_rate, total_points, total_gt_cong, total_pred_cong)
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
        if epoch == 0:
            inital_x_train = x_train[:self.end_of_initial_data_index]
            inital_y_train = y_train[:self.end_of_initial_data_index]
            train_data = utils.data.TensorDataset(inital_x_train, inital_y_train)
            train_iter = utils.data.DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)

            return train_iter
        else:
            current_train = self.train_dataset[:self.end_of_initial_data_index + (self.data_per_step * (epoch - 1))]

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
            print(f"new_x_train shape: {new_x_train.shape}")
            print(f"sampled_x_train shape: {sampled_x_train.shape}")
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
    
    def simple_detect_vehicle_speed_peaks(self, epoch, scaler, y_train, speed_peak = 20):        
        # === STEP 0: Detect peaks in ground truth data (vehicle speed) and restrict some neighbors from being selected===
        ineligible_nodes = set()

        # Only consider the new data that arrived at this epoch (no history or random samples)
        if epoch == 0:
            start_idx = 0
            end_idx = self.end_of_initial_data_index
        else:
            start_idx = self.end_of_initial_data_index + (self.data_per_step * (epoch - 1))
            end_idx = self.end_of_initial_data_index + (self.data_per_step * epoch)

        y_epoch = y_train[start_idx:end_idx]  # shape: [data_per_step, num_nodes]

        if y_epoch.shape[0] == 0:
            return ineligible_nodes  # no new data this epoch

        y_epoch_unscaled = scaler.inverse_transform(
            y_epoch.cpu().numpy().reshape(-1, y_epoch.shape[-1])
        )

        for local_node in self.local_cln_nodes:
            mapped_idx = (self.original_node_map == local_node).nonzero(as_tuple=True)[0].item()

            series = y_epoch_unscaled[:, mapped_idx]  # shape [timesteps] - series of data (vehicle speed) for 1 local node
            peaks, _ = find_peaks(series, prominence=speed_peak)

            if (len(peaks) > 0 and (len(peaks) % 2 == 1)):
                # Node shows vehicle speed spikes → protect its neighbors from being selected in choose_cross_cloudlet_nodes_by_distribution function
                neighbors_mask = (self.original_edge_index[0] == mapped_idx) | (self.original_edge_index[1] == mapped_idx)
                neighbors_src = self.original_edge_index[0, neighbors_mask]
                neighbors_dst = self.original_edge_index[1, neighbors_mask]
                neighbors = torch.cat([neighbors_src, neighbors_dst])
                ineligible_nodes.update(neighbors.tolist())

        print(f"{self.cln_id} - ineligible_nodes (epoch {epoch}): {ineligible_nodes}")
        return ineligible_nodes

    def detect_vehicle_speed_peaks(self,
                               epoch,
                               scaler,
                               y_train,
                               change_window: int = 12,
                               change_delta: float = 20.0,
                               cooldown: int | None = None):
        """
        Parity-based sudden change detector (ground-truth only).

        For each *local* node in this cloudlet:
        - Scan the epoch's time slice for sudden jams/recoveries using a rolling lookback.
        - Collapse consecutive equal events (J/J or R/R).
        - If the number of events is ODD -> mark this node's neighbors as INELIGIBLE (protected).
        - Else (EVEN, incl. 0/2/4...) -> neighbors are ELIGIBLE (not protected).

        Returns
        -------
        ineligible_nodes : set[int]
            Node indices (in original/global indexing) that should NOT be removed
            this epoch (typically neighbors of locally-volatile nodes).
        """
        ineligible_nodes = set()

        # Only consider the new data that arrived at this epoch (no history or random samples)
        if epoch == 0:
            start_idx = 0
            end_idx = self.end_of_initial_data_index
        else:
            start_idx = self.end_of_initial_data_index + (self.data_per_step * (epoch - 1))
            end_idx = self.end_of_initial_data_index + (self.data_per_step * epoch)

        y_epoch = y_train[start_idx:end_idx]  # [data_per_step, num_nodes]
        if y_epoch.shape[0] == 0:
            print(f"{self.cln_id} - ineligible_nodes (epoch {epoch}): {ineligible_nodes}")
            return ineligible_nodes
        
        # unscale to mile/h
        y_epoch_unscaled = scaler.inverse_transform(
            y_epoch.cpu().numpy().reshape(-1, y_epoch.shape[-1])
        )
        T, N = y_epoch_unscaled.shape

        if cooldown is None:
            cooldown = max(1, change_window // 2)

        # helper: add only TRUE neighbors (exclude the node itself)
        def add_neighbors_of(idx: int):
            # edges where idx appears on src: neighbors are dst
            mask_src = (self.original_edge_index[0] == idx)
            # edges where idx appears on dst: neighbors are src
            mask_dst = (self.original_edge_index[1] == idx)
            neigh_from_src = self.original_edge_index[1, mask_src]
            neigh_from_dst = self.original_edge_index[0, mask_dst]
            neigh = torch.cat([neigh_from_src, neigh_from_dst], dim=0)
            ineligible_nodes.update(neigh.tolist())

        # --- per local node: build collapsed event sequence and decide parity ---
        for local_node in self.local_cln_nodes:
            mapped_idx = (self.original_node_map == local_node).nonzero(as_tuple=True)[0].item()
            series = y_epoch_unscaled[:, mapped_idx]  # shape [T]

            if T < 2:
                continue

            # per-node cooldown counter
            cool = 0
            events = []  # collapsed sequence of 'J'/'R'

            for t in range(1, T):
                if cool > 0:
                    cool -= 1
                    continue

                w_start = max(0, t - change_window)
                past = series[w_start:t]              # shape [W]
                if past.size == 0:
                    continue

                cur = series[t]

                # Jam: drop from some past value to current
                jam_best = np.max(past) - cur         # == max(past - cur)
                # Recovery: rise from some past value to current
                rec_best = cur - np.min(past)         # == max(cur - past)

                jam_evt = jam_best >= change_delta
                rec_evt = rec_best >= change_delta

                if not (jam_evt or rec_evt):
                    continue

                # If both would trigger (rare), pick the stronger margin
                if jam_evt and rec_evt:
                    evt = 'J' if jam_best >= rec_best else 'R'
                else:
                    evt = 'J' if jam_evt else 'R'

                # collapse consecutive duplicates
                if not events or events[-1] != evt:
                    events.append(evt)

                cool = cooldown  # start cooldown after an event

            # decision by parity: odd -> protect neighbors; even -> allow removal
            if (len(events) % 2) == 1:
                add_neighbors_of(mapped_idx)
        
        print(f"{self.cln_id} - ineligible_nodes (epoch {epoch}): {ineligible_nodes}")
        return ineligible_nodes

    def choose_random_cross_cloudlet_edges(self):
        if self.cross_cloudlet_edge_index.size(1) == 0:  # No cross-cloudlet edges
            masked_cross_cloudlet_edges = torch.empty((2, 0), dtype=torch.long, device=self.cross_cloudlet_edges.device)
            return masked_cross_cloudlet_edges
        
        masked_cross_cloudlet_edges = torch.rand(self.cross_cloudlet_edge_index.shape[1]) < 0.5 # select random 50% edges

        return masked_cross_cloudlet_edges
    
    def choose_random_cross_cloudlet_edges_after_distribution(self):
        num_all_edges = self.original_cross_cloudlet_edge_index.shape[1]

        # First, determine which edges from the original are still present
        # Use a set for fast lookup
        current_edges_set = {
            (self.cross_cloudlet_edge_index[0, i].item(), self.cross_cloudlet_edge_index[1, i].item())
            for i in range(self.cross_cloudlet_edge_index.shape[1])
        }

        # Create a boolean mask the size of the original index
        valid_mask = torch.zeros(num_all_edges, dtype=torch.bool)

        for i in range(num_all_edges):
            edge = (
                self.original_cross_cloudlet_edge_index[0, i].item(),
                self.original_cross_cloudlet_edge_index[1, i].item()
            )
            if edge in current_edges_set:
                valid_mask[i] = True
        # From valid ones, randomly mask 50%
        random_mask = torch.rand(num_all_edges, device=valid_mask.device) < 0.8
        masked_cross_cloudlet_edges = valid_mask & random_mask

        return masked_cross_cloudlet_edges
    
    def choose_random_cross_cloudlet_nodes_after_distribution(self):
        # Map from node ID → position in original_cross_cloudlet_nodes
        node_id_to_index = {int(n.item()): i for i, n in enumerate(self.original_cross_cloudlet_nodes)}

        # Nodes that remain after distribution-based removal
        remaining_nodes = set(self.cross_cloudlet_nodes.tolist())

        # Sanity check: intersect with original, in case some mismatch
        remaining_nodes &= set(node_id_to_index.keys())

        if not remaining_nodes:
            return torch.zeros_like(self.original_cross_cloudlet_nodes, dtype=torch.bool)
        
        # Sample 50% of the remaining nodes
        remaining_nodes_list = list(remaining_nodes)
        num_to_sample = max(1, len(remaining_nodes_list) // 2)
        sampled_nodes = set(random.sample(remaining_nodes_list, k=num_to_sample))

        # Build the mask
        masked_cross_cloudlet_nodes = torch.zeros(self.original_cross_cloudlet_nodes.size(0), dtype=torch.bool)

        for node in sampled_nodes:
            idx = node_id_to_index[node]
            masked_cross_cloudlet_nodes[idx] = True

        return masked_cross_cloudlet_nodes

    def remove_edges(self, masked_cross_cloudlet_edges):
        if masked_cross_cloudlet_edges.numel() == 0:
            return self.edge_index  # Nothing to remove

        cloudlet_edge_mask = torch.ones(self.edge_index.shape[1], dtype=torch.bool)
        cloudlet_edge_mask[self.cross_cloudlet_edge_mask] = ~masked_cross_cloudlet_edges # Select edges which are NOT REMOVED!!!
        masked_cln_edge_index = self.edge_index[:,cloudlet_edge_mask]
        
        return masked_cln_edge_index
    
    def remove_edges_distribution(self, masked_cross_cloudlet_edges):
        """
        Removes edges from self.edge_index based on a boolean mask over original_cross_cloudlet_edge_index.
        """
        if masked_cross_cloudlet_edges.numel() == 0:
            return self.edge_index  # Nothing to remove

        # Step 1: Get (source, target) pairs of original masked cross-cloudlet edges
        original_edges = self.original_cross_cloudlet_edge_index[:, masked_cross_cloudlet_edges]
        original_edge_tuples = set(
            (original_edges[0, i].item(), original_edges[1, i].item())
            for i in range(original_edges.shape[1])
        )

        # Step 2: Create new mask over self.edge_index by excluding those tuples
        num_edges = self.edge_index.shape[1]
        keep_mask = torch.ones(num_edges, dtype=torch.bool)

        for i in range(num_edges):
            edge = (self.edge_index[0, i].item(), self.edge_index[1, i].item())
            if edge in original_edge_tuples:
                keep_mask[i] = False  # Remove this edge

        masked_cln_edge_index = self.edge_index[:, keep_mask]
        return masked_cln_edge_index
    
    def remove_nodes_distribution(self, masked_cross_cloudlet_nodes):
        """
        Removes all edges from self.edge_index that are connected to the cross-cloudlet nodes 
        selected for masking (via masked_cross_cloudlet_nodes).
        Returns the updated edge_index (masked_cln_edge_index).
        """

        if masked_cross_cloudlet_nodes.numel() == 0 or masked_cross_cloudlet_nodes.sum() == 0:
            return self.edge_index  # Nothing to remove
        
        # Step 1: Get actual node IDs to be removed
        nodes_to_remove = self.original_cross_cloudlet_nodes[masked_cross_cloudlet_nodes]
        nodes_to_remove_set = set(nodes_to_remove.tolist())

        # Step 2: Remove all edges where src or dst is in the nodes_to_remove set
        src, dst = self.edge_index
        keep_mask = ~(
            torch.isin(src, nodes_to_remove) |
            torch.isin(dst, nodes_to_remove)
        )

        masked_cln_edge_index = self.edge_index[:, keep_mask]
        return masked_cln_edge_index
    
    def create_val_iter_from_randomly_removed_cross_cloudlet_nodes_for_node_score_calculation(self, masked_cln_edge_index, stblock_num, Ks, num_nodes, device, n_his, n_pred, batch_size, epoch, num_epochs):
        cln_nodes_subgraph, cln_edge_index, cln_node_map, _ = k_hop_subgraph(
            self.local_cln_nodes,
            stblock_num * (Ks - 1),
            masked_cln_edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes
        )

        cln_nodes_subgraph_cpu = cln_nodes_subgraph.cpu().numpy()

        if (epoch < num_epochs):
            masked_val_dataset = self.train_dataset[:, cln_nodes_subgraph_cpu]
            masked_x_val, masked_y_val = dataloader.data_transform(masked_val_dataset, n_his, n_pred, device)

            masked_x_val = masked_x_val[
                self.end_of_initial_data_index + (self.data_per_step * (epoch)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch + 1))
                ]
            masked_y_val = masked_y_val[
                self.end_of_initial_data_index + (self.data_per_step * (epoch)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch + 1))
                ]
            masked_val_data = utils.data.TensorDataset(masked_x_val, masked_y_val)
            masked_val_iter = utils.data.DataLoader(dataset=masked_val_data, batch_size=batch_size, shuffle=False)

            return masked_val_iter, cln_edge_index, cln_node_map
        else:
            masked_val_dataset = self.val_dataset[:, cln_nodes_subgraph_cpu]
            masked_x_val, masked_y_val = dataloader.data_transform(masked_val_dataset, n_his, n_pred, device)
            masked_val_data = utils.data.TensorDataset(masked_x_val, masked_y_val)
            masked_val_iter = utils.data.DataLoader(dataset=masked_val_data, batch_size=batch_size, shuffle=False)

            return masked_val_iter, cln_edge_index, cln_node_map
    
    def delta_error_between_original_and_masked_for_online_training(self, epoch, zscore, num_epochs, masked_val_iter, masked_edge_index, masked_node_map):
        if (epoch < num_epochs):
            new_x_val = self.x_train[
                self.end_of_initial_data_index + (self.data_per_step * (epoch)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch + 1))
                ]
            new_y_val = self.y_train[
                self.end_of_initial_data_index + (self.data_per_step * (epoch)):
                self.end_of_initial_data_index + (self.data_per_step * (epoch + 1))
                ]

            new_val_data = utils.data.TensorDataset(new_x_val, new_y_val)
            new_val_iter = utils.data.DataLoader(dataset=new_val_data, batch_size=self.batch_size, shuffle=False)

            self.model.eval()
            SUDDEN_EVENT_RATE_original = utility.evaluate_cloudlet_pyg_new_metric_for_node_score(
                self.model, new_val_iter, zscore, self.edge_index, self.node_map
            )
            SUDDEN_EVENT_RATE_masked = utility.evaluate_cloudlet_pyg_new_metric_for_node_score(
                self.model, masked_val_iter, zscore, masked_edge_index, masked_node_map
            )

            d_err = SUDDEN_EVENT_RATE_masked - SUDDEN_EVENT_RATE_original
        else:
            self.model.eval()
            SUDDEN_EVENT_RATE_original = utility.evaluate_cloudlet_pyg_new_metric_for_node_score(
                self.model, self.val_iter, zscore, self.edge_index, self.node_map
            )
            SUDDEN_EVENT_RATE_masked = utility.evaluate_cloudlet_pyg_new_metric_for_node_score(
                self.model, masked_val_iter, zscore, masked_edge_index, masked_node_map
            )

            d_err = SUDDEN_EVENT_RATE_masked - SUDDEN_EVENT_RATE_original

        return d_err