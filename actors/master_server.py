import torch
import torch.utils as utils
from actors import network_actor as na
from messages import messages
from script import utility

class MasterServer(na.NetworkActor):
    def __init__(self, model, cln_models, loss, edge_index, test_iter, val_iter, writer, logs_folder, x_train = [], y_train = [], end_of_initial_data_index = 0, data_per_step = 0, batch_size = 32):
        self.model = model
        self.cln_models = cln_models
        self.loss = loss
        self.edge_index = edge_index
        self.test_iter = test_iter
        self.val_iter = val_iter
        self.writer = writer
        self.logs_folder = logs_folder
        self.x_train = x_train
        self.y_train = y_train
        self.end_of_initial_data_index = end_of_initial_data_index
        self.data_per_step = data_per_step
        self.batch_size = batch_size

        self.best_model = model

    def on_receive(self, message):
        if isinstance(message, messages.CloudletModelMessage):
            for new_cln_model_param, old_cln_model_param in zip(self.cln_models[message.cln_id].parameters(), message.cln_model.parameters()):
                new_cln_model_param.data.copy_(old_cln_model_param.data)
            # self.cln_models[message.cln_id] = message.cln_model
        else:
            assert False, 'unrecognized message type'
    
    @torch.no_grad() 
    def average_model(self):
        # Initialize master model's parameters
        for master_model_param, cln_model_param in zip(self.model.parameters(), self.cln_models[0].parameters()):
            master_model_param.copy_(cln_model_param.data.clone())
        # Accumulate parameters from all cloudlet models
        for cln_model in self.cln_models[1:]:
            for master_model_param, cln_model_param in zip(self.model.parameters(), cln_model.parameters()):
                master_model_param.data.add_(cln_model_param.data)
        # Average accumulated parameters
        for master_model_param in self.model.parameters():
            master_model_param.data.div_(len(self.cln_models))

    def send_model_to_cloudlets(self, cln_actors):
        # Copy and send the new master model to all cloudlet_models
        for cln_actor in cln_actors:
            self.send_to(cln_actor, messages.MasterModelMessage(self.model))

    @torch.no_grad() 
    def test_3d_data(self, zscore, args):
        self.model.eval()
        test_MSE = utility.evaluate_3d_pyg_model_master(self.model, self.loss, self.test_iter, self.edge_index)
        for node_feature_to_evaluate in args.node_features_to_evaluate:
            test_MAE, test_RMSE, test_WMAPE = utility.evaluate_3d_pyg_metric_master(self.model, self.test_iter, zscore, self.edge_index, node_feature_to_evaluate)
            utility.save_test_logs(self.logs_folder, f"master-server_node-feature-{node_feature_to_evaluate}", test_MSE, test_MAE, test_RMSE, test_WMAPE)
            print(f'Master model | Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

    @torch.no_grad() 
    def test_2d_data(self, zscore, args):
        self.model.eval()
        test_MSE = utility.evaluate_pyg_model_master(self.model, self.loss, self.test_iter, self.edge_index)
        test_MAE, test_RMSE, test_WMAPE = utility.evaluate_pyg_metric_master(self.model, self.test_iter, zscore, self.edge_index)
        utility.save_test_logs(self.logs_folder, f"master-server", test_MSE, test_MAE, test_RMSE, test_WMAPE)
        print(f'Master model | Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

    @torch.no_grad() 
    def test_2d_data_using_best_model(self, zscore, args, best_epoch):
        self.best_model.eval()
        test_MSE = utility.evaluate_pyg_model_master(self.best_model, self.loss, self.test_iter, self.edge_index)
        test_MAE, test_RMSE, test_WMAPE = utility.evaluate_pyg_metric_master(self.best_model, self.test_iter, zscore, self.edge_index)
        utility.save_test_logs(self.logs_folder, f"master-server", test_MSE, test_MAE, test_RMSE, test_WMAPE, best_epoch+1)
        print(f'Master model | Epoch {best_epoch+1:03d} | Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

    # disable gradient calculation within a block of code
    # i.e. don't calculate gradients during validation or testing process
    @torch.no_grad()
    def val(self):
        self.model.eval()
        l_sum, n = 0.0, 0
        for x, y in self.val_iter:
            if len(y.shape) == 3:
                y_pred = utility.permute_4d_y_pred_to_3d(self.model, x, self.edge_index)
            elif len(y.shape) == 2:
                y_pred = self.model(x, self.edge_index).view(len(x), -1)
            else:
                assert False, f'Cannot work with data that has {len(y.shape)} dimension'
            l = self.loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return torch.tensor(l_sum / n)
    
    # disable gradient calculation within a block of code
    # i.e. don't calculate gradients during validation or testing process
    @torch.no_grad()
    def val_online_training(self, val_iter):
        self.model.eval()
        l_sum, n = 0.0, 0
        for x, y in val_iter:
            if len(y.shape) == 3:
                y_pred = utility.permute_4d_y_pred_to_3d(self.model, x, self.edge_index)
            elif len(y.shape) == 2:
                y_pred = self.model(x, self.edge_index).view(len(x), -1)
            else:
                assert False, f'Cannot work with data that has {len(y.shape)} dimension'
            l = self.loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return torch.tensor(l_sum / n)

    def store_new_best_model(self):
        for cln_model_param_1, cln_model_param_2 in zip(self.best_model.parameters(), self.model.parameters()):
            cln_model_param_1.data.copy_(cln_model_param_2.data)

    def log_val(self, val_loss, epoch):
        utility.save_val_logs(self.logs_folder, "master_server", epoch+1, "-", "-", val_loss.item(), "-")

    def log_val_metrics(self, epoch, zscore):
        self.model.eval()
        val_MAE, val_RMSE, val_WMAPE = utility.evaluate_pyg_metric_master(self.model, self.val_iter, zscore, self.edge_index)
        utility.save_val_metric_logs(self.logs_folder, "master_server", epoch+1, val_MAE, val_RMSE, val_WMAPE)
        print(f'Master | Epoch: {epoch+1:03d} | MAE {val_MAE:.6f} | RMSE {val_RMSE:.6f} | WMAPE {val_WMAPE:.8f}')

        return val_WMAPE

    def log_val_metrics_online(self, epoch, zscore, val_iter):
        self.model.eval()
        val_MAE, val_RMSE, val_WMAPE = utility.evaluate_pyg_metric_master(self.model, val_iter, zscore, self.edge_index)
        utility.save_val_metric_logs(self.logs_folder, "master_server", epoch+1, val_MAE, val_RMSE, val_WMAPE)
        print(f'Master | Epoch: {epoch:03d} | MAE {val_MAE:.6f} | RMSE {val_RMSE:.6f} | WMAPE {val_WMAPE:.8f}')

        return val_WMAPE

    def log_var_info(self, epoch):
        param_variances = utility.compute_parameter_variance_master(self.model.named_parameters())
        utility.save_variance_logs(self.logs_folder, "master_sever", epoch+1, param_variances)

    def log_validation_metric_for_online_training(self, epoch, zscore):
        if (epoch < 30):
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

            self.log_val_metrics_online(epoch, zscore, new_val_iter)
        else:
            self.log_val_metrics(epoch-1, zscore)

    def calculate_val_online(self, epoch):
        if epoch < 30:
            # Create validation dataset
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

            # Calculate validation loss
            val_loss = self.val_online_training(new_val_iter)

            return val_loss
        elif epoch == 30:
            val_loss = self.val()

            return val_loss