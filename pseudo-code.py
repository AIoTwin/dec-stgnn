import torch

master_model = Model()
num_clouds = 10

local_models = [copy(master_model) for _ in range(num_clouds)] # copy the master model to 10 local models
# separate datasets into 10 regions for 10 local models

for _ in range(epochs): # number if training times
    for (local_model, local_data) in zip(local_models, local_datasets): # go thorugh each model one at a time
        train_one_epoch_locally(local_model, local_data) # train that 1 specific model for that specific local dataset, FIX: we need info about L-hop neighbours (1 node can be in CLI1, while other node an be in CLI2)
    master_model = average_models(local_models) # average all local models into 1 master mode l
    local_models = [copy(master_model) for _ in range(num_clouds)] # copy that new master model into 10 local models (so we get updated model)

