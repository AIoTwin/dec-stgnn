import os
import torch
import numpy as np
import pandas as pd
from tsl.datasets import MetrLA

# Utility functions ################
def print_matrix(matrix):
    return pd.DataFrame(matrix)

def print_model_size(model):
    tot = sum([p.numel() for p in model.parameters() if p.requires_grad])
    out = f"Number of model ({model.__class__.__name__}) parameters:{tot:10d}"
    print("=" * len(out))
    print(out)

dataset = MetrLA(root='./dataset')

print(dataset)

print(f"Sampling period: {dataset.freq}")
print(f"Has missing values: {dataset.has_mask}")
print(f"Percentage of missing values: {(1 - dataset.mask.mean()) * 100:.2f}%")
print(f"Has exogenous variables: {dataset.has_covariates}")
print(f"Covariates: {', '.join(dataset.covariates.keys())}") # you output covariate_name

# Also, the dataset has a covariate attribute (i.e., exogenous variables) – the distance matrix – containing the pairwise distances between sensors.
# You can access covariates by dataset.{covariate_name}:
print_matrix(dataset.dist)

dataset.dataframe()

print(f"Default similarity: {dataset.similarity_score}")
print(f"Available similarity options: {dataset.similarity_options}")
print("==========================================")

sim = dataset.get_similarity("distance")  # or dataset.compute_similarity()

print("Similarity matrix W:")
print_matrix(sim)

# Let’s see what happens with this function call:
# 1.) compute the similarity matrix as before;
# 2.) set to 0 values below 0.1 (threshold=0.1);
# 3.) remove self loops (include_self=False);
# 4.) normalize edge weights by the in degree of nodes (normalize_axis=1);
# 5.) request the sparse COO layout of PyG (layout="edge_index")
connectivity = dataset.get_connectivity(threshold=0.1,
                                        include_self=False,
                                        normalize_axis=1,
                                        layout="edge_index")

edge_index, edge_weight = connectivity

print(f'edge_index {edge_index.shape}:\n', edge_index)
print(f'edge_weight {edge_weight.shape}:\n', edge_weight)

from tsl.ops.connectivity import edge_index_to_adj

adj = edge_index_to_adj(edge_index, edge_weight)
print(f'A {adj.shape}:')
print_matrix(adj)

print(f'Sparse edge weights:\n', adj[edge_index[1], edge_index[0]])

# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

# Building a PyTorch-ready dataset

# In particular, a SpatioTemporalDataset object can be used to achieve the following:
# 1.) Perform data manipulation operations required to feed the data to a PyTorch module
# (e.g., casting data to torch.tensor, handling possibly different shapes, synchronizing temporal data).
#
# 2.) Create (input, target) samples for supervised learning following the sliding window approach.
#
# 3.) Define how data should be arranged in a spatiotemporal graph signal
# (e.g., which are the inputs and targets, how node attributes and covariates variables are mapped into a single graph).
#
# 4.) Preprocess data before creating a spatiotemporal graph signal by appling transformations or scaling operations

# Let’s see how to go from a Dataset to a SpatioTemporalDataset
from tsl.data import SpatioTemporalDataset

torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                      connectivity=connectivity,
                                      mask=dataset.mask,
                                      horizon=12,
                                      window=12,
                                      stride=1)
print(torch_dataset)
# Output before: MetrLA(length=34272, n_nodes=207, n_channels=1)
# Output now: SpatioTemporalDataset(n_samples=34249, n_nodes=207, n_channels=1)
#
# As you can see, the number of samples is not the same as the number of steps we have in the dataset.
# we divided the historic time series with a sliding window of 12 time steps for the lockback window (window=12)
# with a corresponding horizon of 12 time steps (horizon=12)
# Thus, a single sample spans for a total of 24 time steps.
# stride parameters set how many time steps intercurring between two subsequent samples

# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

# Spatiotemporal graph signals in tsl

# We can fetch samples in the same way we fetch elements of a Python list
sample = torch_dataset[0]
print(sample)
# A sample is of type tsl.data.Data, the base class for representing spatiotemporal graph signals in tsl.
# This class extends torch_geometric.data.Data, preserving all its functionalities and adding utilities for spatiotemporal data processing.

# -----------------------------------------------------------------------------------------------------------------------------------

# Input and Target

# Data.input and Data.target provide a view on the unique (shared) storage in Data,
# such that the same key in Data.input and Data.target cannot reference different objects
sample.input.to_dict()
sample.target.to_dict()

# Mask and Transform

# mask and transform are just symbolic links to the corresponding object inside the storage.
# They also expose properties has_mask and has_transform.
if sample.has_mask:
    print(sample.mask)
else:
    print("Sample has no mask.")

if sample.has_transform:
    print(sample.transform)
else:
    print("Sample has no transformation functions.")

# Pattern
# The pattern mapping can be useful to glimpse on how data are arranged. The convention we use is the following:
# 't' stands for the time steps dimension
# 'n' stands for a node dimension
# 'e' stands for the edge dimension
# 'f' stands for a feature dimension
# 'b' stands for the batch dimension
print(sample.pattern)

# Batching spatiotemporal graph signals
# Getting a batch of spatiotemporal graph signals from a single dataset is as simple as accessing multiple elements from a list:
batch = torch_dataset[:5]
print(batch)

# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

# Preparing the dataset for training

# before running an experiment there are two quite common preprocessing steps:
# 1.) splitting the dataset into training/validation/test sets;
# 2.) data preprocessing (scaling/normalizing data, detrending).

# In tsl, these operations are managed by the tsl.data.SpatioTemporalDataModule
# which is based on the LightningDataModule from PyTorch Lightning

# A DataModule allows us to standardize and make consistent training, validation, test splits...
# ...data preparation and transformations across different environments and experiments

# Example:
from tsl.data.datamodule import (SpatioTemporalDataModule,
                                 TemporalSplitter)
from tsl.data.preprocessing import StandardScaler

# Normalize data using mean and std computed over time and node dimensions
scalers = {'target': StandardScaler(axis=(0, 1))}

# Split data sequentially:
#   |------------ dataset -----------|
#   |--- train ---|- val -|-- test --|
splitter = TemporalSplitter(val_len=0.1, test_len=0.2)

dm = SpatioTemporalDataModule(
    dataset=torch_dataset,
    scalers=scalers,
    splitter=splitter,
    batch_size=64,
)

print(dm) # SpatioTemporalDataModule(train_len=None, val_len=None, test_len=None, scalers=[target], batch_size=64)

# We can execute the preprocessing routines by calling the dm.setup() method.
dm.setup()
print(dm) # SpatioTemporalDataModule(train_len=24648, val_len=2728, test_len=6849, scalers=[target], batch_size=64)

# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

# Designing a custom STGNN

import torch.nn as nn

from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding, DiffConv
from einops.layers.torch import Rearrange  # reshape data with Einstein notation

class TimeThenSpaceModel(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 32,
                 rnn_layers: int = 1,
                 gnn_kernel: int = 2):
        super(TimeThenSpaceModel, self).__init__()

        self.encoder = nn.Linear(input_size, hidden_size)

        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)

        self.time_nn = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers,
                           cell='gru',
                           return_only_last_state=True)
        
        self.space_nn = DiffConv(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 k=gnn_kernel)

        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x, edge_index, edge_weight):
        # x: [batch time nodes features]
        x_enc = self.encoder(x)  # linear encoder: x_enc = xΘ + b
        x_emb = x_enc + self.node_embeddings()  # add node-identifier embeddings
        h = self.time_nn(x_emb)  # temporal processing: x=[b t n f] -> h=[b n f]
        z = self.space_nn(h, edge_index, edge_weight)  # spatial processing
        x_out = self.decoder(z)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        x_horizon = self.rearrange(x_out)
        return x_horizon

hidden_size = 32   #@param
rnn_layers = 1     #@param
gnn_kernel = 2     #@param

input_size = torch_dataset.n_channels   # 1 channel
n_nodes = torch_dataset.n_nodes         # 207 nodes
horizon = torch_dataset.horizon         # 12 time steps

stgnn = TimeThenSpaceModel(input_size=input_size,
                           n_nodes=n_nodes,
                           horizon=horizon,
                           hidden_size=hidden_size,
                           rnn_layers=rnn_layers,
                           gnn_kernel=gnn_kernel)
print(stgnn)
print_model_size(stgnn)

# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

# Setting up  training

# The predictor

from tsl.metrics.torch import MaskedMAE, MaskedMAPE
from tsl.engines import Predictor

loss_fn = MaskedMAE()

metrics = {'mae': MaskedMAE(),
           'mape': MaskedMAPE(),
           'mae_at_15': MaskedMAE(at=2),  # '2' indicates the third time step,
                                          # which correspond to 15 minutes ahead
           'mae_at_30': MaskedMAE(at=5),
           'mae_at_60': MaskedMAE(at=11)}

# setup predictor
predictor = Predictor(
    model=stgnn,                   # our initialized model
    optim_class=torch.optim.Adam,  # specify optimizer to be used...
    optim_kwargs={'lr': 0.001},    # ...and parameters for its initialization
    loss_fn=loss_fn,               # which loss function to be used
    metrics=metrics                # metrics to be logged during train/val/test
)

from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(save_dir="logs", name="tsl_intro", version=0)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath='logs',
    save_top_k=1,
    monitor='val_mae',
    mode='min',
)

trainer = pl.Trainer(max_epochs=100,
                     logger=logger,
                     gpus=1 if torch.cuda.is_available() else None,
                     limit_train_batches=100,  # end an epoch after 100 updates
                     callbacks=[checkpoint_callback])

trainer.fit(predictor, datamodule=dm)

# Testing
predictor.load_model(checkpoint_callback.best_model_path)
predictor.freeze()

trainer.test(predictor, datamodule=dm);