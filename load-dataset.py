import os
import pandas as pd
import scipy.sparse as sp

# a csv file where 1st row is temporal, and 2nd row is each node
# so if we extract data from 1st row, we exact temporal data for all nodes
# this means that each node has only 1 node feature

dataset_name = "metr-la"

dataset_path = './data'
dataset_path = os.path.join(dataset_path, dataset_name)
dataset = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

print(dataset.shape)

dataset_path = './data'
dataset_path = os.path.join(dataset_path, dataset_name)
adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))

print(adj.shape)

num_rows, num_cols = adj.shape

# Iterate over all indices
# for row in range(num_rows):
#     for col in range(num_cols):
#         value = adj[row, col]
#         if value == 0:
#             print(f"Value at ({row}, {col}): {value}")

num_zeros = 0
num_greater_than_zero_smaller_than_one = 0

for col in range(num_cols):
    value = adj[0, col]
    if value == 0:
        num_zeros+=1
    elif value > 0 and value < 1:
        num_greater_than_zero_smaller_than_one+=1
    if value >= 0.95 and value <= 1:
        print(f"Value at ({0}, {col}): {value}")

print(f"Number of 0s: {num_zeros}")
print(f"Number of greater than 0, but smaller than 1: {num_greater_than_zero_smaller_than_one}")