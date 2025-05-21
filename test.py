import numpy as np
import scipy.sparse as sp
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.utils as utils
import tqdm
from script import utility, dataloader
from torch_geometric.utils import from_scipy_sparse_matrix, k_hop_subgraph
from sklearn import preprocessing
import networkx as nx
import matplotlib.pyplot as plt
import csv
from scipy.sparse.linalg import eigs
from collections import deque

dataset_name = 'pemsd4'
cln_num = 10
stblock_num = 2
time_sequence = 34271
val_and_test_rate = 0.15

def get_adjacency_matrix(num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    edges_df = os.path.join(dataset_path, 'edges.csv')
    print(f"edges_df: {edges_df}")
    with open(edges_df, 'r') as f:
        print(f"I opened it")
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    print(f"edges: {edges}")
    print(f"edge count: {len(edges)}")
    for i, j in edges:
        A[i, j] = 1

    return A

def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])

# Function to load edge file into a networkx graph
def load_graph_from_csv():
    # load dataset from specific file path
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    edges_df = pd.read_csv(os.path.join(dataset_path, 'edges.csv'))
    
    # Create a directed graph from the DataFrame
    G = nx.from_pandas_edgelist(edges_df, source='from', target='to', edge_attr='cost', create_using=nx.DiGraph())
    
    return G

# Function to plot degree distribution of nodes in a graph
def plot_degree_distribution(G):
    degrees = dict(G.degree())  # Get degrees of all nodes
    degree_values = list(degrees.values())
    print(f"degree_values: {degree_values}")
    
    # Plot degree distribution
    # plt.hist(degree_values, bins=30, alpha=0.75)
    # plt.title("Degree Distribution of Nodes")
    # plt.xlabel("Degree")
    # plt.ylabel("Frequency")
    # plt.show()

def find_nodes_with_edges_to_node(graph, target_node):
    nodes_with_edges_to_target = []
    
    for node in graph.nodes():
        if node != target_node:  # Skip the target node itself
            if graph.has_edge(node, target_node):
                nodes_with_edges_to_target.append(node)
    
    return nodes_with_edges_to_target


def find_all_isolated_nodes(graph):
    isolated_nodes = []
    
    for node in graph.nodes():
        if not any(edge[1] == node for edge in graph.in_edges(node)):  # Check if node has no incoming edges
            isolated_nodes.append(node)
    
    return isolated_nodes

if __name__ == "__main__":
    THIS IS TAKEN FROM ASTGCN
    adj_matrix = get_adjacency_matrix(307)
    print(f"adj_matrix: {adj_matrix}")

    L_tilde = scaled_Laplacian(adj_matrix)
    print(f"L_tilde: {L_tilde}")

    Load edge file into a networkx graph
    graph = load_graph_from_csv()
    
    # Example: Print basic information about the graph
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")

    # Plot degree distribution of nodes
    plot_degree_distribution(graph)

    neighbors_of_node = list(graph.neighbors(44))
    print(f"neighbors_of_node 44: {neighbors_of_node}")

    neighbors_of_node = list(graph.neighbors(7))
    print(f"neighbors_of_node 7: {neighbors_of_node}")
    
    nodes_with_edges_to_target = find_nodes_with_edges_to_node(graph, 44)
    print(f"nodes_with_edges_to_target 44: {nodes_with_edges_to_target}")

    nodes_with_edges_to_target = find_nodes_with_edges_to_node(graph, 7)
    print(f"nodes_with_edges_to_target 7: {nodes_with_edges_to_target}")

    all_isolated_nodes = find_all_isolated_nodes(graph) # find nodes with no incoming edges
    print(f"all_isolated_nodes: {all_isolated_nodes}")
