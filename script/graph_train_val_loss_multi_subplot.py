import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# FLOPs per epoch for METR-LA and PeMS-BAY
flops_data = {
    "PeMS-BAY": {
        "central": 4032147724800,
        "trad_fl_aggregation": 1113096,
        "individual_cloudlets": [
            3336553972992, 4006345476864, 3969134837760,
            4031152569600, 4031152569600, 3845099374080, 4031152569600
        ],
        "server_free_aggregation": [556548] * 7,
        "gossip_aggregation": [417411] * 7
    },
    "METR-LA": {
        "central": 1688542599168,
        "trad_fl_aggregation": 1113096,
        "individual_cloudlets": [
            1663446316032, 1239430588416, 1687908761856,
            1687908761856, 1500363343872, 1687908761856, 1435130155008
        ],
        "server_free_aggregation": [556548, 417411, 556548, 695685, 417411, 556548, 417411],
        "gossip_aggregation": [417411] * 7
    }
}

sharey = "none" # "row" |"none"

dataset = "metr-la_pems-bay"
loss_type = "val"

output_directory = "plots"
output_file_name = f"plot_{loss_type}_loss_{dataset}_multi_subplot.png"

if dataset == "metr-la_pems-bay":
    csv_file_paths_1 = {
        "METR-LA - 15min": {
            "Centralized": "2024-11-01_15-16-08_metr-la_pred-15min_his-60min_centralized/val/central.csv",
            "Traditional FL": "2024-11-01_15-48-29_metr-la_pred-15min_his-60min_semi-dec-fl-distance/val/",
            "Server-free FL": "2024-11-01_15-48-42_metr-la_15min_ray-semi-dec-distance/val/",
            "Gossip Learning": "2024-11-01_15-48-46_metr-la_15min_ray-semi-dec-gossip-learning-distance/val/"
        },
        "PeMS-BAY - 15min": {
            "Centralized": "2024-11-01_10-40-34_pems-bay_pred-15min_his-60min_centralized/val/central.csv",
            "Traditional FL": "2024-11-06_14-37-13_pems-bay_pred-15min_his-60min_semi-dec-fl-distance/val/",
            "Server-free FL": "2024-11-06_09-25-58_pems-bay_15min_ray-semi-dec-distance/val/",
            "Gossip Learning": "2024-11-06_09-26-12_pems-bay_15min_ray-semi-dec-gossip-learning-distance/val/"
        }
    }

    csv_file_paths_2 = {
        "METR-LA - 15min": {
            "Centralized": "2024-11-01_15-16-08_metr-la_pred-15min_his-60min_centralized/val/central.csv",
            "Traditional FL": "2024-11-01_15-48-29_metr-la_pred-15min_his-60min_semi-dec-fl-distance/val/",
            "Server-free FL": "2024-11-01_15-48-42_metr-la_15min_ray-semi-dec-distance/val/",
            "Gossip Learning": "2024-11-01_15-48-46_metr-la_15min_ray-semi-dec-gossip-learning-distance/val/"
        },
        "PeMS-BAY - 15min": {
            "Centralized": "2024-11-01_10-40-34_pems-bay_pred-15min_his-60min_centralized/val/central.csv",
            "Traditional FL": "2024-11-06_14-37-13_pems-bay_pred-15min_his-60min_semi-dec-fl-distance/val/",
            "Server-free FL": "2024-11-06_09-25-58_pems-bay_15min_ray-semi-dec-distance/val/",
            "Gossip Learning": "2024-11-06_09-26-12_pems-bay_15min_ray-semi-dec-gossip-learning-distance/val/"
        }
    }
else:
    print(f"Dataset doesn't exist: {dataset}!")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

full_output_directory = os.path.join(base_dir, 'logs', output_directory)
print(f"full_output_directory: {full_output_directory}")

if not os.path.exists(full_output_directory):
    os.makedirs(full_output_directory)

# Helper function to calculate cumulative FLOPs
def calculate_flops(dataset_name, method):
    flops = flops_data[dataset_name]
    
    if method == "Centralized":
        flops_per_epoch = flops["central"]
    elif method == "Traditional FL":
        flops_per_epoch = sum(flops["individual_cloudlets"]) + flops["trad_fl_aggregation"]
    elif method == "Server-free FL":
        flops_per_epoch = sum(flops["individual_cloudlets"]) + sum(flops["server_free_aggregation"])
    elif method == "Gossip Learning":
        flops_per_epoch = sum(flops["individual_cloudlets"]) + sum(flops["gossip_aggregation"])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute cumulative FLOPs for 40 epochs
    cumulative_flops = [flops_per_epoch * (epoch + 1) for epoch in range(40)]
    return cumulative_flops

# Create a 2x3 grid for subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=sharey)
axes = axes.ravel()  # Flatten the 2D array of axes for easy iteration

plot_index = 0
for experiment_name, methods in csv_file_paths_1.items():
    ax = axes[plot_index]  # Select subplot axis for each experiment
    dataset_name = "METR-LA" if "METR-LA" in experiment_name else "PeMS-BAY"

    for method, path in methods.items():
        full_path = os.path.join(base_dir, 'logs', path)
        if plot_index < 2:  # First two subplots over FLOPS
            x_values = calculate_flops(dataset_name, method)
        else:  # Last two subplots over epochs
            x_values = list(range(1, 41))
        
        if full_path.endswith(".csv"):
            # Directly read and plot from the single CSV file
            data = pd.read_csv(full_path)
            y_values = data['Train Loss'] if loss_type == "train" else data['Val Loss']
        else:
            # Calculate the average train and validation loss from files in the directory
            train_losses, val_losses = [], []
            for file in os.listdir(full_path):
                if file.endswith(".csv") and file != "master_server.csv":
                    data = pd.read_csv(os.path.join(full_path, file))
                    train_losses.append(data['Train Loss'].values)
                    val_losses.append(data['Val Loss'].values)
            
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)
            
            y_values = avg_train_loss if loss_type == "train" else avg_val_loss

        ax.plot(x_values, y_values, label=f'{method}')

    # Set title and labels for each subplot
    ax.set_title(f"{experiment_name}")
    # ax.legend()
    ax.set_xlabel('FLOPs')
    if plot_index % 2 == 0:
        ax.set_ylabel('Loss')
    # ax.set_ylabel('Loss')
    plot_index += 1  # Move to the next subplot for the next experiment

for experiment_name, methods in csv_file_paths_2.items():
    ax = axes[plot_index]  # Select subplot axis for each experiment
    dataset_name = "METR-LA" if "METR-LA" in experiment_name else "PeMS-BAY"

    for method, path in methods.items():
        full_path = os.path.join(base_dir, 'logs', path)
        if plot_index < 2:  # First two subplots over FLOPS
            x_values = calculate_flops(dataset_name, method)
        else:  # Last two subplots over epochs
            x_values = list(range(1, 41))
        
        if full_path.endswith(".csv"):
            # Directly read and plot from the single CSV file
            data = pd.read_csv(full_path)
            y_values = data['Train Loss'] if loss_type == "train" else data['Val Loss']
        else:
            # Calculate the average train and validation loss from files in the directory
            train_losses, val_losses = [], []
            for file in os.listdir(full_path):
                if file.endswith(".csv") and file != "master_server.csv":
                    data = pd.read_csv(os.path.join(full_path, file))
                    train_losses.append(data['Train Loss'].values)
                    val_losses.append(data['Val Loss'].values)
            
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)
            
            y_values = avg_train_loss if loss_type == "train" else avg_val_loss

        ax.plot(x_values, y_values, label=f'{method}')

    # Set title and labels for each subplot
    ax.set_title(f"{experiment_name}")
    # ax.legend()
    ax.set_xlabel('Epochs')
    if plot_index % 2 == 0:
        ax.set_ylabel('Loss')
    # ax.set_ylabel('Loss')
    plot_index += 1  # Move to the next subplot for the next experiment

# Adjust layout and save the plot
plt.tight_layout()
# Get unique legend labels and handles
handles, labels = fig.axes[0].get_legend_handles_labels()
unique_labels = []
unique_handles = []
for h, l in zip(handles, labels):
    if l not in unique_labels:
        unique_labels.append(l)
        unique_handles.append(h)

# Adjust legend placement
# fig.legend(unique_handles, unique_labels, loc='center left', bbox_to_anchor=(1.00, 0.95))
fig.legend(unique_handles,
           unique_labels,
           loc="lower center", 
           ncol=7, 
           fontsize=14, 
           bbox_to_anchor=(0.5, -0.03))
full_output_file = os.path.join(full_output_directory, output_file_name)
print(f"Plot saved to: {full_output_file}")
plt.savefig(full_output_file, bbox_inches='tight')
plt.close()