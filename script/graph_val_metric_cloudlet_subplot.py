import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sharey = "none" # "row" |"none"

dataset = "metr-la_pems-bay"
metric_type = "WMAPE"

if metric_type == "WMAPE":
    plot_label = "WMAPE [%]"
    y_min, y_max = 5, 20
elif metric_type == "MAE":
    plot_label = "MAE [mile/h]"
elif metric_type == "RMSE":
    plot_label = "RMSE [mile$^2$/h$^2$]"
else:
    print(f"Error finding metric_type {metric_type}")
    sys.exit(1)

output_directory = "plots"
output_file_name = f"plot_val_metric_{metric_type}_{dataset}_cloudlet_subplot.png"

if dataset == "metr-la":
    csv_file_paths = {
        "Server-Free FL - 15min": "2024-11-01_15-48-42_metr-la_15min_ray-semi-dec-distance/val_metric/",
        "Gossip Learning - 15min": "2024-11-01_15-48-46_metr-la_15min_ray-semi-dec-gossip-learning-distance/val_metric/",

        "Server-Free FL - 30min": "2024-11-03_10-48-00_metr-la_30min_ray-semi-dec-distance/val_metric/",
        "Gossip Learning - 30min": "2024-11-03_10-48-12_metr-la_30min_ray-semi-dec-gossip-learning-distance/val_metric/",

        "Server-Free FL - 60min": "2024-11-03_15-20-33_metr-la_60min_ray-semi-dec-distance/val_metric/",
        "Gossip Learning - 60min": "2024-11-03_15-20-45_metr-la_60min_ray-semi-dec-gossip-learning-distance/val_metric/"
    }
elif dataset == "pems-bay":
    csv_file_paths = {
        "Server-Free FL - 15min": "2024-11-06_09-25-58_pems-bay_15min_ray-semi-dec-distance/val_metric/",
        "Gossip Learning - 15min": "2024-11-06_09-26-12_pems-bay_15min_ray-semi-dec-gossip-learning-distance/val_metric/",

        "Server-Free FL - 30min": "2024-11-07_10-02-12_pems-bay_30min_ray-semi-dec-distance/val_metric/",
        "Gossip Learning - 30min": "2024-11-07_10-02-12_pems-bay_30min_ray-semi-dec-gossip-learning-distance/val_metric/",

        "Server-Free FL - 60min": "2024-11-07_14-55-53_pems-bay_60min_ray-semi-dec-distance/val_metric/",
        "Gossip Learning - 60min": "2024-11-07_14-56-04_pems-bay_60min_ray-semi-dec-gossip-learning-distance/val_metric/"
    }
elif dataset == "metr-la_pems-bay":
    # original adj matrix
    csv_file_paths = {
        "Centralized - METR-LA - 15min": "2024-11-01_15-16-08_metr-la_pred-15min_his-60min_centralized/val_metric/",
        "Traditional FL - METR-LA - 15min": "2024-11-22_13-38-47_metr-la_pred-15min_his-60min_semi-dec-fl-distance/val_metric/",
        "Server-Free FL - METR-LA - 15min": "2024-11-01_15-48-42_metr-la_15min_ray-semi-dec-distance/val_metric/",
        "Gossip Learning - METR-LA - 15min": "2024-11-01_15-48-46_metr-la_15min_ray-semi-dec-gossip-learning-distance/val_metric/",

        "Centralized - METR-LA - 60min": "2024-11-03_15-20-14_metr-la_pred-60min_his-60min_centralized/val_metric/",
        "Traditional FL - METR-LA - 60min": "2024-11-23_13-59-41_metr-la_pred-60min_his-60min_semi-dec-fl-distance/val_metric/",
        "Server-Free FL - METR-LA - 60min": "2024-11-03_15-20-33_metr-la_60min_ray-semi-dec-distance/val_metric/",
        "Gossip Learning - METR-LA - 60min": "2024-11-03_15-20-45_metr-la_60min_ray-semi-dec-gossip-learning-distance/val_metric/",

        "Centralized - PeMS-BAY - 15min": "2024-11-01_10-40-34_pems-bay_pred-15min_his-60min_centralized/val_metric/",
        "Traditional FL - PeMS-BAY - 15min": "2024-11-23_14-01-43_pems-bay_pred-15min_his-60min_semi-dec-fl-distance/val_metric/",
        "Server-Free FL - PeMS-BAY - 15min": "2024-11-06_09-25-58_pems-bay_15min_ray-semi-dec-distance/val_metric/",
        "Gossip Learning - PeMS-BAY - 15min": "2024-11-06_09-26-12_pems-bay_15min_ray-semi-dec-gossip-learning-distance/val_metric/",

        "Centralized - PeMS-BAY - 60min": "2024-11-07_14-55-43_pems-bay_pred-60min_his-60min_centralized/val_metric/",
        "Traditional FL - PeMS-BAY - 60min": "2024-11-23_14-05-46_pems-bay_pred-60min_his-60min_semi-dec-fl-distance/val_metric/",
        "Server-Free FL - PeMS-BAY - 60min": "2024-11-07_14-55-53_pems-bay_60min_ray-semi-dec-distance/val_metric/",
        "Gossip Learning - PeMS-BAY - 60min": "2024-11-07_14-56-04_pems-bay_60min_ray-semi-dec-gossip-learning-distance/val_metric/",
    }
    # adj matrix direct 0 & 1
    # csv_file_paths = {
    #     "Centralized - METR-LA - 15min": "2025-01-10_14-53-49_metr-la_pred-15min_his-60min_centralized/val_metric/",
    #     "Traditional FL - METR-LA - 15min": "2025-01-13_10-09-32_metr-la_pred-15min_his-60min_semi-dec-fl-distance/val_metric/",
    #     "Server-Free FL - METR-LA - 15min": "2025-01-14_10-26-38_metr-la_15min_ray-semi-dec-distance/val_metric/",
    #     "Gossip Learning - METR-LA - 15min": "2025-01-15_15-45-35_metr-la_15min_ray-semi-dec-gossip-learning-distance/val_metric/",

    #     "Original Centralized - METR-LA - 15min": "2024-11-01_15-16-08_metr-la_pred-15min_his-60min_centralized/val_metric/",
    #     "Original Traditional FL - METR-LA - 15min": "2024-11-22_13-38-47_metr-la_pred-15min_his-60min_semi-dec-fl-distance/val_metric/",
    #     "Original Server-Free FL - METR-LA - 15min": "2024-11-01_15-48-42_metr-la_15min_ray-semi-dec-distance/val_metric/",
    #     "Original Gossip Learning - METR-LA - 15min": "2024-11-01_15-48-46_metr-la_15min_ray-semi-dec-gossip-learning-distance/val_metric/",

    #     "Centralized - METR-LA - 60min": "2025-01-16_10-29-18_metr-la_pred-60min_his-60min_centralized/val_metric/",
    #     "Traditional FL - METR-LA - 60min": "2025-01-16_13-47-14_metr-la_pred-60min_his-60min_semi-dec-fl-distance/val_metric/",
    #     "Server-Free FL - PeMS-BAY - 15min": "2024-11-06_09-25-58_pems-bay_15min_ray-semi-dec-distance/val_metric/",
    #     "Gossip Learning - PeMS-BAY - 15min": "2024-11-06_09-26-12_pems-bay_15min_ray-semi-dec-gossip-learning-distance/val_metric/",

    #     "Original Centralized - METR-LA - 60min": "2024-11-03_15-20-14_metr-la_pred-60min_his-60min_centralized/val_metric/",
    #     "Original Traditional FL - METR-LA - 60min": "2024-11-23_13-59-41_metr-la_pred-60min_his-60min_semi-dec-fl-distance/val_metric/",
    #     "Original Server-Free FL - METR-LA - 60min": "2024-11-03_15-20-33_metr-la_60min_ray-semi-dec-distance/val_metric/",
    #     "Original Gossip Learning - METR-LA - 60min": "2024-11-03_15-20-45_metr-la_60min_ray-semi-dec-gossip-learning-distance/val_metric/",
    # }
    # csv_file_paths = {
    #     "Original Centralized - METR-LA - 15min": "2024-11-01_15-16-08_metr-la_pred-15min_his-60min_centralized/val_metric/",
    #     "Centralized with no neighbours - METR-LA - 15min": "2025-01-31_10-16-36_metr-la_pred-15min_his-60min_centralized/val_metric/"
    # }
else:
    print(f"Dataset doesn't exist: {dataset}!")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

full_output_directory = os.path.join(base_dir, 'logs', output_directory)
print(f"full_output_directory: {full_output_directory}")

if not os.path.exists(full_output_directory):
    os.makedirs(full_output_directory)

# Create a 4x4 grid for subplots
fig, axes = plt.subplots(4, 4, figsize=(20, 20), sharey=sharey)
axes = axes.ravel()  # Flatten the 2D array of axes for easy iteration

# Loop through each file path and plot in each subplot
for idx, (title, csv_file) in enumerate(csv_file_paths.items()):
    path = os.path.join(base_dir, 'logs', csv_file)

    # Check if path is valid
    if not os.path.isdir(path):
        print(f"Error: Directory {path} does not exist.")
        continue

    # Load all CSV files in the directory
    csv_list = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]
    csv_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  # Sort by cloudlet ID

    # Plot each CSV file in its subplot
    ax = axes[idx]  # Select the subplot
    for csv_file in csv_list:
        try:
            df = pd.read_csv(csv_file)
            cloudlet_id = Path(csv_file).stem
            if cloudlet_id.isdigit():
                cloudlet_name = f"Cloudlet {int(cloudlet_id) + 1}"

                if metric_type == "WMAPE":
                    df['WMAPE'] *= 100  # Convert to percentage if WMAPE
                    ax.plot(df['Epoch'], df['WMAPE'], label=cloudlet_name, linestyle='-', marker='o')

                elif metric_type == "MAE":
                    ax.plot(df['Epoch'], df['MAE'], label=cloudlet_name, linestyle='-', marker='o')

            # Set plot labels and title
            ax.set_xlabel('Epoch')
            if idx % 4 == 0:
                ax.set_ylabel(plot_label)
            # ax.set_ylim(y_min, y_max)
            ax.set_title(title)
            # ax.legend(loc="upper right")
        except Exception as e:
            print(f"Error loading the CSV file {csv_file}: {e}")
            continue

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