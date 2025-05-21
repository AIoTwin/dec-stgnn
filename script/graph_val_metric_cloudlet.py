import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

dataset = "metr-la"
pred_min = "pred-60min"
metric_type = "WMAPE"
setup_type = "gossip-learning" # centralized | server-free-fl | gossip-learning

if metric_type == "WMAPE":
    plot_label = "%"
elif metric_type == "MAE":
    plot_label = "mile/h"
elif metric_type == "RMSE":
    plot_label = "mile$^2$/h$^2$"
else:
    print(f"Error finding metric_type {metric_type}")
    sys.exit(1)

output_directory = "plots"
output_file_name = f"plot_val_metric_{metric_type}_{pred_min}_{dataset}_{setup_type}_cloudlet.png"

if dataset == "metr-la" and pred_min == "pred-15min" and setup_type == "centralized":
    csv_file = "2024-11-01_15-16-08_metr-la_pred-15min_his-60min_centralized/val_metric/"

elif dataset == "metr-la" and pred_min == "pred-15min" and setup_type == "server-free-fl":
    csv_file = "2024-11-01_15-48-42_metr-la_15min_ray-semi-dec-distance/val_metric/"

elif dataset == "metr-la" and pred_min == "pred-15min" and setup_type == "gossip-learning":
    csv_file = "2024-11-01_15-48-46_metr-la_15min_ray-semi-dec-gossip-learning-distance/val_metric/"

# -----------------------------------------------------------------------------------------------------

elif dataset == "metr-la" and pred_min == "pred-30min" and setup_type == "centralized":
    csv_file = "2024-11-03_10-47-36_metr-la_pred-30min_his-60min_centralized/val_metric/"

elif dataset == "metr-la" and pred_min == "pred-30min" and setup_type == "server-free-fl":
    csv_file = "2024-11-03_10-48-00_metr-la_30min_ray-semi-dec-distance/val_metric/"

elif dataset == "metr-la" and pred_min == "pred-30min" and setup_type == "gossip-learning":
    csv_file = "2024-11-03_10-48-12_metr-la_30min_ray-semi-dec-gossip-learning-distance/val_metric/"

# -----------------------------------------------------------------------------------------------------

elif dataset == "metr-la" and pred_min == "pred-60min" and setup_type == "centralized":
    csv_file = "2024-11-03_15-20-14_metr-la_pred-60min_his-60min_centralized/val_metric/"

elif dataset == "metr-la" and pred_min == "pred-60min" and setup_type == "server-free-fl":
    csv_file = "2024-11-03_15-20-33_metr-la_60min_ray-semi-dec-distance/val_metric/"

elif dataset == "metr-la" and pred_min == "pred-60min" and setup_type == "gossip-learning":
    csv_file = "2024-11-03_15-20-45_metr-la_60min_ray-semi-dec-gossip-learning-distance/val_metric/"

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

elif dataset == "pems-bay" and pred_min == "pred-15min" and setup_type == "centralized":
    csv_file = "2024-10-30_15-08-52_pems-bay_pred-15min_his-60min_centralized/val_metric/"

elif dataset == "pems-bay" and pred_min == "pred-15min" and setup_type == "server-free-fl":
    csv_file = "2024-09-30_09-10-13_pems-bay_15min_ray-semi-dec-distance/val_metric/"

elif dataset == "pems-bay" and pred_min == "pred-15min" and setup_type == "gossip-learning":
    csv_file = "2024-10-03_09-28-25_pems-bay_15min_ray-semi-dec-gossip-learning-distance/val_metric/"

# -----------------------------------------------------------------------------------------------------

elif dataset == "pems-bay" and pred_min == "pred-30min" and setup_type == "centralized":
    csv_file = "2024-10-25_09-31-01_pems-bay_pred-30min_his-60min_centralized/val_metric/"

elif dataset == "pems-bay" and pred_min == "pred-30min" and setup_type == "server-free-fl":
    csv_file = "2024-10-09_09-21-59_pems-bay_30min_ray-semi-dec-distance/val_metric/"

elif dataset == "pems-bay" and pred_min == "pred-30min" and setup_type == "gossip-learning":
    csv_file = "2024-10-10_15-37-20_pems-bay_30min_ray-semi-dec-gossip-learning-distance/val_metric/"

# -----------------------------------------------------------------------------------------------------

elif dataset == "pems-bay" and pred_min == "pred-60min" and setup_type == "centralized":
    csv_file = "2024-10-25_09-32-19_pems-bay_pred-60min_his-60min_centralized/val_metric/"

elif dataset == "pems-bay" and pred_min == "pred-60min" and setup_type == "server-free-fl":
    csv_file = "2024-10-16_08-50-01_pems-bay_60min_ray-semi-dec-distance/val_metric/"

elif dataset == "pems-bay" and pred_min == "pred-60min" and setup_type == "gossip-learning":
    csv_file = "2024-10-17_10-45-13_pems-bay_60min_ray-semi-dec-gossip-learning-distance/val_metric/"

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

else:
    print(f"No combination exists for {dataset} | {pred_min} | {setup_type}!!!")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

full_output_directory = os.path.join(base_dir, 'logs', output_directory)
print(f"full_output_directory: {full_output_directory}")

# Create the output directory if it doesn't exist
if not os.path.exists(full_output_directory):
    os.makedirs(full_output_directory)

# Plotting parameters
wmape_color = "red"
plt.figure(figsize=(10, 6))

path = os.path.join(base_dir, 'logs', csv_file)
print(f"Processing: {path}")

# Check if the file exists
if os.path.isfile(path):
    csv_list = [path]
elif os.path.isdir(path):
    csv_list = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]
else:
    print(f"Error: The path {path} does not exist.")
    sys.exit(1)

# Check if there are CSV files found
if not csv_list:
    print(f"Error: No CSV files found in {path}.")
    sys.exit(1)

# Sort csv files in ascending order
csv_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

# Plot each CSV file (representing each cloudlet) individually
for idx, csv_file in enumerate(csv_list):
    try:
        df = pd.read_csv(csv_file)
        cloudlet_id = Path(csv_file).stem
        if cloudlet_id.isdigit():
            cloudlet_name = f"Cloudlet {int(cloudlet_id) + 1}"

            if metric_type == "MAE":
                plt.plot(df['Epoch'], df['MAE'], label=cloudlet_name, linestyle='-', marker='o')
            elif metric_type == "RMSE":
                plt.plot(df['Epoch'], df['RMSE'], label=cloudlet_name, linestyle='-', marker='o')
            elif metric_type == "WMAPE":
                # Scale WMAPE to percentage
                df['WMAPE'] *= 100
                # Plot WMAPE for each cloudlet
                plt.plot(df['Epoch'], df['WMAPE'], label=cloudlet_name, linestyle='-', marker='o')
            else:
                print(f"Error finding metric_type {metric_type}")
                sys.exit(1)
    except Exception as e:
        print(f"Error loading the CSV file {csv_file}: {e}")
        continue

# Adding labels, title, and legend
plt.xlabel('Epoch')
plt.ylabel(plot_label)  # WMAPE is represented as percentage
plt.title(f'{metric_type}')

# y_min, y_max = plt.gca().get_ylim()
# plt.gca().set_ylim(y_min, y_max + (y_max - y_min) * 0.2)
plt.legend(loc="upper right")

# Save the plot
full_output_file = os.path.join(full_output_directory, output_file_name)
print(f"Plot saved to: {full_output_file}")
plt.savefig(full_output_file)
plt.close()