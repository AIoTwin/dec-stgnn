# Example of running this script:
# python graph_val_multi.py

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Specify the maximum epoch to consider
max_epoch = -1

dataset = "metr-la"
pred_min = "pred-60min"
loss_type = "val" # train | val

# Change variables (specifically csv_files) to output different files
output_directory = "plots"
output_file_name = f"plot_{loss_type}_loss_joint_{pred_min}_{dataset}.png"

# metr-la & 15min pred
# csv_files = [ "2024-11-01_15-16-08_metr-la_pred-15min_his-60min_centralized/val/central.csv",
#             "2024-11-01_15-48-29_metr-la_pred-15min_his-60min_semi-dec-fl-distance/val/", # master server doesn't have train loss, so we average cloudlet loss
#             "2024-11-01_15-48-42_metr-la_15min_ray-semi-dec-distance/val/",
#             "2024-11-01_15-48-46_metr-la_15min_ray-semi-dec-gossip-learning-distance/val/"]

# metr-la & 30min pred
# csv_files = [ "2024-11-03_10-47-36_metr-la_pred-30min_his-60min_centralized/val/central.csv",
#             "2024-11-03_10-47-40_metr-la_pred-30min_his-60min_semi-dec-fl-distance/val/", # master server doesn't have train loss, so we average cloudlet loss
#             "2024-11-03_10-48-00_metr-la_30min_ray-semi-dec-distance/val/",
#             "2024-11-03_10-48-12_metr-la_30min_ray-semi-dec-gossip-learning-distance/val/"]

# metr-la & 60min pred
csv_files = [ "2024-11-03_15-20-14_metr-la_pred-60min_his-60min_centralized/val/central.csv",
            "2024-11-03_15-20-30_metr-la_pred-60min_his-60min_semi-dec-fl-distance/val/", # master server doesn't have train loss, so we average cloudlet loss
            "2024-11-03_15-20-33_metr-la_60min_ray-semi-dec-distance/val/",
            "2024-11-03_15-20-45_metr-la_60min_ray-semi-dec-gossip-learning-distance/val/" ]

# pems-bay & 15min pred
# csv_files = [ "2024-10-30_15-08-52_pems-bay_pred-15min_his-60min_centralized/val/central.csv",
#             "2024-10-30_15-08-54_pems-bay_pred-15min_his-60min_semi-dec-fl-distance/val/", # master server doesn't have train loss, so we average cloudlet loss
#             "2024-10-30_15-08-58_pems-bay_15min_ray-semi-dec-distance/val/",
#             "2024-10-30_15-08-57_pems-bay_15min_ray-semi-dec-gossip-learning-distance/val/"]
# csv_files = [ "2024-10-31_17-48-05_pems-bay_pred-15min_his-60min_centralized/val/central.csv",
#             "2024-10-31_17-50-07_pems-bay_pred-15min_his-60min_centralized/val/central.csv",
#             "2024-11-01_10-40-34_pems-bay_pred-15min_his-60min_centralized/val/central.csv"]

# pems-bay & 30min pred
# csv_files = [ "2024-10-11_10-32-55_pems-bay_30min_centralized/val/central.csv",
#             "2024-10-09_09-19-43_pems-bay_30min_semi-dec-fl-distance/val/", # master server doesn't have train loss, so we average cloudlet loss
#             "2024-10-09_09-21-59_pems-bay_30min_ray-semi-dec-distance/val/",
#             "2024-10-10_15-37-20_pems-bay_30min_ray-semi-dec-gossip-learning-distance/val/"]

# pems-bay & 60min pred
# csv_files = [ "2024-10-14_09-38-10_pems-bay_pred-60min_his-60_centralized/val/central.csv",
#             "2024-10-15_08-45-58_pems-bay_pred-60min_his-60min_semi-dec-fl-distance/val/", # master server doesn't have train loss, so we average cloudlet loss
#             "2024-10-16_08-50-01_pems-bay_60min_ray-semi-dec-distance/val/",
#             "2024-10-17_10-45-13_pems-bay_60min_ray-semi-dec-gossip-learning-distance/val/" ]

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

full_output_directory = os.path.join(base_dir, 'logs', output_directory)
print(f"full_output_directory: {full_output_directory}")

# Create the output directory if it doesn't exist
if not os.path.exists(full_output_directory):
    os.makedirs(full_output_directory)

colors = ['blue', 'orange', 'green', 'red']

# Plotting the data
plt.figure(figsize=(10, 6))

labels = ['Centralized', 'Traditional FL', 'Server-free FL', 'Gossip learning']

for i, relative_path in enumerate(csv_files):
    path = os.path.join(base_dir, 'logs', relative_path)

    print(f"Processing: {path}")

    # Check if the file exists
    if os.path.isfile(path):
        csv_list = [path]
    elif os.path.isdir(path):
        csv_list = [os.path.join(path, file) for file in os.listdir(path)
                    if file.endswith('.csv') and not 'master_server' in file]
    else:
        print(f"Error: The path {path} does not exist.")
        sys.exit(1)

    # Check if there are CSV files found
    if not csv_list:
        print(f"Error: No CSV files found in {path}.")
        sys.exit(1)
    
    # Load and average the CSV files
    df_list = []
    for csv_file in csv_list:
        try:
            df = pd.read_csv(csv_file)
            df_list.append(df)
        except Exception as e:
            print(f"Error loading the CSV file {csv_file}: {e}")
            sys.exit(1)

    # Concatenate all DataFrames and calculate mean and std deviation of Val Loss by epoch
    if df_list:
        combined_df = pd.concat(df_list)

        if max_epoch >= 1:
            # Filter the DataFrame to include only epochs up to the specified max_epoch
            combined_df = combined_df[combined_df['Epoch'] <= max_epoch]

        avg_df_train = combined_df.groupby('Epoch', as_index=False)['Train Loss'].mean()
        std_df_train = combined_df.groupby('Epoch', as_index=False)['Train Loss'].std()
        avg_df_val = combined_df.groupby('Epoch', as_index=False)['Val Loss'].mean()
        std_df_val = combined_df.groupby('Epoch', as_index=False)['Val Loss'].std()

        print(f"Averaged Dataframe:\n{avg_df_train.head()}")

        train_label = labels[i] if i < len(labels) else f'Train Loss {i+1}'
        val_label = labels[i] if i < len(labels) else f'Val Loss {i+1}'
        # Plot the averaged Val Loss
        if loss_type == "train":
            plt.plot(avg_df_train['Epoch'], avg_df_train['Train Loss'], label=train_label, color=colors[i])
            # plt.fill_between(avg_df['Epoch'], 
            #                  avg_df['Train Loss'] - std_df['Train Loss'], 
            #                  avg_df['Train Loss'] + std_df['Train Loss'], 
            #                  color=colors[i], alpha=0.3)
        elif loss_type == "val":
            plt.plot(avg_df_val['Epoch'], avg_df_val['Val Loss'], label=val_label, color=colors[i])
        else:
            print(f"Error finding loss_type: {loss_type}")
            sys.exit(1)

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Save the plot
full_output_file = os.path.join(full_output_directory, output_file_name)

print(f"Plot saved to: {full_output_file}")
plt.savefig(full_output_file)
plt.close()