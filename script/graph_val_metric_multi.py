import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Specify the maximum epoch to consider
max_epoch = -1

dataset = "metr-la"
pred_min = "pred-60min"
metric_type = "RMSE"
# plot_label = "mile$^2$/h$^2$" # MAE - mile/h, RMSE - mile$^2$/h$^2$, WMAPE - %

if metric_type == "WMAPE":
    plot_label = "%"
elif metric_type == "MAE":
    plot_label = "mile/h"
elif metric_type == "RMSE":
    plot_label = "mile$^2$/h$^2$"
else:
    print(f"Error finding metric_type {metric_type}")
    sys.exit(1)

# Change variables (specifically csv_files) to output different files
output_directory = "plots"
output_file_name = f"plot_val_metric_{metric_type}_{pred_min}_{dataset}.png"

# metr-la & 15min pred
# csv_files = [ "2024-11-01_15-16-08_metr-la_pred-15min_his-60min_centralized/val_metric/central.csv",
#             "2024-11-01_15-48-29_metr-la_pred-15min_his-60min_semi-dec-fl-distance/val_metric/master_server.csv",
#             "2024-11-01_15-48-42_metr-la_15min_ray-semi-dec-distance/val_metric/",
#             "2024-11-01_15-48-46_metr-la_15min_ray-semi-dec-gossip-learning-distance/val_metric/"]

# metr-la & 30min pred
# csv_files = [ "2024-11-03_10-47-36_metr-la_pred-30min_his-60min_centralized/val_metric/central.csv",
#             "2024-11-03_10-47-40_metr-la_pred-30min_his-60min_semi-dec-fl-distance/val_metric/master_server.csv",
#             "2024-11-03_10-48-00_metr-la_30min_ray-semi-dec-distance/val_metric/",
#             "2024-11-03_10-48-12_metr-la_30min_ray-semi-dec-gossip-learning-distance/val_metric/"]

# metr-la & 60min pred
csv_files = [ "2024-11-03_15-20-14_metr-la_pred-60min_his-60min_centralized/val_metric/central.csv",
            "2024-11-03_15-20-30_metr-la_pred-60min_his-60min_semi-dec-fl-distance/val_metric/master_server.csv",
            "2024-11-03_15-20-33_metr-la_60min_ray-semi-dec-distance/val_metric/",
            "2024-11-03_15-20-45_metr-la_60min_ray-semi-dec-gossip-learning-distance/val_metric/" ]
csv_files = [ "2024-11-01_15-16-08_metr-la_pred-15min_his-60min_centralized/val_metric/central.csv",
            "2025-01-31_10-16-36_metr-la_pred-15min_his-60min_centralized/val_metric/central.csv" ]

# pems-bay & 15min pred
# csv_files = [ "2024-10-31_17-48-05_pems-bay_pred-15min_his-60min_centralized/val_metric/central.csv",
#             "2024-10-30_15-08-54_pems-bay_pred-15min_his-60min_semi-dec-fl-distance/val_metric/master_server.csv",
#             "2024-10-30_15-08-58_pems-bay_15min_ray-semi-dec-distance/val_metric/",
#             "2024-10-30_15-08-57_pems-bay_15min_ray-semi-dec-gossip-learning-distance/val_metric/"]
# csv_files = [ "2024-10-31_17-48-05_pems-bay_pred-15min_his-60min_centralized/val_metric/central.csv",
#             "2024-10-31_17-50-07_pems-bay_pred-15min_his-60min_centralized/val_metric/central.csv",
#             "2024-11-01_10-40-34_pems-bay_pred-15min_his-60min_centralized/val_metric/central.csv"]

# pems-bay & 30min pred
# csv_files = [ "2024-10-29_08-56-56_pems-bay_pred-30min_his-60min_centralized/val_metric/central.csv",
#             "2024-10-29_08-57-00_pems-bay_pred-30min_his-60min_semi-dec-fl-distance/val_metric/master_server.csv",
#             "2024-10-29_08-57-11_pems-bay_30min_ray-semi-dec-distance/val_metric/",
#             "2024-10-29_08-57-25_pems-bay_30min_ray-semi-dec-gossip-learning-distance/val_metric/"]

# pems-bay & 60min pred
# csv_files = [ "2024-10-30_14-20-21_pems-bay_pred-60min_his-60min_centralized/val_metric/central.csv",
#             "2024-10-29_15-54-29_pems-bay_pred-60min_his-60min_semi-dec-fl-distance/val_metric/master_server.csv",
#             "2024-10-29_15-54-35_pems-bay_60min_ray-semi-dec-distance/val_metric/",
#             "2024-10-29_15-54-35_pems-bay_60min_ray-semi-dec-gossip-learning-distance/val_metric/" ]

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
# labels = ['Old hyperparameters', 'New hyperparameters', 'Newer hyperparameters']

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

        mae_avg_df = combined_df.groupby('Epoch', as_index=False)['MAE'].mean()
        mae_std_df = combined_df.groupby('Epoch', as_index=False)['MAE'].std()

        rmse_avg_df = combined_df.groupby('Epoch', as_index=False)['RMSE'].mean()
        rmse_std_df = combined_df.groupby('Epoch', as_index=False)['RMSE'].std()

        wmape_avg_df = combined_df.groupby('Epoch', as_index=False)['WMAPE'].mean() * 100
        wmape_std_df = combined_df.groupby('Epoch', as_index=False)['WMAPE'].std() * 100

        print(f"Averaged MAE Dataframe:\n{mae_avg_df.head()}")
        print(f"Averaged RMSE Dataframe:\n{rmse_avg_df.head()}")
        print(f"Averaged WMAPE Dataframe:\n{wmape_avg_df.head()}")

        metric_label = labels[i] if i < len(labels) else f'Metric {metric_type} {i+1}'

        # Plot any type of val metric
        if metric_type == "MAE":
            plt.plot(mae_avg_df['Epoch'], mae_avg_df['MAE'], label=metric_label, color=colors[i])
            # plt.fill_between(mae_avg_df['Epoch'], 
            #                     mae_avg_df['MAE'] - mae_std_df['MAE'], 
            #                     mae_avg_df['MAE'] + mae_std_df['MAE'], 
            #                     color=mae_color, alpha=0.3)

        elif metric_type == "RMSE":
            plt.plot(rmse_avg_df['Epoch'], rmse_avg_df['RMSE'], label=metric_label, color=colors[i])
            # plt.fill_between(rmse_avg_df['Epoch'], 
            #                     rmse_avg_df['RMSE'] - rmse_std_df['RMSE'], 
            #                     rmse_avg_df['RMSE'] + rmse_std_df['RMSE'], 
            #                     color=rmse_color, alpha=0.3)

        elif metric_type == "WMAPE":
            plt.plot(wmape_avg_df['Epoch'] / 100, wmape_avg_df['WMAPE'], label=metric_label, color=colors[i])
            # plt.fill_between(wmape_avg_df['Epoch'] / 100, 
            #                     wmape_avg_df['WMAPE'] - wmape_std_df['WMAPE'], 
            #                     wmape_avg_df['WMAPE'] + wmape_std_df['WMAPE'], 
            #                     color=colors[i], alpha=0.3)
        else:
            print(f"Error finding metric_type {metric_type}")
            sys.exit(1)

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel(plot_label)
plt.title(f'{metric_type}')
plt.legend()

# Save the plot
full_output_file = os.path.join(full_output_directory, output_file_name)

print(f"Plot saved to: {full_output_file}")
plt.savefig(full_output_file)
plt.close()