# Example of running this script:
# python graph_val_train_single.py

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

output_directory = "plots"
output_file_name = "plot_val_wmape.png"
#csv_file = "2024-09-06_11-33-42_metr-la_ray-semi-dec/val_metric/"
csv_file = "2024-09-10_09-01-00_metr-la_centralized/val_metric/"

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

full_output_directory = os.path.join(base_dir, 'logs', output_directory)
print(f"full_output_directory: {full_output_directory}")

# Create the output directory if it doesn't exist
if not os.path.exists(full_output_directory):
    os.makedirs(full_output_directory)

mae_color = "blue"
rmse_color = "orange"
wmape_color = "red"

# Plotting the data
plt.figure(figsize=(10, 6))

mae_label = "MAE"
rmse_label = "RMSE"
wmape_label = "WMAPE"

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
    mae_avg_df = combined_df.groupby('Epoch', as_index=False)['MAE'].mean()
    mae_std_df = combined_df.groupby('Epoch', as_index=False)['MAE'].std()

    rmse_avg_df = combined_df.groupby('Epoch', as_index=False)['RMSE'].mean()
    rmse_std_df = combined_df.groupby('Epoch', as_index=False)['RMSE'].std()

    wmape_avg_df = combined_df.groupby('Epoch', as_index=False)['WMAPE'].mean() * 100
    wmape_std_df = combined_df.groupby('Epoch', as_index=False)['WMAPE'].std() * 100

    print(f"Averaged MAE Dataframe:\n{mae_avg_df.head()}")
    print(f"Averaged RMSE Dataframe:\n{rmse_avg_df.head()}")
    print(f"Averaged WMAPE Dataframe:\n{wmape_avg_df.head()}")

    # Plot any type of val metric
    # plt.plot(mae_avg_df['Epoch'], mae_avg_df['MAE'], label=mae_label, color=mae_color)
    # plt.fill_between(mae_avg_df['Epoch'], 
    #                     mae_avg_df['MAE'] - mae_std_df['MAE'], 
    #                     mae_avg_df['MAE'] + mae_std_df['MAE'], 
    #                     color=mae_color, alpha=0.3)
    
    # plt.plot(rmse_avg_df['Epoch'], rmse_avg_df['RMSE'], label=rmse_label, color=rmse_color)
    # plt.fill_between(rmse_avg_df['Epoch'], 
    #                     rmse_avg_df['RMSE'] - rmse_std_df['RMSE'], 
    #                     rmse_avg_df['RMSE'] + rmse_std_df['RMSE'], 
    #                     color=rmse_color, alpha=0.3)
    
    plt.plot(wmape_avg_df['Epoch'] / 100, wmape_avg_df['WMAPE'], label=wmape_label, color=wmape_color)
    plt.fill_between(wmape_avg_df['Epoch'] / 100, 
                        wmape_avg_df['WMAPE'] - wmape_std_df['WMAPE'], 
                        wmape_avg_df['WMAPE'] + wmape_std_df['WMAPE'], 
                        color=wmape_color, alpha=0.3)

# Adding labels and title
plt.xlabel('Epoch')
# plt.ylabel('mile/h') # MAE
# plt.ylabel('mile^2/h^2') # RMSE
plt.ylabel('%') # WMAPE
plt.title('Metric')
plt.legend()

# Save the plot
full_output_file = os.path.join(full_output_directory, output_file_name)

print(f"Plot saved to: {full_output_file}")
plt.savefig(full_output_file)
plt.close()