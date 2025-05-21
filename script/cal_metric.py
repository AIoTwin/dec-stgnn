import pandas as pd
import sys
import os

dataset = "metr-la"
cloudlets_test_files = "2024-11-03_15-20-45_metr-la_60min_ray-semi-dec-gossip-learning-distance/test"

if dataset == "pems-bay":
    # Cloudlet test dataset sizes
    cloudlet_sizes = {
        0: 289229,
        1: 211059,
        2: 257961,
        3: 461203,
        4: 586275,
        5: 398667,
        6: 336131,
    }
elif dataset == "metr-la":
    cloudlet_sizes = {
        0: 118220,
        1: 169620,
        2: 133640,
        3: 143920,
        4: 195320,
        5: 174760,
        6: 128500,
    }
else:
    print(f"Dataset doesn't exist: {dataset}!")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

path = os.path.join(base_dir, 'logs', cloudlets_test_files)

print(f"Processing: {path}")

# Check if the file exists
if os.path.isfile(path):
    csv_list = [path]
elif os.path.isdir(path):
    csv_list = [os.path.join(path, file) for file in os.listdir(path)
                if file.endswith('.csv') and not 'master-server' in file]
else:
    print(f"Error: The path {path} does not exist.")
    sys.exit(1)

# Sort csv files in ascending order
csv_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

# Check if there are CSV files found
if not csv_list:
    print(f"Error: No CSV files found in {path}.")
    sys.exit(1)

# Calculate weighted average
total_weight_mae = 0
weighted_sum_mae = 0

total_weight_rmse = 0
weighted_sum_rmse = 0

total_weight_wmape = 0
weighted_sum_wmape = 0

for csv_file in csv_list:
    # Extract cloudlet ID from the filename (assuming cloudlet ID is in the filename as 'cloudlet_X.csv')
    cloudlet_id = int(os.path.basename(csv_file).split('-')[1].split('.')[0])
    
    if cloudlet_id not in cloudlet_sizes:
        print(f"Warning: Cloudlet ID {cloudlet_id} not found in sizes dictionary. Skipping file {csv_file}.")
        continue

    # Read the CSV file
    try:
        data = pd.read_csv(csv_file)
        
        # Get the last row's metric value
        mae_value = data["MAE"].iloc[-1]
        rmse_value = data["RMSE"].iloc[-1]
        wmape_value = data["WMAPE"].iloc[-1]
        
        # Compute sum of values multiplied by cloudlet size
        weighted_sum_mae += mae_value * cloudlet_sizes[cloudlet_id]
        total_weight_mae += cloudlet_sizes[cloudlet_id]

        weighted_sum_rmse += rmse_value * cloudlet_sizes[cloudlet_id]
        total_weight_rmse += cloudlet_sizes[cloudlet_id]

        weighted_sum_wmape += wmape_value * cloudlet_sizes[cloudlet_id]
        total_weight_wmape += cloudlet_sizes[cloudlet_id]
    except Exception as e:
        print(f"Error processing file {csv_file}: {e}")
        continue

# Compute final weighted average
if total_weight_mae > 0:
    weighted_avg_mae = weighted_sum_mae / total_weight_mae
    print(f"Weighted Average MAE: {weighted_avg_mae}")
if total_weight_rmse > 0:
    weighted_avg_rmse = weighted_sum_rmse / total_weight_rmse
    print(f"Weighted Average RMSE: {weighted_avg_rmse}")
if total_weight_wmape > 0:
    weighted_avg_wmape = weighted_sum_wmape / total_weight_wmape
    print(f"Weighted Average WMAPE: {weighted_avg_wmape * 100}")
else:
    print("Error: Total weight is zero. Cannot compute weighted average.")