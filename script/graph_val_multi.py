# Example of running this script:
# python graph_val_multi.py

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Specify the maximum epoch to consider
max_epoch = -1

# Change variables (specifically csv_files) to output different files
output_directory = "plots"
output_file_name = "plot_val_loss_joint.png"
# csv_files = [ "2024-07-08_11-03-10_metr-la_centralized/val/central.csv", "2024-08-23_10-28-42_metr-la_semi-dec-fl/val/", "2024-08-23_09-05-46_metr-la_ray-semi-dec/val/" ]
# csv_files = [ "2024-09-02_08-43-37_metr-la_ray-semi-dec-distance/val/", "2024-08-30_11-18-10_metr-la_ray-semi-dec-distance/val/" ]
csv_files = [ "2024-08-23_09-05-46_metr-la_ray-semi-dec/val/", "2024-09-04_13-11-19_metr-la_ray-semi-dec-gossip-learning/val/" ]

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

full_output_directory = os.path.join(base_dir, 'logs', output_directory)
print(f"full_output_directory: {full_output_directory}")

# Create the output directory if it doesn't exist
if not os.path.exists(full_output_directory):
    os.makedirs(full_output_directory)

# colors = ['blue', 'orange', 'green']
colors = ['blue', 'orange']

# Plotting the data
plt.figure(figsize=(10, 6))

# labels = ['Centralized', 'FL', 'Server-free FL']
labels = ['Server-free FL', 'Gossip learning']

for i, relative_path in enumerate(csv_files):
    path = os.path.join(base_dir, 'logs', relative_path)

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

        if max_epoch >= 1:
            # Filter the DataFrame to include only epochs up to the specified max_epoch
            combined_df = combined_df[combined_df['Epoch'] <= max_epoch]

        avg_df = combined_df.groupby('Epoch', as_index=False)['Val Loss'].mean()
        std_df = combined_df.groupby('Epoch', as_index=False)['Val Loss'].std()

        print(f"Averaged Dataframe:\n{avg_df.head()}")

        train_label = labels[i] if i < len(labels) else f'Train Loss {i+1}'

        # Plot the averaged Val Loss
        plt.plot(avg_df['Epoch'], avg_df['Val Loss'], label=train_label, color=colors[i])
        plt.fill_between(avg_df['Epoch'], 
                         avg_df['Val Loss'] - std_df['Val Loss'], 
                         avg_df['Val Loss'] + std_df['Val Loss'], 
                         color=colors[i], alpha=0.3)

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Over Epochs')
plt.legend()

# Save the plot
full_output_file = os.path.join(full_output_directory, output_file_name)

print(f"Plot saved to: {full_output_file}")
plt.savefig(full_output_file)
plt.close()