# Example of running this script:
# python graph_communication_trainable_parameters.py

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Change variables (specifically csv_files) to output different files
output_directory = "plots"
output_file_name = "plot_communication_node_features_transfer.png"
csv_files = ["2024-08-23_10-28-42_metr-la_semi-dec-fl/communication_size/", "2024-08-23_09-05-46_metr-la_ray-semi-dec/communication_size/"]

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

full_output_directory = os.path.join(base_dir, 'logs', output_directory)
print(f"full_output_directory: {full_output_directory}")

# Create the output directory if it doesn't exist
if not os.path.exists(full_output_directory):
    os.makedirs(full_output_directory)

colors = ['orange', 'green']

# Plotting the data
plt.figure(figsize=(10, 6))

labels = ['FL', 'Server-free FL']

for i, relative_path in enumerate(csv_files):
    path = os.path.join(base_dir, 'logs', relative_path)

    print(f"Processing: {path}")

    # Check if the file exists
    if os.path.isfile(path):
        csv_list = [path]
    elif os.path.isdir(path):
        csv_list = [os.path.join(path, file) for file in os.listdir(path) 
                    if file.endswith('.csv') and '_total_transfer_size_node_features' in file]
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

    # Concatenate all DataFrames and calculate mean of transfer by epoch
    if df_list:
        combined_df = pd.concat(df_list)
        avg_df = combined_df.groupby('Epoch', as_index=False)['Megabytes'].sum()

        print(f"Averaged Dataframe:\n{avg_df.head()}")

        label = labels[i] if i < len(labels) else f'Node Features Transfer Size {i+1}'

        # Plot the averaged Val Loss
        plt.plot(avg_df['Epoch'], avg_df['Megabytes'], label=label, color=colors[i])

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('Megabytes')
plt.title('Node Features Transfer Size Over Epochs')
plt.legend()

# Save the plot
full_output_file = os.path.join(full_output_directory, output_file_name)

print(f"Plot saved to: {full_output_file}")
plt.savefig(full_output_file)
plt.close()