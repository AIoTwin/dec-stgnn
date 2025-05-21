# Example of running this script:
# python graph_val_train_single.py

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

output_directory = "plots"
output_file_name = "plot_val_train_loss_single.png"
csv_file = "2024-10-30_15-08-54_pems-bay_pred-15min_his-60min_semi-dec-fl-distance/val/master_server.csv"

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

full_output_directory = os.path.join(base_dir, 'logs', output_directory)
print(f"full_output_directory: {full_output_directory}")

# Create the output directory if it doesn't exist
if not os.path.exists(full_output_directory):
    os.makedirs(full_output_directory)

val_color = "blue"
train_color = "orange"

# Plotting the data
plt.figure(figsize=(10, 6))

val_label = "Gossip Learning Validation Loss"
train_label = "Gossip Learning Training Loss"

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
    val_avg_df = combined_df.groupby('Epoch', as_index=False)['Val Loss'].mean()
    val_std_df = combined_df.groupby('Epoch', as_index=False)['Val Loss'].std()

    train_avg_df = combined_df.groupby('Epoch', as_index=False)['Train Loss'].mean()
    train_std_df = combined_df.groupby('Epoch', as_index=False)['Train Loss'].std()

    print(f"Averaged Validation Dataframe:\n{val_avg_df.head()}")
    print(f"Averaged Train Dataframe:\n{train_avg_df.head()}")

    # Plot the averaged Val and Train Loss
    plt.plot(val_avg_df['Epoch'], val_avg_df['Val Loss'], label=val_label, color=val_color)
    plt.fill_between(val_avg_df['Epoch'], 
                        val_avg_df['Val Loss'] - val_std_df['Val Loss'], 
                        val_avg_df['Val Loss'] + val_std_df['Val Loss'], 
                        color=val_color, alpha=0.3)
    plt.plot(train_avg_df['Epoch'], train_avg_df['Train Loss'], label=train_label, color=train_color)
    plt.fill_between(train_avg_df['Epoch'], 
                        train_avg_df['Train Loss'] - train_std_df['Train Loss'], 
                        train_avg_df['Train Loss'] + train_std_df['Train Loss'], 
                        color=train_color, alpha=0.3)

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation & Train Loss Over Epochs')
plt.legend()

# Save the plot
full_output_file = os.path.join(full_output_directory, output_file_name)

print(f"Plot saved to: {full_output_file}")
plt.savefig(full_output_file)
plt.close()