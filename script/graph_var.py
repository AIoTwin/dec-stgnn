# Example of running this script:
# python graph_var.py 2024-06-24_15-54-57_pemsd4_semi-dec-fl/var/0.csv 2024-06-24_15-54-57_pemsd4_semi-dec-fl/var_plot 0.png

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Ensure the script is called with the correct number of arguments
if len(sys.argv) != 4:
    print("Usage: python graph.py <relative_path_to_csv> <output_directory> <output_file_name>")
    sys.exit(1)

relative_path = sys.argv[1]
output_directory = sys.argv[2]
output_file_name = sys.argv[3]

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
csv_file = os.path.join(base_dir, 'logs', relative_path)

# Print the path for debugging purposes
print(f"Reading from: {csv_file}")

# Check if the file exists
if not os.path.isfile(csv_file):
    print(f"Error: The file {csv_file} does not exist.")
    sys.exit(1)

# Load the CSV file into a DataFrame
try:
    df = pd.read_csv(csv_file)
    print(f"Dataframe loaded successfully:\n{df.head()}")
except Exception as e:
    print(f"Error loading the CSV file: {e}")
    sys.exit(1)

full_output_directory = os.path.join(base_dir, 'logs', output_directory)
print(f"full_output_directory: {full_output_directory}")
# Create the output directory if it doesn't exist
if not os.path.exists(full_output_directory):
    os.makedirs(full_output_directory)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Variance'], label='Variance')

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('Variance')
plt.title('Variance Mean Over Epochs')
plt.legend()

# Save the plot
# full_output_directory = os.path.join(output_directory, output_file_name)
full_output_file = os.path.join(full_output_directory, output_file_name)

print(f"Plot saved to: {full_output_file}")
plt.savefig(full_output_file)
