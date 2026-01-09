import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import re
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta

sharey = "none" # "row" |"none"

dataset = "pems-bay_pemsd7-m"
metric_type = "WMAPE"

training_setup = "full-graph-conn" # full-graph-conn | no-graph-conn | node-score

# time resolution of samples (minutes per sample)
step_minutes = 5

# ---- PeMSD-7M timebase ----
# pems7d_data_per_step_short = 70
# pems7d_data_per_step_long = 140

pems7d_initial_train_len = 4537
pems7d_train_len = 10137
pems7d_valid_len = 2534
pems7d_total_len = pems7d_train_len + pems7d_valid_len

pems7d_start_datetime = datetime.strptime("01.05.2012 00:00", "%d.%m.%Y %H:%M")  # 01.05.2012 – 30.06.2012

# ---- PeMS-BAY timebase ----
# pemsbay_data_per_step_short = 70
# pemsbay_data_per_step_long = 140

pemsbay_initial_train_len = 29692
pemsbay_train_len = 41692
pemsbay_valid_len = 10423
pemsbay_total_len = pemsbay_train_len + pemsbay_valid_len

pemsbay_start_datetime = datetime.strptime("01.01.2017 00:00", "%d.%m.%Y %H:%M")  # 01.01.2017 – 30.06.2017

if metric_type == "WMAPE":
    plot_label = "WMAPE [%]"
    y_min, y_max = 5, 20
    subfolder = "val_metric"
    value_column = "WMAPE"
elif metric_type == "MAE":
    plot_label = "MAE [mile/h]"
    subfolder = "val_metric"
    value_column = "MAE"
elif metric_type == "RMSE":
    plot_label = "RMSE [mile$^2$/h$^2$]"
    subfolder = "val_metric"
    value_column = "RMSE"
elif metric_type == "SCSR":
    plot_label = "SCSR [%]"
    subfolder = "new_val_metric"
    value_column = "Total sudden change in speed rate"
else:
    print(f"Error finding metric_type {metric_type}")
    sys.exit(1)

output_directory = "plots"
output_file_name = f"plot_val_metric_{training_setup}_{metric_type}_{dataset}_cloudlet_subplot.png"

if training_setup == "full-graph-conn":
    csv_file_paths = {
        "Traditional FL - PeMS-BAY - 15min": "2025-09-05_16-04-07_pems-bay_pred-15min_dps-140_trad-fl-distance_online-training_no_algorithm",
        "Server-Free FL - PeMS-BAY - 15min": "2025-09-06_05-00-38_pems-bay_pred-15min_dps-140_server-free-fl-distance_online-training_no_algorithm",
        "Gossip Learning - PeMS-BAY - 15min": "2025-09-06_17-59-17_pems-bay_pred-15min_dps-140_gossip-learning-distance_online-training_no_algorithm",

        "Traditional FL - PeMS-BAY - 60min": "2025-09-05_19-03-50_pems-bay_pred-60min_dps-140_trad-fl-distance_online-training_no_algorithm",
        "Server-Free FL - PeMS-BAY - 60min": "2025-09-06_08-05-08_pems-bay_pred-60min_dps-140_server-free-fl-distance_online-training_no_algorithm",
        "Gossip Learning - PeMS-BAY - 60min": "2025-09-06_21-03-28_pems-bay_pred-60min_dps-140_gossip-learning-distance_online-training_no_algorithm",

        "Traditional FL - PeMSD-7M - 15min": "2025-09-06_01-03-18_pemsd7-m_pred-15min_dps-140_trad-fl-distance_online-training_no_algorithm",
        "Server-Free FL - PeMSD-7M - 15min": "2025-09-06_14-13-47_pemsd7-m_pred-15min_dps-140_server-free-fl-distance_online-training_no_algorithm",
        "Gossip Learning - PeMSD-7M - 15min": "2025-09-07_03-11-24_pemsd7-m_pred-15min_dps-140_gossip-learning-distance_online-training_no_algorithm",

        "Traditional FL - PeMSD-7M - 60min": "2025-09-06_02-22-13_pemsd7-m_pred-60min_dps-140_trad-fl-distance_online-training_no_algorithm",
        "Server-Free FL - PeMSD-7M - 60min": "2025-09-06_15-28-48_pemsd7-m_pred-60min_dps-140_server-free-fl-distance_online-training_no_algorithm",
        "Gossip Learning - PeMSD-7M - 60min": "2025-09-07_04-26-26_pemsd7-m_pred-60min_dps-140_gossip-learning-distance_online-training_no_algorithm",
    }
elif training_setup == "node-score":
    csv_file_paths = {
        "Traditional FL - PeMS-BAY - 15min": "2025-09-10_09-49-06_pems-bay_pred-15min_dps-140_trad-fl-distance_online-training_with_algorithm",
        "Server-Free FL - PeMS-BAY - 15min": "2025-09-12_09-32-23_pems-bay_pred-15min_dps-140_server-free-fl-distance_online-training_with_algorithm",
        "Gossip Learning - PeMS-BAY - 15min": "2025-09-18_12-03-51_pems-bay_pred-15min_dps-140_gossip-learning-distance_online-training_with_algorithm",

        "Traditional FL - PeMS-BAY - 60min": "2025-09-10_13-00-40_pems-bay_pred-60min_dps-140_trad-fl-distance_online-training_with_algorithm",
        "Server-Free FL - PeMS-BAY - 60min": "2025-09-12_13-32-51_pems-bay_pred-60min_dps-140_server-free-fl-distance_online-training_with_algorithm",
        "Gossip Learning - PeMS-BAY - 60min": "2025-09-18_15-45-43_pems-bay_pred-60min_dps-140_gossip-learning-distance_online-training_with_algorithm",

        "Traditional FL - PeMSD-7M - 15min": "2025-09-10_20-02-01_pemsd7-m_pred-15min_dps-140_trad-fl-distance_online-training_with_algorithm",
        "Server-Free FL - PeMSD-7M - 15min": "2025-09-12_23-28-09_pemsd7-m_pred-15min_dps-140_server-free-fl-distance_online-training_with_algorithm",
        "Gossip Learning - PeMSD-7M - 15min": "2025-09-19_01-18-27_pemsd7-m_pred-15min_dps-140_gossip-learning-distance_online-training_with_algorithm",

        "Traditional FL - PeMSD-7M - 60min": "2025-09-10_21-15-40_pemsd7-m_pred-60min_dps-140_trad-fl-distance_online-training_with_algorithm",
        "Server-Free FL - PeMSD-7M - 60min": "2025-09-13_00-39-09_pemsd7-m_pred-60min_dps-140_server-free-fl-distance_online-training_with_algorithm",
        "Gossip Learning - PeMSD-7M - 60min": "2025-09-19_02-27-38_pemsd7-m_pred-60min_dps-140_gossip-learning-distance_online-training_with_algorithm",
    }
else:
    print(f"Training setup doesn't exist: {training_setup}!")
    sys.exit(1)

# ---------------- PATHS (two levels up) ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))  # <= one more os.pardir
logs_root = os.path.join(base_dir, "logs")

full_output_directory = os.path.join(logs_root, output_directory)
os.makedirs(full_output_directory, exist_ok=True)
print(f"full_output_directory: {full_output_directory}")

# ---------------- TIME HELPERS ----------------
def epoch_to_datetime_pems7d(epoch: int, data_per_step: int):
    """Convert epoch index (1-based) to wall-clock datetime for PeMSD-7M."""
    base_minutes = pems7d_initial_train_len * step_minutes
    minutes_after = epoch * data_per_step * step_minutes
    return pems7d_start_datetime + timedelta(minutes=base_minutes + minutes_after)

def epoch_to_datetime_pemsbay(epoch: int, data_per_step: int):
    """Convert epoch index (1-based) to wall-clock datetime for PeMS-BAY."""
    base_minutes = pemsbay_initial_train_len * step_minutes
    minutes_after = epoch * data_per_step * step_minutes
    return pemsbay_start_datetime + timedelta(minutes=base_minutes + minutes_after)

def parse_dataset_and_horizon(title: str):
    """Return ('pemsbay'|'pems7d', 'short'|'long') based on panel title."""
    ds = "pemsbay" if "PeMS-BAY" in title else "pems7d"
    hz = "short" if "15min" in title else "long" if "60min" in title else None
    return ds, hz

def epochs_to_datetimes(title: str, epochs_series: pd.Series):
    ds, hz = parse_dataset_and_horizon(title)
    dps = 140
    if ds == "pemsbay":
        return [epoch_to_datetime_pemsbay(int(e), dps) for e in epochs_series]
    else:
        return [epoch_to_datetime_pems7d(int(e), dps) for e in epochs_series]

# ---------------- OTHER HELPERS ----------------
def extract_cloudlet_id_from_masked(pathlike: str) -> int:
    """masked_{id}.csv -> id (int)."""
    m = re.search(r"(\d+)", Path(pathlike).stem)
    return int(m.group(1)) if m else 10**9  # big sentinel to push unknowns to the end

# Map the "none" string to a proper False for sharey
sharey_arg = False if isinstance(sharey, str) and sharey.lower() == "none" else sharey

# ---------------- PLOTTING ----------------
rows, cols = 4, 3
fig, axes = plt.subplots(rows, cols, figsize=(20, 15), sharey=sharey_arg)
axes = axes.ravel()

legend_handles = {}
legend_order = []

items = list(csv_file_paths.items())  # keep insertion order

# Loop through each file path and plot in each subplot
for idx, (title, run_folder_name) in enumerate(items):
    if idx >= rows * cols:
        print(f"Skipping extra entry '{title}' because grid is {rows}x{cols}.")
        continue

    ax = axes[idx]

    run_dir = os.path.join(logs_root, run_folder_name)
    if not os.path.isdir(run_dir):
        print(f"Error: Directory {run_dir} does not exist.")
        continue
    
    metric_dir = os.path.join(run_dir, subfolder)
    if not os.path.isdir(metric_dir):
        print(f"Warning: '{subfolder}' not found in {run_dir} (title: '{title}'). Skipping.")
        continue

    csv_list = [
        os.path.join(metric_dir, f)
        for f in os.listdir(metric_dir)
        if f.endswith(".csv") and f.startswith("masked_")
    ]
    csv_list.sort(key=lambda p: extract_cloudlet_id_from_masked(p))

    for csv_path in csv_list:
        try:
            df = pd.read_csv(csv_path)

            if value_column not in df.columns or "Epoch" not in df.columns:
                print(f"Warning: needed columns not found in {csv_path}. Skipping.")
                continue

            # ---- SCSR-specific filtering: keep only epochs with >0 sudden-change count ----
            if metric_type == "SCSR":
                if "Total sudden change in speed count" not in df.columns:
                    print(f"Warning: 'Total sudden change in speed count' not in {csv_path}. Skipping.")
                    continue
                df = df.loc[df["Total sudden change in speed count"] > 0].copy()
                if df.empty:
                    # nothing to plot for this cloudlet in this run
                    continue

            # x-axis as datetime
            epochs = pd.to_numeric(df["Epoch"], errors="coerce").astype("Int64").dropna().astype(int)
            datetimes = epochs_to_datetimes(title, epochs)

            y = df.loc[epochs.index, value_column].copy()

            if metric_type == "WMAPE":
                y = y * 100.0
            elif metric_type == "SCSR":
                y_numeric = pd.to_numeric(y, errors="coerce")
                if y_numeric.max(skipna=True) is not None and y_numeric.max(skipna=True) <= 1.0:
                    y = y * 100.0

            cid = extract_cloudlet_id_from_masked(csv_path)
            cloudlet_name = f"Cloudlet {cid + 1}" if cid != 10**9 else "Cloudlet ?"

            # plot
            ln, = ax.plot(datetimes, y, marker="o", markersize=2, linestyle='-', label=cloudlet_name)

            # collect legend handles uniquely by label
            if cloudlet_name not in legend_handles:
                legend_handles[cloudlet_name] = ln
                legend_order.append(cloudlet_name)

        except Exception as e:
            print(f"Error loading CSV '{csv_path}': {e}")
            continue

    ax.set_xlabel("")
    if idx % cols == 0:
        ax.set_ylabel(plot_label)
    ax.set_title(title)
    # date axis formatting
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    # if metric_type == "WMAPE":
    #     ax.set_ylim(y_min, y_max)

# Remove any unused axes (if fewer than rows*cols)
for j in range(len(items), rows * cols):
    fig.delaxes(axes[j])

plt.tight_layout()

# Legend at the bottom using unique labels
handles_in_order = [legend_handles[lbl] for lbl in legend_order]
fig.legend(
    handles_in_order,
    legend_order,
    loc="lower center",
    ncol=min(7, len(legend_order)),
    fontsize=14,
    bbox_to_anchor=(0.5, -0.05),
)

full_output_file = os.path.join(full_output_directory, output_file_name)
print(f"Plot saved to: {full_output_file}")
plt.savefig(full_output_file, bbox_inches="tight", dpi=200)
plt.close()