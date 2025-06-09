import pandas as pd
import matplotlib.pyplot as plt
import glob
import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config("/data/temporary/leopard/source/evaluation/pathology-leopard-evaluation/config/config.yaml")
    datasets_dict = config["datasets"]
    dataset_names_dict = config["dataset_names"]

    data_to_plot = []
    dataset_titles = []

    bins = range(0, 25, 2)
    x_limits = (0, 24)

    for dataset_dict in datasets_dict:
        dataset = list(dataset_dict.keys())[0]
        dataset_name = dataset_names_dict[dataset]

        file_path = os.path.join(config["ground_truth_path"], dataset + ".csv")
        df = pd.read_csv(file_path)

        if 'follow_up_years' not in df.columns or 'event' not in df.columns:
            print(f"Skipping {file_path} as required columns are missing.")
            continue

        followup_event_0 = df[df['event'] == 0]['follow_up_years']
        followup_event_1 = df[df['event'] == 1]['follow_up_years']

        data_to_plot.append((followup_event_0, followup_event_1, len(df), dataset_name))

    # Set up the plot: each dataset is a row, two columns (event=0 and event=1)
    n_datasets = len(data_to_plot)
    fig, axes = plt.subplots(nrows=n_datasets, ncols=2, figsize=(14, 5 * n_datasets), sharex=True)

    for i, (event_0, event_1, n, name) in enumerate(data_to_plot):
        # Determine max y limit for both subplots of the row
        counts_0, _ = pd.Series(event_0).value_counts(bins=bins, sort=False).values, bins
        counts_1, _ = pd.Series(event_1).value_counts(bins=bins, sort=False).values, bins
        y_max = max(max(counts_0), max(counts_1)) + 1

        ax0 = axes[i, 0] if n_datasets > 1 else axes[0]
        ax1 = axes[i, 1] if n_datasets > 1 else axes[1]

        ax0.hist(event_0, bins=bins, alpha=0.5, color='blue', edgecolor='black')
        ax0.set_title(f"{name} (n={n}): Event = 0")
        ax0.set_ylabel('Frequency')
        ax0.set_xlim(x_limits)
        ax0.set_ylim(0, y_max)
        ax0.grid(axis='y', linestyle='--', alpha=0.7)

        ax1.hist(event_1, bins=bins, alpha=0.5, color='red', edgecolor='black')
        ax1.set_title(f"{name} (n={n}): Event = 1")
        ax1.set_xlim(x_limits)
        ax1.set_ylim(0, y_max)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("/data/temporary/leopard/source/evaluation/results/time_to_follow_distribution_datasets.png")
    plt.show()

if __name__ == "__main__":
    main()
