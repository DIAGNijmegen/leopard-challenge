import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config("/data/temporary/leopard/source/evaluation/pathology-leopard-evaluation/config/config.yaml")

    datasets_dict = config["datasets"]
    dataset_names_dict = config["dataset_names"]

    num_datasets = len(datasets_dict)
    fig, axes = plt.subplots(num_datasets, 2, figsize=(14, 5 * num_datasets), sharex=True)

    bins = range(0, 7, 1)
    x_limits = (0, 6)

    for idx, dataset_dict in enumerate(datasets_dict):
        dataset = list(dataset_dict.keys())[0]
        dataset_name = dataset_names_dict[dataset]

        file_path = os.path.join(config["ground_truth_path"], dataset + ".csv")
        df = pd.read_csv(file_path)

        if 'isup' not in df.columns or 'event' not in df.columns:
            print(f"Skipping {file_path} as required columns are missing.")
            continue

        followup_event_0 = df[df['event'] == 0]['isup']
        followup_event_1 = df[df['event'] == 1]['isup']

        ax0 = axes[idx, 0] if num_datasets > 1 else axes[0]
        ax1 = axes[idx, 1] if num_datasets > 1 else axes[1]

        # Get histogram counts to normalize y-axis
        counts_0, _ = pd.cut(followup_event_0, bins=bins, right=False).value_counts(sort=False).values, bins
        counts_1, _ = pd.cut(followup_event_1, bins=bins, right=False).value_counts(sort=False).values, bins
        max_y = max(max(counts_0), max(counts_1)) + 1  # Add margin

        ax0.hist(followup_event_0, bins=bins, alpha=0.5, color='blue', edgecolor='black')
        ax0.set_xlabel('ISUP')
        ax0.set_ylabel('Frequency')
        ax0.set_xlim(x_limits)
        ax0.set_ylim(0, max_y)
        ax0.grid(axis='y', linestyle='--', alpha=0.7)
        ax0.set_title(f'{dataset_name} (n={len(df)}): Event = 0')

        ax1.hist(followup_event_1, bins=bins, alpha=0.5, color='red', edgecolor='black')
        ax1.set_xlabel('ISUP')
        ax1.set_ylabel('Frequency')
        ax1.set_xlim(x_limits)
        ax1.set_ylim(0, max_y)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_title(f'{dataset_name} (n={len(df)}): Event = 1')

    plt.tight_layout()
    plt.savefig("/data/temporary/leopard/source/evaluation/results/isup_distribution_datasets.png")
    plt.show()

if __name__ == "__main__":
    main()
