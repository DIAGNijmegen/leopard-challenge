import os
import json
import yaml
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, statistics
import matplotlib
matplotlib.use('Agg')  # Add this line
import matplotlib.pyplot as plt
%matplotlib inline
from lifelines.plotting import add_at_risk_counts


# Load configuration file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Read ground truth data
def load_ground_truth(ground_truth_path):
    return pd.read_csv(ground_truth_path, dtype={"case_id": str})

# Read team predictions
def load_predictions(team, dataset, input_path):
    team_path = os.path.join(input_path, team, dataset)
    predictions = {}
    for file in os.listdir(team_path):
        if file.endswith('.json'):
            case_id = file.replace('.json', '')
            with open(os.path.join(team_path, file), 'r') as f:
                predictions[case_id] = json.load(f)
    return predictions

# Assign risk groups
def assign_risk_groups(predictions):
    values = np.array(list(predictions.values()))
    low_thresh = np.percentile(values, 33)
    high_thresh = np.percentile(values, 66)
    risk_groups = {}
    
    for case_id, time in predictions.items():
        if time <= low_thresh:
            risk_groups[case_id] = 'High'
        elif time <= high_thresh:
            risk_groups[case_id] = 'Intermediate'
        else:
            risk_groups[case_id] = 'Low'
    
    return risk_groups

# Plot all Kaplan-Meier curves in a grid

def plot_all_kaplan_meier(config):
    teams = config['teams']
    team_names = config['team_names']
    dataset_names = config['dataset_names']
    datasets = [list(dataset_dict.keys())[0] for dataset_dict in config['datasets']]
    dataset_x_limits = {"cologne": (0, 6), "brazil": (0, 6), "plco": (0, 14), "radboud": (0, 12)}
    colors = {'High': '#D1495B', 'Intermediate': '#f0a202', 'Low': '#4A6670'}

    for team in teams:
        fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 6), sharey=True)

        if len(datasets) == 1:
            axes = [axes]  # ensure axes is iterable

        for j, dataset in enumerate(datasets):
            ground_truth = load_ground_truth(os.path.join(config['ground_truth_path'], dataset + ".csv"))
            predictions = load_predictions(team, dataset, config['input_dir'])
            risk_groups = assign_risk_groups(predictions)

            kmfs = {}
            ax = axes[j]

            for group in ['Low', 'Intermediate', 'High']:
                mask = ground_truth['case_id'].isin([k for k, v in risk_groups.items() if v == group])
                group_data = ground_truth[mask]

                if len(group_data) > 0:
                    kmf = KaplanMeierFitter()
                    kmf.fit(group_data['follow_up_years'], event_observed=group_data['event'], label=group)
                    kmf.plot_survival_function(ax=ax, color=colors[group],lw=2.5, ci_alpha=0.09, show_censors=True)
                    kmfs[group] = kmf

            ax.set_title(f'{team_names[team]} - {dataset_names[dataset]} ({len(ground_truth)} cases)')
            ax.set_xlabel('Follow-up Time (Years)')
            ax.set_ylabel('BCR-Free Survival Probability')
            ax.legend(loc='lower left')
            ax.grid(True)


            if dataset in dataset_x_limits:
                ax.set_xlim(dataset_x_limits[dataset])

            # Add at risk counts properly
            if kmfs:
                add_at_risk_counts(*kmfs.values(), ax=ax)

        plt.tight_layout()
        output_path = os.path.join(config['output_dir'], f'kaplan_meier_{team}.png')
        plt.savefig(output_path)
        plt.close(fig)


# Main execution function
def main(config_path):
    config = load_config(config_path)
    plot_all_kaplan_meier(config)
    
if __name__ == '__main__':
    main("/data/temporary/leopard/source/evaluation/pathology-leopard-evaluation/config/config.yaml")
