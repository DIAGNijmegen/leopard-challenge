import os
import json
import yaml  # fixed import
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test  # added import
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from lifelines.plotting import add_at_risk_counts
%matplotlib inline

# Load configuration file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Read ground truth data
def load_ground_truth(ground_truth_path):
    return pd.read_csv(ground_truth_path, dtype={"case_id": str})

# Read team predictions into a DataFrame
def load_all_predictions(teams, dataset, input_dir):
    dfs = []
    for team in teams:
        team_path = os.path.join(input_dir, team, dataset)
        preds = {}
        for fname in os.listdir(team_path):
            if fname.endswith('.json'):
                case_id = fname.replace('.json', '')
                with open(os.path.join(team_path, fname), 'r') as f:
                    preds[case_id] = json.load(f)
        df = pd.Series(preds, name=team)
        dfs.append(df)
    return pd.concat(dfs, axis=1)

# Assign risk groups given explicit thresholds
def assign_risk_groups(scores, low_thresh, high_thresh):
    risk = {}
    for case_id, val in scores.items():
        if val <= low_thresh:
            risk[case_id] = 'High'
        elif val <= high_thresh:
            risk[case_id] = 'Intermediate'
        else:
            risk[case_id] = 'Low'
    return risk

# Plot ensemble Kaplan-Meier curves across datasets with log-rank p-values at bottom
# Using cutoff thresholds 

def plot_ensemble_kaplan_meier(config):
    teams = config['ensemble_teams']
    dataset_names = config['dataset_names']
    datasets = list(dataset_names.keys())
    dataset_x_limits = {"cologne": (0, 6), "brazil": (0, 6), "plco": (0, 15), "radboud": (0, 15)}
    colors = {'High': '#D1495B', 'Intermediate': '#f0a202', 'Low': '#4A6670'}

  

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 6), sharey=True)
    if len(datasets) == 1:
        
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        # Load truth and predictions
        gt = load_ground_truth(os.path.join(config['ground_truth_path'], f"{dataset}.csv"))
        preds_df = load_all_predictions(teams, dataset, config['input_dir'])

        # Z-score normalize per team across cases
        preds_z = preds_df.apply(lambda col: (col - col.mean()) / col.std(), axis=0)
        # Ensemble: mean per case
        ensemble_scores = preds_z.mean(axis=1)
        
        #fig, axes = plt.subplots(1, 1, figsize=(7 * len(datasets), 6))
       
        #plt.hist(ensemble_scores)
        #plt.axvline(x=low_thresh,color='red', linestyle='--', linewidth=2)
        #plt.axvline(x=high_thresh,color='red', linestyle='--', linewidth=2)        
        #plt.show()

        # Assign risk groups using Radboud thresholds
        low_thresh = ensemble_scores.quantile(0.33)
        high_thresh = ensemble_scores.quantile(0.66)
        risk_groups = assign_risk_groups(ensemble_scores, low_thresh, high_thresh)
        
        counts = {}
        for item in list(risk_groups.values()):
            counts[item] = counts.get(item, 0) + 1
        print('dataset:',dataset,'counts:',counts)

        # Fit and plot KM curves
        kmfs = {}
        for group in ['Low', 'Intermediate', 'High']:
            case_ids = [cid for cid, grp in risk_groups.items() if grp == group]
            mask = gt['case_id'].isin(case_ids)
            data_grp = gt[mask]
            if not data_grp.empty:
                kmf = KaplanMeierFitter()
                kmf.fit(data_grp['follow_up_years'], event_observed=data_grp['event'], label=group)
                kmf.plot_survival_function(ax=ax, color=colors[group], lw=2.5, ci_alpha=0.09, show_censors=True)
                kmfs[group] = kmf

        # Log-rank test between High and Low risk groups
        high_ids = [cid for cid, grp in risk_groups.items() if grp == 'High']
        low_ids = [cid for cid, grp in risk_groups.items() if grp == 'Low']
        data_high = gt[gt['case_id'].isin(high_ids)]
        data_low = gt[gt['case_id'].isin(low_ids)]
   
        if not data_high.empty and not data_low.empty:
            result = logrank_test(
                data_high['follow_up_years'], data_low['follow_up_years'],
                event_observed_A=data_high['event'], event_observed_B=data_low['event']
            )
            p = result.p_value
            ax.text(
                0.95, 0.05, f"Log-rank p = {p:.3e}",
                transform=ax.transAxes, ha='right', va='bottom', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.7)
            )

        ax.set_title(f"{dataset_names[dataset]} ({len(gt)} cases)")
        ax.set_xlabel('Follow-up Time (Years)')
        ax.set_ylabel('BCR-Free Survival Probability')
        ax.legend(loc='lower left')
        ax.grid(True)
        

        if dataset in dataset_x_limits:
            ax.set_xlim(dataset_x_limits[dataset])

        if kmfs:
            add_at_risk_counts(*kmfs.values(), ax=ax)

    plt.tight_layout()
    out = os.path.join(config['output_dir'], 'kaplan_meier_ensemble_with_pvalue.png')
    plt.savefig(out)
    plt.close(fig)
    plt.show()


# Main

def main(config_path):
    config = load_config(config_path)
    plot_ensemble_kaplan_meier(config)

if __name__ == '__main__':
    main('/data/temporary/leopard/source/evaluation/pathology-leopard-evaluation/config/config.yaml')
