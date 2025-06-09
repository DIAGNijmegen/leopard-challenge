import os
import json
import pandas as pd
import numpy as np
import yaml
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# Function to load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Function to load ground truth
def load_ground_truth(dataset, ground_truth_path):
    file_path = os.path.join(ground_truth_path, f"{dataset}_clinical_standardized_capra_s.csv")
    return pd.read_csv(file_path, dtype={"case_id": str})

# Function to count unique cases
def get_unique_case_count(df, case_column='case_id'):
    return df[case_column].nunique()

# Load predictions
def load_predictions(input_dir, teams, datasets):
    predictions = {}
    for team in teams:
        predictions[team] = {}
        for ds_dict in datasets:
            ds = list(ds_dict.keys())[0]
            path = os.path.join(input_dir, team, ds)
            if not os.path.exists(path):
                continue
            preds = {}
            expected = list(ds_dict.values())[0]
            files = [f for f in os.listdir(path) if f.endswith('.json')]
            if len(files) != expected:
                print(f"WARNING: {team}/{ds} has {len(files)} preds, expected {expected}")
            for fn in files:
                cid = fn.replace('.json', '')
                with open(os.path.join(path, fn), 'r') as f:
                    preds[cid] = json.load(f)
            if preds:
                predictions[team][ds] = preds
    return predictions

# Teams order for LaTeX tables
TEAMS_ORDER = [
    'MEVIS-ProSurvival','MartelLab','Paicon','LEOPARD Baseline',
    'AIRA Matrix','HITSZLab','KatherTeam','QuIIL Lab','IUComPath'
]

# Compute Cox metrics
def compute_cox_metrics(predictions, datasets, ground_truth_path, team_names, dataset_names):
    results = {}
    for ds_dict in datasets:
        ds = list(ds_dict.keys())[0]
        ds_label = dataset_names[ds]
        gt_df = load_ground_truth(ds, ground_truth_path)
        for team, team_data in predictions.items():
            if ds not in team_data:
                continue
            preds = team_data[ds]
            df = gt_df[gt_df['case_id'].isin(preds)].copy()
            if df.empty:
                continue
            df['prediction'] = df['case_id'].map(preds)
            df = df[['prediction','capra_s_score','follow_up_years','event','case_id']]

            # Univariate Cox
            uni = CoxPHFitter()
            uni.fit(df[['prediction','follow_up_years','event']], duration_col='follow_up_years', event_col='event')
            hr_u = uni.hazard_ratios_['prediction']
            lo_u = uni.summary.loc['prediction','exp(coef) lower 95%']
            hi_u = uni.summary.loc['prediction','exp(coef) upper 95%']
            p_u  = uni.summary.loc['prediction','p']

            # Multivariate Cox (prediction only)
            multi = CoxPHFitter()
            multi.fit(df[['prediction','capra_s_score','follow_up_years','event']], duration_col='follow_up_years', event_col='event')
            hr_m = multi.hazard_ratios_['prediction']
            lo_m = multi.summary.loc['prediction','exp(coef) lower 95%']
            hi_m = multi.summary.loc['prediction','exp(coef) upper 95%']
            p_m  = multi.summary.loc['prediction','p']

            team_label = team_names[team]
            n = get_unique_case_count(df)
            results.setdefault(team_label, {})[ds_label] = {
                'hr_u': hr_u, 'lo_u': lo_u, 'hi_u': hi_u, 'p_u': p_u,
                'hr_m': hr_m, 'lo_m': lo_m, 'hi_m': hi_m, 'p_m': p_m,
                'n': n
            }
    return results

# Flatten to DataFrame for CSV

def format_results(results):
    rows = []
    for team, ds_dict in results.items():
        for ds, vals in ds_dict.items():
            rows.append({
                'Team': team, 'Dataset': ds,
                'hr_u': vals['hr_u'], 'lo_u': vals['lo_u'], 'hi_u': vals['hi_u'], 'p_u': vals['p_u'],
                'hr_m': vals['hr_m'], 'lo_m': vals['lo_m'], 'hi_m': vals['hi_m'], 'p_m': vals['p_m'],
                'n': vals['n']
            })
    return pd.DataFrame(rows).sort_values(['Dataset','Team'])

# Save outputs: CSV unchanged, LaTeX one table per dataset

def save_results(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # CSV
    df.to_csv(os.path.join(output_dir,'cox_metrics_capra_s.csv'), index=False)

    # LaTeX: separate tables
    tex_file = os.path.join(output_dir,'cox_metrics_capra_s.tex')
    with open(tex_file,'w') as f:
        for ds in df['Dataset'].unique():
            sub = df[df['Dataset']==ds].set_index('Team').reindex(TEAMS_ORDER)
            n_cases = int(sub['n'].iloc[0]) if not sub['n'].isnull().all() else 0
            tbl = pd.DataFrame(index=TEAMS_ORDER)
            tbl['HR (univariate)'] = sub.apply(
                lambda r: f"${r.hr_u:.3f}_{{({r.lo_u:.3f},{r.hi_u:.3f})}}$", axis=1)
       
            tbl['p-value (univariate)'] = sub['p_u'].apply(lambda p: f"{p:.1e}")
            tbl['HR (multivariate)'] = sub.apply(
                lambda r: f"${r.hr_m:.3f}_{{({r.lo_m:.3f},{r.hi_m:.3f})}}$", axis=1)
            
        
            tbl['p-value (multivariate)'] = sub['p_m'].apply(lambda p: f"{p:.1e}")
            caption = f"Cox Proportional Hazard models analysis of AI models predictions with CAPRA-S for {ds} (n={n_cases})."
            f.write(tbl.to_latex(escape=False, caption=caption, label=f"tab:{ds.replace(' ','_')}") + "\n")

# Main entry point
if __name__=='__main__':
    # Hardcoded config path as before
    main_config = "/data/temporary/leopard/source/evaluation/pathology-leopard-evaluation/config/config.yaml"
    cfg = load_config(main_config)
    preds = load_predictions(cfg['input_dir'], cfg['teams'], cfg['datasets'])
    results = compute_cox_metrics(preds, cfg['datasets'], cfg['clinical_variables'], cfg['team_names'], cfg['dataset_names'])
    df = format_results(results)
    print(df)
    save_results(df, cfg['output_dir'])
