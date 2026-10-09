# Source: run.ipynb, cell 13
# AI + CAPRA-S C-index per ensemble team, Cox models frozen on the RUMC calibration set.

import os
import json
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import yaml

# Algorithm + CAPRA-S C-index per ensemble team, with Cox models fit on the RUMC calibration set
# (RUMC Tuning + RUMC Internal Validation) and frozen; algorithm predictions and CAPRA-S are
# z-scored with calibration-set mean/SD
CLINICAL_COL = 'capra_s_score'   # column in <clinical_variables>/<dataset>_capra_s_median.csv
CLINICAL_LABEL = 'CAPRA-S'
OUTPUT_NAME = 'c_index_model_capra_s_combined_results_median_calib'

# Function to load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Function to load ground truth
def load_ground_truth(dataset, ground_truth_path):
    file_path = os.path.join(ground_truth_path, f"{dataset}_capra_s_median.csv")
    return pd.read_csv(file_path, dtype={"case_id": str})[['case_id', 'event', 'follow_up_years', CLINICAL_COL]]

# Function to load one folder of predictions (one JSON per case), inverted so higher = higher risk
def load_prediction_dir(path):
    preds = {}
    for file_name in os.listdir(path):
        if file_name.endswith('.json'):
            with open(os.path.join(path, file_name), 'r') as f:
                preds[file_name.replace('.json', '')] = -json.load(f)
    return preds

# Function to load predictions of the ensemble teams: test datasets + RUMC Tuning split
def load_predictions(config):
    predictions = {}
    for team in config['ensemble_teams']:
        predictions[team] = {}
        for dataset_dict in config['datasets']:
            dataset = next(iter(dataset_dict))
            dataset_path = os.path.join(config['input_dir'], team, dataset)
           
            predictions[team][dataset] = load_prediction_dir(dataset_path)
            expected = dataset_dict[dataset]
            if len(predictions[team][dataset]) != expected:
                print(f"Warning: {team}/{dataset} expected {expected} preds, found {len(predictions[team][dataset])}")
        tuning_path = os.path.join(config['validation_dir'], team, config['tuning_predictions_subfolder'])
        predictions[team][config['tuning_dataset']] = load_prediction_dir(tuning_path)
    return predictions

# Function to build one team's per-case table (prediction, clinical variable, outcome) over all datasets
def build_team_df(team_preds, datasets, ground_truth_path):
    frames = []
    for dataset in datasets:
     
        gt = load_ground_truth(dataset, ground_truth_path)
        gt = gt[gt['case_id'].isin(team_preds[dataset].keys())].copy()
        gt['prediction'] = gt['case_id'].map(team_preds[dataset])
        gt['dataset'] = dataset
        frames.append(gt)
    return pd.concat(frames, ignore_index=True)

# Function to compute C-index of the frozen algorithm + clinical-variable Cox model on every dataset
def compute_c_index(predictions, config):
    calib_datasets = [config['tuning_dataset'], 'radboud']
    eval_datasets = [next(iter(d)) for d in config['datasets']]  # RUMC, PLCO, IMP, UHC
    cols = ['prediction', CLINICAL_COL, 'follow_up_years', 'event']

    results = {}
    for team, team_preds in predictions.items():
        df = build_team_df(team_preds, list(dict.fromkeys(calib_datasets + eval_datasets)), config['clinical_variables'])
        calib = df['dataset'].isin(calib_datasets)

        # z-score with calibration-set mean/SD, frozen and applied to every dataset
        for col in ['prediction', CLINICAL_COL]:
            df[col] = (df[col] - df.loc[calib, col].mean()) / df.loc[calib, col].std(ddof=1)

        # Cox model fit once on the calibration set and frozen
        cph = CoxPHFitter().fit(df.loc[calib, cols], duration_col='follow_up_years', event_col='event')
        print(f"{team}: calibration n={int(calib.sum())}, events={int(df.loc[calib, 'event'].sum())}")

        official_team_name = config['team_names'][team]
        results[official_team_name] = {}
        for dataset in eval_datasets:
            data = df[df['dataset'] == dataset]
           
            c_index = concordance_index(data['follow_up_years'], -cph.predict_partial_hazard(data[cols]), data['event'])
            results[official_team_name][config['dataset_names'][dataset]] = c_index
    return results

# Function to format results into a DataFrame
def format_results(results):
    df = pd.DataFrame(results).T  # Transpose so that teams are rows
    df = df.reset_index().rename(columns={'index': 'Team'})

    # Calculate average and standard deviation of C-index across datasets
    dataset_columns = df.columns.drop('Team')
    df['average_c_index'] = df[dataset_columns].mean(axis=1)
    df['std_c_index'] = df[dataset_columns].std(axis=1)

    # Create formatted column "average (+/- std)"
    df['Average C-index'] = (
        '$' + df['average_c_index'].map('{:.3f}'.format) +
        r' _{(\pm ' + df['std_c_index'].map('{:.3f}'.format) + ')}$'
    )

    # Sort by average C-index
    df = df.sort_values(by='average_c_index', ascending=False).reset_index(drop=True)
    return df

# Function to save results
def save_results(results_df, output_dir, dataset_columns):
    os.makedirs(output_dir, exist_ok=True)

    formatted_results_df = results_df.drop(columns=['average_c_index', 'std_c_index'])
    formatted_results_df[dataset_columns] = formatted_results_df[dataset_columns].apply(
        lambda col: col.map(lambda x: f"${x:.3f}$" if pd.notnull(x) else "--"))

    caption = (f"C-index of each ensemble algorithm combined with {CLINICAL_LABEL}. Algorithm predictions and "
               f"{CLINICAL_LABEL} are standardised with RUMC calibration-set (RUMC Tuning + RUMC Internal Validation) "
               "statistics; Cox models are fit on the calibration set and frozen (RUMC is part of the calibration set). "
               "Average C-index is the mean ($\\pm$ SD) across datasets.")
    latex_path = os.path.join(output_dir, f'{OUTPUT_NAME}.tex')
    csv_path = os.path.join(output_dir, f'{OUTPUT_NAME}.csv')
    print(formatted_results_df.to_string(index=False))

    with open(latex_path, 'w') as f:
        f.write(formatted_results_df.to_latex(index=False, escape=False, caption=caption,
                                              label=f"tab:{OUTPUT_NAME}"))
    results_df.to_csv(csv_path, index=True)
    print(f"Saved {csv_path} and {latex_path}")

# Main function
def main(config_path):
    config = load_config(config_path)
    predictions = load_predictions(config)
    results = compute_c_index(predictions, config)
    results_df = format_results(results)
    dataset_columns = [config['dataset_names'][next(iter(d))] for d in config['datasets']]
    save_results(results_df, config['output_dir'], dataset_columns)

if __name__ == '__main__':
    config_path = "/Users/khrystynafaryna/Documents/leopard-rebuttal/config.yaml"
    main(config_path)
