import os 
import json
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from scipy.stats import bootstrap
import yaml
from sklearn.utils import resample

# Function to load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to load ground truth
def load_ground_truth(dataset, ground_truth_path):
    file_path = f"{ground_truth_path}{dataset}.csv"
    return pd.read_csv(file_path, dtype={"case_id": str})

# Function to load predictions
def load_predictions(input_dir, teams, datasets):
    predictions = {}
    for team in teams:
        predictions[team] = {}
        for dataset_dict in datasets:
            dataset = list(dataset_dict.keys())[0]
            dataset_path = os.path.join(input_dir, team, dataset)
            if not os.path.exists(dataset_path):
                continue
            team_dataset_preds = {}
            
            if len(os.listdir(dataset_path)) == list(dataset_dict.values())[0]:
                for file_name in os.listdir(dataset_path):
                    if file_name.endswith('.json'):
                        case_id = file_name.replace('.json', '')
                        with open(os.path.join(dataset_path, file_name), 'r') as f:
                            team_dataset_preds[case_id] = json.load(f)
            else:
                print(f"Dataset {dataset} predictions for team {team} are incomplete, expected {list(dataset_dict.values())[0]} files, found {len(os.listdir(dataset_path))} files")
            
            if team_dataset_preds:
                predictions[team][dataset] = team_dataset_preds
    return predictions

# Function to compute C-index and standard deviation
def compute_c_index(predictions, datasets, ground_truth_path, official_team_names, official_dataset_names):
    results = {}
    results_csv = {}
    teams = set(predictions.keys())
    dataset_names = [list(dataset_dict.keys())[0] for dataset_dict in datasets]

    
    for team in teams:
        print('official_team_names',official_team_names)
        
        official_team_name = official_team_names[team]
        
        results[official_team_name] = {'Team': official_team_name}
        results_csv[official_team_name] = {'Team': official_team_name}
        c_index_list = []
        
        for dataset in dataset_names:
            official_dataset_name = official_dataset_names[dataset]
            if dataset not in predictions[team]:
                results[official_team_name][official_dataset_name] = 'N/A'
                results_csv[official_team_name][official_dataset_name] = 'N/A'
                
                continue
            
            preds = predictions[team][dataset]
            ground_truth = load_ground_truth(dataset, ground_truth_path)
            ground_truth_filtered = ground_truth[ground_truth['case_id'].isin(preds.keys())].copy()
            
            if ground_truth_filtered.empty:
                results[official_team_name][official_dataset_name] = 'N/A'
                results_csv[official_team_name][official_dataset_name] = 'N/A'
                continue
            
            ground_truth_filtered['prediction'] = ground_truth_filtered['case_id'].map(preds)
            c_index, ci_lower, ci_upper = bootstrap_c_index(
                events=ground_truth_filtered['event'].values, 
                times=ground_truth_filtered['follow_up_years'].values, 
                predictions=ground_truth_filtered['prediction'].values, 
                n_bootstraps=1000
            )
            ci_formatted = f"${c_index:.3f}_{{({ci_lower:.3f}; {ci_upper:.3f})}}$"
            results[official_team_name][official_dataset_name] = ci_formatted
            results_csv[official_team_name][official_dataset_name+"_c_index"] = c_index
            results_csv[official_team_name][official_dataset_name+"_ci_upper"] = ci_upper
            results_csv[official_team_name][official_dataset_name+"_ci_lower"] = ci_lower
            c_index_list.append(c_index)
        
        # Compute average C-index and standard deviation
        if c_index_list:
            avg_c_index = np.mean(c_index_list)
            std_c_index = np.std(c_index_list, ddof=1)  # Sample standard deviation
            results[official_team_name]['Average C-index'] = f"${avg_c_index:.3f}_{{(\pm {std_c_index:.3f})}}$"
            results_csv[official_team_name]['Average C-index'] = avg_c_index
            results_csv[official_team_name]['stdev_c_index'] = std_c_index
        else:
            results[official_team_name]['Average C-index'] = 'N/A'
            results_csv[official_team_name]['Average C-index'] = 'N/A'
            results_csv[official_team_name]['stdev_c_index'] = 'N/A'
    
    df_results = pd.DataFrame.from_dict(results, orient='index').reset_index(drop=True)
    
    # Sort teams by average C-index (ignoring 'N/A' values)
    df_results = df_results[df_results['Average C-index'] != 'N/A']
    df_results = df_results.sort_values(by='Average C-index', ascending=False)
    
    df_results_csv = pd.DataFrame.from_dict(results_csv, orient='index').reset_index(drop=True)
    
    # Sort teams by average C-index (ignoring 'N/A' values)
    df_results_csv = df_results_csv[df_results_csv['Average C-index'] != 'N/A']
    df_results_csv = df_results_csv.sort_values(by='Average C-index', ascending=False)
    
    return df_results, df_results_csv

# Function to compute bootstrap confidence intervals
def bootstrap_c_index(events, times, predictions, n_bootstraps):
    original_c_index = concordance_index(times, predictions, events)
    c_index_bootstrap = np.zeros(n_bootstraps)
    
    for i in range(n_bootstraps):
        indices = resample(range(len(events)), replace=True, n_samples=len(events))
        c_index_bootstrap[i] = concordance_index(times[indices], predictions[indices], events[indices])
    
    ci_lower, ci_upper = np.percentile(c_index_bootstrap, [2.5, 97.5])
    return original_c_index, ci_lower, ci_upper

# Function to save results
def save_results(results, results_csv, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_path = os.path.join(output_dir, 'c_index_results.csv')
    latex_path = os.path.join(output_dir, 'c_index_results.tex')
    
    
    
    with open(latex_path, 'w') as f:
        latex_output = results.to_latex(index=False, escape=False)
        #latex_output = latex_output.replace("_", "\\_")  # Ensuring LaTeX-friendly formatting
        f.write(latex_output)
        
    results_csv.to_csv(csv_path, index=False)


# Main function
def main(config_path):
    config = load_config(config_path)
    predictions = load_predictions(config['input_dir'], config['teams'], config['datasets'])
    results,results_csv = compute_c_index(predictions, config['datasets'], config['ground_truth_path'],config['team_names'],config['dataset_names'])
    print(results)
    save_results(results,results_csv, config['output_dir'])

if __name__ == '__main__':
    import sys
    config_path = "/data/temporary/leopard/source/evaluation/pathology-leopard-evaluation/config/config.yaml"
    main(config_path)
