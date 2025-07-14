import os
import json
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from scipy.stats import bootstrap
import yaml
from sklearn.utils import resample
import matplotlib.pyplot as plt
from scipy.stats import zscore

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
                continue  # Skip if directory does not exist
            team_dataset_preds = {}
            

            if len(os.listdir(dataset_path)) == list(dataset_dict.values())[0]:
                for file_name in os.listdir(dataset_path):
                    if file_name.endswith('.json'):
                        case_id = file_name.replace('.json', '')
                        with open(os.path.join(dataset_path, file_name), 'r') as f:
                            team_dataset_preds[case_id] = json.load(f)
            else:
                print(f"Dataset {dataset} predictions for team {team} are incomplete, expected {list(dataset_dict.values())[0]} files, found {len(os.listdir(dataset_path))} files")
            #print('loaded dataset',dataset)
            if team_dataset_preds:  # Ensure predictions are not empty
                #print(team_dataset_preds.shape)
                
                predictions[team][dataset] = team_dataset_preds
            #print('predictions[team][dataset]',len(predictions[team][dataset]))
    return predictions

# Function to compute ensemble and individual C-indexes
def compute_c_indexes(predictions, datasets, ground_truth_path):
    results = []
    for dataset_dict in datasets:
        dataset = list(dataset_dict.keys())[0]
        ground_truth = load_ground_truth(dataset, ground_truth_path)
        ground_truth = ground_truth.sort_values(by='case_id', ascending=True)

        valid_case_ids = ground_truth['case_id'].tolist()
        #print('valid_case_ids',valid_case_ids)
        all_preds = {}
        team_c_indexes = {}

        for team, team_data in predictions.items():
            if dataset not in team_data or not team_data[dataset]:
                continue
            
            preds = team_data[dataset]
            #print("preds",preds)
       
            #case_ids = [case_id for case_id in preds.keys() if case_id in valid_case_ids]
            case_values_bf = np.array([preds[case_id] for case_id in valid_case_ids])
            print("case_values_bf")
            plt.hist(case_values_bf)
            plt.show()
        
        
            case_values = zscore(np.array([preds[case_id] for case_id in valid_case_ids]))
            print("case_values")
            plt.hist(case_values)
            plt.show()
        

            #print("case_values",case_values)
            if len(case_values) == 0:
                continue
            
            for case_id, value in zip(valid_case_ids, case_values):
                #print('in  loop',case_id, value)
                if case_id not in all_preds:
                    all_preds[case_id] = []
                all_preds[case_id].append(value)
            #print("all_preds",all_preds)

            ground_truth_filtered = ground_truth[ground_truth['case_id'].isin(valid_case_ids)].copy()
            if not ground_truth_filtered.empty:
                c_index, ci_lower, ci_upper = bootstrap_c_index(
                    ground_truth_filtered['event'].values,
                    ground_truth_filtered['follow_up_years'].values,
                    case_values,
                    n_bootstraps=1000
                )
                team_c_indexes[team] = {'c_index': c_index, 'ci_lower': ci_lower, 'ci_upper': ci_upper}

        if not all_preds:
            continue

        #case_ids = list(all_preds.keys())
        ensemble_risks=[]

        ensemble_risks = np.array([np.mean(pred_list) for pred_list in all_preds.values()])
        ground_truth_filtered = ground_truth[ground_truth['case_id'].isin(valid_case_ids)].copy()
        if ground_truth_filtered.empty:
            continue
        
        ground_truth_filtered['ensemble_risk'] = ensemble_risks
        num_cases = len(valid_case_ids)

        # Compute ensemble C-index
        ensemble_c_index, ensemble_ci_lower, ensemble_ci_upper = bootstrap_c_index(
            ground_truth_filtered['event'].values,
            ground_truth_filtered['follow_up_years'].values,
            ground_truth_filtered['ensemble_risk'].values,
            n_bootstraps=1000
        )

        result_entry = {
            '#cases': num_cases,
            'dataset': dataset,
            'ensemble_c_index': ensemble_c_index,
            'ci_lower': ensemble_ci_lower,
            'ci_upper': ensemble_ci_upper
        }

        for team, metrics in team_c_indexes.items():
            result_entry[f'{team}_c_index'] = metrics['c_index']
            result_entry[f'{team}_ci_lower'] = metrics['ci_lower']
            result_entry[f'{team}_ci_upper'] = metrics['ci_upper']
        
        results.append(result_entry)

    return pd.DataFrame(results) if results else pd.DataFrame(columns=['dataset', 'ensemble_c_index', 'ci_lower', 'ci_upper'])

# Function to compute bootstrap C-index
def bootstrap_c_index(events, times, predictions, n_bootstraps):
    original_c_index = concordance_index(times, predictions, events)
    c_index_bootstrap = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        indices = resample(range(len(events)), replace=True, n_samples=len(events))
        c_index_bootstrap[i] = concordance_index(times[indices], predictions[indices], events[indices])

    ci_lower, ci_upper = np.percentile(c_index_bootstrap, [2.5, 97.5])
    return original_c_index, ci_lower, ci_upper

# Function to save results
def save_results(results, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_path = os.path.join(output_dir, 'c_index_ensemble_results.csv')
    latex_path = os.path.join(output_dir, 'c_index_ensemble_results.tex')
    
    results.to_csv(csv_path, index=False)
    with open(latex_path, 'w') as f:
        f.write(results.to_latex(index=False))

# Main function
def main(config_path):
    config = load_config(config_path)
    predictions = load_predictions(config['input_dir'], config['ensemble_teams'], config['datasets'])
    results = compute_c_indexes(predictions, config['datasets'], config['ground_truth_path'])
    print(results)
    save_results(results, config['output_dir'])

if __name__ == '__main__':
    import sys
    config_path = "/data/temporary/leopard/source/evaluation/pathology-leopard-evaluation/config/config.yaml"
    main(config_path)
