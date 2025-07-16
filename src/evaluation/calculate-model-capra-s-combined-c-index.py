import os
import json
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import yaml
from sklearn.utils import resample

# Function to load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to load ground truth
def load_ground_truth(dataset, ground_truth_path):
    file_path = f"{ground_truth_path}{dataset}_clinical_standardized_capra_s_postsubmission.csv"
    return pd.read_csv(file_path, dtype={"case_id": str})

# Function to get the number of unique cases
def get_unique_case_count(df, case_column='case_id'):
    return df[case_column].nunique()

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
                print(f"Dataset {dataset} predictions for team {team} are incomplete")
            if team_dataset_preds:
                predictions[team][dataset] = team_dataset_preds
    return predictions

# Function to compute C-index
def compute_c_index(predictions, datasets, ground_truth_path, official_team_names, official_dataset_names):
    results = {}
    for dataset_dict in datasets:
        dataset = list(dataset_dict.keys())[0]
        official_dataset_name = official_dataset_names[dataset]
        ground_truth = load_ground_truth(dataset, ground_truth_path)
        for team, team_data in predictions.items():
            
            

            if dataset not in team_data or not team_data[dataset]:
                continue
            official_team_name = official_team_names[team]
            
            
            preds = team_data[dataset]
            ground_truth_filtered = ground_truth[ground_truth['case_id'].isin(preds.keys())].copy()
            if ground_truth_filtered.empty:
                continue
            ground_truth_filtered['prediction'] = ground_truth_filtered['case_id'].map(preds)
            data = ground_truth_filtered[['prediction', 'event', 'follow_up_years', 'capra_s_score']]
            num_cases = get_unique_case_count(ground_truth_filtered)
            cph = CoxPHFitter()
            cph.fit(data, duration_col='follow_up_years', event_col='event')
            c_index = concordance_index(data['follow_up_years'], -cph.predict_partial_hazard(data), data['event'])
            if official_team_name not in results:
                results[official_team_name] = {}
            results[official_team_name][official_dataset_name] = c_index
    return results

# Function to format results into a DataFrame
def format_results(results):
    df = pd.DataFrame(results).T  # Transpose so that teams are rows
    print(df)

    # Add "Team" column by resetting the index
    df = df.reset_index().rename(columns={'index': 'Team'})

    #print(df)

    # Calculate average and standard deviation of C-index across datasets
    dataset_columns = df.columns.drop('Team')  # All dataset columns, excluding 'Team'
    print('1',df[dataset_columns])
    df['average_c_index'] = df[dataset_columns].mean(axis=1)
    print(df['average_c_index'] )
    
    df['std_c_index'] = df[dataset_columns].std(axis=1)
    print('2',df[dataset_columns])
    print(df['std_c_index'] )

    # Create formatted column "average (+/- std)"
    df['Average C-index'] = (
        '$' + df['average_c_index'].round(3).astype(str) +
        ' _{(\pm ' + df['std_c_index'].round(3).astype(str) + ')}$'
    )

    # Sort by average C-index
    df = df.sort_values(by='average_c_index', ascending=False).reset_index(drop=True)
    return df


# Function to save results
def save_results(results_df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # List of columns to format
    columns_to_format = ["RUMC", "PLCO","IMP","UHC"]  # replace with your actual column names

    # Apply formatting only to specified columns
    formatted_results_df = results_df.copy().drop(columns=['average_c_index', 'std_c_index'])  # Avoid modifying original DataFrame
   

    formatted_results_df[columns_to_format] = formatted_results_df[columns_to_format].applymap(
        lambda x: f"${x:.3f}$" if pd.notnull(x) else "--")
    
    latex_path = os.path.join(output_dir, 'c_index_model_capra_s_combined_results.tex')
    csv_path = os.path.join(output_dir, 'c_index_model_capra_s_combined_results.csv')
    print(formatted_results_df)
    
    with open(latex_path, 'w') as f:
        f.write(formatted_results_df.to_latex(index=True, escape=False))
    
    results_df.to_csv(csv_path, index=True)

# Main function
def main(config_path):
    config = load_config(config_path)
    predictions = load_predictions(config['input_dir'], config['teams'], config['datasets'])
    results = compute_c_index(predictions, config['datasets'], config['clinical_variables'],config['team_names'],config['dataset_names'])
    results_df = format_results(results)
    #print(results_df)
    save_results(results_df, config['output_dir'])

if __name__ == '__main__':
    import sys
    config_path = "/data/temporary/leopard/source/evaluation/pathology-leopard-evaluation/config/config.yaml"
    main(config_path)
