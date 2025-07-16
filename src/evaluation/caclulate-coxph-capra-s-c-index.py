import yaml
import pandas as pd
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from pathlib import Path

# Load configuration from YAML file
def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to get the number of unique cases
def get_unique_case_count(df, case_column='case_id'):
    return df[case_column].nunique()

# Compute C-index
def compute_c_index(df, isup_col, time_col, event_col):
    data = df[[isup_col, time_col, event_col]]
        

    # Fit the Cox Proportional Hazards model
    cph = CoxPHFitter()
    cph.fit(data, duration_col='follow_up_years', event_col='event')

    # Display the summary of the Cox model
    #print(cph.summary)

    # Calculate the concordance index (C-index)
    c_index = concordance_index(data['follow_up_years'], -cph.predict_partial_hazard(data), data['event'])

   
    return c_index#concordance_index(df[time_col], -df[isup_col], df[event_col])

# Main function
def main(config_path, output_csv):
    # Load config
    config = load_config(config_path)
    dataset_dict = config['datasets']
    datasets = [list(i.keys())[0] for i in dataset_dict]
    print(datasets)
    ground_truth_path = config.get("clinical_variables", "")
    
    results = []
    
    for dataset in datasets:
        dataset_path = f"{ground_truth_path}{dataset}_clinical_standardized_capra_s_postsubmission.csv"
        

        df = pd.read_csv(dataset_path)
        
        num_cases = get_unique_case_count(df)
        c_index = compute_c_index(df, "capra_s_score", "follow_up_years", "event")
        print(num_cases,dataset,c_index)
        results.append({"dataset": dataset, "c_index": c_index,"#cases":num_cases})

    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(config['output_dir']+'/'+output_csv), index=False)
    print(f"Results saved to {output_csv}")

# Run the script
if __name__ == "__main__":
    config_file = "/data/temporary/leopard/source/evaluation/pathology-leopard-evaluation/config/config.yaml"  # Adjust path as needed
    output_file = "c_index_capra_results.csv"
    main(config_file, output_file)
