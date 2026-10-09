# Source: run.ipynb, cell 12
# As team_plus_capra_s_c_index.py, plus HR (95% CI) and p-value of the prediction.
import os
import json
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import yaml

# Same as the cell above, but the per-dataset Cox model adjusts for CAPRA-S instead of ISUP (prediction + CAPRA-S):
# C-index plus the HR (95% CI) and Wald p-value of the prediction in that model. Predictions are inverted (higher = higher risk) and
# prediction/CAPRA-S are z-scored per dataset so HRs are per 1 SD and comparable across teams;
# this changes neither the C-index nor the p-values
COVARIATES = ['prediction', 'capra_s_score']
CLINICAL_LABEL = 'CAPRA-S score'
OUTPUT_NAME = 'c_index_hr_model_capra_s_combined_results_median'

# Function to load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Function to load ground truth
def load_ground_truth(dataset, ground_truth_path):
    file_path = f"{ground_truth_path}{dataset}_capra_s_median.csv"
    return pd.read_csv(file_path, dtype={"case_id": str})

# Function to load predictions (only datasets with the expected number of cases, as above)
def load_predictions(input_dir, teams, datasets):
    predictions = {}
    for team in teams:
        predictions[team] = {}
        for dataset_dict in datasets:
            dataset, expected = next(iter(dataset_dict.items()))
            dataset_path = os.path.join(input_dir, team, dataset)
            
            if len(os.listdir(dataset_path)) != expected:
                print(f"Dataset {dataset} predictions for team {team} are incomplete")
             
            team_dataset_preds = {}
            for file_name in os.listdir(dataset_path):
                if file_name.endswith('.json'):
                    with open(os.path.join(dataset_path, file_name), 'r') as f:
                        team_dataset_preds[file_name.replace('.json', '')] = json.load(f)
            if team_dataset_preds:
                predictions[team][dataset] = team_dataset_preds
    return predictions

# Function to fit the per-dataset Cox model and collect C-index, HRs and p-values (one row per team/dataset)
def compute_cox_metrics(predictions, datasets, ground_truth_path, official_team_names, official_dataset_names):
    rows = []
    for dataset_dict in datasets:
        dataset = next(iter(dataset_dict))
        ground_truth = load_ground_truth(dataset, ground_truth_path)
        for team, team_data in predictions.items():
            preds = team_data.get(dataset)
          
            data = ground_truth[ground_truth['case_id'].isin(preds.keys())].copy()
        
            data['prediction'] = -data['case_id'].map(preds)
            data = data[COVARIATES + ['event', 'follow_up_years']]
            data[COVARIATES] = (data[COVARIATES] - data[COVARIATES].mean()) / data[COVARIATES].std(ddof=1)

            cph = CoxPHFitter().fit(data, duration_col='follow_up_years', event_col='event')
            c_index = concordance_index(data['follow_up_years'], -cph.predict_partial_hazard(data), data['event'])

            row = {'Team': official_team_names[team], 'Dataset': official_dataset_names[dataset],
                   'n': len(data), 'events': int(data['event'].sum()), 'c_index': c_index}
            for cov in COVARIATES:
                hr, lower, upper, p = cph.summary.loc[cov, ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
                row.update({f'hr_{cov}': hr, f'hr_{cov}_lower': lower, f'hr_{cov}_upper': upper, f'p_{cov}': p})
            rows.append(row)
    return pd.DataFrame(rows)

def fmt_hr(hr, lower, upper, p, latex=False):
    if not latex:
        return f"{hr:.2f} [{lower:.2f}, {upper:.2f}]"
    stars = '*' * sum(p < t for t in (0.05, 0.01, 0.001))
    superscript = f"^{{{stars}}}" if stars else ''
    return f"${hr:.2f}{superscript}_{{[{lower:.2f},{upper:.2f}]}}$"

def fmt_p(p):
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"

# Function to format results: per dataset C-index, HR (95% CI) and p of the prediction, plus average C-index.
# In LaTeX the p-values are shown as stars on the HR so the table fits the page width (exact values are in the CSV)
def format_results(metrics, dataset_order, latex=False):
    hr_column = 'HR (95\\% CI)' if latex else 'HR (95% CI)'
    sub_columns = ['C-index', hr_column] + ([] if latex else ['p'])
    formatted = metrics.assign(**{
        'C-index': metrics['c_index'].map(lambda x: f"${x:.3f}$" if latex else f"{x:.3f}"),
        hr_column: [fmt_hr(*v, latex=latex) for v in
                    metrics[['hr_prediction', 'hr_prediction_lower', 'hr_prediction_upper', 'p_prediction']].itertuples(index=False)],
        'p': metrics['p_prediction'].map(fmt_p),
    })
    dataset_order = [d for d in dataset_order if d in set(metrics['Dataset'])]
    df = (formatted.pivot(index='Team', columns='Dataset', values=sub_columns)
          .swaplevel(axis=1)
          .reindex(columns=pd.MultiIndex.from_product([dataset_order, sub_columns]))
          .fillna('--'))

    # Average and standard deviation of C-index across datasets
    c_index = metrics.pivot(index='Team', columns='Dataset', values='c_index')
    average, std = c_index.mean(axis=1), c_index.std(axis=1)
    df[('Average C-index', '')] = [f"${a:.3f} _{{(\\pm {s:.3f})}}$" if latex else f"{a:.3f} (± {s:.3f})"
                                   for a, s in zip(average[df.index], std[df.index])]

    # Sort by average C-index
    df = df.loc[average.sort_values(ascending=False).index]
    df.index.name = 'Team'
    return df.reset_index()

# Function to write the LaTeX table, sized to the page width like the manuscript's other wide tables
def to_latex_table(df):
    datasets = list(dict.fromkeys(d for d, _ in df.columns[1:-1]))
    n_sub = (len(df.columns) - 2) // len(datasets)
    caption = (f"C-index of per-cohort Cox models with the AI prediction and {CLINICAL_LABEL}, and hazard ratio (HR) of the "
               f"AI prediction in that model. The subscripts indicate 95\\% CIs. AI prediction and {CLINICAL_LABEL} are "
               "standardised within each cohort; HRs are per 1 SD increase in predicted risk. "
               "Wald test: * $p<0.05$, ** $p<0.01$, *** $p<0.001$. Average C-index: mean ($\\pm$ SD) across cohorts.")
    lines = [
        '\\begin{table}[!htbp]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{4pt}',
        f'\\caption{{{caption}}}', f'\\label{{tab:{OUTPUT_NAME}}}',
        '\\resizebox{\\textwidth}{!}{%',
        f'\\begin{{tabular}}{{l{"c" * (len(df.columns) - 1)}}}',
        '\\toprule',
        ' & '.join([''] + [f'\\multicolumn{{{n_sub}}}{{c}}{{{d}}}' for d in datasets] + ['']) + ' \\\\',
        ' '.join(f'\\cmidrule(lr){{{2 + i * n_sub}-{1 + (i + 1) * n_sub}}}' for i in range(len(datasets))),
        ' & '.join(['Team'] + [sub for _, sub in df.columns[1:-1]] + ['Average C-index']) + ' \\\\',
        '\\midrule',
        *[' & '.join(map(str, row)) + ' \\\\' for row in df.itertuples(index=False)],
        '\\bottomrule', '\\end{tabular}}', '\\end{table}',
    ]
    return '\n'.join(lines) + '\n'

# Function to save results
def save_results(metrics, latex_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{OUTPUT_NAME}.tex'), 'w') as f:
        f.write(to_latex_table(latex_df))
    metrics.to_csv(os.path.join(output_dir, f'{OUTPUT_NAME}.csv'), index=False)

# Main function
def main(config_path):
    config = load_config(config_path)
    predictions = load_predictions(config['input_dir'], config['teams'], config['datasets'])
    metrics = compute_cox_metrics(predictions, config['datasets'], config['clinical_variables'],
                                  config['team_names'], config['dataset_names'])
    dataset_order = [config['dataset_names'][next(iter(d))] for d in config['datasets']]
    with pd.option_context('display.width', None, 'display.max_columns', None):
        print(format_results(metrics, dataset_order).to_string(index=False))
    save_results(metrics, format_results(metrics, dataset_order, latex=True), config['output_dir'])

if __name__ == '__main__':
    config_path = "/Users/khrystynafaryna/Documents/leopard-rebuttal/config.yaml"
    main(config_path)
