#1. addCaculate Cox PH for ensemble per dataset
import os
import json
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from scipy.stats import bootstrap
import yaml
from scipy.stats import zscore
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
from sklearn.utils import resample
# Python’s built-in RNG
import random
random.seed(1)
%matplotlib inline

# NumPy’s RNG
np.random.seed(1)

# Function to load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Function to load ground truth
def load_ground_truth(dataset, ground_truth_path):
    file_path = os.path.join(ground_truth_path, f"{dataset}_clinical_standardized_capra_s.csv")#might need to be corrected
    relevant_cols_df = pd.read_csv(file_path, dtype={"case_id": str})[['case_id','event','follow_up_years','capra_s_score']]
    return relevant_cols_df

def zscore_normalize_dict(data):
    import numpy as np

    values = np.array(list(data.values()))
    mean = values.mean()
    std = values.std()

    if std == 0:
        # Avoid division by zero
        return {k: 0 for k in data}

    normalized_data = {k: (v - mean) / std for k, v in data.items()}
    return normalized_data


def invert_pred_dict(data):
    import numpy as np

    values = np.array(list(data.values()))
  


    normalized_data = {k: v*(-1.0) for k, v in data.items()}
    return normalized_data


# Function to load predictions
def load_predictions(input_dir, teams, datasets):
    predictions = {}
    for team in teams:
        predictions[team] = {}
        for dataset_dict in datasets:
            dataset = next(iter(dataset_dict))
            dataset_path = os.path.join(input_dir, team, dataset)
            if not os.path.isdir(dataset_path):
                continue

            files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
            expected = dataset_dict[dataset]
            if len(files) != expected:
                print(f"Warning: {team}/{dataset} expected {expected} preds, found {len(files)}")

            team_preds = {}
            for fn in files:
                cid = fn[:-5]
                with open(os.path.join(dataset_path, fn)) as f:
                    team_preds[cid] = json.load(f)

            if team_preds:
                
                #print(team_preds)
                plt.hist(team_preds.values())
                plt.show()
                predictions[team][dataset] = invert_pred_dict(zscore_normalize_dict(team_preds))
                plt.hist(zscore_normalize_dict(team_preds).values())
                plt.show()
    return predictions


def calculate_p_value_permutation(
    events:     np.ndarray,
    times:      np.ndarray,
    preds1:     np.ndarray,
    preds2:     np.ndarray,
    n_permutations: int = 1000,
    random_state:   int = 1
) -> tuple:
    """
    Compare two models via a paired‐permutation test on the C‐index.

    Parameters
    ----------
    events : array-like of shape (n_samples,)
        Event indicators (1=event, 0=censored).
    times : array-like of shape (n_samples,)
        Follow-up times.
    preds1 : array-like of shape (n_samples,)
        Risk scores (or predictions) from model 1.
    preds2 : array-like of shape (n_samples,)
        Risk scores (or predictions) from model 2.
    n_permutations : int, default=1000
        Number of random permutations.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    original_diff : float
        C-index(preds1) − C-index(preds2) on the full sample.
    p_value : float
        Two‐sided p‐value for H₀: no difference, computed as
        P(|Δ*| ≥ |Δ_obs|).
    null_distribution : np.ndarray
        Array of length `n_permutations` with permuted Δ values.
    """
    rng = np.random.RandomState(random_state)

    # 1) Compute observed C‐indices and their difference
    c1_orig = concordance_index(times, preds1, events)
    c2_orig = concordance_index(times, preds2, events)
    original_diff = c1_orig - c2_orig

    # 2) Build null distribution by paired‐swap permutation
    n = len(events)
    S = np.vstack([preds1, preds2]).T  # shape (n, 2)
    null_diffs = np.zeros(n_permutations, dtype=float)

    for i in range(n_permutations):
        # for each sample, flip a fair coin: if True, swap the two model scores
        swap_mask = rng.rand(n) < 0.5
       
        permuted = S.copy()
        permuted[swap_mask] = permuted[swap_mask][:, ::-1]

        c1p = concordance_index(times, permuted[:, 0], events)
        c2p = concordance_index(times, permuted[:, 1], events)
        null_diffs[i] = c1p - c2p

    # 3) Two‐sided p‐value
    p_value = np.mean(np.abs(null_diffs) >= abs(original_diff))

    return original_diff, p_value, null_diffs

# Function to compute bootstrap confidence intervals
def bootstrap_c_index(events, times, predictions, n_bootstraps):
    original_c_index = concordance_index(times, predictions, events)
    c_index_bootstrap = np.zeros(n_bootstraps)
    
    for i in range(n_bootstraps):
        indices = resample(range(len(events)), replace=True, n_samples=len(events))
        c_index_bootstrap[i] = concordance_index(times[indices], predictions[indices], events[indices])
    
    ci_lower, ci_upper = np.percentile(c_index_bootstrap, [2.5, 97.5])
    return original_c_index, ci_lower, ci_upper

def compute_global_cox_metrics(preds_df, gt_df):
    # Merge predictions with ground truth across all datasets
    merged = pd.merge(
        preds_df,
        gt_df,
        on=['case_id'],
        how='inner'
    )
    print('preds_df:',preds_df.keys(),len(preds_df))
    print('gt_df:',gt_df.keys(),len(gt_df))
    print('merged:',merged.keys(),len(merged))
    
    results = {}

    df = merged[['prediction', 'capra_s_score', 'follow_up_years', 'event']].copy()
    
    data = df[['prediction', 'follow_up_years', 'event']]
    cph = CoxPHFitter().fit(data, 'follow_up_years', 'event')
    #c_index = concordance_index(data['follow_up_years'], -cph.predict_partial_hazard(data), data['event'])
    c_index, c_index_l, c_index_u = bootstrap_c_index(data['event'],
                                                      data['follow_up_years'], 
                                                      -cph.predict_partial_hazard(data), 
                                                      n_bootstraps=1000)
    
    data_c = df[['capra_s_score', 'follow_up_years', 'event']]
    cph_c = CoxPHFitter().fit(data_c, 'follow_up_years', 'event')
    #c_index_c = concordance_index(data_c['follow_up_years'], -cph_c.predict_partial_hazard(data_c), data_c['event'])
    c_index_c, c_index_c_l, c_index_c_u = bootstrap_c_index(data_c['event'], 
                                                            data_c['follow_up_years'], 
                                                            -cph_c.predict_partial_hazard(data_c), 
                                                            n_bootstraps=1000)
  
        
    # Univariate Cox (prediction only)
    uni = CoxPHFitter()
    uni.fit(df[['prediction', 'follow_up_years', 'event']],
            duration_col='follow_up_years',
            event_col='event')
    hr_u_p = uni.hazard_ratios_['prediction']
    lo_u_p = uni.summary.loc['prediction', 'exp(coef) lower 95%']
    hi_u_p = uni.summary.loc['prediction', 'exp(coef) upper 95%']
    p_u_p  = uni.summary.loc['prediction', 'p']
    
    uni.fit(df[['capra_s_score', 'follow_up_years', 'event']],
            duration_col='follow_up_years',
            event_col='event')
    
    hr_u_c = uni.hazard_ratios_['capra_s_score']
    lo_u_c = uni.summary.loc['capra_s_score', 'exp(coef) lower 95%']
    hi_u_c = uni.summary.loc['capra_s_score', 'exp(coef) upper 95%']
    p_u_c  = uni.summary.loc['capra_s_score', 'p']
    
    # Multivariate Cox (prediction + CAPRA-S)
    multi = CoxPHFitter()
    multi.fit(df[['prediction', 'capra_s_score', 'follow_up_years', 'event']],
              duration_col='follow_up_years',
              event_col='event')
    hr_m_p = multi.hazard_ratios_['prediction']
    lo_m_p = multi.summary.loc['prediction', 'exp(coef) lower 95%']
    hi_m_p = multi.summary.loc['prediction', 'exp(coef) upper 95%']
    p_m_p  = multi.summary.loc['prediction', 'p']

    hr_m_c = multi.hazard_ratios_['capra_s_score']
    lo_m_c = multi.summary.loc['capra_s_score', 'exp(coef) lower 95%']
    hi_m_c = multi.summary.loc['capra_s_score', 'exp(coef) upper 95%']
    p_m_c = multi.summary.loc['capra_s_score', 'p']
    
    data_ai_capra_c = df[['prediction','capra_s_score', 'follow_up_years', 'event']]
    
    merged[['case_id','prediction','capra_s_score', 'follow_up_years', 'event']].copy().to_csv(
        "/data/temporary/leopard/source/evaluation/results/ensembled_predictions_all_datasets_capra_s_gt.csv")
    

    #c_index_ai_capra = concordance_index(df['follow_up_years'],
    #-multi.predict_partial_hazard(df[['prediction', 'capra_s_score', 
    #'follow_up_years', 'event']]), df['event'])
    c_index_ai_capra, c_index_ai_capra_l, c_index_ai_capra_u = bootstrap_c_index(data_ai_capra_c['event'], 
                                                            data_ai_capra_c['follow_up_years'], 
                                                            -multi.predict_partial_hazard(data_ai_capra_c), 
                                                            n_bootstraps=1000)   
    print()
    
    
    
    original_diff_permutation_test, p_value_permutation_test, null_diffs = calculate_p_value_permutation(data_ai_capra_c['event'], 
                                                                                 data_ai_capra_c['follow_up_years'],
                                                                                 -multi.predict_partial_hazard(data_ai_capra_c),
                                                                                 #-multi.predict_partial_hazard(data_ai_capra_c), 
                                                                                 -cph_c.predict_partial_hazard(data_c), 
                                                                                 n_permutations=1000, random_state=1)
    
    print('original_diff, p_value',original_diff_permutation_test, p_value_permutation_test)
    

    plt.hist(null_diffs, bins=30, alpha=0.7)
    # Add red vertical line at the observed difference
    plt.axvline(
        original_diff_permutation_test,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Observed Δ = {original_diff_permutation_test:.3f}'
    )
    plt.legend()
    plt.xlabel('Δ (C-index₁ − C-index₂)')
    plt.ylabel('Frequency')
    plt.title('Null distribution of Δ under permutation')
    plt.show()
    print(np.shape(-cph_c.predict_partial_hazard(data_c)))
    
    plt.hist(-cph_c.predict_partial_hazard(data_c))
    plt.show()
    print(np.shape(-multi.predict_partial_hazard(data_ai_capra_c)))
    plt.hist(-multi.predict_partial_hazard(data_ai_capra_c))
    plt.show()
 
    
    n = df['case_id'].nunique() if 'case_id' in df.columns else len(df)
    results = {
        'c_index': c_index,
        'c_index_u': c_index_u,
        'c_index_l': c_index_l,
        'c_index_c': c_index_c,
        'c_index_c_u': c_index_c_u,
        'c_index_c_l': c_index_c_l,
        'c_index_ai_capra': c_index_ai_capra,
        'c_index_ai_capra_u': c_index_ai_capra_u,
        'c_index_ai_capra_l': c_index_ai_capra_l,
        'hr_u_p': hr_u_p, 'lo_u_p': lo_u_p, 'hi_u_p': hi_u_p, 'p_u_p': p_u_p,
        'hr_u_c': hr_u_c, 'lo_u_c': lo_u_c, 'hi_u_c': hi_u_c, 'p_u_c': p_u_c,
        'hr_m_p': hr_m_p, 'lo_m_p': lo_m_p, 'hi_m_p': hi_m_p, 'p_m_p': p_m_p,
        'hr_m_c': hr_m_c, 'lo_m_c': lo_m_c, 'hi_m_c': hi_m_c, 'p_m_c': p_m_c,
        'p_value_permutation_test': p_value_permutation_test,
        'n': n
    }
    return results



def format_global_results(vals):
    rows = []
    rows.append({
         
            'AI ensemble HR (univariate)':      f"${vals['hr_u_p']:.3f}_{{({vals['lo_u_p']:.3f},{vals['hi_u_p']:.3f})}}$",
            'AI ensemble p-value (univariate)': f"{vals['p_u_p']:.1e}",
            'AI ensemble HR (multivariate)':    f"${vals['hr_m_p']:.3f}_{{({vals['lo_m_p']:.3f},{vals['hi_m_p']:.3f})}}$",
            'AI ensemble p-value (multivariate)': f"{vals['p_m_p']:.1e}",
            'CAPRA-S HR (univariate)':      f"${vals['hr_u_c']:.3f}_{{({vals['lo_u_c']:.3f},{vals['hi_u_c']:.3f})}}$",
            'CAPRA-S p-value (univariate)': f"{vals['p_u_c']:.1e}",
            'CAPRA-S HR (multivariate)':    f"${vals['hr_m_c']:.3f}_{{({vals['lo_m_c']:.3f},{vals['hi_m_c']:.3f})}}$",
            'CAPRA-S p-value (multivariate)': f"{vals['p_m_c']:.1e}",
            'C-index AI ensemble': vals['c_index'],
            'C-index CI lower AI ensemble': vals['c_index_l'],
            'C-index CI upper AI ensemble': vals['c_index_u'],        
            'C-index CAPRA-S': vals['c_index_c'],
            'C-index CI lower CAPRA-S': vals['c_index_c_l'],
            'C-index CI upper CAPRA-S': vals['c_index_c_u'],
            'C-index AI Ensemble + CAPRA-S': vals['c_index_ai_capra'],
            'C-index CI lower AI Ensemble + CAPRA-S': vals['c_index_ai_capra_l'],
            'C-index CI upper AI Ensemble + CAPRA-S': vals['c_index_ai_capra_u'],
            'The p-value - H0 C-index models CAPRA-S == C-index CAPRA-S+AI ensemble': vals['p_value_permutation_test'],
            'n': vals['n']
        })
    return pd.DataFrame(rows)

def save_global_results(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # CSV
    df.reset_index().to_csv(os.path.join(output_dir, 'total_cox_metrics_capra_s_global.csv'), index=False)
    # LaTeX
    caption = f"Cox PH models on combined datasets (n={df['n'].max()} total cases)."
    with open(os.path.join(output_dir, 'total_cox_metrics_capra_s_global.tex'), 'w') as f:
        f.write(df.drop(columns='n').to_latex(escape=False, caption=caption, label="tab:global_cox"))



# Function to compute per-dataset and overall C-index
def compute_ensemble_coxph(predictions, datasets, ground_truth_path):
    per_ds_results = []
    # accumulators for overall
    all_events, all_times, all_preds, all_capra_s, all_case_ids = [], [], [], [], []

    for ds_dict in datasets:
        ds = next(iter(ds_dict))
        gt = load_ground_truth(ds, ground_truth_path)
        # collect per-case across teams
        combined = {}
        
        for team, tdata in predictions.items():
            # for each team grediction for a particular dataset loop through each case_id and  corrresponding prediction 
            for cid, pred in tdata.get(ds, {}).items():
                # for each case_id append prdection of all teams for this case
                combined.setdefault(cid, []).append(pred)

        if not combined:
            continue

        # only keep cases with ground truth
        cids = [cid for cid in combined if cid in set(gt['case_id'])]
        if not cids:
            continue

        # ensemble prediction = mean then z-score
        
        print('combined',len(combined),len(list(combined.values())[0]))
        print('combined val',len(combined),list(combined.values())[0])
        preds = np.array([np.mean(combined[cid]) for cid in cids])
        print('ensemble combined preds',preds.shape)
        print('ensemble combined preds val',preds[0])
        #preds = preds)

        #order ground truth based on cids
        
        sub = gt.set_index('case_id').loc[cids]
        case_ids = sub.index.values#sub['case_id'].values
        events = sub['event'].values
        times  = sub['follow_up_years'].values
        capra_s  = zscore(sub['capra_s_score'].values)
        
        

        # append to overall
        all_case_ids.extend(case_ids)
        all_events.extend(events)
        all_times.extend(times)
        all_preds.extend(preds)
        all_capra_s.extend(capra_s)

    gt_df = pd.DataFrame()
    preds_df = pd.DataFrame()
    preds_df['case_id'] = all_case_ids
    preds_df['prediction'] = all_preds
    gt_df['case_id'] = all_case_ids
    gt_df['event'] = all_events
    gt_df['follow_up_years'] = all_times
    gt_df['capra_s_score'] = all_capra_s
          
    results = compute_global_cox_metrics(preds_df, gt_df)
    
    return results

# Function to save results
def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results.to_csv(os.path.join(output_dir, 'all_ensemble_c_index_results.csv'), index=False)
    with open(os.path.join(output_dir, 'all_ensemble_c_index_results.tex'), 'w') as f:
        f.write(results.to_latex(index=False))

# Main function
def main(config_path):
    cfg = load_config(config_path)
    preds = load_predictions(cfg['input_dir'], cfg['ensemble_teams'], cfg['datasets'])
    results = compute_ensemble_coxph(preds, cfg['datasets'], cfg['clinical_variables'])
    print(results)
    results = format_global_results(results)
    print(results)
    save_global_results(results, cfg['output_dir'])
    

if __name__ == '__main__':

    import sys
    config_path = "/data/temporary/leopard/source/evaluation/pathology-leopard-evaluation/config/config.yaml"
    main(config_path)
