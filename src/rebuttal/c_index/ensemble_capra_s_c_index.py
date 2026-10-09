import os
import json
import logging
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import permutation_test
from sklearn.utils import resample
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import zscore

# Configuration
np.random.seed(1)
random = __import__('random')
random.seed(1)
logging.basicConfig(level=logging.INFO)

# Utility Functions
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_ground_truth(dataset, ground_truth_path):
    file_path = os.path.join(ground_truth_path, f"{dataset}_capra_s_median.csv")
    return pd.read_csv(file_path, dtype={"case_id": str})[['case_id', 'event', 'follow_up_years', 'capra_s_score']]

def zscore_normalize_dict(data):
    values = np.array(list(data.values()))
    std = values.std(ddof=1)
    mean = values.mean()
    return {k: 0 if std == 0 else (v - mean) / std for k, v in data.items()}

def invert_pred_dict(data):
    return {k: -v for k, v in data.items()}

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
                logging.warning(f"{team}/{dataset} expected {expected} preds, found {len(files)}")

            team_preds = {}
            for fn in files:
                cid = fn[:-5]
                with open(os.path.join(dataset_path, fn)) as f:
                    team_preds[cid] = json.load(f)

            if team_preds:
                predictions[team][dataset] = invert_pred_dict(team_preds)
    return predictions

def load_tuning_predictions(cfg, predictions):
    """Add the RUMC Tuning split (<validation_dir>/<team>/<tuning_predictions_subfolder>) under cfg['tuning_dataset']."""
    for team in predictions:
        path = os.path.join(cfg['validation_dir'], team, cfg['tuning_predictions_subfolder'])
        team_preds = {}
        for fn in os.listdir(path):
            if fn.endswith('.json'):
                with open(os.path.join(path, fn)) as f:
                    team_preds[fn[:-5]] = json.load(f)
        predictions[team][cfg['tuning_dataset']] = invert_pred_dict(team_preds)
    return predictions

def calculate_p_value_permutation(events, times, preds1, preds2, n_permutations=1000, random_state=1):
    def c_index_diff(data1, data2):
        c1 = concordance_index(times, data1, events)
        c2 = concordance_index(times, data2, events)
        return c1 - c2

    result = permutation_test(
        (preds1, preds2),
        statistic=c_index_diff,
        permutation_type='samples',
        n_resamples=n_permutations,
        alternative='two-sided',
        random_state=random_state
    )

    observed_diff = c_index_diff(preds1, preds2)
    p_value = result.pvalue
    return observed_diff, p_value, result.null_distribution

def bootstrap_c_index(events, times, predictions, n_bootstraps):
    original_c_index = concordance_index(times, predictions, events)
    c_index_bootstrap = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        indices = resample(range(len(events)), replace=True, n_samples=len(events))
        c_index_bootstrap[i] = concordance_index(times[indices], predictions[indices], events[indices])

    return original_c_index, *np.percentile(c_index_bootstrap, [2.5, 97.5])

def compute_global_cox_metrics(calib_df, eval_df):
    # Cox models are fit on the RUMC calibration set and frozen; C-indexes are computed on the evaluation datasets
    cal = calib_df[['prediction', 'capra_s_score', 'follow_up_years', 'event']]
    df = eval_df[['prediction', 'capra_s_score', 'follow_up_years', 'event']].copy()

    # AI ensemble only
    data = df[['prediction', 'follow_up_years', 'event']]
    cph = CoxPHFitter().fit(cal[['prediction', 'follow_up_years', 'event']], 'follow_up_years', 'event')
    c_index, c_index_l, c_index_u = bootstrap_c_index(data['event'], data['follow_up_years'], -cph.predict_partial_hazard(data), 1000)

    # CAPRA-S only
    data_c = df[['capra_s_score', 'follow_up_years', 'event']]
    cph_c = CoxPHFitter().fit(cal[['capra_s_score', 'follow_up_years', 'event']], 'follow_up_years', 'event')
    c_index_c, c_index_c_l, c_index_c_u = bootstrap_c_index(data_c['event'], data_c['follow_up_years'], -cph_c.predict_partial_hazard(data_c), 1000)

    # Univariate
    uni = CoxPHFitter().fit(cal[['prediction', 'follow_up_years', 'event']], 'follow_up_years', 'event')
    hr_u_p = uni.hazard_ratios_['prediction']
    lo_u_p, hi_u_p, p_u_p = uni.summary.loc['prediction', ['exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]

    uni_c = CoxPHFitter().fit(cal[['capra_s_score', 'follow_up_years', 'event']], 'follow_up_years', 'event')
    hr_u_c = uni_c.hazard_ratios_['capra_s_score']
    lo_u_c, hi_u_c, p_u_c = uni_c.summary.loc['capra_s_score', ['exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]

    # Multivariate
    multi = CoxPHFitter().fit(cal, 'follow_up_years', 'event')
    hr_m_p, lo_m_p, hi_m_p, p_m_p = multi.summary.loc['prediction', ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
    hr_m_c, lo_m_c, hi_m_c, p_m_c = multi.summary.loc['capra_s_score', ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]

    c_index_ai_capra, c_index_ai_capra_l, c_index_ai_capra_u = bootstrap_c_index(
        df['event'], df['follow_up_years'], -multi.predict_partial_hazard(df), 1000)

    original_diff, p_val, null_diffs = calculate_p_value_permutation(
        df['event'], df['follow_up_years'],
        -multi.predict_partial_hazard(df),
        -cph_c.predict_partial_hazard(data_c)
    )

    plt.hist(null_diffs, bins=30, alpha=0.7)
    plt.axvline(original_diff, color='red', linestyle='--', linewidth=2,
                label=f'Observed Δ = {original_diff:.3f}')
    plt.legend()
    plt.title('Null distribution of Δ under permutation')
    plt.xlabel('Δ (C-index₁ − C-index₂)')
    plt.ylabel('Frequency')
    plt.show()

    return {
        'c_index': c_index, 'c_index_l': c_index_l, 'c_index_u': c_index_u,
        'c_index_c': c_index_c, 'c_index_c_l': c_index_c_l, 'c_index_c_u': c_index_c_u,
        'c_index_ai_capra': c_index_ai_capra, 'c_index_ai_capra_l': c_index_ai_capra_l, 'c_index_ai_capra_u': c_index_ai_capra_u,
        'hr_u_p': hr_u_p, 'lo_u_p': lo_u_p, 'hi_u_p': hi_u_p, 'p_u_p': p_u_p,
        'hr_u_c': hr_u_c, 'lo_u_c': lo_u_c, 'hi_u_c': hi_u_c, 'p_u_c': p_u_c,
        'hr_m_p': hr_m_p, 'lo_m_p': lo_m_p, 'hi_m_p': hi_m_p, 'p_m_p': p_m_p,
        'hr_m_c': hr_m_c, 'lo_m_c': lo_m_c, 'hi_m_c': hi_m_c, 'p_m_c': p_m_c,
        'p_value_permutation_test': p_val,
        'n': len(df)
    }

def format_global_results(vals):
    return pd.DataFrame([{
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
    }])

def save_global_results(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'total_cox_metrics_capra_s_global_rebuttal_median_nat_com.csv'), index=False)
    caption = f"Cox PH models fit on the RUMC calibration set, evaluated on combined external datasets, RUMC excluded (n={df['n'].max()} total cases)."
    with open(os.path.join(output_dir, 'total_cox_metrics_capra_s_global_rebuttal_median_nat_com.tex'), 'w') as f:
        f.write(df.drop(columns='n').to_latex(escape=False, caption=caption, label="tab:global_cox"))

def compute_ensemble_coxph(predictions, datasets, ground_truth_path, cfg):
    # RUMC calibration set (RUMC Tuning + RUMC Internal Validation), as in calibration.ipynb:
    # normalisation and Cox models are fit on it and frozen; the evaluation datasets are only scored
    calib_datasets = [cfg['tuning_dataset'], 'radboud']
    eval_datasets = [next(iter(d)) for d in datasets if next(iter(d)) not in calib_datasets]  # PLCO, IMP, UHC; RUMC excluded
    all_events, all_times, all_preds, all_capra_s, all_case_ids, all_ds = [], [], [], [], [], []

    for ds in dict.fromkeys(calib_datasets + eval_datasets):
        gt = load_ground_truth(ds, ground_truth_path)
        print('clinical variables:',gt.head())

        combined = {}
        for team, tdata in predictions.items():
            for cid, pred in tdata.get(ds, {}).items():
                combined.setdefault(cid, []).append(pred)

        if not combined:
            continue

        cids = [cid for cid in combined if cid in set(gt['case_id'])]
        if not cids:
            continue

        preds = np.array([combined[cid] for cid in cids])
        sub = gt.set_index('case_id').loc[cids]
        case_ids = sub.index.values
        all_case_ids.extend(case_ids)
        all_events.extend(sub['event'].values)
        all_times.extend(sub['follow_up_years'].values)
        all_capra_s.extend(sub['capra_s_score'].values)
        all_preds.extend(preds)
        all_ds.extend([ds] * len(cids))

    raw_preds = np.array(all_preds)
    calib = np.isin(all_ds, calib_datasets)
    evaluated = np.isin(all_ds, eval_datasets)
    # per-team z-score with calibration-set mean/SD
    stds = raw_preds[calib].std(axis=0, ddof=1)
    stds[stds == 0] = 1  # Prevent division by zero
    norm_preds = ((raw_preds - raw_preds[calib].mean(axis=0)) / stds).mean(axis=1)

    df = pd.DataFrame({
        'case_id': all_case_ids,
        'prediction': norm_preds,
        'event': all_events,
        'follow_up_years': all_times,
        'capra_s_score': all_capra_s
    })
    # Cox covariates to zero mean / unit variance with calibration-set statistics
    for col in ['prediction', 'capra_s_score']:
        df[col] = (df[col] - df.loc[calib, col].mean()) / df.loc[calib, col].std(ddof=1)

    return compute_global_cox_metrics(df[calib].reset_index(drop=True), df[evaluated].reset_index(drop=True))

def main(config_path):
    print("main")
    cfg = load_config(config_path)
    print("read config")
    preds = load_predictions(cfg['input_dir'], cfg['ensemble_teams'], cfg['datasets'])
    print("read preds")
    preds = load_tuning_predictions(cfg, preds)
    print("read tuning npreds")
    raw_results = compute_ensemble_coxph(preds, cfg['datasets'], cfg['clinical_variables'], cfg)
    print("computed cox")
    formatted_results = format_global_results(raw_results)
    save_global_results(formatted_results, cfg['output_dir'])
    print(formatted_results)

if __name__ == '__main__':
    
    config_path = "/Users/khrystynafaryna/Documents/leopard-rebuttal/config.yaml"
    main(config_path)
