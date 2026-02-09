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

def minmax_normalize_dict(data):
    values = np.array(list(data.values()), dtype=float)
    vmin, vmax = values.min(), values.max()
    vrange = vmax - vmin
    return {k: 0.0 if vrange == 0 else (float(v) - vmin) / vrange for k, v in data.items()}

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
def get_reference_minmax(predictions, teams, reference_dataset="radboud"):
    """
    Compute per-team (column-wise) min/max using ONLY reference_dataset predictions.
    Returns (mins, maxs) aligned to `teams` order.
    """
    mins = np.full(len(teams), np.inf, dtype=float)
    maxs = np.full(len(teams), -np.inf, dtype=float)

    for j, team in enumerate(teams):
        tpreds = predictions.get(team, {}).get(reference_dataset, {})
        if not tpreds:
            raise ValueError(f"No predictions found for team '{team}' on reference dataset '{reference_dataset}'")

        vals = np.array(list(tpreds.values()), dtype=float)
        mins[j] = vals.min()
        maxs[j] = vals.max()
        print('team',team,'mins',mins[j],'maxs',maxs[j])

    return mins, maxs


def scale_with_reference_minmax(raw_preds, ref_mins, ref_maxs, clip=True):
    """
    Scale each column j by (x - ref_mins[j]) / (ref_maxs[j] - ref_mins[j]).
    If range == 0, outputs 0 for that column.
    Optionally clip to [0, 1].
    """
    ref_mins = np.asarray(ref_mins, dtype=float)
    ref_maxs = np.asarray(ref_maxs, dtype=float)
    ranges = ref_maxs - ref_mins

    scaled = np.zeros_like(raw_preds, dtype=float)
    for j in range(raw_preds.shape[1]):
        if ranges[j] == 0:
            scaled[:, j] = 0.0
        else:
            scaled[:, j] = (raw_preds[:, j] - ref_mins[j]) / ranges[j]

    if clip:
        scaled = np.clip(scaled, 0.0, 1.0)

    return scaled

def bootstrap_c_index(events, times, predictions, n_bootstraps):
    original_c_index = concordance_index(times, predictions, events)
    c_index_bootstrap = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        indices = resample(range(len(events)), replace=True, n_samples=len(events))
        c_index_bootstrap[i] = concordance_index(times[indices], predictions[indices], events[indices])

    return original_c_index, *np.percentile(c_index_bootstrap, [2.5, 97.5])

def compute_global_cox_metrics(preds_df, gt_df):
    merged = pd.merge(preds_df, gt_df, on='case_id', how='inner')

    df = merged[['prediction', 'capra_s_score', 'follow_up_years', 'event']].copy()

    # AI ensemble only
    data = df[['prediction', 'follow_up_years', 'event']]
    cph = CoxPHFitter().fit(data, 'follow_up_years', 'event')
    c_index, c_index_l, c_index_u = bootstrap_c_index(data['event'], data['follow_up_years'], -cph.predict_partial_hazard(data), 1000)

    # CAPRA-S only
    data_c = df[['capra_s_score', 'follow_up_years', 'event']]
    cph_c = CoxPHFitter().fit(data_c, 'follow_up_years', 'event')
    c_index_c, c_index_c_l, c_index_c_u = bootstrap_c_index(data_c['event'], data_c['follow_up_years'], -cph_c.predict_partial_hazard(data_c), 1000)

    # Univariate
    uni = CoxPHFitter().fit(data, 'follow_up_years', 'event')
    hr_u_p = uni.hazard_ratios_['prediction']
    lo_u_p, hi_u_p, p_u_p = uni.summary.loc['prediction', ['exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]

    uni_c = CoxPHFitter().fit(data_c, 'follow_up_years', 'event')
    hr_u_c = uni_c.hazard_ratios_['capra_s_score']
    lo_u_c, hi_u_c, p_u_c = uni_c.summary.loc['capra_s_score', ['exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]

    # Multivariate
    multi = CoxPHFitter().fit(df[['prediction', 'capra_s_score', 'follow_up_years', 'event']], 'follow_up_years', 'event')
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
    df.to_csv(os.path.join(output_dir, 'total_cox_metrics_capra_s_global_minimax.csv'), index=False)
    caption = f"Cox PH models on combined datasets (n={df['n'].max()} total cases)."
    with open(os.path.join(output_dir, 'total_cox_metrics_capra_s_global_minimax.tex'), 'w') as f:
        f.write(df.drop(columns='n').to_latex(escape=False, caption=caption, label="tab:global_cox"))

def compute_ensemble_coxph(predictions, datasets, ground_truth_path, teams, reference_dataset="radboud"):
    # --- Reference scaling from radboud only (per team/column) ---
    ref_mins, ref_maxs = get_reference_minmax(predictions, teams, reference_dataset=reference_dataset)
    print(f"[Reference: {reference_dataset}] mins per team:", ref_mins)
    print(f"[Reference: {reference_dataset}] maxs per team:", ref_maxs)

    all_events, all_times, all_preds, all_capra_s, all_case_ids = [], [], [], [], []

    for ds_dict in datasets:
        ds = next(iter(ds_dict))
        gt = load_ground_truth(ds, ground_truth_path)
        gt_ids = set(gt['case_id'])

        # Build fixed-length vectors in the same order as `teams`
        per_case_vec = {}  # cid -> np.array shape (n_teams,), filled with nan until set
        for j, team in enumerate(teams):
            team_ds_preds = predictions.get(team, {}).get(ds, {})
            if not team_ds_preds:
                continue

            for cid, pred in team_ds_preds.items():
                if cid not in gt_ids:
                    continue
                if cid not in per_case_vec:
                    per_case_vec[cid] = np.full(len(teams), np.nan, dtype=float)
                per_case_vec[cid][j] = float(pred)

        if not per_case_vec:
            continue

        # Keep only cases that have predictions from ALL teams (no NaNs)
        cids = [cid for cid, vec in per_case_vec.items() if np.all(~np.isnan(vec))]
        if not cids:
            continue

        raw_preds = np.vstack([per_case_vec[cid] for cid in cids])  # (n_cases, n_teams)
        print('Raw preds shape',raw_preds.shape)
        sub = gt.set_index('case_id').loc[cids]

        # Scale using RADBOUD-derived min/max (per team/column)
        scaled_preds = scale_with_reference_minmax(raw_preds, ref_mins, ref_maxs, clip=True)

        # Ensemble = mean across team columns
        norm_preds = scaled_preds.mean(axis=1)
        print('Norm preds shape',norm_preds.shape)

        all_case_ids.extend(sub.index.values)
        all_events.extend(sub['event'].values)
        all_times.extend(sub['follow_up_years'].values)
        all_capra_s.extend(sub['capra_s_score'].values)
        all_preds.extend(norm_preds)

    preds_df = pd.DataFrame({'case_id': all_case_ids, 'prediction': np.array(all_preds, dtype=float)})
    gt_df = pd.DataFrame({
        'case_id': all_case_ids,
        'event': all_events,
        'follow_up_years': all_times,
        'capra_s_score': all_capra_s
    })

    return compute_global_cox_metrics(preds_df, gt_df)

def main(config_path):
    cfg = load_config(config_path)
    
    preds = load_predictions(cfg['input_dir'], cfg['ensemble_teams'], cfg['datasets'])
    raw_results = compute_ensemble_coxph(
                preds,
                cfg['datasets'],
                cfg['clinical_variables'],
                teams=cfg['ensemble_teams'],
                reference_dataset="radboud"
                )
    
    formatted_results = format_global_results(raw_results)
    save_global_results(formatted_results, cfg['output_dir'])
    print(formatted_results)

if __name__ == '__main__':
    config_path = "/data/pathology/projects/leopard/source/evaluation/pathology-leopard-evaluation/config/config.yaml"
    main(config_path)
