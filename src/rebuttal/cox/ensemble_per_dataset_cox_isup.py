# Source: run.ipynb, cell 28
# Per-dataset univariate/multivariate Cox table, AI Ensemble + ISUP (frozen calibration-set models).
import os
import sys
import json
import logging
import yaml
import pandas as pd
import numpy as np
from scipy.stats import permutation_test
from sklearn.utils import resample
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# Configuration
np.random.seed(1)
random = __import__('random')
random.seed(1)
logging.basicConfig(level=logging.INFO)

AI_COLS = ['prediction', 'follow_up_years', 'event']
ISUP_COLS = ['isup', 'follow_up_years', 'event']
AI_ISUP_COLS = ['prediction', 'isup', 'follow_up_years', 'event']

# Utility Functions
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_ground_truth(dataset, ground_truth_path):
    # ISUP grade (1-5, continuous covariate) from the same cleaned files as the CAPRA-S table, so the cases are identical
    file_path = os.path.join(ground_truth_path, f"{dataset}_capra_s_median.csv")
    gt = pd.read_csv(file_path, dtype={"case_id": str})[['case_id', 'event', 'follow_up_years', 'ISUP']]
    return gt.rename(columns={'ISUP': 'isup'})

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

def fit_cox(df, cols):
    return CoxPHFitter().fit(df[cols], 'follow_up_years', 'event')

def hr_summary(cph, covariate):
    # (HR, CI lower, CI upper, p-value)
    return tuple(cph.summary.loc[covariate, ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])

def build_ensemble_df(predictions, datasets, ground_truth_path, cfg):
    # RUMC calibration set (RUMC Tuning + RUMC Internal Validation), as in calibration.ipynb:
    # normalisation statistics are computed on it and frozen; every dataset is normalised with them
    calib_datasets = [cfg['tuning_dataset'], 'radboud']
    eval_datasets = [next(iter(d)) for d in datasets]  # RUMC, PLCO, IMP, UHC
    all_events, all_times, all_preds, all_isup, all_case_ids, all_ds = [], [], [], [], [], []

    for ds in dict.fromkeys(calib_datasets + eval_datasets):
        gt = load_ground_truth(ds, ground_truth_path)

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
        all_case_ids.extend(sub.index.values)
        all_events.extend(sub['event'].values)
        all_times.extend(sub['follow_up_years'].values)
        all_isup.extend(sub['isup'].values)
        all_preds.extend(preds)
        all_ds.extend([ds] * len(cids))

    raw_preds = np.array(all_preds)
    calib = np.isin(all_ds, calib_datasets)
    # per-team z-score with calibration-set mean/SD
    stds = raw_preds[calib].std(axis=0, ddof=1)
    stds[stds == 0] = 1  # Prevent division by zero
    norm_preds = ((raw_preds - raw_preds[calib].mean(axis=0)) / stds).mean(axis=1)

    df = pd.DataFrame({
        'case_id': all_case_ids,
        'dataset': all_ds,
        'prediction': norm_preds,
        'event': all_events,
        'follow_up_years': all_times,
        'isup': all_isup
    })
    # Cox covariates to zero mean / unit variance with calibration-set statistics
    for col in ['prediction', 'isup']:
        df[col] = (df[col] - df.loc[calib, col].mean()) / df.loc[calib, col].std(ddof=1)

    return df[calib].reset_index(drop=True), df, eval_datasets

def compute_dataset_cox_metrics(calib_df, ds_df, n_bootstraps=1000):
    """C-indexes: Cox models fit on the calibration set and frozen, scored on ds_df.
    HRs / p-values: univariate and multivariate Cox models fit on ds_df itself."""
    events, times = ds_df['event'].values, ds_df['follow_up_years'].values

    risk = {}
    for name, cols in [('ai', AI_COLS), ('isup', ISUP_COLS), ('ai_isup', AI_ISUP_COLS)]:
        frozen = fit_cox(calib_df, cols)
        risk[name] = -frozen.predict_partial_hazard(ds_df[cols]).values

    vals = {'n': len(ds_df)}
    for name, r in risk.items():
        vals[f'c_index_{name}'] = bootstrap_c_index(events, times, r, n_bootstraps)
    _, vals['p_perm'], _ = calculate_p_value_permutation(events, times, risk['ai_isup'], risk['isup'])

    vals['hr_u_ai'] = hr_summary(fit_cox(ds_df, AI_COLS), 'prediction')
    vals['hr_u_isup'] = hr_summary(fit_cox(ds_df, ISUP_COLS), 'isup')
    multi = fit_cox(ds_df, AI_ISUP_COLS)
    vals['hr_m_ai'] = hr_summary(multi, 'prediction')
    vals['hr_m_isup'] = hr_summary(multi, 'isup')
    return vals

def fmt_ci(v, lo, hi, *_):
    return f"${v:.3f}_{{[{lo:.3f},{hi:.3f}]}}$"

TABLE_ROWS = [
    ('N',                                   lambda v: str(v['n'])),
    ('C-index AI Ensemble',                 lambda v: fmt_ci(*v['c_index_ai'])),
    ('C-index ISUP',                        lambda v: fmt_ci(*v['c_index_isup'])),
    ('C-index ISUP + AI Ensemble',          lambda v: fmt_ci(*v['c_index_ai_isup'])),
    ('HR AI Ensemble Univariate',           lambda v: fmt_ci(*v['hr_u_ai'])),
    ('p-value AI Ensemble Univariate',      lambda v: f"{v['hr_u_ai'][3]:.1e}"),
    ('HR ISUP Univariate',                  lambda v: fmt_ci(*v['hr_u_isup'])),
    ('p-value ISUP Univariate',             lambda v: f"{v['hr_u_isup'][3]:.1e}"),
    ('HR AI Ensemble Multivariate',         lambda v: fmt_ci(*v['hr_m_ai'])),
    ('p-value AI Ensemble Multivariate',    lambda v: f"{v['hr_m_ai'][3]:.1e}"),
    ('HR ISUP Multivariate',                lambda v: fmt_ci(*v['hr_m_isup'])),
    ('p-value ISUP Multivariate',           lambda v: f"{v['hr_m_isup'][3]:.1e}"),
    ('p-value C-index ISUP + AI Ensemble vs ISUP (permutation)', lambda v: f"{v['p_perm']:.3f}"),
]

def format_dataset_results(results, dataset_names):
    table = pd.DataFrame({
        dataset_names.get(ds, ds): [fmt(vals) for _, fmt in TABLE_ROWS]
        for ds, vals in results.items()
    })
    table.insert(0, 'Dataset', [label for label, _ in TABLE_ROWS])
    return table

def flatten_results(results, dataset_names):
    rows = []
    for ds, vals in results.items():
        row = {'dataset': dataset_names.get(ds, ds)}
        for k, v in vals.items():
            if np.ndim(v) == 0:
                row[k] = v
            else:
                row.update({k + suffix: x for suffix, x in zip(['', '_l', '_u', '_p'], v)})
        rows.append(row)
    return pd.DataFrame(rows)

def save_dataset_results(table, raw, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, 'per_dataset_cox_metrics_isup_rebuttal_median_nat_com')
    table.to_csv(base + '.csv', index=False)
    raw.to_csv(base + '_raw.csv', index=False)
    caption = ("Univariate and multivariate Cox Proportional Hazard models analysis of AI Ensemble and ISUP grade. "
               "The subscripts indicate 95\\% CIs. AI Ensemble and ISUP grade are standardised with RUMC calibration-set "
               "(RUMC Tuning + RUMC Internal Validation) statistics; HRs are per calibration-set SD. "
               "C-indexes are computed with Cox models fit on the RUMC calibration set and frozen (RUMC is part of the calibration set); "
               "HRs and p-values are from Cox models fit on each dataset.")
    with open(base + '.tex', 'w') as f:
        f.write(table.to_latex(index=False, escape=False, caption=caption, label="tab:per_dataset_cox_isup"))

def main(config_path):
    cfg = load_config(config_path)
    preds = load_predictions(cfg['input_dir'], cfg['ensemble_teams'], cfg['datasets'])
    preds = load_tuning_predictions(cfg, preds)
    calib_df, df, eval_datasets = build_ensemble_df(preds, cfg['datasets'], cfg['clinical_variables'], cfg)
    logging.info(f"Calibration set: n={len(calib_df)}, events={int(calib_df['event'].sum())}")

    results = {}
    for ds in eval_datasets:
        ds_df = df[df['dataset'] == ds].reset_index(drop=True)
        if ds_df.empty:
            continue
        logging.info(f"{ds}: n={len(ds_df)}, events={int(ds_df['event'].sum())}")
        results[ds] = compute_dataset_cox_metrics(calib_df, ds_df)

    table = format_dataset_results(results, cfg['dataset_names'])
    save_dataset_results(table, flatten_results(results, cfg['dataset_names']), cfg['output_dir'])
    print(table.to_string(index=False))

if __name__ == '__main__':
    config_path = "/Users/khrystynafaryna/Documents/leopard-rebuttal/config.yaml"
    main(config_path)

