import os
import json
import logging
import yaml
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# C-indexes of AI Ensemble, CAPRA-S and CAPRA-S + AI Ensemble within ISUP subgroups of RUMC, PLCO, IMP and UHC.
# Normalisation statistics and Cox models come from the RUMC calibration set (RUMC Tuning + RUMC Internal Validation,
# all ISUP grades) and are frozen; every (dataset, ISUP subgroup) is only scored with them
logging.basicConfig(level=logging.INFO)

MODELS = [  # (label, Cox model columns)
    ('AI Ensemble',           ['prediction', 'follow_up_years', 'event']),
    ('CAPRA-S',               ['capra_s_score', 'follow_up_years', 'event']),
    ('CAPRA-S + AI Ensemble', ['prediction', 'capra_s_score', 'follow_up_years', 'event']),
]
ISUP_SUBGROUPS = [(f'ISUP {g}', [g]) for g in range(1, 6)] + [  # (label, ISUP grades)
    ('ISUP low ($\\leq$2)', [1, 2]),
    ('ISUP high ($>$2)',    [3, 4, 5]),
]
OUTPUT_NAME = 'c_index_isup_subgroups_capra_s_rebuttal_median_nat_com_lh'

# Utility Functions
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_ground_truth(dataset, ground_truth_path):
    file_path = os.path.join(ground_truth_path, f"{dataset}_capra_s_median.csv")
    return pd.read_csv(file_path, dtype={"case_id": str})[['case_id', 'event', 'follow_up_years', 'capra_s_score', 'ISUP']]

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

def build_ensemble_df(predictions, datasets, ground_truth_path, cfg):
    # RUMC calibration set (RUMC Tuning + RUMC Internal Validation), as in calibration.ipynb:
    # normalisation statistics are computed on it and frozen; every dataset is normalised with them
    calib_datasets = [cfg['tuning_dataset'], 'radboud']
    eval_datasets = [next(iter(d)) for d in datasets]  # RUMC, PLCO, IMP, UHC
    all_events, all_times, all_preds, all_capra_s, all_isup, all_case_ids, all_ds = [], [], [], [], [], [], []

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
        all_capra_s.extend(sub['capra_s_score'].values)
        all_isup.extend(sub['ISUP'].values)
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
        'capra_s_score': all_capra_s,
        'ISUP': all_isup  # raw grade, only used to define the subgroups
    })
    # Cox covariates to zero mean / unit variance with calibration-set statistics
    for col in ['prediction', 'capra_s_score']:
        df[col] = (df[col] - df.loc[calib, col].mean()) / df.loc[calib, col].std(ddof=1)

    return df[calib].reset_index(drop=True), df, eval_datasets

def add_frozen_risks(calib_df, df):
    # one Cox model per MODELS entry, fit on the calibration set and frozen; -partial hazard so higher = longer survival
    for name, cols in MODELS:
        frozen = CoxPHFitter().fit(calib_df[cols], 'follow_up_years', 'event')
        df[f'risk {name}'] = -frozen.predict_partial_hazard(df[cols]).values
    return df

def safe_c_index(times, scores, events):
    # NaN when there are no comparable pairs (e.g. no events in the subgroup)
    try:
        return concordance_index(times, scores, events)
    except ZeroDivisionError:
        return np.nan

def subgroup_c_indexes(sub_df, n_bootstraps=1000, random_state=1):
    """C-index and 95% bootstrap CI of every frozen model on one (dataset, ISUP subgroup);
    all models are scored on the same bootstrap resamples."""
    events, times = sub_df['event'].values, sub_df['follow_up_years'].values
    rng = np.random.RandomState(random_state)
    resamples = [rng.randint(0, len(sub_df), len(sub_df)) for _ in range(n_bootstraps)]

    vals = {'n': len(sub_df), 'events': int(events.sum())}
    undefined = 0
    for name, _ in MODELS:
        scores = sub_df[f'risk {name}'].values
        c_index = safe_c_index(times, scores, events)
        if np.isnan(c_index):
            vals[name] = (np.nan, np.nan, np.nan)
            continue
        boot = np.array([safe_c_index(times[idx], scores[idx], events[idx]) for idx in resamples])
        undefined = np.isnan(boot).sum()  # same for every model: comparable pairs depend only on times/events
        vals[name] = (c_index, *np.nanpercentile(boot, [2.5, 97.5]))
    if undefined:
        logging.warning(f"{undefined}/{n_bootstraps} bootstrap resamples without comparable pairs skipped")
    return vals

def fmt_ci(v, lo, hi):
    return 'N/A' if np.isnan(v) else f"${v:.3f}_{{[{lo:.3f},{hi:.3f}]}}$"

def format_results(results, dataset_names):
    # rows: (ISUP subgroup, N (events) / model); columns: datasets
    index, rows = [], []
    for label, _ in ISUP_SUBGROUPS:
        index.append((label, 'N (events)'))
        rows.append({dataset_names.get(ds, ds): f"{r[label]['n']} ({r[label]['events']})" for ds, r in results.items()})
        for name, _ in MODELS:
            index.append((label, f'C-index {name}'))
            rows.append({dataset_names.get(ds, ds): fmt_ci(*r[label][name]) for ds, r in results.items()})
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index, names=['ISUP subgroup', '']))

def flatten_results(results, dataset_names):
    rows = []
    for ds, subgroups in results.items():
        for label, vals in subgroups.items():
            for name, _ in MODELS:
                c_index, lo, hi = vals[name]
                rows.append({'dataset': dataset_names.get(ds, ds), 'isup_subgroup': label, 'model': name,
                             'n': vals['n'], 'events': vals['events'],
                             'c_index': c_index, 'c_index_l': lo, 'c_index_u': hi})
    return pd.DataFrame(rows)

def save_results(table, raw, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, OUTPUT_NAME)
    table.to_csv(base + '.csv')
    raw.to_csv(base + '_raw.csv', index=False)
    caption = ("C-indexes of AI Ensemble, CAPRA-S and CAPRA-S + AI Ensemble within ISUP grade subgroups. "
               "The subscripts indicate 95\\% bootstrap CIs. AI Ensemble and CAPRA-S are standardised with RUMC calibration-set "
               "(RUMC Tuning + RUMC Internal Validation) statistics, and the Cox models are fit on the RUMC calibration set "
               "(all ISUP grades) and frozen (RUMC is part of the calibration set). "
               "N/A: C-index undefined (no events in the subgroup).")
    with open(base + '.tex', 'w') as f:
        f.write(table.to_latex(escape=False, multirow=True, index_names=False, column_format='ll' + 'c' * table.shape[1],
                               caption=caption, label="tab:c_index_isup_subgroups"))

def main(config_path):
    cfg = load_config(config_path)
    preds = load_predictions(cfg['input_dir'], cfg['ensemble_teams'], cfg['datasets'])
    preds = load_tuning_predictions(cfg, preds)
    calib_df, df, eval_datasets = build_ensemble_df(preds, cfg['datasets'], cfg['clinical_variables'], cfg)
    logging.info(f"Calibration set: n={len(calib_df)}, events={int(calib_df['event'].sum())}")
    df = add_frozen_risks(calib_df, df)

    results = {}
    for ds in eval_datasets:
        ds_df = df[df['dataset'] == ds]
        results[ds] = {}
        for label, grades in ISUP_SUBGROUPS:
            sub_df = ds_df[ds_df['ISUP'].isin(grades)]
            logging.info(f"{ds} / {label}: n={len(sub_df)}, events={int(sub_df['event'].sum())}")
            results[ds][label] = subgroup_c_indexes(sub_df)

    table = format_results(results, cfg['dataset_names'])
    save_results(table, flatten_results(results, cfg['dataset_names']), cfg['output_dir'])
    print(table.to_string())

if __name__ == '__main__':
    config_path = "/Users/khrystynafaryna/Documents/leopard-rebuttal/config.yaml"
    main(config_path)
