# Source: run.ipynb, cell 17
# AI Ensemble / CAPRA-S / combined metrics overall and within risk groups and ISUP groups.
# Notebook note: Check what this is for

import os
import json
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
from sklearn.utils import resample
import random
import yaml
from decimal import Decimal

# -------------- Helpers --------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_ground_truth(dataset, gt_dir):
    # Keeping your filename pattern as-is
    path = os.path.join(gt_dir, f"{dataset}_capra_s_median.csv")
    df = pd.read_csv(path, dtype={"case_id": str})

    # Include ISUP
    # Make sure ISUP is numeric-ish; keep NaNs if missing.
    if "ISUP" in df.columns:
        df["ISUP"] = pd.to_numeric(df["ISUP"], errors="coerce")
    else:
        raise KeyError(f"Ground truth file {path} does not contain required column 'ISUP'.")

    return df[['case_id', 'event', 'follow_up_years', 'capra_s_score', 'ISUP']]

def zscore_normalize_dict(data, m, s):
    # m, s: frozen RUMC calibration-set mean/SD (see zscore_with_calibration_stats)
    if s == 0:
        print('stdev == 0')
        return {k: 0.0 for k in data}
    return {k: (v - m)/s for k, v in data.items()}

def invert_pred_dict(data):
    return {k: -v for k, v in data.items()}

# --- NEW: map ISUP -> low/intermediate/high ---
def isup_to_risk_group(isup):
    """
    Common mapping:
      low = ISUP 1
      intermediate = ISUP 2-3
      high = ISUP 4-5
    Adjust if your definition differs.
    """
    if pd.isna(isup):
        return np.nan
    isup = int(isup)
    if isup in (1,2):
        return "Low"
    elif isup == 3:
        return "Intermediate"
    elif isup in (4, 5):
        return "High"
    return np.nan


# -------------- Load Predictions --------------
def load_predictions(input_dir, teams, datasets):
    preds = {}
    for team in teams:
        preds[team] = {}
        for d in datasets:
            ds = next(iter(d))
            exp = d[ds]
            ds_path = os.path.join(input_dir, team, ds)
            if not os.path.isdir(ds_path):
                continue
            files = [f for f in os.listdir(ds_path) if f.endswith('.json')]
            if len(files) != exp:
                print(f"Warning: {team}/{ds} expected {exp}, found {len(files)}")
            raw = {}
            for fn in files:
                cid = fn[:-5]
                with open(os.path.join(ds_path, fn)) as f:
                    raw[cid] = json.load(f)
            if raw:
                preds[team][ds] = invert_pred_dict(raw)
    return preds

def load_tuning_predictions(cfg, preds):
    """Add the RUMC Tuning split (<validation_dir>/<team>/<tuning_predictions_subfolder>) under cfg['tuning_dataset']."""
    for team in preds:
        path = os.path.join(cfg['validation_dir'], team, cfg['tuning_predictions_subfolder'])
        raw = {}
        for fn in os.listdir(path):
            if fn.endswith('.json'):
                with open(os.path.join(path, fn)) as f:
                    raw[fn[:-5]] = json.load(f)
        preds[team][cfg['tuning_dataset']] = invert_pred_dict(raw)
    return preds

def zscore_with_calibration_stats(preds, calib_datasets, calib_ids):
    """z-score each team, on every dataset, with its mean/SD on the RUMC calibration set, as in calibration.ipynb."""
    for team, team_preds in preds.items():
        vals = np.array([v for ds in calib_datasets for cid, v in team_preds[ds].items() if cid in calib_ids])
        m, s = vals.mean(), vals.std(ddof=1)
        for ds in team_preds:
            team_preds[ds] = zscore_normalize_dict(team_preds[ds], m, s)
    return preds



# -------------- Compute Metrics --------------
def fit_frozen_models(pred_df, gt_df):
    """Cox models fit once on the RUMC calibration set and frozen, as in calibration.ipynb."""
    df = pd.merge(pred_df, gt_df, on='case_id')
    return {
        'ai': CoxPHFitter().fit(df[['prediction', 'follow_up_years', 'event']], 'follow_up_years', 'event'),
        'capra_s': CoxPHFitter().fit(df[['capra_s_score', 'follow_up_years', 'event']], 'follow_up_years', 'event'),
        'joint': CoxPHFitter().fit(df[['prediction', 'capra_s_score', 'follow_up_years', 'event']], 'follow_up_years', 'event'),
    }

def compute_cox_metrics(pred_df, gt_df, models):
    df = pd.merge(pred_df, gt_df, on='case_id')

    # AI model
    X = df[['prediction', 'follow_up_years', 'event']]
    ai_cph = models['ai']
    ai_h = -ai_cph.predict_partial_hazard(X).to_numpy().reshape(-1)
    ai_c = concordance_index(
        df['follow_up_years'].to_numpy(),
        ai_h,
        df['event'].to_numpy()
    )

    # CAPRA-S
    C = df[['capra_s_score', 'follow_up_years', 'event']]
    cs_cph = models['capra_s']
    cs_h = -cs_cph.predict_partial_hazard(C).to_numpy().reshape(-1)
    cs_c = concordance_index(
        df['follow_up_years'].to_numpy(),
        cs_h,
        df['event'].to_numpy()
    )

    # Combined
    M = df[['prediction', 'capra_s_score', 'follow_up_years', 'event']]
    multi = models['joint']
    comb_h = -multi.predict_partial_hazard(M).to_numpy().reshape(-1)
    comb_c = concordance_index(
        df['follow_up_years'].to_numpy(),
        comb_h,
        df['event'].to_numpy()
    )

    return {
        'N (Total Dataset)': int(df.shape[0]),
        'Events (Total Dataset)': int(df['event'].sum()),
        'C-index AI Ensemble (Total Dataset)': ai_c,
        'C-index CAPRA-S (Total Dataset)': cs_c,
        'C-index CAPRA-S + AI Ensemble (Total Dataset)': comb_c,
    }

# --- MODIFIED: supports ISUP strata OR risk-group strata; can disable CI ---
def compute_cindex_by_isup(
    pred_df,
    gt_df,
    models,
    group_mode="isup"):
    
    df = pd.merge(pred_df, gt_df, on='case_id')
    df = df.dropna(subset=["ISUP", "event", "follow_up_years", "capra_s_score", "prediction"]).copy()

    if group_mode == "risk":
        df["ISUP_GROUP"] = df["ISUP"].apply(isup_to_risk_group)
        #df = df.dropna(subset=["ISUP_GROUP"])
        
        order = ['Low', 'Intermediate', 'High']
        strata = sorted(df["ISUP_GROUP"].unique(),key=lambda x: order.index(x))
        def suffix(g): return f" (ISUP-group={g})"
        def subset(g): return df[df["ISUP_GROUP"] == g]
    else:
        
        df.drop(df[df["ISUP"].isin([1])].index, inplace=True)
        strata = sorted(df["ISUP"].unique())
        
        def suffix(g): return f" (ISUP={int(g)})"
        def subset(g): return df[df["ISUP"] == g]

    rows = {}
    print(strata)

    for g in strata:
        sdf = subset(g)
       
        # AI
        X = sdf[['prediction', 'follow_up_years', 'event']]
        print(len(X['event']))
        print(X['event'].value_counts())
        ai_cph = models['ai']
        ai_h = -ai_cph.predict_partial_hazard(X).to_numpy().reshape(-1)
        ai_c = concordance_index(
            sdf['follow_up_years'].to_numpy(),
            ai_h,
            sdf['event'].to_numpy()
        )

        # CAPRA-S
        C = sdf[['capra_s_score', 'follow_up_years', 'event']]
        cs_cph = models['capra_s']
        cs_h = -cs_cph.predict_partial_hazard(C).to_numpy().reshape(-1)
        cs_c = concordance_index(
            sdf['follow_up_years'].to_numpy(),
            cs_h,
            sdf['event'].to_numpy()
        )

        
        # Combined
        M = sdf[['prediction', 'capra_s_score', 'follow_up_years', 'event']]
        multi = models['joint']
        comb_h = -multi.predict_partial_hazard(M).to_numpy().reshape(-1)
        comb_c = concordance_index(
            sdf['follow_up_years'].to_numpy(),
            comb_h,
            sdf['event'].to_numpy()
        )

        suf = suffix(g)
        rows[f'N{suf}'] = int(sdf.shape[0])
        rows[f'Events{suf}'] = int(sdf['event'].sum())
        rows[f'C-index AI Ensemble{suf}'] = ai_c
        rows[f'C-index CAPRA-S{suf}'] = cs_c
        rows[f'C-index CAPRA-S + AI Ensemble{suf}'] = comb_c
        

    return rows

# -------------- Ensemble & Save --------------
def build_ensemble(preds, ds, gt_dir):
    """Ensemble = mean of the per-team z-scores -> (pred_df, ground truth of the same cases), or None."""
    gt = load_ground_truth(ds, gt_dir)

    all_pred = {}
    for team_preds in preds.values():
        for cid, scores in team_preds.get(ds, {}).items():
            all_pred.setdefault(cid, []).append(scores)

    valid = [cid for cid in all_pred if cid in set(gt['case_id'])]
    if not valid:
        return None

    pred_vals = np.array([np.mean(all_pred[c]) for c in valid])
    sub = gt.set_index('case_id').loc[valid].reset_index()
    pred_df = pd.DataFrame({'case_id': valid, 'prediction': pred_vals})
    return pred_df, sub

def standardise_covariates(pred_df, sub, cov_stats):
    """Ensemble and CAPRA-S to zero mean / unit variance with the frozen calibration-set mean/SD."""
    pred_df['prediction'] = (pred_df['prediction'] - cov_stats['prediction'][0]) / cov_stats['prediction'][1]
    sub['capra_s_score'] = (sub['capra_s_score'] - cov_stats['capra_s_score'][0]) / cov_stats['capra_s_score'][1]
    return pred_df, sub

def compute_and_save(preds, datasets, gt_dir, out_dir, cfg):
    # RUMC calibration set (RUMC Tuning + RUMC Internal Validation), as in calibration.ipynb:
    # normalisation and Cox models are fit on it and frozen, then applied to every dataset
    calib_datasets = [cfg['tuning_dataset'], 'radboud']
    calib_ids = set(pd.concat([load_ground_truth(ds, gt_dir) for ds in calib_datasets])['case_id'])
    preds = zscore_with_calibration_stats(preds, calib_datasets, calib_ids)
    calib = [build_ensemble(preds, ds, gt_dir) for ds in calib_datasets]
    calib_pred_df = pd.concat([p for p, _ in calib], ignore_index=True)
    calib_sub = pd.concat([s for _, s in calib], ignore_index=True)
    cov_stats = {'prediction': (calib_pred_df['prediction'].mean(), calib_pred_df['prediction'].std(ddof=1)),
                 'capra_s_score': (calib_sub['capra_s_score'].mean(), calib_sub['capra_s_score'].std(ddof=1))}
    models = fit_frozen_models(*standardise_covariates(calib_pred_df, calib_sub, cov_stats))

    records = []
    for d in datasets:
        ds = next(iter(d))
        built = build_ensemble(preds, ds, gt_dir)
        if built is None:
            continue
        pred_df, sub = standardise_covariates(*built, cov_stats)

        # Original overall metrics
        rec = compute_cox_metrics(pred_df, sub, models)



        #  low / intermediate / high risk groups WITHOUT CI ---
        risk_rows = compute_cindex_by_isup(
            pred_df,
            sub,
            models,
            group_mode="risk"
        )

        rec.update(risk_rows)

        isup_rows = compute_cindex_by_isup(
            pred_df,
            sub,
            models,
            group_mode="isup"
        )
        
        rec.update(isup_rows)       
        
        print(cfg["dataset_names"][ds])
        rec['Dataset'] = cfg["dataset_names"][ds]
        records.append(rec)
        

    # Build DataFrame
    df = pd.DataFrame(records).set_index('Dataset').T

    print(df)
    
    df_fmt = df.copy()



    # Format p-values and other numeric metrics
    def fmt(metric, x):
        if pd.isna(x):
            return x

        # N and Events → integers
        if metric.startswith("N") or metric.startswith("Events"):
            return str(int(round(x)))

        # All C-index values → 3 decimals
        if isinstance(x, float):
            return f"{x:.3f}"

        return x


    for metric in df.index:
        df_fmt.loc[metric] = df.loc[metric].map(lambda v: fmt(metric, v))

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'combined_metrics_capra_s_median_isup_group.csv')
    out_tex = os.path.join(out_dir, 'combined_metrics_capra_s_median_isup_group.tex')
    df_fmt.to_csv(out_csv)
    df_fmt.to_latex(out_tex, index=True, escape=False)
    print(f"Saved combined metrics to {out_csv} and {out_tex}")

    return df_fmt

# -------------- Main --------------
def main(cfg_path):
    cfg = load_config(cfg_path)
    preds = load_predictions(cfg['input_dir'], cfg['ensemble_teams'], cfg['datasets'])
    preds = load_tuning_predictions(cfg, preds)
    out = cfg.get('output_dir', '.')
    compute_and_save(preds, cfg['datasets'], cfg['clinical_variables'], out, cfg)

if __name__ == '__main__':
    import sys
    main("/Users/khrystynafaryna/Documents/leopard-rebuttal/config.yaml")
