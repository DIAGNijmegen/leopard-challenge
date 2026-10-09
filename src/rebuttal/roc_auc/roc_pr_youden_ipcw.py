# Source: run.ipynb, cell 26
# ROC/PR curves, AUC/AP and Youden-threshold metrics at a fixed horizon (IPCW).

import os
import json
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.utils import resample
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score
)
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv
import random
import yaml
import math
from decimal import Decimal

MODEL_COLS = {
    "AI": ['prediction', 'follow_up_years', 'event'],
    "CAPRA-S": ['capra_s_score', 'follow_up_years', 'event'],
    "CAPRA-S + AI": ['prediction', 'capra_s_score', 'follow_up_years', 'event'],
}
CALIB_NAME = "RUMC Calibration"

# -------------- Helpers --------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_ground_truth(dataset, gt_dir):
    path = os.path.join(gt_dir, f"{dataset}_capra_s_median.csv")
    df = pd.read_csv(path, dtype={"case_id": str})
    return df[['case_id', 'event', 'follow_up_years', 'capra_s_score']]

def invert_pred_dict(data):
    return {k: -v for k, v in data.items()}

# ---------- Outcome helper ----------
def outcome_within_horizon(event, follow_up_years, horizon_years):
    """1 if event occurred within horizon_years, else 0."""
    e = np.asarray(event).astype(int)
    t = np.asarray(follow_up_years).astype(float)
    return ((e == 1) & (t <= horizon_years)).astype(int)

def ipcw_horizon_weights(event, follow_up_years, horizon_years):
    """
    Inverse-probability-of-censoring weights for the outcome at horizon_years
    (Uno et al., same weighting as sksurv's cumulative_dynamic_auc):
      cases    (event at or before horizon):           1 / G(T_i)
      controls (followed beyond horizon):              1 / G(horizon)
      censored at or before horizon without an event:  0 (status unknown)
    G = Kaplan-Meier estimate of the censoring distribution of the same cohort.
    """
    e = np.asarray(event).astype(bool)
    t = np.asarray(follow_up_years).astype(float)
    cens = CensoringDistributionEstimator().fit(Surv.from_arrays(event=e, time=t))

    case = e & (t <= horizon_years)
    control = t > horizon_years
    w = np.zeros(len(t))
    w[case] = 1.0 / cens.predict_proba(t[case])
    if control.any():
        w[control] = 1.0 / cens.predict_proba(np.array([horizon_years]))[0]
    return w

def horizon_outcome_ipcw(df, horizon_years):
    """Binary outcome at horizon_years and IPCW weights, restricted to cases with known status (weight > 0)."""
    y_true = outcome_within_horizon(df['event'], df['follow_up_years'], horizon_years)
    w = ipcw_horizon_weights(df['event'], df['follow_up_years'], horizon_years)
    known = w > 0
    return y_true[known], w[known], known

# -------------- Load Predictions --------------
def load_predictions(input_dir, teams, datasets):
    # raw (inverted) predictions; normalised later with frozen calibration-set statistics
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

# =========================
# Ensemble with frozen calibration-set statistics (as in per_dataset_cox_table.py)
# =========================
def build_ensemble_df(preds, datasets, gt_dir, cfg):
    # RUMC calibration set (RUMC Tuning + RUMC Internal Validation):
    # normalisation statistics are computed on it and frozen; every dataset is normalised with them
    calib_datasets = [cfg['tuning_dataset'], 'radboud']
    eval_datasets = [next(iter(d)) for d in datasets]  # RUMC, PLCO, IMP, UHC
    all_events, all_times, all_preds, all_capra_s, all_case_ids, all_ds = [], [], [], [], [], []

    for ds in dict.fromkeys(calib_datasets + eval_datasets):
        gt = load_ground_truth(ds, gt_dir)

        combined = {}
        for team_preds in preds.values():
            for cid, score in team_preds.get(ds, {}).items():
                combined.setdefault(cid, []).append(score)

        valid = [cid for cid in combined if cid in set(gt['case_id'])]
        if not valid:
            continue

        sub = gt.set_index('case_id').loc[valid]
        all_case_ids.extend(sub.index.values)
        all_events.extend(sub['event'].values)
        all_times.extend(sub['follow_up_years'].values)
        all_capra_s.extend(sub['capra_s_score'].values)
        all_preds.extend(np.array([combined[cid] for cid in valid]))
        all_ds.extend([ds] * len(valid))

    raw_preds = np.array(all_preds)
    calib = np.isin(all_ds, calib_datasets)
    # per-team z-score with calibration-set mean/SD
    means = raw_preds[calib].mean(axis=0)
    stds = raw_preds[calib].std(axis=0, ddof=1)
    stds[stds == 0] = 1  # Prevent division by zero
    for team, m, s in zip(preds, means, stds):
        print(f"Calibration stats {team}: mean={m:.4f}, std={s:.4f}")
    norm_preds = ((raw_preds - means) / stds).mean(axis=1)

    df = pd.DataFrame({
        'case_id': all_case_ids,
        'dataset': all_ds,
        'prediction': norm_preds,
        'event': all_events,
        'follow_up_years': all_times,
        'capra_s_score': all_capra_s
    })
    # Cox covariates to zero mean / unit variance with calibration-set statistics
    for col in ['prediction', 'capra_s_score']:
        df[col] = (df[col] - df.loc[calib, col].mean()) / df.loc[calib, col].std(ddof=1)

    return df[calib].reset_index(drop=True), df, eval_datasets

# =========================
# Frozen Cox models (fit once on the calibration set)
# =========================
def fit_frozen_models(calib_df):
    return {model: CoxPHFitter().fit(calib_df[cols], 'follow_up_years', 'event')
            for model, cols in MODEL_COLS.items()}

def predict_risk(models, df):
    return {model: models[model].predict_partial_hazard(df[cols]).astype(float).values
            for model, cols in MODEL_COLS.items()}

# =========================
# Metrics helpers (ROC/PR + Youden threshold)
# =========================
def _safe_div(num, den):
    return float(num) / float(den) if den else np.nan

def metrics_at_threshold(y_true, scores, threshold, sample_weight=None):
    """
    Compute confusion-matrix metrics at a given threshold.
    Predict positive if score >= threshold.
    With sample_weight (IPCW), tp/fp/tn/fn are weighted counts.
    Returns: ppv/precision, npv, recall/sensitivity, specificity, etc.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    w = np.ones(len(y_true)) if sample_weight is None else np.asarray(sample_weight).astype(float)

    y_pred = (scores >= threshold).astype(int)

    tp = float(np.sum(w * ((y_pred == 1) & (y_true == 1))))
    fp = float(np.sum(w * ((y_pred == 1) & (y_true == 0))))
    tn = float(np.sum(w * ((y_pred == 0) & (y_true == 0))))
    fn = float(np.sum(w * ((y_pred == 0) & (y_true == 1))))

    sensitivity = _safe_div(tp, tp + fn)  # recall
    specificity = _safe_div(tn, tn + fp)
    precision   = _safe_div(tp, tp + fp)  # PPV
    npv         = _safe_div(tn, tn + fn)
    recall      = sensitivity
    ppv         = precision

    return {
        "npv": npv,
        "ppv": ppv,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }

def youden_optimal_threshold(y_true, scores, sample_weight=None):
    """
    Compute Youden's J = sensitivity + specificity - 1.
    Uses roc_curve; returns (best_threshold, best_J, fpr, tpr, thresholds, auc).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    fpr, tpr, thresholds = roc_curve(y_true, scores, sample_weight=sample_weight)
    auc = roc_auc_score(y_true, scores, sample_weight=sample_weight)

    # Youden J = TPR - FPR
    J = tpr - fpr
    best_idx = int(np.nanargmax(J))
    best_thr = float(thresholds[best_idx])
    best_J = float(J[best_idx])

    return best_thr, best_J, fpr, tpr, thresholds, float(auc)

def calibration_youden_thresholds(models, calib_df, horizon_years):
    """Youden-optimal threshold per model (IPCW ROC), selected on the calibration set only."""
    y_true, w, known = horizon_outcome_ipcw(calib_df, horizon_years)
    thresholds = {}
    for model, scores in predict_risk(models, calib_df).items():
        thr, J, *_ = youden_optimal_threshold(y_true, scores[known], sample_weight=w)
        thresholds[model] = thr
        print(f"{model}: calibration Youden threshold={thr:.4f} (J={J:.3f})")
    return thresholds

def compute_all_metrics(ds_df, models, thresholds, horizon_years=5):
    """
    Scores ds_df with the frozen Cox models; computes IPCW PR + ROC curves and the IPCW metrics at the
    calibration-set Youden thresholds (not re-optimised on ds_df). Cases censored before the horizon
    without an event have unknown status and get weight 0; the observed cases are up-weighted instead.
    """
    y_true, w, known = horizon_outcome_ipcw(ds_df, horizon_years)
    prevalence = float(np.average(y_true, weights=w))  # IPCW estimate of P(event <= horizon)
    N = int(ds_df.shape[0])
    n_censored = int((~known).sum())

    scores_by_model = {model: s[known] for model, s in predict_risk(models, ds_df).items()}

    # PR + ROC + metrics at calibration Youden threshold
    pr = {}
    roc = {}
    youden = {}

    for model, scores in scores_by_model.items():
        # PR
        precision, recall, _ = precision_recall_curve(y_true, scores, sample_weight=w)
        ap = average_precision_score(y_true, scores, sample_weight=w) if np.any(y_true == 1) else np.nan
        pr[model] = {"precision": precision, "recall": recall, "ap": float(ap) if np.isfinite(ap) else np.nan}

        # ROC + AUC
        fpr, tpr, _ = roc_curve(y_true, scores, sample_weight=w)
        auc = roc_auc_score(y_true, scores, sample_weight=w)
        roc[model] = {"fpr": fpr, "tpr": tpr, "auc": float(auc) if np.isfinite(auc) else np.nan}

        # Metrics at the frozen calibration-set Youden threshold
        thr = thresholds[model]
        if np.isfinite(thr):
            m = metrics_at_threshold(y_true, scores, thr, sample_weight=w)
        else:
            m = {
                "npv": np.nan, "ppv": np.nan, "precision": np.nan, "recall": np.nan,
                "specificity": np.nan, "sensitivity": np.nan
            }

        youden[model] = {
            "threshold": float(thr) if np.isfinite(thr) else np.nan,
            "J": m["sensitivity"] + m["specificity"] - 1,
            **m
        }

    return pr, roc, youden, prevalence, N, n_censored

# =========================
# Plot helpers
# =========================
def _get_model_color(model):
    if model == "AI":
        return cm.get_cmap("Reds")(0.65)
    if model == "CAPRA-S":
        return cm.get_cmap("Blues")(0.65)
    if model == "CAPRA-S + AI":
        return cm.get_cmap("Greens")(0.65)
    return cm.get_cmap("Greys")(0.65)

# =========================
# Plot ROC (left) + PR (right)
# =========================
def plot_roc_pr_subplots(metrics_by_dataset, out_dir, horizon_years=2):
    """
    One figure with 2 columns per dataset row:
      - Left: ROC curves
      - Right: PR curves
    Produces (n_datasets x 2) subplots (e.g., 8 datasets -> 8x2).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Enforced order (only keep datasets that exist)
    order = ["RUMC", "PLCO", "IMP", "UHC"]
    datasets = [d for d in order if d in metrics_by_dataset]
    extras = [d for d in metrics_by_dataset.keys() if d not in datasets]
    datasets += sorted(extras)

    n = len(datasets)
    if n == 0:
        print("No datasets to plot.")
        return

    fig, axes = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=(12.0, 3.6 * n),
        sharex=False,
        sharey=False
    )

    if n == 1:
        axes = np.array([axes])

    for i, ds in enumerate(datasets):
        ax_roc = axes[i, 0]
        ax_pr  = axes[i, 1]

        roc, pr, prevalence, N, _ = metrics_by_dataset[ds]

        ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        for model in ["AI", "CAPRA-S", "CAPRA-S + AI"]:
            if model not in roc:
                continue
            auc = roc[model]["auc"]
            label = f"{model} (AUC={auc:.3f})" if np.isfinite(auc) else f"{model} (AUC=nan)"
            ax_roc.plot(roc[model]["fpr"], roc[model]["tpr"], color=_get_model_color(model), label=label)

        ax_roc.set_title(f"{ds} (N={N}) — ROC")
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.grid(True, alpha=0.25)
        ax_roc.legend(loc="lower right")

        ax_pr.hlines(prevalence, 0, 1, linestyles="--", linewidth=1)
        for model in ["AI", "CAPRA-S", "CAPRA-S + AI"]:
            if model not in pr:
                continue
            ap = pr[model]["ap"]
            label = f"{model} (AP={ap:.3f})" if np.isfinite(ap) else f"{model} (AP=nan)"
            ax_pr.plot(pr[model]["recall"], pr[model]["precision"], color=_get_model_color(model), label=label)

        ax_pr.set_title(f"{ds} (N={N}) — PR")
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision (PPV)")
        ax_pr.grid(True, alpha=0.25)
        ax_pr.legend(loc="upper left")

    #fig.suptitle(f"IPCW ROC (left) and Precision–Recall (right) at {horizon_years} years "
    #            f"(Cox models frozen on RUMC calibration set)", y=1.002)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"roc_pr_all_datasets_{horizon_years}y_calib_ipcw.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC+PR subplot figure to {out_path}")

# =========================
# compute_and_save: ensemble normalisation, Cox models and Youden thresholds frozen on the
# RUMC calibration set (Tuning + Internal Validation), then applied to every dataset;
# all ROC/PR/threshold metrics are IPCW-weighted for censoring before the horizon
# =========================
def compute_and_save(preds, datasets, gt_dir, out_dir, cfg):
    horizon_years = 5 # <-- choose horizon here

    calib_df, df, eval_datasets = build_ensemble_df(preds, datasets, gt_dir, cfg)
    print(f"Calibration set: n={len(calib_df)}, events={int(calib_df['event'].sum())}")

    models = fit_frozen_models(calib_df)
    thresholds = calibration_youden_thresholds(models, calib_df, horizon_years)

    metrics_by_dataset = {}
    youden_rows = []

    # Calibration set first (where the thresholds were selected), then every evaluation dataset
    cohorts = [(CALIB_NAME, calib_df)]
    for ds in eval_datasets:
        ds_df = df[df['dataset'] == ds].reset_index(drop=True)
        if not ds_df.empty:
            cohorts.append((cfg["dataset_names"][ds], ds_df))

    for ds_name, ds_df in cohorts:
        print("Dataset:", ds_name)
        plt.hist(ds_df['prediction'])
        plt.show()

        pr, roc, youden, prevalence, N, n_censored = compute_all_metrics(ds_df, models, thresholds, horizon_years=horizon_years)
        print(f"N={N}, censored before {horizon_years}y (weight 0)={n_censored}, IPCW prevalence={prevalence:.3f}")

        if ds_name != CALIB_NAME:
            metrics_by_dataset[ds_name] = (roc, pr, prevalence, N, n_censored)

        # Collect Youden threshold metrics for CSV/LaTeX (one row per model per dataset)
        for model in ["AI", "CAPRA-S", "CAPRA-S + AI"]:
            yr = youden.get(model, {})
            youden_rows.append({
                "dataset": ds_name,
                "model": model,  # keep so rows are interpretable
                "threshold value": yr.get("threshold", np.nan),
                "# of cases": N,
                "npv": yr.get("npv", np.nan),
                "ppv": yr.get("ppv", np.nan),
                "precision": yr.get("precision", np.nan),
                "recall": yr.get("recall", np.nan),
                "specificity": yr.get("specificity", np.nan),
                "sensitivity": yr.get("sensitivity", np.nan),
            })

    # Plot ROC+PR curves
    os.makedirs(out_dir, exist_ok=True)
    plot_roc_pr_subplots(metrics_by_dataset, out_dir, horizon_years=horizon_years)

    # Save Youden threshold metrics CSV + LaTeX (without precision column)
    if youden_rows:
        youden_df = pd.DataFrame(youden_rows)

        # CSV (full, includes precision)
        out_csv = os.path.join(out_dir, f"youden_threshold_metrics_{horizon_years}_y_calib_ipcw.csv")
        youden_df.to_csv(out_csv, index=False)
        print(f"Saved Youden threshold metrics to {out_csv}")

        # LaTeX (drop precision)
        latex_df = youden_df.drop(columns=["precision"], errors="ignore")

        caption = (f"Metrics at the Youden-optimal threshold for {horizon_years}-year outcome. "
                   "AI Ensemble and CAPRA-S are standardised with RUMC calibration-set "
                   "(RUMC Tuning + RUMC Internal Validation) statistics; Cox models are fit on the calibration set "
                   "and frozen; thresholds are selected on the calibration set and applied unchanged to every dataset "
                   "(RUMC is part of the calibration set). All metrics use inverse-probability-of-censoring weighting: "
                   "cases censored before the horizon without an event are excluded and observed cases are weighted "
                   "by the inverse Kaplan-Meier probability of remaining uncensored.")
        latex_str = latex_df.to_latex(
            index=False,
            escape=True,
            float_format=lambda x: f"{x:.3f}" if np.isfinite(x) else "",
            caption=caption,
            label="tab:youden_calib_ipcw"
        )

        out_tex = os.path.join(out_dir, f"youden_threshold_metrics_{horizon_years}_y_calib_ipcw.tex")
        with open(out_tex, "w") as f:
            f.write(latex_str)
        print(f"Saved Youden threshold metrics LaTeX table to {out_tex}")

    # Optional: save AUC + AP summary table
    rows = []
    for ds_name, (roc, pr, prevalence, N, n_censored) in metrics_by_dataset.items():
        rows.append({
            "Dataset": ds_name,
            "N": N,
            "Censored before horizon": n_censored,
            "Prevalence (IPCW)": prevalence,
            "AUC_AI": roc["AI"]["auc"],
            "AUC_CAPRA-S": roc["CAPRA-S"]["auc"],
            "AUC_CAPRA-S + AI": roc["CAPRA-S + AI"]["auc"],
            "AP_AI": pr["AI"]["ap"],
            "AP_CAPRA-S": pr["CAPRA-S"]["ap"],
            "AP_CAPRA-S + AI": pr["CAPRA-S + AI"]["ap"],
        })
    if rows:
        df = pd.DataFrame(rows).set_index("Dataset")
        out_csv = os.path.join(out_dir, f"auc_ap_summary_{horizon_years}_y_calib_ipcw.csv")
        df.to_csv(out_csv)
        print(f"Saved AUC/AP summary to {out_csv}")

    return metrics_by_dataset

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
