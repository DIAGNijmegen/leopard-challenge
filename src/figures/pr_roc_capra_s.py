import os
import json
import pandas as pd
import numpy as np
from scipy.stats import zscore
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.utils import resample
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score
)
import random
import yaml
import math
from decimal import Decimal

# -------------- Helpers --------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_ground_truth(dataset, gt_dir):
    path = os.path.join(gt_dir, f"{dataset}_capra_s_median.csv")
    df = pd.read_csv(path, dtype={"case_id": str})
    return df[['case_id', 'event', 'follow_up_years', 'capra_s_score']]

def zscore_normalize_dict(data):
    vals = np.array(list(data.values()))
    m, s = vals.mean(), vals.std()
    if s == 0:
        print('stdev == 0')
        return {k: 0.0 for k in data}
    return {k: (v - m) / s for k, v in data.items()}

def invert_pred_dict(data):
    return {k: -v for k, v in data.items()}

# ---------- Outcome helper ----------
def outcome_within_horizon(event, follow_up_years, horizon_years):
    """1 if event occurred within horizon_years, else 0."""
    e = np.asarray(event).astype(int)
    t = np.asarray(follow_up_years).astype(float)
    return ((e == 1) & (t <= horizon_years)).astype(int)

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
                norm = zscore_normalize_dict(raw)
                inv = invert_pred_dict(norm)
                preds[team][ds] = inv
    return preds

# =========================
# Metrics helpers (ROC/PR + Youden threshold)
# =========================
def _safe_div(num, den):
    return float(num) / float(den) if den else np.nan

def metrics_at_threshold(y_true, scores, threshold):
    """
    Compute confusion-matrix metrics at a given threshold.
    Predict positive if score >= threshold.
    Returns: ppv/precision, npv, recall/sensitivity, specificity, etc.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    y_pred = (scores >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

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

def youden_optimal_threshold(y_true, scores):
    """
    Compute Youden's J = sensitivity + specificity - 1.
    Uses roc_curve; returns (best_threshold, best_J, fpr, tpr, thresholds, auc).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    # ROC undefined if only one class present
    #has_both = (np.any(y_true == 1) and np.any(y_true == 0))
    #if not has_both:
    #    fpr = np.array([0.0, 1.0])
    #    tpr = np.array([0.0, 1.0])
    #    thresholds = np.array([np.inf, -np.inf])
    #    return np.nan, np.nan, fpr, tpr, thresholds, np.nan

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    # Youden J = TPR - FPR
    J = tpr - fpr
    best_idx = int(np.nanargmax(J))
    best_thr = float(thresholds[best_idx])
    best_J = float(J[best_idx])

    return best_thr, best_J, fpr, tpr, thresholds, float(auc)

def compute_all_metrics(pred_df, gt_df, horizon_years=5):
    """
    Fits Cox models once; computes PR + ROC + Youden-optimal threshold metrics
    for AI / CAPRA-S / CAPRA-S + AI at a chosen horizon.
    """
    df = pd.merge(pred_df, gt_df, on='case_id')

    # Risk scores from Cox models
    X = df[['prediction', 'follow_up_years', 'event']]
    ai_cph = CoxPHFitter().fit(X, 'follow_up_years', 'event')
    ai_risk = (ai_cph.predict_partial_hazard(X)).astype(float).values

    C = df[['capra_s_score', 'follow_up_years', 'event']]
    cs_cph = CoxPHFitter().fit(C, 'follow_up_years', 'event')
    cs_risk = (cs_cph.predict_partial_hazard(C)).astype(float).values

    M = df[['prediction', 'capra_s_score', 'follow_up_years', 'event']]
    multi = CoxPHFitter().fit(M, 'follow_up_years', 'event')
    comb_risk = (multi.predict_partial_hazard(M)).astype(float).values

    # Binary outcome at horizon
    y_true = outcome_within_horizon(df['event'], df['follow_up_years'], horizon_years)
    prevalence = float(np.mean(y_true))
    N = int(df.shape[0])

    scores_by_model = {
        "AI": ai_risk,
        "CAPRA-S": cs_risk,
        "CAPRA-S + AI": comb_risk,
    }

    # PR + ROC + Youden
    pr = {}
    roc = {}
    youden = {}

    for model, scores in scores_by_model.items():
        # PR
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores) if np.any(y_true == 1) else np.nan
        pr[model] = {"precision": precision, "recall": recall, "ap": float(ap) if np.isfinite(ap) else np.nan}

        # ROC + AUC + Youden threshold
        thr, J, fpr, tpr, thresholds, auc = youden_optimal_threshold(y_true, scores)
        roc[model] = {"fpr": fpr, "tpr": tpr, "auc": float(auc) if np.isfinite(auc) else np.nan}

        # Metrics at Youden threshold
        if np.isfinite(thr):
            m = metrics_at_threshold(y_true, scores, thr)
        else:
            m = {
                "npv": np.nan, "ppv": np.nan, "precision": np.nan, "recall": np.nan,
                "specificity": np.nan, "sensitivity": np.nan
            }

        youden[model] = {
            "threshold": float(thr) if np.isfinite(thr) else np.nan,
            "J": float(J) if np.isfinite(J) else np.nan,
            **m
        }

    return pr, roc, youden, prevalence, N

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

        roc, pr, prevalence, N = metrics_by_dataset[ds]

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
        ax_pr.legend(loc="lower left")

    fig.suptitle(f"ROC (left) and Precision–Recall (right) at {horizon_years} years", y=1.002)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"roc_pr_all_datasets_{horizon_years}y.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC+PR subplot figure to {out_path}")

# =========================
# MODIFIED: compute_and_save now also writes CSV + LaTeX (no precision column)
# =========================
def compute_and_save(preds, datasets, gt_dir, out_dir, cfg):
    horizon_years = 1 # <-- choose horizon here

    metrics_by_dataset = {}
    youden_rows = []

    for d in datasets:
        ds = next(iter(d))
        gt = load_ground_truth(ds, gt_dir)

        all_pred = {}
        for team_preds in preds.values():
            for cid, score in team_preds.get(ds, {}).items():
                all_pred.setdefault(cid, []).append(score)

        valid = [cid for cid in all_pred if cid in set(gt['case_id'])]
        if not valid:
            continue

        pred_vals = np.array([np.mean(all_pred[c]) for c in valid])

        print("Dataset:", d)
        plt.hist(pred_vals)
        plt.show()

        sub = gt.set_index('case_id').loc[valid].reset_index()
        sub['capra_s_score'] = zscore(sub['capra_s_score'])

        pred_df = pd.DataFrame({'case_id': valid, 'prediction': pred_vals})

        pr, roc, youden, prevalence, N = compute_all_metrics(pred_df, sub, horizon_years=horizon_years)

        ds_name = cfg["dataset_names"][ds]
        print(ds_name)

        metrics_by_dataset[ds_name] = (roc, pr, prevalence, N)

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
        out_csv = os.path.join(out_dir, f"youden_threshold_metrics_{horizon_years}_y_.csv")
        youden_df.to_csv(out_csv, index=False)
        print(f"Saved Youden threshold metrics to {out_csv}")

        # LaTeX (drop precision)
        latex_df = youden_df.drop(columns=["precision"], errors="ignore")

        # Optional: nicer numeric formatting
        latex_str = latex_df.to_latex(
            index=False,
            escape=True,
            float_format=lambda x: f"{x:.3f}" if np.isfinite(x) else ""
        )

        out_tex = os.path.join(out_dir, f"youden_threshold_metrics_{horizon_years}_y_.tex")
        with open(out_tex, "w") as f:
            f.write(latex_str)
        print(f"Saved Youden threshold metrics LaTeX table to {out_tex}")

    # Optional: save AUC + AP summary table
    rows = []
    for ds_name, (roc, pr, prevalence, N) in metrics_by_dataset.items():
        rows.append({
            "Dataset": ds_name,
            "N": N,
            "Prevalence": prevalence,
            "AUC_AI": roc["AI"]["auc"],
            "AUC_CAPRA-S": roc["CAPRA-S"]["auc"],
            "AUC_CAPRA-S + AI": roc["CAPRA-S + AI"]["auc"],
            "AP_AI": pr["AI"]["ap"],
            "AP_CAPRA-S": pr["CAPRA-S"]["ap"],
            "AP_CAPRA-S + AI": pr["CAPRA-S + AI"]["ap"],
        })
    if rows:
        df = pd.DataFrame(rows).set_index("Dataset")
        out_csv = os.path.join(out_dir, f"auc_ap_summary_{horizon_years}_y_.csv")
        df.to_csv(out_csv)
        print(f"Saved AUC/AP summary to {out_csv}")

    return metrics_by_dataset

# -------------- Main --------------
def main(cfg_path):
    cfg = load_config(cfg_path)
    preds = load_predictions(cfg['input_dir'], cfg['ensemble_teams'], cfg['datasets'])

    out = cfg.get('output_dir', '.')
    compute_and_save(preds, cfg['datasets'], cfg['clinical_variables'], out, cfg)

if __name__ == '__main__':
    import sys
    main("/data/pathology/projects/leopard/source/evaluation/pathology-leopard-evaluation/config/config.yaml")
