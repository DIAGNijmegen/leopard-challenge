import os
import json
import pandas as pd
import numpy as np
from scipy.stats import zscore
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve, average_precision_score  # NEW
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
    return df[['case_id', 'event', 'follow_up_years', 'ISUP']]

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
# NEW: compute Precision-Recall data at a single horizon
# =========================
def compute_pr_data(pred_df, gt_df, horizon_years=5):
    """
    Returns PR curve data for AI / ISUP / ISUP + AI at a chosen horizon (years).
    """
    df = pd.merge(pred_df, gt_df, on='case_id')

    # Risk scores from Cox models
    X = df[['prediction', 'follow_up_years', 'event']]
    ai_cph = CoxPHFitter().fit(X, 'follow_up_years', 'event')
    ai_risk = (ai_cph.predict_partial_hazard(X)).astype(float).values

    C = df[['ISUP', 'follow_up_years', 'event']]
    cs_cph = CoxPHFitter().fit(C, 'follow_up_years', 'event')
    cs_risk = (cs_cph.predict_partial_hazard(C)).astype(float).values

    M = df[['prediction', 'ISUP', 'follow_up_years', 'event']]
    multi = CoxPHFitter().fit(M, 'follow_up_years', 'event')
    comb_risk = (multi.predict_partial_hazard(M)).astype(float).values

    # Binary outcome at horizon
    y_true = outcome_within_horizon(df['event'], df['follow_up_years'], horizon_years)

    # Precision-Recall curves
    pr = {}
    for name, scores in [
        ("AI", ai_risk),
        ("ISUP", cs_risk),
        ("ISUP + AI", comb_risk),
    ]:
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores) if np.any(y_true == 1) else np.nan
        pr[name] = {
            "precision": precision,
            "recall": recall,
            "ap": ap
        }

    prevalence = float(np.mean(y_true))  # baseline precision
    return pr, prevalence, int(df.shape[0])

# =========================
# NEW: plot Precision-Recall curves with subplots (all datasets)
# =========================
def _get_model_color(model):
    if model == "AI":
        return cm.get_cmap("Reds")(0.65)
    if model == "ISUP":
        return cm.get_cmap("Blues")(0.65)
    if model == "ISUP + AI":
        return cm.get_cmap("Greens")(0.65)
    return cm.get_cmap("Greys")(0.65)

def plot_precision_recall_subplots(pr_by_dataset, out_dir, horizon_years=2):
    """
    One figure with subplots (one per dataset), each subplot shows PR curves for:
      - AI (red)
      - ISUP (blue)
      - ISUP + AI (green)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Enforced order (only keep datasets that exist)
    order = ["RUMC", "PLCO", "IMP", "UHC"]
    datasets = [d for d in order if d in pr_by_dataset]
    # If any other datasets exist, append them at the end (stable)
    extras = [d for d in pr_by_dataset.keys() if d not in datasets]
    datasets += sorted(extras)

    n = len(datasets)
    if n == 0:
        print("No datasets to plot.")
        return

    # Grid layout
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.0 * ncols, 4.0 * nrows),
        sharex=True,
        sharey=True
    )
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, ds in enumerate(datasets):
        ax = axes[i]
        pr, prevalence, N = pr_by_dataset[ds]

        # Baseline prevalence line (random classifier)
        ax.hlines(prevalence, 0, 1, linestyles="--", linewidth=1)

        for model in ["AI", "ISUP", "ISUP + AI"]:
            if model not in pr:
                continue
            ax.plot(
                pr[model]["recall"],
                pr[model]["precision"],
                color=_get_model_color(model),
                label=f"{model} (AP={pr[model]['ap']:.3f})" if np.isfinite(pr[model]["ap"]) else f"{model} (AP=nan)"
            )
            
        ax.legend(loc="upper right", bbox_to_anchor=(1, 1))

        ax.set_title(f"{ds} (N={N})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)

    # Turn off unused subplots
    for j in range(n, nrows * ncols):
        axes[j].axis("off")

    fig.supxlabel("Recall")
    fig.supylabel("Precision (PPV)")
    fig.suptitle(f"Precision–Recall curves at {horizon_years} years", y=1.02)

    # One shared legend (grab from first used axis)
    handles, labels = None, None
    for ax in axes[:n]:
        h, l = ax.get_legend_handles_labels()
        if len(h) > 0:
            handles, labels = h, l
            break
    #if handles is not None:
    #    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.02, 1.02))

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"precision_recall_all_datasets_isup_{horizon_years}y.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PR subplot figure to {out_path}")

# =========================
# MODIFIED: compute_and_save now plots PR curves instead of PPV/NPV curves
# =========================
def compute_and_save(preds, datasets, gt_dir, out_dir, cfg):
    horizon_years = 1  # <-- choose PR horizon here (e.g., 2, 3, 5, etc.)

    pr_by_dataset = {}

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
        print("Dataset:",d)
        plt.hist(pred_vals)
        plt.show()
        sub = gt.set_index('case_id').loc[valid].reset_index()
        sub['ISUP'] = zscore(sub['ISUP'])

        pred_df = pd.DataFrame({'case_id': valid, 'prediction': pred_vals})

        pr, prevalence, N = compute_pr_data(pred_df, sub, horizon_years=horizon_years)

        ds_name = cfg["dataset_names"][ds]
        print(ds_name)
        pr_by_dataset[ds_name] = (pr, prevalence, N)

    # Plot PR curves (subplots)
    os.makedirs(out_dir, exist_ok=True)
    plot_precision_recall_subplots(pr_by_dataset, out_dir, horizon_years=horizon_years)

    # Optional: also save AP summary table (handy for reporting)
    ap_rows = []
    for ds_name, (pr, prevalence, N) in pr_by_dataset.items():
        ap_rows.append({
            "Dataset": ds_name,
            "N": N,
            "Prevalence": prevalence,
            "AP_AI": pr["AI"]["ap"],
            "AP_ISUP": pr["ISUP"]["ap"],
            "AP_ISUP + AI": pr["ISUP + AI"]["ap"],
        })
    if ap_rows:
        ap_df = pd.DataFrame(ap_rows).set_index("Dataset")
        out_csv = os.path.join(out_dir, f"average_precision_summary_isup_{horizon_years}y.csv")
        ap_df.to_csv(out_csv)
        print(f"Saved AP summary to {out_csv}")

    return pr_by_dataset

# -------------- Main --------------
def main(cfg_path):
    cfg = load_config(cfg_path)
    preds = load_predictions(cfg['input_dir'], cfg['ensemble_teams'], cfg['datasets'])

    out = cfg.get('output_dir', '.')
    compute_and_save(preds, cfg['datasets'], cfg['clinical_variables'], out, cfg)

if __name__ == '__main__':
    import sys
    main("/data/pathology/projects/leopard/source/evaluation/pathology-leopard-evaluation/config/config.yaml")
