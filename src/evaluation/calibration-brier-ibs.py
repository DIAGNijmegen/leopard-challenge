import os
import json
import logging
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression

# Configuration
np.random.seed(1)
random = __import__('random')
random.seed(1)
logging.basicConfig(level=logging.INFO)

# -------------------------
# Utility Functions
# -------------------------
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_ground_truth(dataset, ground_truth_path):
    file_path = os.path.join(ground_truth_path, f"{dataset}_capra_s_median.csv")
    return pd.read_csv(file_path, dtype={"case_id": str})[['case_id', 'event', 'follow_up_years', 'capra_s_score']]

# --- Step 1: time -> inverse risk score ---
def time_to_risk_dict(pred_time_dict, method="neg"):  # method: "neg" or "inv"
    out = {}
    eps = 1e-9
    for k, t in pred_time_dict.items():
        t = float(t)
        if method == "neg":
            out[k] = -t
        elif method == "inv":
            out[k] = 1.0 / max(t, eps)
        else:
            raise ValueError("method must be 'neg' or 'inv'")
    return out

def load_predictions(input_dir, teams, datasets, time_to_risk_method="neg"):
    """
    Loads per-case JSON values which are assumed to be predicted times.
    Converts to risk score r via r=-t or r=1/t.
    """
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

            team_pred_times = {}
            for fn in files:
                cid = fn[:-5]
                with open(os.path.join(dataset_path, fn)) as f:
                    team_pred_times[cid] = json.load(f)

            if team_pred_times:
                predictions[team][dataset] = time_to_risk_dict(team_pred_times, method=time_to_risk_method)

    return predictions

# -------------------------
# Reference scaling (same as you had)
# -------------------------
def get_reference_minmax(predictions, teams, reference_dataset="radboud"):
    mins = np.full(len(teams), np.inf, dtype=float)
    maxs = np.full(len(teams), -np.inf, dtype=float)

    for j, team in enumerate(teams):
        tpreds = predictions.get(team, {}).get(reference_dataset, {})
        if not tpreds:
            raise ValueError(f"No predictions found for team '{team}' on reference dataset '{reference_dataset}'")

        vals = np.array(list(tpreds.values()), dtype=float)
        mins[j] = vals.min()
        maxs[j] = vals.max()
        print('team', team, 'mins', mins[j], 'maxs', maxs[j])

    return mins, maxs

def scale_with_reference_minmax(raw_preds, ref_mins, ref_maxs, clip=True):
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

# -------------------------
# Step 2: Cox recalibration on radboud
# -------------------------
def fit_cox_recalibration(cal_df):
    """
    cal_df must have columns: ['r', 'follow_up_years', 'event']
    Fits Cox: h(t|r)=h0(t)*exp(gamma*r)
    """
    cph = CoxPHFitter()
    cph.fit(cal_df[['r', 'follow_up_years', 'event']], duration_col='follow_up_years', event_col='event')
    gamma = float(cph.params_['r'])
    return cph, gamma

def predict_survival_at_times(cph, df_with_r, times_years):
    """
    Returns:
      S: (n, len(times_years)) survival probs
      risk: (n, len(times_years)) risk = 1-S
    """
    surv = cph.predict_survival_function(df_with_r[['r']], times=times_years)  # shape: (len(times), n)
    S = surv.values.T  # (n, len(times))
    R = 1.0 - S
    return S, R

# -------------------------
# Step 3: Calibration + Brier/IBS with censoring (IPCW)
# -------------------------
def fit_censoring_km(times, events):
    """
    Fit KM for censoring distribution G(t)=P(C>=t).
    Here "censoring event" indicator is (1-event).
    """
    km = KaplanMeierFitter()
    censor_event = 1 - np.asarray(events, dtype=int)
    km.fit(times, event_observed=censor_event)
    return km

def ipcw_weights_at_t(times, events, t, km_censor):
    """
    Standard IPCW weights for Brier score at horizon t.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)

    # G(u) evaluated at u = min(T_i, t)
    u = np.minimum(times, t)
    G_u = km_censor.survival_function_at_times(u).values
    G_u = np.maximum(G_u, 1e-9)  # avoid division by zero

    # contribution depends on whether observed beyond t or event before t
    y = ((times <= t) & (events == 1)).astype(int)  # event by t
    # weights:
    # if T_i <= t: weight = 1/G(T_i)
    # if T_i >  t: weight = 1/G(t)
    G_t = float(km_censor.survival_function_at_times([t]).values[0])
    G_t = max(G_t, 1e-9)

    w = np.where(times <= t, 1.0 / G_u, 1.0 / G_t)
    return y, w

def brier_score_ipcw(times, events, pred_risk_at_t, t, km_censor):
    y, w = ipcw_weights_at_t(times, events, t, km_censor)
    pred = np.asarray(pred_risk_at_t, dtype=float)
    return np.sum(w * (y - pred) ** 2) / np.sum(w)

def integrated_brier_score_ipcw(times, events, pred_risk_matrix, grid_times, km_censor):
    """
    pred_risk_matrix: (n, len(grid_times)) predicted risk at each grid time
    IBS = integral_0^tau BS(t) dt / tau (approximated by trapezoid)
    """
    bs = []
    for j, t in enumerate(grid_times):
        bs.append(brier_score_ipcw(times, events, pred_risk_matrix[:, j], t, km_censor))
    bs = np.asarray(bs, dtype=float)
    tau = float(grid_times[-1])
    return np.trapz(bs, grid_times) / tau

def calibration_intercept_slope_ipcw(times, events, pred_risk_at_t, t, km_censor):
    """
    Fit weighted logistic regression: y ~ a + b * logit(pred)
    Returns (a, b)
    """
    pred = np.clip(np.asarray(pred_risk_at_t, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(pred / (1 - pred)).reshape(-1, 1)  # logit
    y, w = ipcw_weights_at_t(times, events, t, km_censor)

    # LogisticRegression includes intercept by default
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(x, y, sample_weight=w)

    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_[0, 0])
    return intercept, slope

def plot_calibration_curve(times, events, pred_risk_at_t, t, km_censor, n_bins=10, title=None):
    """
    Bin by predicted risk and compute observed event prob by t with IPCW estimate.
    """
    pred = np.asarray(pred_risk_at_t, dtype=float)
    df = pd.DataFrame({"pred": pred, "time": times, "event": events})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")

    xs, ys = [], []
    for b, g in df.groupby("bin"):
        # mean predicted risk in bin
        xs.append(g["pred"].mean())

        # IPCW observed risk in bin
        y, w = ipcw_weights_at_t(g["time"].values, g["event"].values, t, km_censor)
        obs = np.sum(w * y) / np.sum(w)
        ys.append(obs)

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter(xs, ys)
    plt.plot(xs, ys)
    plt.xlabel(f"Predicted risk by {t:.0f}y")
    plt.ylabel(f"Observed risk by {t:.0f}y (IPCW)")
    plt.title(title if title else f"Calibration curve at {t:.0f}y")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.show()

# -------------------------
# Your ensemble building + now: calibrate on radboud, evaluate on all
# -------------------------
def compute_ensemble_predictions(predictions, datasets, ground_truth_path, teams, reference_dataset="radboud"):
    """
    Returns merged dataframe with:
      case_id, dataset, r_raw (ensemble risk score), event, follow_up_years, capra_s_score
    NOTE: At this stage r_raw is the ensemble score (after per-team scaling + mean).
    Cox recalibration happens later using radboud rows only.
    """
    ref_mins, ref_maxs = get_reference_minmax(predictions, teams, reference_dataset=reference_dataset)

    rows = []

    for ds_dict in datasets:
        ds = next(iter(ds_dict))
        gt = load_ground_truth(ds, ground_truth_path)
        gt_ids = set(gt['case_id'])

        # Build fixed-length vectors in the same order as `teams`
        per_case_vec = {}
        for j, team in enumerate(teams):
            team_ds_preds = predictions.get(team, {}).get(ds, {})
            if not team_ds_preds:
                continue

            for cid, pred_r in team_ds_preds.items():
                if cid not in gt_ids:
                    continue
                if cid not in per_case_vec:
                    per_case_vec[cid] = np.full(len(teams), np.nan, dtype=float)
                per_case_vec[cid][j] = float(pred_r)

        if not per_case_vec:
            continue

        # only cases with all teams present
        cids = [cid for cid, vec in per_case_vec.items() if np.all(~np.isnan(vec))]
        if not cids:
            continue

        raw_preds = np.vstack([per_case_vec[cid] for cid in cids])  # (n_cases, n_teams)
        sub = gt.set_index('case_id').loc[cids]

        # scale columns using radboud mins/maxs
        scaled_preds = scale_with_reference_minmax(raw_preds, ref_mins, ref_maxs, clip=True)
        ens_r = scaled_preds.mean(axis=1)  # ensemble risk score (monotone)

        for cid, r, ev, fu, cap in zip(sub.index.values, ens_r, sub["event"].values, sub["follow_up_years"].values, sub["capra_s_score"].values):
            rows.append({
                "case_id": cid,
                "dataset": ds,
                "r": float(r),
                "event": int(ev),
                "follow_up_years": float(fu),
                "capra_s_score": float(cap),
            })

    return pd.DataFrame(rows)

def evaluate_calibration_and_brier(df_all, calibration_dataset="radboud", horizons=(2.0, 5.0, 10.0), ibs_tau=10.0, ibs_grid_n=200):
    """
    Fits Cox recalibration on calibration_dataset only, then evaluates metrics on ALL rows (and per dataset if you want).
    """
    # --- Fit Cox recalibration on radboud only ---
    cal = df_all[df_all["dataset"] == calibration_dataset].copy()
    if cal.empty:
        raise ValueError(f"No rows found for calibration dataset '{calibration_dataset}'")

    cph_cal, gamma = fit_cox_recalibration(cal)
    print(f"[Cox recalibration] gamma (slope) = {gamma:.4f}")

    # --- Predict risks at horizons for everyone ---
    times_eval = np.array(list(horizons), dtype=float)
    _, R_h = predict_survival_at_times(cph_cal, df_all, times_eval)  # (n, len(horizons))

    # --- Censoring KM for IPCW (fit on evaluation set or per-dataset; here: ALL) ---
    km_censor = fit_censoring_km(df_all["follow_up_years"].values, df_all["event"].values)

    # --- Metrics ---
    out_rows = []
    for j, t in enumerate(times_eval):
        pred_risk = R_h[:, j]

        bs = brier_score_ipcw(df_all["follow_up_years"].values, df_all["event"].values, pred_risk, t, km_censor)
        intercept, slope = calibration_intercept_slope_ipcw(df_all["follow_up_years"].values, df_all["event"].values, pred_risk, t, km_censor)

        out_rows.append({
            "horizon_years": float(t),
            "brier_ipcw": float(bs),
            "calibration_intercept_ipcw": float(intercept),
            "calibration_slope_ipcw": float(slope),
        })

        plot_calibration_curve(
            df_all["follow_up_years"].values,
            df_all["event"].values,
            pred_risk,
            t,
            km_censor,
            n_bins=10,
            title=f"Ensemble calibration at {t:.0f}y (Cox recal on {calibration_dataset})"
        )

    # IBS over grid up to tau
    grid = np.linspace(1e-6, ibs_tau, ibs_grid_n)
    _, R_grid = predict_survival_at_times(cph_cal, df_all, grid)
    ibs = integrated_brier_score_ipcw(df_all["follow_up_years"].values, df_all["event"].values, R_grid, grid, km_censor)

    results = pd.DataFrame(out_rows)
    results["ibs_ipcw_0_tau"] = float(ibs)
    results["cox_recal_gamma"] = float(gamma)
    results["calibration_dataset"] = calibration_dataset
    results["n_total"] = len(df_all)
    results["n_cal"] = len(cal)

    return results

def main(config_path):
    cfg = load_config(config_path)

    # Step 1 happens here (time -> risk)
    preds = load_predictions(
        cfg['input_dir'],
        cfg['ensemble_teams'],
        cfg['datasets'],
        time_to_risk_method=cfg.get("time_to_risk_method", "neg")  # "neg" or "inv"
    )

    # Build ensemble risk score per case across all datasets
    df_all = compute_ensemble_predictions(
        preds,
        cfg['datasets'],
        cfg['clinical_variables'],
        teams=cfg['ensemble_teams'],
        reference_dataset="radboud"
    )

    # Step 2 & 3: recalibrate on radboud, evaluate calibration + Brier/IBS
    results = evaluate_calibration_and_brier(
        df_all,
        calibration_dataset="radboud",
        horizons=(2.0, 5.0, 10.0),
        ibs_tau=10.0,
        ibs_grid_n=200
    )

    os.makedirs(cfg['output_dir'], exist_ok=True)
    out_csv = os.path.join(cfg['output_dir'], "ensemble_calibration_brier_ibs.csv")
    results.to_csv(out_csv, index=False)
    print(results)
    print("Saved:", out_csv)

if __name__ == '__main__':
    config_path = "/data/pathology/projects/leopard/source/evaluation/pathology-leopard-evaluation/config/config.yaml"
    main(config_path)
