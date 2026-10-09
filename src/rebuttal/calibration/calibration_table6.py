# Source: calibration.ipynb, cells 0 + 1
# RESULT 2 -- Table 6: fixed-horizon calibration (O/E, IPCW Brier, calibration curves).
# Layout: cell 1's settings, then the helper cell 0, then the rest of cell 1.

# ======================================================================
# RESULT 2 -- Table 6: fixed-horizon calibration
#
#   The challenge models were optimised for discrimination, so they emit a
#   continuous risk SCORE, not a probability. To judge clinical applicability we
#   need predicted values that can be compared with observed event rates, i.e.
#   fixed-horizon BCR RISKS. We obtain them by fitting Cox proportional hazards
#   models on the RUMC CALIBRATION SET (RUMC Tuning set + RUMC
#   Internal Validation set).
#
#   Predictors: the top-5 AI models (config["ensemble_teams"]) + their Ensemble
#   = 6, plus CAPRA-S. For EACH of the 6 AI predictors three Cox models are fit:
#       (1) capra_s  CAPRA-S alone
#       (2) ai       AI model alone
#       (3) joint    AI + CAPRA-S combined
#   All covariates are standardised to zero mean and unit variance using RUMC
#   CALIBRATION SET statistics (frozen; a held-out cohort is never standardised
#   on its own mean/SD), so every coefficient is a per-SD log hazard ratio.
#
#   BCR risk at horizon t* in {3, 5} years:
#
#       F(t* | x) = 1 - exp( -H0_hat(t*) * exp(eta_hat(x)) )
#
#   with H0_hat the BRESLOW baseline cumulative hazard estimated on the RUMC
#   calibration set and eta_hat the linear predictor. Computed explicitly in
#   fixed_horizon_risk() rather than via lifelines' predict_survival_function,
#   so the code reads like the formula -- and because the two are NOT the same:
#   lifelines reads H0 off its grid with np.interp, i.e. linearly BETWEEN event
#   times, while the Breslow estimator is a step function (H0(t*) is the last
#   jump at or before t*). STEP 5 verifies our H0 against lifelines' at the
#   event times, where they must agree exactly, and prints the interpolation
#   gap at the horizons. 
#   Two prediction modes, reported side by side:
#     raw_risk          applies the CALIBRATION-SET H0 unchanged to each
#                       held-out cohort. Nothing about the cohort is used.
#                       This is the honest, fully-frozen number.
#     intercept-        re-estimates H0 on the held-out cohort by the Breslow
#     recalibrated_risk estimator while keeping the LINEAR PREDICTOR frozen.
#                       This corrects cohort-specific baseline event-rate
#                       differences without altering patient rank ordering, and
#                       so separates INTERCEPT miscalibration -- addressable by
#                       local baseline-hazard recalibration -- from SLOPE
#                       miscalibration, which would require refitting. 
#   The calibration slope (Cox coefficient of the frozen eta_hat refit on the
#   held-out cohort; 1.0 = no slope miscalibration) is reported per row, so the
#   intercept/slope distinction above is readable off Table 6 itself.
#
#
# Reader's map -- every STEP displays something you can eyeball:
#   STEP 1  config + paths        -> path-existence table (stops if a share is unmounted)
#   STEP 2  predictors + cohorts  -> spec table ("what does 'ensemble' expand to?"), per-(team, dataset) file audit: found vs expected
#   STEP 3  calibration set       -> MUST match the expected tuning + RUMC case count; frozen z-scores, 3 models x coefficients, and the frozen H0(t*)
#   STEP 4  Table 6               -> per-tertile O/E + IPCW Brier, raw vs intercept-recalibrated
#   STEP 5  display / plot / save -> calibration curves (one figure per horizon x risk kind:
#                                    cohorts down, AI predictors across, the 3 Cox models as
#                                    3 lines per panel), tables, LaTeX, PROBLEM REPORT
# ======================================================================


CONFIG_PATH   = "/Users/khrystynafaryna/Documents/leopard/config/config-mac.yaml"  # same config file the real pipeline (main.py) uses
SUFFIX        = "median"                  # reads <dataset>_capra_s_<SUFFIX>.csv ground truth
HORIZONS      = (3, 5)                    # t* in {3, 5} years (paper Sec 2.5)
DRILLDOWN_KEY = "ensemble"                # predictor whose STEP 4 worked example is shown
                                          # (the STEP 5 curves cover every predictor)
STRICT        = True                      # raise at the end if any ERROR was recorded
CALIB_DATASET = "radboud"                 # RUMC Internal Validation: part of the calibration set, never held out
MIN_EPV       = 10                        # events-per-variable below this -> warning (Cox rule of thumb)


# EXPECTED_TUNING_N / EXPECTED_CALIB_N / EXPECTED_CALIB_EVENTS are dataset-specific,
# so they are read from the config in STEP 1.

# The validation folder also holds predictions for cases that are not in the
# tuning ground truth; True = restrict the standardisation to the modelled cases.

RESTRICT_DEV_TO_GT = True

# Only 5 of the 9 configured teams - exactly config["ensemble_teams"], and ensemble

ONLY_ENSEMBLE_TEAMS = True

# --- reliability floors for a fixed-horizon calibration number --------
MIN_AT_RISK_COHORT  = 50                  # patients still at risk at the horizon, whole cohort
MIN_AT_RISK_TERTILE = 10                  # patients still at risk at the horizon, per tertile
MIN_EVENTS_BY_H     = 10                  # events observed by the horizon, whole cohort


import os
import json
import time as _time
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display

from lifelines import CoxPHFitter, KaplanMeierFitter
from sksurv.metrics import brier_score
from sksurv.util import Surv

# %matplotlib inline  # notebook-only, disabled in script
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
pd.set_option("display.max_colwidth", 200)   # so the PROBLEM REPORT's messages are never truncated




def read_prediction_dir(dirpath):
    """Read <dirpath>/*.json -> ({case_id: raw float}, [file-level complaints])."""
    values = {}
    for fn in sorted(os.listdir(dirpath)):              # sorted -> deterministic across machines
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(dirpath, fn)) as f:
            v = json.load(f)
        values[fn[:-5]] = float(v)                      # filename minus ".json" is the case_id
    return values


def load_split(root, teams, dataset, expected=None, invert=True):
    """{team: {case_id: value}} for one <root>/<team>/<dataset> folder, plus one audit row per team."""
    out = {}
    for team in teams:
        path = os.path.join(root, team, dataset)
        values = read_prediction_dir(path)
        
        out[team] = {k: -v for k, v in values.items()} if invert else dict(values)
    return out




def read_ground_truth(dataset):
    """Ground-truth CSV for one dataset, with the columns the Cox models need validated."""
    path = os.path.join(config["clinical_variables"], f"{dataset}_capra_s_{SUFFIX}.csv")
    
    gt = pd.read_csv(path, dtype={"case_id": str})
    return gt


def zscore(raw, mean, sd):
    """(v - mean) / sd, case by case, guarding the degenerate cases."""

    return {k: (v - mean) / sd for k, v in raw.items()}


def combine_scores(case_ids, per_team_scaled):
    """Ensemble.Average each case's AVAILABLE per-team z-scores -> ({case_id: score}, {case_id: n teams used})."""
    combined, n_used = {}, {}
    for cid in case_ids:
        vals = [d[cid] for d in per_team_scaled.values() if cid in d]
        if vals:
            combined[cid] = float(np.mean(vals))
            n_used[cid] = len(vals)
    return combined, n_used



def breslow_cumulative_hazard(time, event, log_partial_hazard):
    """Breslow baseline cumulative hazard H0 for a GIVEN linear predictor (coefficients unchanged)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    explp = np.exp(np.asarray(log_partial_hazard, dtype=float))
    event_times = np.unique(time[event == 1])
    #print("****************************************")
    #print(f"event_times: {event_times}")
    cum_hazard = np.empty(len(event_times))
    running = 0.0
    for i, tk in enumerate(event_times):
        d_k = np.sum((time == tk) & (event == 1))        # events exactly at tk (ties)
        risk_set = explp[time >= tk].sum()               # everyone still at risk at tk
        if risk_set <= 0:
            raise ValueError(f"Invalid risk set at time {tk}: risk_set={risk_set}")
        running += d_k / risk_set 
        cum_hazard[i] = running
    return event_times, cum_hazard


def step_lookup(grid_times, grid_values, t):
    """Right-continuous step lookup: value at the largest grid time <= t, else 0."""
    grid_times = np.asarray(grid_times)
    mask = grid_times <= t
    return float(grid_values[mask][-1]) if mask.any() else print(f"Error", "step_lookup", f"t={t} is before the first grid time {grid_times.min()}; earlier version of code would returning 0.0")



def fixed_horizon_risk(H0_at_t, lp):
    """F(t* | x) = 1 - exp(-H0_hat(t*) * exp(eta_hat)) -- the formula, written out."""
    return 1.0 - np.exp(-H0_at_t * np.exp(np.asarray(lp, dtype=float)))


def fit_frozen_models(df):
    """Fit the three Cox models on the calibration set and freeze each one's Breslow H0(t*)."""
    fitted = {}
    for name, cols in MODEL_COVARIATES.items():
        #print("cols", cols)
        # get betas from coxph
        cph = CoxPHFitter().fit(df[cols + ["follow_up_years", "event"]], "follow_up_years", "event")
        # get hazards from betas from coxph and the calibration set
        lp = np.asarray(cph.predict_log_partial_hazard(df[cols])).ravel()
        # get baseline cumulative hazard from the calibration set gt and the hazards with the Breslow estimator
        # NOTE check what happens to H0 after 13 years
        h0_times, H0 = breslow_cumulative_hazard(df["follow_up_years"], df["event"], lp)
        #print(f"h0_times: {h0_times}, H0: {H0}")
    
        #print('step lookup at horizons', {h: step_lookup(h0_times, H0, h) for h in HORIZONS})
        fitted[name] = {"cph": cph, "cols": cols, "h0_times": h0_times, "H0": H0,
                        "H0_at": {h: step_lookup(h0_times, H0, h) for h in HORIZONS}}
    return fitted



def calib_raw_for(team):
    """One team's calibration-set predictions = tuning split merged with RUMC Internal Validation.

    With RESTRICT_DEV_TO_GT, predictions for case_ids that are in no calibration
    ground-truth file are dropped here -- see the note at the top of the cell.
    """
    tuning, rumc = tuning_raw.get(team, {}), cohort_raw[CALIB_DATASET].get(team, {})
    
    merged = {**tuning, **rumc}
    if not RESTRICT_DEV_TO_GT:
        return merged
    return {cid: v for cid, v in merged.items() if cid in CALIB_IDS}



def standardise_covariates(df, cov_stats):
    """Add the z-scored covariate columns using FROZEN calibration-set (mean, SD)."""
    out = df.copy()
    for zcol, src in COVARIATE_SOURCES.items():
        mean, sd = cov_stats[src]
        out[zcol] = (out[src] - mean) / sd
    return out


def build_calibration(teams, label):
    """Frozen standardisation stats + the calibration-set DataFrame for an ensemble AI predictor.

    Returns (df, per-team prediction stats, covariate stats, per-team stats table).
    Two standardisations happen here and they are not the same thing:
      * per TEAM, so several teams' raw scores can be averaged on a common scale;
      * per COVARIATE (the averaged AI score, and CAPRA-S), so the Cox covariates
        have zero mean and unit variance on the calibration set, as specified.
    """
    raw = {t: p for t in teams for p in [calib_raw_for(t)] if p}

    stats, scaled, rows_ = {}, {}, []
    for team, preds in raw.items():
        arr = np.fromiter(preds.values(), dtype=float)
        mean, sd = float(arr.mean()), float(arr.std(ddof=1))
        
        stats[team] = (mean, sd)

        plt.show()
        scaled[team] = zscore(preds, mean, sd)

              
        n_tuning = len(set(preds) & set(gt_tuning["case_id"]))
        rows_.append({"team": team, "n_pred_used": len(preds), "of which tuning": n_tuning,
                      "of which RUMC": len(preds) - n_tuning, "calib_mean": mean, "calib_sd": sd})
        # The check the unmounted-share failure would trip even if STEP 3 passed.
        

    score, n_used = combine_scores(CALIB_IDS, scaled)
    df = gt_calib[gt_calib["case_id"].isin(score)].copy()
    df["score"] = df["case_id"].map(score)
    df["n_teams"] = df["case_id"].map(n_used)

    # Covariate standardisation, estimated HERE and frozen for every cohort.

    cov_stats = {}
    
    for src in COVARIATE_SOURCES.values(): 
        cov_stats[src] = (float(df[src].mean()), float(df[src].std(ddof=1)))

    df = standardise_covariates(df, cov_stats)


    keep = [c for c in ["case_id", "event", "follow_up_years", "capra_s_score", 
                        "score", "capra_s_z", "ai_z", "n_teams"] if c in df.columns]
    df = df[keep].reset_index(drop=True)

    return df, stats, cov_stats, pd.DataFrame(rows_)

def build_cohort(dataset, teams, stats, cov_stats):
    """Score one held-out cohort with the CALIBRATION SET's mean/SD -- frozen, never the cohort's own."""
    gt = GT[dataset]
    scaled = {}
    for team in teams:
        #print(f"Inside build_cohort: {teams}")
        preds = cohort_raw[dataset].get(team)
        scaled[team] = zscore(preds, *stats[team])
          
    # compute ensemble
    score, n_used = combine_scores(set(gt["case_id"]), scaled)
    df = gt[gt["case_id"].isin(score)].copy()
    df["score"] = df["case_id"].map(score)
    df["n_teams"] = df["case_id"].map(n_used)
    # ensure covariates (capra_s_score, and ensemble score) are standardised to zero mean and unit variance using the calibration set's statistics
    df = standardise_covariates(df, cov_stats)           # frozen covariate z-scores
    keep = [c for c in ["case_id", "event", "follow_up_years", "capra_s_score",
                        "score", "capra_s_z", "ai_z", "n_teams"] if c in df.columns]
    return df[keep].reset_index(drop=True), len(scaled)


def km_event_rate(time, event, t):
    """Observed event probability by t -> (1-KM(t), n at risk at t, group max follow-up, extrapolated?).

    `extrapolated` is True when the group's follow-up ends BEFORE t: KM is then
    held flat from its last observation, so 'observed' understates the true rate
    by an unknown amount. The original code returned that number with no signal.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    
    sf = KaplanMeierFitter().fit(time, event_observed=event).survival_function_.iloc[:, 0]
    
    # Mask the sf.index to only include times <= t
    idx = sf.index[sf.index <= t]
    # Take the sf value corresponding to the highest value of the index(the KM estimate at t=indicated horizon)
    surv = float(sf.loc[idx[-1]]) 
    return 1.0 - surv, int((time >= t).sum()), float(time.max()), float(time.max()) < t



# [script fix] moved up: SUMMARY_METRICS below needs them
TEX_DP_OE    = 2            # decimals for O/E ratios
TEX_DP_BRIER = 3            # decimals for Brier scores

SUMMARY_MODEL_LABELS = {"capra_s": "CAPRA-S", "ai": "AI", "joint": "AI+CAPRA-S"}
SUMMARY_METRICS = [("oe_overall_raw", "O/E raw", TEX_DP_OE),
                   ("oe_overall_recal", "O/E recal", TEX_DP_OE),
                   ("brier_recalibrated", "Brier", TEX_DP_BRIER)]
SUMMARY_CAPTION = (
    "Calibration of absolute BCR risk predictions at 3 and 5 years. O/E: observed/expected ratio "
    "(Kaplan--Meier event probability at the horizon divided by the mean predicted risk). "
    "Raw: development-set baseline hazard applied unchanged to the test cohort. "
    "Recal: intercept-recalibrated predictions (per-cohort Breslow baseline hazard re-estimated, "
    "linear predictor frozen). Brier: IPCW-weighted Brier score computed on recalibrated predictions.")

def generate_calibration_summary_table(table6, csv_path, tex_path):
    """Write the compact calibration table -- predictor blocks, cohort x model rows -- as CSV and LaTeX."""
    T6 = {(r.Predictor, r.Cohort, r.Model, r.horizon_years): r for r in table6.itertuples()}
    n_cols = 2 + len(SUMMARY_METRICS) * len(HORIZONS)

    lines = [
        "% requires \\usepackage{booktabs}",
        r"\begin{table}[htbp]", r"\centering", r"\small",
        f"\\caption{{{SUMMARY_CAPTION}}}", r"\label{tab:calibration_summary}",
        r"\begin{tabular}{ll" + "c" * (n_cols - 2) + "}", r"\toprule",
        " & & " + " & ".join(rf"\multicolumn{{{len(SUMMARY_METRICS)}}}{{c}}{{{h} years}}" for h in HORIZONS)
        + r" \\",
        " ".join(rf"\cmidrule(lr){{{3 + len(SUMMARY_METRICS) * i}-{2 + len(SUMMARY_METRICS) * (i + 1)}}}"
                 for i in range(len(HORIZONS))),
        "Cohort & Model & " + " & ".join(name for _ in HORIZONS for _, name, _ in SUMMARY_METRICS) + r" \\",
    ]

    csv_rows = []
    for predictor_label in PREDICTOR_LABELS:
        lines += [r"\midrule",
                  rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{tex_escape(predictor_label)}}}}} \\"]
        first_cohort = True
        for cohort_label in COHORT_ORDER:
            first_of_cohort = True               # the cohort label only on its first model row
            for model_name in MODEL_ORDER:
                got = {h: T6.get((predictor_label, cohort_label, model_name, h)) for h in HORIZONS}
                if all(r is None for r in got.values()):
                    continue
                if first_of_cohort and not first_cohort:
                    lines.append(r"\addlinespace")
                csv_row = {"Predictor": predictor_label, "Cohort": cohort_label,
                           "Model": SUMMARY_MODEL_LABELS[model_name]}
                cells = [tex_escape(cohort_label) if first_of_cohort else "",
                         tex_escape(SUMMARY_MODEL_LABELS[model_name])]
                for h, r in got.items():
                    for col, name, dp in SUMMARY_METRICS:
                        v = np.nan if r is None else float(getattr(r, col))
                        csv_row[f"{name} {h}y"] = round(v, dp) if np.isfinite(v) else np.nan
                        cells.append(tex_num(v, dp))
                csv_rows.append(csv_row)
                lines.append(" & ".join(cells) + r" \\")
                first_of_cohort = first_cohort = False

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines) + "\n"

    summary = pd.DataFrame(csv_rows)
    display(summary)
    summary.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(summary)} rows)")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"Saved {tex_path}")
    print("\n" + "-" * 78 + "\ncopy-paste LaTeX below (calibration summary)\n" + "-" * 78)
    print(tex)


def tertile_calibration(time, event, risk, t):
    """Per-tertile observed vs expected risk at horizon t -> one tidy DataFrame.

    Tertile edges use exact 100/3 and 200/3 percentiles (the pipeline used
    33.33/66.67). With heavy ties in the predicted risk np.digitize can still
    return unequal groups, so `n` is reported rather than assumed to be n/3.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    edges = np.percentile(risk, [0.0, 100.0 / 3.0, 200.0 / 3.0, 100.0])
    edges[0] -= 1e-9                                     # so the minimum lands in tertile 1
    grp = np.digitize(risk, edges[1:-1])                 # 0, 1, 2

    out = []
    for k in range(3):
        m = grp == k
        # print(f"grp {grp}, m {m}, len(time[m]) {len(time[m])}, len(event[m]) {len(event[m])}, len(risk[m]) {len(risk[m])}")
        obs, at_risk, gmax, extrap = km_event_rate(time[m], event[m], t)
        exp = float(risk[m].mean()) #if m.any() else np.nan
        out.append({"tertile": k + 1, "n": int(m.sum()),
                    "risk_lo": float(risk[m].min()), # if m.any() else np.nan,
                    "risk_hi": float(risk[m].max()), # if m.any() else np.nan,
                    "expected": exp, "observed": obs,
                    "oe": obs / exp, #if (exp and exp > 0) else np.nan,
                    "n_at_risk_at_t": at_risk, "max_fu": gmax, "km_extrapolated": extrap})
    return pd.DataFrame(out)


def ipcw_brier(time, event, risk, t, where):
    """IPCW (Graf et al.) Brier score at t; censoring distribution from this same cohort."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    #
    #t_used = min(t, time.max() * 0.999)
    # formatting in the right way for sksurv.brier_score, which wants a structured array
    y = Surv.from_arrays(event=event.astype(bool), time=time)
   
    _, scores = brier_score(y, y, (1.0 - risk).reshape(-1, 1), [t])  # [t_used])  # wants SURVIVAL, not risk
    return float(scores[0])


def ipcw_brier_calib_censoring(time, event, risk, t, calib_time, calib_event, where):
    """IPCW (Graf et al.) Brier score at t on a held-out cohort; censoring distribution from the CALIBRATION set.

    Same estimator as ipcw_brier (sksurv.metrics.brier_score: G read off as a
    right-continuous step, BCR-by-t cases weighted 1/G(T_i), event-free-at-t
    controls 1/G(t), censored-before-t contribute 0), except that G -- the
    Kaplan-Meier of censoring -- is fitted on the calibration set, not on the
    cohort being scored.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    calib_time = np.asarray(calib_time, dtype=float)
    calib_event = np.asarray(calib_event, dtype=int)
    if t > calib_time.max():
        raise ValueError(f"{where}: t={t} is beyond the calibration set's last follow-up "
                         f"({calib_time.max():.2f}y); its censoring distribution is not estimable there")

    # censoring distribution G: Kaplan-Meier with the event indicator reversed, on the calibration set
    G = KaplanMeierFitter().fit(calib_time, event_observed=1 - calib_event).survival_function_.iloc[:, 0]
    G_times, G_vals = G.index.to_numpy(dtype=float), G.to_numpy(dtype=float)
    G_at = lambda s: G_vals[np.searchsorted(G_times, s, side="right") - 1]   # value at the largest time <= s

    is_case = (time <= t) & (event == 1)                 # BCR by t
    is_control = time > t                                # still event-free at t
    G_case, G_t = G_at(time[is_case]), float(G_at(t))
    if G_t <= 0 or (G_case <= 0).any():
        raise ValueError(f"{where}: calibration-set censoring survival is 0 at or before t={t}")

    surv = 1.0 - risk                                    # predicted SURVIVAL at t, as in ipcw_brier
    score = np.zeros(len(time))
    score[is_case] = surv[is_case] ** 2 / G_case
    score[is_control] = (1.0 - surv[is_control]) ** 2 / G_t
    return float(score.mean())


# ======================================================================
# Table 6 outputs -- tables, LaTeX and figures, called from STEP 9
# ======================================================================

# Table 6 columns shown on screen and written to the rounded CSV.
VIEW = ["Cohort", "Predictor", "Model", "horizon_years", "n", "events_by_h", "n_at_risk_at_h",
        "observed_overall", "expected_raw", "oe_overall_raw",
        "oe_raw_t1", "oe_raw_t2", "oe_raw_t3", "brier_raw",
        "oe_overall_recal", "oe_recal_t1", "oe_recal_t2", "oe_recal_t3", "brier_recalibrated",
        "H0_ratio"] #"cal_slope"]


# colour/marker per Cox model, fixed across all four calibration-curve figures
CURVE_STYLE = {"capra_s": ("#8172B3", "^"), "ai": ("#4C72B0", "o"), "joint": ("#DD8452", "s")}
RISK_KINDS = {"raw": "frozen (raw) risk", "intercept_recalibrated": "intercept-recalibrated risk"}


def generate_table_6(table6, output_path):
    """Display Table 6 (the VIEW columns, with a legend) and write every column to CSV."""
    print("O/E > 1 = the model UNDER-predicts risk; O/E < 1 = it OVER-predicts. "
          "t1/t2/t3 are tertiles of predicted risk (low to high).")
    print("*_raw   = calibration-set H0 applied unchanged (fully frozen).")
    print("*_recal = intercept-recalibrated: H0 re-estimated on the held-out cohort, linear predictor frozen.")
    print("          Fit on the same patients it scores -- optimistic by construction.")
    print("H0_ratio  = H0_cohort(t*) / H0_calib(t*): the size of the pure intercept (baseline event rate) shift.")

    display(table6[VIEW])
    table6.to_csv(output_path, index=False)
    print(f"\nSaved {output_path} ({len(table6)} rows)")


def generate_table_6_rounded(table6, output_path):
    """The VIEW columns of Table 6, rounded, for pasting into a spreadsheet or a Word table."""
    round_map = {c: TEX_DP_OE for c in table6.columns if c.startswith("oe_")}
    round_map.update({c: TEX_DP_BRIER for c in table6.columns if c.startswith("brier")})
    round_map.update({c: TEX_DP_OE for c in ["observed_overall", "expected_raw", "expected_recal",
                                             "H0_ratio"] if c in table6.columns})#"cal_slope"] if c in table6.columns})
    table6[VIEW].round(round_map).to_csv(output_path, index=False)
    print(f"Saved {output_path}")


def generate_table_6_tertiles(tertile_store, output_path):
    """Every tertile of every Table 6 row, long-form -- so a reviewer can rebuild any O/E."""
    tert_long = pd.concat(
        [tb.assign(Predictor=p, Cohort=c, Model=m, horizon_years=h, risk_kind=nm)
         for (p, c, m, h), store in tertile_store.items() for nm, tb in store.items()],
        ignore_index=True,
    )
    tert_long.to_csv(output_path, index=False)
    print(f"Saved {output_path} ({len(tert_long)} rows)")


# --- LaTeX export: paper-ready, booktabs, three models side by side ----
# One row per (predictor, cohort, horizon); the three Cox models of the
# calibration analysis -- CAPRA-S alone, AI alone, AI + CAPRA-S -- sit side by
# side, which is the comparison the paragraph describes. Per-tertile O/E lives
# in table6_calibration_tertiles.csv; each cell here shows calibration-in-the-
# large plus the tertile range, which is what fits at manuscript width.
def tex_escape(s):
    """Escape the characters LaTeX would otherwise interpret, e.g. in team names."""
    out = str(s).replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        out = out.replace(a, b)
    return out


def tex_num(v, dp):
    """A number, or an em-dash when it could not be computed -- never a bare 'nan'."""
    return "--" if v is None or not np.isfinite(v) else f"${v:.{dp}f}$"


def tex_range(lo, hi, dp):
    """Tertile range as lo--hi, or an em-dash if either end is missing."""
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return "--"
    return f"${lo:.{dp}f}$--${hi:.{dp}f}$"


def generate_table_6_latex(table6, output_path, kind="raw"):
    """Write Table 6 as LaTeX for one risk kind: 'raw' (frozen) or 'recal' (intercept-recalibrated)."""
    oe_col = "oe_overall_raw" if kind == "raw" else "oe_overall_recal"
    lo_col = "oe_raw_min" if kind == "raw" else "oe_recalibrated_min"
    hi_col = "oe_raw_max" if kind == "raw" else "oe_recalibrated_max"
    br_col = "brier_raw" if kind == "raw" else "brier_recalibrated"

    # (Predictor, Cohort, horizon, Model) -> row, so the three models sit on one line
    T6 = {(r.Predictor, r.Cohort, r.horizon_years, r.Model): r for r in table6.itertuples()}

    if kind == "raw":
        calib_desc = (f"{EXPECTED_TUNING_N}-case {TUNING_LABEL} set + {len(GT[CALIB_DATASET])}-case "
                      f"{CALIB_PART_LABEL} set = {EXPECTED_CALIB_N} patients, {EXPECTED_CALIB_EVENTS} events")
        label = "tab:calibration"
        caption = (f"Fixed-horizon BCR calibration on the held-out cohorts, frozen predictions. "
                   f"For each AI predictor three Cox models are fit on the RUMC calibration set "
                   f"({calib_desc}): CAPRA-S alone, the AI score alone, and the two combined. "
                   f"Covariates are standardised to zero mean and unit variance using RUMC calibration "
                   f"set statistics. Risks are $\\hat{{F}}(t^{{\\star}}\\,|\\,x) = 1 - "
                   f"\\exp(-\\hat{{H}}_0(t^{{\\star}})\\exp(\\hat{{\\eta}}))$ with $\\hat{{H}}_0$ the "
                   f"Breslow baseline cumulative hazard of the calibration set, applied unchanged to "
                   f"each held-out cohort. O/E is observed ($1-$Kaplan-Meier) over expected (mean "
                   f"predicted risk); values above 1 indicate under-prediction. Range is across "
                   f"tertiles of predicted risk.")
    else:
        label = "tab:calibration_recal"
        caption = (f"Fixed-horizon BCR calibration after intercept recalibration. Identical to "
                   f"Table~\\ref{{tab:calibration}} except that $\\hat{{H}}_0$ is re-estimated on each "
                   f"held-out cohort by the Breslow estimator while the linear predictor $\\hat{{\\eta}}$ "
                   f"stays frozen, correcting cohort-specific baseline event rates without altering "
                   f"patient rank ordering.")

    lines = [
        "% requires \\usepackage{booktabs, graphicx}",
        r"\begin{table}[htbp]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrr" + "ccr" * len(MODEL_ORDER) + "}", r"\toprule",
        " & & & & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{MODEL_LABELS[m]}}}" for m in MODEL_ORDER)
        + r" \\",
        " ".join(rf"\cmidrule(lr){{{6 + 3 * i}-{8 + 3 * i}}}" for i in range(len(MODEL_ORDER))),
        r"Predictor & Cohort & $t^{\star}$ (y) & $n$ & Events & "
        + " & ".join(["O/E & O/E range & Brier"] * len(MODEL_ORDER)) + r" \\",
        r"\midrule",
    ]

    n_flagged = 0
    for _, predictor_label, _ in SPECS:
        body = []
        for cohort_label in COHORT_ORDER:
            first_of_cohort = True                   # repeat the predictor/cohort label only once per block
            for h in HORIZONS:
                got = [T6.get((predictor_label, cohort_label, h, m)) for m in MODEL_ORDER]
                if any(r is None for r in got):
                    continue
                ref = got[0]
                flag = "" #if ref.reliable else r"^{\dagger}"
                n_flagged += 0 #if ref.reliable else 1
                cells = [tex_escape(predictor_label) if not body else "",
                         tex_escape(cohort_label) if first_of_cohort else "",
                         f"${h}{flag}$", f"{ref.n:,}", f"{ref.events_by_h:,}"]
                for r in got:
                    cells += [tex_num(getattr(r, oe_col), TEX_DP_OE),
                              tex_range(getattr(r, lo_col), getattr(r, hi_col), TEX_DP_OE),
                              tex_num(getattr(r, br_col), TEX_DP_BRIER)]
                body.append(" & ".join(cells) + r" \\")
                first_of_cohort = False
        if not body:
            continue
        if lines[-1] != r"\midrule":
            lines.append(r"\addlinespace")
        lines += body

    lines += [r"\bottomrule", r"\end{tabular}}"]
    notes = []

    if n_flagged:
        notes.append(f"$^{{\\dagger}}$ Fewer than {MIN_AT_RISK_COHORT} patients still at risk, fewer than "
                     f"{MIN_EVENTS_BY_H} events observed by this horizon, or a tertile with fewer than "
                     f"{MIN_AT_RISK_TERTILE} at risk; these estimates rest on the tail of the "
                     f"Kaplan-Meier curve.")
    if notes:
        lines.append(r"\begin{minipage}{\textwidth}\footnotesize " + " ".join(notes) + r"\end{minipage}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines) + "\n"

    with open(output_path, "w") as f:
        f.write(tex)
    print(f"Saved {output_path}")
    kind_desc = "frozen / raw risk" if kind == "raw" else "intercept-recalibrated risk"
    print("\n" + "-" * 78 + f"\ncopy-paste LaTeX below ({kind_desc})\n" + "-" * 78)
    print(tex)


# --- calibration curves: one figure per (horizon, risk kind) ----------
# FOUR figures: {3y, 5y} x {frozen (raw), intercept-recalibrated}. Within a
# figure the held-out cohorts run DOWN the rows and the AI predictors (top-5
# models + Ensemble) run ACROSS the columns, and each panel carries the three
# Cox models of the calibration analysis as three lines -- CAPRA-S alone, AI
# alone, AI + CAPRA-S. That is the layout the question needs: "does adding the
# AI score to CAPRA-S change calibration, and does the answer hold across
# models and cohorts?" is read off one panel and then scanned along a row,
# rather than reconstructed by flipping between figures. Splitting raw from
# recalibrated into separate figures (they were two lines in one panel before)
# is what frees the three lines for the models; the two are also not comparable
# point-for-point anyway -- the recalibrated one is fit on the patients it
# scores.
def plot_figure_calibration_curves(tertile_store, output_path, h, kind):
    """Calibration curves at horizon h for one risk kind: 'raw' or 'intercept_recalibrated'."""
    # STEP 8 computes no row for a cohort whose follow-up ends before t*,
    # so a horizon's figure has only the cohorts that actually have panels.
    cohorts_h = [c for c in COHORT_ORDER
                 if any((p, c, m, h) in tertile_store
                        for p in PREDICTOR_LABELS for m in MODEL_ORDER)]

    fig, axes = plt.subplots(len(cohorts_h), len(PREDICTOR_LABELS),
                             figsize=(2.55 * len(PREDICTOR_LABELS) + 1.0,
                                      2.55 * len(cohorts_h) + 1.8),
                             squeeze=False)
    for ri, cohort_label in enumerate(cohorts_h):
        # One axis scale per COHORT row. Baseline event rates differ by an
        # order of magnitude between cohorts, so a figure-wide limit would
        # squash whole rows into the corner; a per-panel limit would instead
        # make the six predictors of a row silently incomparable.
        row_vals = np.concatenate(
            [tertile_store[(p, cohort_label, m, h)][kind][["expected", "observed"]]
             .to_numpy(dtype=float).ravel()
             for p in PREDICTOR_LABELS for m in MODEL_ORDER
             if (p, cohort_label, m, h) in tertile_store])
        hi = np.nanmax(row_vals) if np.isfinite(row_vals).any() else 0.02   # all-NaN must not break the axis
        lim = max(0.02, float(hi) * 1.15)

      

        for ci, predictor_label in enumerate(PREDICTOR_LABELS):
            ax = axes[ri][ci]
            have = [m for m in MODEL_ORDER
                    if (predictor_label, cohort_label, m, h) in tertile_store]
            if not have:                        # predictor missing for this cohort entirely
                ax.set_axis_off()               # blank panel, not a lone diagonal
                continue
            ax.plot([0, lim], [0, lim], color="grey", lw=1, ls="--", zorder=1)
            for model_name in have:
                tb = tertile_store[(predictor_label, cohort_label, model_name, h)][kind]
                colour, marker = CURVE_STYLE[model_name]
                ax.plot(tb["expected"], tb["observed"], marker=marker, ms=5.5, lw=1.3,
                        color=colour, zorder=3)
                # the same red x as before: a tertile whose O/E is tail-of-the-KM noise
                #for r in tb.itertuples():
                #    if r.km_extrapolated or r.n_at_risk_at_t < MIN_AT_RISK_TERTILE:
                #        ax.plot(r.expected, r.observed, marker="x", ms=10, color="#C44E52",
                #                mew=2, zorder=4)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.tick_params(labelsize=7)
            if ri == 0:
                ax.set_title(predictor_label, fontsize=9)
            if ri == len(cohorts_h) - 1:
                ax.set_xlabel("predicted (mean risk)", fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"{cohort_label}\nobserved (1 - KM)"
                              + (""), fontsize=9, color="black") 
            else:
                ax.set_yticklabels([])

    handles = ([Line2D([], [], color=CURVE_STYLE[m][0], marker=CURVE_STYLE[m][1], ms=5.5, lw=1.3,
                       label=MODEL_LABELS[m]) for m in MODEL_ORDER]
               + [Line2D([], [], color="grey", lw=1, ls="--", label="perfect calibration")#,
                  #Line2D([], [], color="#C44E52", marker="x", ms=10, mew=2, ls="none",
                  #       label=f"tertile with < {MIN_AT_RISK_TERTILE} at risk or follow-up "
                  #             f"ending before {h}y")
                               ])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
   
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")
    plt.show()


def plot_figure_calibration_in_the_large(table6, output_path):
    """Frozen-risk O/E for every predictor x (cohort, horizon), one heatmap panel per Cox model."""
    pivots = {}
    for model_name in MODEL_ORDER:
        sub = table6[table6["Model"] == model_name]
        if not sub.empty:
            pivots[model_name] = sub.pivot_table(index="Predictor", columns=["Cohort", "horizon_years"],
                                                 values="oe_overall_raw", aggfunc="first")
    if not pivots:
        print("warn", "plot", "no finite calibration-in-the-large value to plot")
        return

    all_oe = np.concatenate([p.to_numpy(dtype=float).ravel() for p in pivots.values()])
    with np.errstate(divide="ignore", invalid="ignore"):
        finite_log = np.log2(all_oe[np.isfinite(all_oe) & (all_oe > 0)])
    vmax = max(float(np.abs(finite_log).max()), 1e-3) if finite_log.size else 1.0
    first = next(iter(pivots.values()))
    fig, axes = plt.subplots(1, len(pivots),
                             figsize=(1.15 * first.shape[1] * len(pivots) + 3.6, 0.4 * first.shape[0] + 2.4),
                             squeeze=False)
    for ax, (model_name, piv) in zip(axes[0], pivots.items()):
        oe = piv.to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_oe = np.log2(np.where(oe > 0, oe, np.nan))     # O/E of 0 -> blank cell, not -inf
        ax.imshow(log_oe, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([f"{c}\n{h}y" for c, h in piv.columns], fontsize=8)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index if ax is axes[0][0] else [""] * piv.shape[0], fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = oe[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(MODEL_LABELS[model_name], fontsize=10)
    fig.suptitle("Calibration-in-the-large, frozen (raw) risk -- observed / expected; "
                 "1.00 is perfect, > 1 under-predicts", fontsize=10)
    fig.colorbar(axes[0][-1].images[0], ax=axes[0].tolist(), label="log2(O/E)", fraction=0.03)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.show()


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


# ======================================================================
# STEP 1 -- config and input paths
# ======================================================================
banner("STEP 1  config and input paths")


with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Expected size of the calibration set -- dataset-specific, so kept in the config
EXPECTED_TUNING_N     = config["expected_tuning_n"]          # rows in <tuning_dataset>_capra_s_<SUFFIX>.csv
EXPECTED_CALIB_N      = config["expected_calibration_n"]     # RUMC Tuning + RUMC Internal Validation
EXPECTED_CALIB_EVENTS = config["expected_calibration_events"]

TUNING_DATASET   = config.get("tuning_dataset", "radboud_tuning")            # GT name for the tuning split
TUNING_SUBFOLDER = config.get("tuning_predictions_subfolder", "validation")  # its prediction subfolder

# Human-readable names for the two halves of the calibration set.
_DCV = config.get("dataset_clinical_variables", {})
TUNING_LABEL = _DCV.get(TUNING_DATASET, "RUMC Tuning")
CALIB_PART_LABEL = _DCV.get(f"{CALIB_DATASET}_test",
                            config["dataset_names"].get(CALIB_DATASET, CALIB_DATASET))
CALIB_SET_LABEL = f"RUMC calibration set ({TUNING_LABEL} + {CALIB_PART_LABEL})"


paths = pd.DataFrame([
    {"key": "input_dir",          "role": "held-out cohort predictions", "path": config["input_dir"]},
    {"key": "validation_dir",     "role": "tuning-split predictions",    "path": config["validation_dir"]},
    {"key": "clinical_variables", "role": "ground-truth CSVs",           "path": config["clinical_variables"]},
    {"key": "output_dir",         "role": "where results are written",   "path": config["output_dir"]},
])
paths["exists"] = paths["path"].map(os.path.isdir)


print(f"\ntuning ground truth : {TUNING_DATASET}_capra_s_{SUFFIX}.csv   (expected {EXPECTED_TUNING_N} cases)")
print(f"tuning predictions  : {config['validation_dir']}/<team>/{TUNING_SUBFOLDER}/*.json")
print(f"calibration set     : {CALIB_SET_LABEL} = {TUNING_DATASET} + {CALIB_DATASET}")
print(f"                      (fits coefficients AND the Breslow baseline hazard; never held out)")
print(f"horizons t*         : {HORIZONS} years")


# ======================================================================
# STEP 2 -- predictors and held-out cohorts, data loading
# ======================================================================
banner("STEP 2  predictors and held-out cohorts")

# config["datasets"] is a list of single-key dicts: [{"<dataset>": <n_cases>}, ...]
DATASET_SIZES = {next(iter(d)): d[next(iter(d))] for d in config["datasets"]}
EVAL_DATASETS = [ds for ds in DATASET_SIZES if ds != CALIB_DATASET]          # PLCO, IMP, UHC

TEAMS = list(config["ensemble_teams"]) if ONLY_ENSEMBLE_TEAMS else list(config["teams"])

team_names = config.get("team_names", {})
SPECS = [(t, team_names.get(t, t), [t]) for t in TEAMS]
#SPECS.append(("ensemble", "Ensemble", list(config["ensemble_teams"])))
SPECS.insert(0, ("ensemble", "Ensemble", list(config["ensemble_teams"])))
SPEC_LOOKUP = {key: (label, teams) for key, label, teams in SPECS}

tuning_raw = load_split(config["validation_dir"], TEAMS, TUNING_SUBFOLDER, invert=True)
cohort_raw = {}                                          # {dataset: {team: {case_id: value}}}
for ds in DATASET_SIZES:
    cohort_raw[ds] = load_split(config["input_dir"], TEAMS, ds,
                                       expected=DATASET_SIZES[ds], invert=True)
  

# ======================================================================
# STEP 3 -- RUMC calibration set: RUMC tuning + RUMC internal validation,
#           and the frozen zero-mean/unit-variance covariate statistics
# ======================================================================
banner("STEP 3  RUMC calibration set (frozen standardisation)")


GT = {ds: read_ground_truth(ds) for ds in list(DATASET_SIZES) + [TUNING_DATASET]}

# --- the tuning half, asserted explicitly -----------------------------
gt_tuning = GT[TUNING_DATASET]
gt_calib = pd.concat([gt_tuning, GT[CALIB_DATASET]], ignore_index=True)
CALIB_IDS = set(gt_calib["case_id"])
COVARIATE_SOURCES = {"capra_s_z": "capra_s_score", "ai_z": "score"}
DRILL_LABEL, DRILL_TEAMS = SPEC_LOOKUP[DRILLDOWN_KEY]

# The three models the calibration analysis is defined on, per AI predictor.
MODEL_COVARIATES = {
    "capra_s": ["capra_s_z"],              # (1) CAPRA-S alone
    "ai":      ["ai_z"],                   # (2) AI model alone
    "joint":   ["capra_s_z", "ai_z"],      # (3) AI + CAPRA-S combined
}
MODEL_LABELS = {"capra_s": "CAPRA-S alone", "ai": "AI alone", "joint": "AI + CAPRA-S"}
MODEL_ORDER = list(MODEL_COVARIATES)

support_rows = []
for ds in EVAL_DATASETS:
    gt = GT[ds]
    tt, ee = gt["follow_up_years"].to_numpy(), gt["event"].to_numpy()
    for h in HORIZONS:
        at_risk = int((tt >= h).sum())
        ev_by_h = int(((ee == 1) & (tt <= h)).sum())
        support_rows.append({
            "cohort": config["dataset_names"].get(ds, ds), "horizon_years": h, "n": len(gt),
            "max_fu": float(tt.max()), "median_fu": float(np.median(tt)),
            "n_at_risk_at_h": at_risk, "pct_at_risk": at_risk / len(gt),
            "events_by_h": ev_by_h,
            "reliable": at_risk >= MIN_AT_RISK_COHORT and ev_by_h >= MIN_EVENTS_BY_H,
        })
SUPPORT = pd.DataFrame(support_rows)

# ======================================================================
# STEP 4 -- Table 6
# ======================================================================
banner("STEP 4  Table 6  (per-tertile O/E and IPCW Brier, 3 models per predictor)")

rows, tertile_store, skipped = [], {}, []
# Loop through teams
for key, label, teams in SPECS:
    print(f"key: {key}, label: {label}, teams: {', '.join(teams)}")

    # get calibration set stats
    pcalib, pstats, pcov, _ = build_calibration(teams, label)

    #print(f"calibration set: pcalib{pcalib}, pstats: {pstats}, pcov: {pcov}")

    # Get frozeb Cox models fit on calibration data
    pmodels = fit_frozen_models(pcalib)

    print(f"fitted models: {list(pmodels.keys())}")

    #Loop through heldout cohorts
    for ds in EVAL_DATASETS:
        print(f"\ncohort: {ds} ({config['dataset_names'].get(ds, ds)})")
        #Get held-out cohort standardized with calibration set stats, also computed ensemble, that is again nnormalized with xcalibration set to have 0 mean 1 std
        cdf, n_teams_used = build_cohort(ds, teams, pstats, pcov)

        # print(f"cohort data: cdf{cdf.head()}, n_teams_used: {n_teams_used}")
        cohort_label = config["dataset_names"].get(ds, ds)

        tt, ee = cdf["follow_up_years"].to_numpy(), cdf["event"].to_numpy()
        print(f"follow-up years: {len(tt)}, ex. {tt[0]}, event: {len(ee)}, ex. {ee[0]}")

        # The calibration slope does not depend on the horizon -- one fit per
        # (predictor, cohort, model), reused by both horizon rows.
        # = {name: calibration_slope(pmodels[name], cdf[MODEL_COVARIATES[name]], tt, ee,
        #                                  f"{label}/{cohort_label}/{name}")
        #          for name in MODEL_ORDER}
        # Loop through horizon

        for h in HORIZONS:
            sup = SUPPORT[(SUPPORT["cohort"] == cohort_label) & (SUPPORT["horizon_years"] == h)].iloc[0]
              
            obs_all, at_risk_all, _, _ = km_event_rate(tt, ee, h)

            # oop through models: CAPRA-S alone, AI alone, AI + CAPRA-S combined

            for model_name in MODEL_ORDER:

                covars = MODEL_COVARIATES[model_name]
                # model is frozen fit on calibration set
                model = pmodels[model_name]
                print(f"\nmodel: {model_name} ({MODEL_LABELS[model_name]}), covariates: {covars}")
                print(f"cdf: {cdf}")
                # X are predictions for the held-out cohort standardized with the calibration set stats, and the covariates for the model
                X = cdf[covars]
                print(f"X shape: {X.shape}, ex. {X.iloc[0].to_dict()}")
                where = f"{label}/{cohort_label}/{model_name}/{h}y"

                
                # Raw risk (1-survival) using the frozen H0 from the calibration set at t={h}, applied to the held-out cohort. 
                # Model is frozen fit on calibration set, and the linear predictor is frozen (no refit on held-out cohort).
                risk_raw = fixed_horizon_risk(model["H0_at"][h], np.asarray(model["cph"].predict_log_partial_hazard(X)).ravel())

                # Intercept-recalibrated risk: re-estimate H0 on the held-out cohort while keeping the linear predictor frozen
                lp = np.asarray(model["cph"].predict_log_partial_hazard(X)).ravel() 
                event_times, cum_hazard = breslow_cumulative_hazard(tt, ee, lp)
                H0_t = step_lookup(event_times, cum_hazard, h)
                risk_recal, H0_cohort = fixed_horizon_risk(H0_t, lp), H0_t
             
                # Tertiles of predicted risk, with observed vs expected and O/E per tertile. The "raw" tertiles are computed on the frozen H0, the "intercept_recalibrated" tertiles on the recalibrated H0. The latter is what a local recalibration would produce, and is reported as an upper bound on what a local recalibration could achieve.
                tert_raw = tertile_calibration(tt, ee, risk_raw, h)
                tert_recal = tertile_calibration(tt, ee, risk_recal, h)
                tertile_store[(label, cohort_label, model_name, h)] = {"raw": tert_raw,
                                                                      "intercept_recalibrated": tert_recal}


                H0_calib = model["H0_at"][h]
                row = {
                    "Cohort": cohort_label, "Predictor": label,
                    "Model": model_name, "Model_label": MODEL_LABELS[model_name],
                    "horizon_years": h,
                    "n": len(cdf), "n_dropped": len(GT[ds]) - len(cdf), "events": int(ee.sum()),
                    "teams_used": n_teams_used, "teams_expected": len(teams),
                    "calib_n": len(pcalib), "calib_events": int(pcalib["event"].sum()),  # full calibration set, or internal validation alone?
                    "n_at_risk_at_h": at_risk_all, "events_by_h": int(sup["events_by_h"]),
                    #"reliable": bool(sup["reliable"]) and tert_raw["n_at_risk_at_t"].min() >= MIN_AT_RISK_TERTILE,
                    "observed_overall": obs_all,
                    # calibration-in-the-large: the single most readable calibration number
                    "expected_raw": float(risk_raw.mean()), "oe_overall_raw": obs_all / risk_raw.mean(),
                    "expected_recal": float(risk_recal.mean()),
                    "oe_overall_recal": obs_all / risk_recal.mean(), # if risk_recal.mean() > 0 else np.nan,
                    "brier_raw": ipcw_brier(tt, ee, risk_raw, h, where),
                    "brier_recalibrated": ipcw_brier(tt, ee, risk_recal, h, where),
                    #"brier_raw": ipcw_brier_calib_censoring(tt, ee, risk_raw, h,
                    #                                        pcalib["follow_up_years"], pcalib["event"], where),
                    #"brier_recalibrated": ipcw_brier_calib_censoring(tt, ee, risk_recal, h,
                    #                                                 pcalib["follow_up_years"], pcalib["event"], where),
                    # the intercept/slope decomposition, in numbers
                    "H0_calib_at_h": H0_calib, "H0_cohort_at_h": H0_cohort,
                    "H0_ratio": H0_cohort / H0_calib # if H0_calib > 0 else np.nan,
                    #"cal_slope": slopes[model_name],
                }
                # All three tertiles, not just min/max: min/max cannot tell you
                # whether calibration drifts monotonically or only the top
                # tertile is off, which is the question Table 6 exists to answer.
                for i in range(3):
                    row[f"oe_raw_t{i + 1}"] = tert_raw["oe"].iloc[i]
                    row[f"oe_recal_t{i + 1}"] = tert_recal["oe"].iloc[i]
                row["oe_raw_min"] = tert_raw["oe"].min()          # pipeline-compatible columns, kept
                row["oe_raw_max"] = tert_raw["oe"].max()
                row["oe_recalibrated_min"] = tert_recal["oe"].min()
                row["oe_recalibrated_max"] = tert_recal["oe"].max()
                rows.append(row)

  


# ======================================================================
# STEP 5 -- table, calibration curves, save, 
# ======================================================================
banner("STEP 5  Table 6")

table6 = pd.DataFrame(rows)

expected_rows = len(SPECS) * len(SUPPORT) * len(MODEL_COVARIATES)

OUTPUT_DIR = os.path.join(config["output_dir"], "calibration_utility")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Row/column order of the figures and the LaTeX table.
COHORT_ORDER = [config["dataset_names"].get(d, d) for d in EVAL_DATASETS]
PREDICTOR_LABELS = [label for _, label, _ in SPECS]

generate_table_6(table6, os.path.join(OUTPUT_DIR, "table6_calibration.csv"))

for kind in RISK_KINDS:
    for h in HORIZONS:
        plot_figure_calibration_curves(
            tertile_store, os.path.join(OUTPUT_DIR, f"table6_calibration_curves_{h}y_{kind}.png"), h, kind)

plot_figure_calibration_in_the_large(table6, os.path.join(OUTPUT_DIR, "table6_calibration_in_the_large.png"))

generate_table_6_tertiles(tertile_store, os.path.join(OUTPUT_DIR, "table6_calibration_tertiles.csv"))
generate_table_6_rounded(table6, os.path.join(OUTPUT_DIR, "table6_calibration_rounded.csv"))

# The raw table is the result; the recalibrated one is fit on the patients it
# scores and is reported as an upper bound on local re-baselining.
generate_table_6_latex(table6, os.path.join(OUTPUT_DIR, "table6_calibration.tex"), kind="raw")
generate_table_6_latex(table6, os.path.join(OUTPUT_DIR, "table6_calibration_intercept_recalibrated.tex"),
                       kind="recal")

generate_calibration_summary_table(table6, os.path.join(OUTPUT_DIR, "calibration_summary.csv"),
                                   os.path.join(OUTPUT_DIR, "calibration_summary.tex"))
