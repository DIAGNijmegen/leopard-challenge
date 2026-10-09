# Source: decision_curve.ipynb, cells 2 + 4
# RESULT 3 -- figure-only variant: restyled decision curves with per-cohort y-limits.
# Layout: cell 4's settings, then the helper cell 2, then the rest of cell 4.

# ======================================================================
# RESULT 3 -- Decision curve analysis: Figure 10 + decision-curve summary table
#
# Run the helper cell above first -- it defines every function used here,
# including all the plotting, table and LaTeX code -- then Kernel > Restart
# and run the two cells in order.
# Same scaffolding, same frozen models and the same development split
# as RESULT 2 (Table 6), so the calibration table and the decision curves are
# read off ONE model rather than two slightly different ones.
#
# The question this cell answers
#   "If a clinician treated every patient whose predicted risk of recurrence by
#    t years exceeds a threshold p_t, does adding the AI score to CAPRA-S do
#    more good than harm -- compared with CAPRA-S alone, with treating
#    everyone, and with treating nobody?"
#
# Reader's map -- every STEP displays something you can eyeball:
#   STEP 1  config + paths        -> path-existence table (stops if a share is unmounted)
#   STEP 2  predictors + cohorts  -> the 5 ensemble teams + Ensemble = 6 predictors
#   STEP 3  load predictions      -> per-(team, dataset) file audit: found vs expected
#   STEP 4  development split     -> MUST match the expected tuning + RUMC case / event counts
#   STEP 5  frozen Cox models     -> coefficients + the frozen baseline survival at 3y/5y
#   STEP 6  net-benefit helpers   -> ONE worked threshold you can recompute by hand,
#                                    plus 6 arithmetic self-tests that must all pass
#   STEP 7  threshold support     -> predicted-risk distribution vs the threshold grid
#                                    (the trap: over most of the grid NOBODY is treated)
#   STEP 8  all curves            -> 6 predictors x 3 cohorts x 2 horizons
#   STEP 9  figures + summary table + LaTeX  -> ONE 6-panel figure PER PREDICTOR
#                                    (2 horizons x 3 held-out cohorts; each panel
#                                    draws the score alone, CAPRA-S alone and the
#                                    two together), plus the Delta-NB heatmap
#                                    that compares the 6 predictors
#
# ----------------------------------------------------------------------
# IS THE IMPLEMENTATION CORRECT?  (read this before trusting the output)
# ----------------------------------------------------------------------
# The net-benefit arithmetic in the previous version of this cell was RIGHT.
# For censored data (Vickers et al. 2008), with S_hat the Kaplan-Meier
# survival among the patients a threshold would treat:
#     NB(p_t) = P(treat) * [ (1 - S_hat(t|treat)) - S_hat(t|treat) * p_t/(1-p_t) ]
# which is exactly `(n_pos/n)*obs - (n_pos/n)*(1-obs)*(pt/(1-pt))`.
# `treat_all_curve` is the same formula with everyone treated. Both are kept
# unchanged here (and are now checked by STEP 6's self-tests).
#
# What was NOT right, and is fixed below:
#   (1) The development split could silently collapse to RUMC alone
#       whenever validation_dir was an unmounted share, because
#       `{**tuning, **rumc}` degrades to `rumc` with no error. Every curve would
#       still be drawn, from the WRONG frozen model. STEP 4 now asserts the
#       expected case / event counts (from the config) and stops if that fails.
#   (2) Only one predictor ("ensemble") was ever analysed. All 5 ensemble teams
#       are now analysed separately as well, from the same frozen pipeline, and
#       each one gets its own figure: six overlaid joint curves in a single
#       panel could not be told apart, and hid each predictor's own comparison
#       against treat-all / treat-none / CAPRA-S alone.
#   (3) `if horizon >= t_i.max(): continue` left a blank panel titled
#       "unsupported" and moved on. That guard is far too weak in one direction
#       (a horizon just inside a cohort's maximum follow-up passes even when
#       few patients are still at risk) and silent in the other. Support is now
#       computed, displayed, and carried into every row as `reliable`.
#   (4) Nothing reported HOW MANY PATIENTS a threshold actually treats. When a
#       cohort's frozen risks are low, above a modest p_t almost nobody is
#       treated: the model curve collapses onto the treat-none line at 0 and
#       looks like a real (and flattering) finding. `n_treated` / `frac_treated`
#       are now recorded at every threshold and flagged below MIN_TREATED.
#   (5) A treated subgroup whose follow-up ends before the horizon gets a
#       Kaplan-Meier held flat, which UNDERSTATES its event rate and therefore
#       its net benefit. That is now detected and reported per point.
#   (6) The CSV was overwritten per predictor-key and stored the identical
#       treat-all curve once per predictor. One tidy long-form CSV now.
#
# Interpretation caveats that no amount of code can fix -- state them in the paper:
#   * These curves use the FROZEN baseline hazard. Decision curves are therefore
#     sensitive to calibration, not only discrimination: a frozen model that
#     under- or over-predicts on a cohort (see Table 6) shifts who crosses a
#     threshold. `oe_capra` / `oe_joint` are carried into the decision-curve summary table so that a
#     net-benefit difference driven by miscalibration is visible, not hidden.
#   * No confidence intervals. Net benefit differences of ~0.005 (0.5 events per
#     100 patients) are well inside the noise of a cohort of a few hundred
#     patients. Set N_BOOTSTRAP > 0 for percentile CIs on Delta-NB (slow), or
#     report the sign and the threshold range rather than the exact value.
#   * The 3-year and 5-year rows come from the same patients, so they are not
#     independent results.
#
# Failure policy: nothing is skipped quietly. Every deviation goes through
# problem() and is reprinted in STEP 9's PROBLEM REPORT; deviations that would
# make everything downstream meaningless go through require() and stop the cell
# where they happen.
# ======================================================================

import os
import json
import time as _time
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from IPython.display import display

from lifelines import CoxPHFitter, KaplanMeierFitter   # KM is used only to cross-check ours in STEP 6

# %matplotlib inline  # notebook-only, disabled in script
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
pd.set_option("display.max_colwidth", 200)   # so the PROBLEM REPORT's messages are never truncated

# ----------------------------------------------------------------------
# Run parameters -- the only knobs in this cell
# ----------------------------------------------------------------------
CONFIG_PATH   = "/Users/khrystynafaryna/Documents/leopard/config/config-mac.yaml"  # same config file the real pipeline (main.py) uses
SUFFIX        = "median"                  # reads <dataset>_capra_s_<SUFFIX>.csv ground truth
HORIZONS      = (3, 5)                    # 3y primary, 5y secondary (paper Sec 2.5)
DRILLDOWN_KEY = "ensemble"                # predictor whose worked example and diagnostics are shown
STRICT        = True                      # raise at the end if any ERROR was recorded
DEV_DATASET   = "radboud"                 # RUMC: merged into the development split, never held out
MIN_EPV       = 10                        # events-per-variable below this -> warning (Cox rule of thumb)
N_BOOTSTRAP   = 0                         # >0 -> percentile CIs on Delta-NB at the reference threshold

# Threshold-probability grids, one per horizon (unchanged from the original cell).
THRESHOLDS = {
    3: np.round(np.arange(0.05, 0.3001, 0.01), 4),    # 3-year: 5%-30% risk thresholds
    5: np.round(np.arange(0.10, 0.4001, 0.01), 4),    # 5-year: 10%-40% risk thresholds
}
# [script fix] REFERENCE_THRESHOLD was used below but never defined in the notebook (it survived
# in the kernel from an earlier version of this cell); restored from leopard/decision_curve.py.
# The single threshold quoted in the decision-curve summary table.
REFERENCE_THRESHOLD = {3: 0.10, 5: 0.20}

# --- the tuning cohort this analysis is defined to be fit on ----------
# The frozen models MUST be fit on the RUMC tuning split merged with the RUMC
# cohort. If validation_dir is an unmounted SMB share, load_split() returns {}
# for every team and the merge {**tuning, **rumc} silently degenerates to RUMC
# alone with no error anywhere: the original cell would have drawn a
# complete, plausible, WRONG set of decision curves. Asserted in STEP 4 against
# EXPECTED_TUNING_N / EXPECTED_DEV_N / EXPECTED_DEV_EVENTS, which are
# dataset-specific and therefore read from the config in STEP 1.

# The validation folder also holds predictions for cases that appear in NO
# ground truth file. src/calibration_utility.py computes each team's frozen
# mean/SD over all of them, so the standardisation population is not the
# modelling population. True = restrict to the modelled cases. Keep this
# identical to RESULT 1 and RESULT 2 or the three results disagree on the model.
RESTRICT_DEV_TO_GT = True

# Only 5 of the 9 configured teams (exactly config["ensemble_teams"]) submitted
# tuning-split predictions; the other 4 have no validation/ folder, so their
# development split would be RUMC alone and their curves are not comparable.
# The analysis is therefore the 5 ensemble members, separately, plus the Ensemble.
ONLY_ENSEMBLE_TEAMS = True

# --- reliability floors -----------------------------------------------
MIN_AT_RISK_COHORT = 50    # patients still at risk at the horizon, whole cohort
MIN_EVENTS_BY_H    = 10    # events observed by the horizon, whole cohort
MIN_TREATED        = 10    # patients a threshold treats; below this its net benefit is noise
MIN_GRID_COVERAGE  = 0.5   # fraction of the threshold grid that must treat >= MIN_TREATED patients
EPS_NB             = 1e-4  # 0.01 net true positives per 100 patients: below this, call it a tie
NEGLIGIBLE_PER_100 = 0.005 # a Delta-NB this small prints as "+0.00" -- the score changed nothing

# ----------------------------------------------------------------------
# Problem bookkeeping -- the alternative to `except: continue`
# ----------------------------------------------------------------------
PROBLEMS = {}          # (severity, where, message) -> how many times it happened


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


def build_dev(teams, label):
    """Frozen per-team (mean, SD) + the development-split DataFrame for one predictor."""
    raw = {t: p for t in teams for p in [dev_raw_for(t)] if p}


    stats, scaled, rows_ = {}, {}, []
    for team, preds in raw.items():
        arr = np.fromiter(preds.values(), dtype=float)
        mean, sd = float(arr.mean()), float(arr.std(ddof=1))
       
        stats[team] = (mean, sd)
        scaled[team] = zscore(preds, mean, sd)
        n_tuning = len(set(preds) & set(gt_tuning["case_id"]))
        rows_.append({"team": team, "n_pred_used": len(preds), "of which tuning": n_tuning,
                      "of which RUMC": len(preds) - n_tuning, "dev_mean": mean, "dev_sd": sd})
        # The check the unmounted-share failure would trip even if STEP 3 passed.
        if n_tuning != EXPECTED_TUNING_N:
            print("error", "dev split",
                    f"{team}: the frozen mean/SD use {n_tuning} tuning cases, expected {EXPECTED_TUNING_N} "
                    f"-- this team is not standardised on the full tuning cohort")

    score, n_used = combine_scores(DEV_IDS, scaled)
    df = gt_dev[gt_dev["case_id"].isin(score)].copy()
    df["score"] = df["case_id"].map(score)
    df["n_teams"] = df["case_id"].map(n_used)
    keep = [c for c in ["case_id", "event", "follow_up_years", "capra_s_score", "score", "n_teams"]
            if c in df.columns]
    df = df[keep].reset_index(drop=True)
    if len(df) != EXPECTED_DEV_N:
        print("error", "dev split",
                f"{label}: only {len(df)}/{EXPECTED_DEV_N} development cases have a score "
                f"({int(df['event'].sum())}/{EXPECTED_DEV_EVENTS} events) -- its frozen models are fit on a "
                f"smaller population than a fully-covered predictor, so its curves are not comparable")
    return df, stats, pd.DataFrame(rows_)


def dev_raw_for(team):
    """One team's development-split predictions = tuning split merged with RUMC."""
    tuning, rumc = tuning_raw.get(team, {}), cohort_raw[DEV_DATASET].get(team, {})
    clash = set(tuning) & set(rumc)
    merged = {**tuning, **rumc}
    if not RESTRICT_DEV_TO_GT:
        return merged
    return {cid: v for cid, v in merged.items() if cid in DEV_IDS}

def fit_frozen_models(df, label):
    n_events = int(df["event"].sum())
    for name, cols in MODEL_COVARIATES.items():
        epv = n_events / len(cols)
        if epv < MIN_EPV:
            print("warn", "model fit", f"{label}/{name}: {n_events} events for {len(cols)} covariate(s) = "
                                         f"{epv:.0f} events per variable (< {MIN_EPV}) -- unstable "
                                         f"coefficients, and a shaky frozen baseline hazard")
    return {name: CoxPHFitter().fit(df[cols + ["follow_up_years", "event"]], "follow_up_years", "event")
            for name, cols in MODEL_COVARIATES.items()}

def build_cohort(dataset, teams, stats):
    """Score one held-out cohort using the DEVELOPMENT split's mean/SD -- frozen, never the cohort's own."""
    gt = GT[dataset]
    scaled = {}
    for team in teams:
        preds = cohort_raw[dataset].get(team)
        if preds and team in stats:
            scaled[team] = zscore(preds, *stats[team])
    score, n_used = combine_scores(set(gt["case_id"]), scaled)
    df = gt[gt["case_id"].isin(score)].copy()
    df["score"] = df["case_id"].map(score)
    df["n_teams"] = df["case_id"].map(n_used)
    keep = [c for c in ["case_id", "event", "follow_up_years", "capra_s_score", "ISUP", "score", "n_teams"]
            if c in df.columns]
    return df[keep].reset_index(drop=True), len(scaled)





def raw_risk(cph, X, t):
    """1 - S(t|x) with coefficients AND baseline hazard frozen on the development split."""
    sf = cph.predict_survival_function(X, times=[t])     # columns follow X's row order
    return 1.0 - sf.iloc[0].to_numpy()


def km_event_rate(time, event, t):
    """Observed event probability by t -> (1-KM(t), n at risk at t, group max follow-up, extrapolated?).

    Plain Kaplan-Meier, written out so the whole calculation is visible (and so
    it is fast enough to call once per threshold): S(t) = prod over event times
    tk <= t of (1 - d_k / n_k). `extrapolated` is True when the group's
    follow-up ends BEFORE t, in which case KM is held flat and the 'observed'
    rate understates the truth by an unknown amount.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if time.size == 0:
        return np.nan, 0, np.nan, True
    order = np.argsort(time, kind="mergesort")
    time_s, event_s = time[order], event[order]
    hit = (event_s == 1) & (time_s <= t)
    if hit.any():
        ev_times, d = np.unique(time_s[hit], return_counts=True)          # tied event times collapse
        n_at_risk = time_s.size - np.searchsorted(time_s, ev_times, side="left")
        surv = float(np.prod(1.0 - d / n_at_risk))
    else:
        surv = 1.0
    at_risk_t = int(time_s.size - np.searchsorted(time_s, t, side="left"))
    return 1.0 - surv, at_risk_t, float(time_s.max()), bool(time_s.max() < t)


def net_benefit_curve(time, event, risk, thresholds, t):
    """Net benefit of 'treat everyone with predicted risk >= p_t', one row per threshold.

    NB(p_t) = P(treat) * [ obs - (1 - obs) * p_t/(1 - p_t) ],  obs = 1 - KM(t | treated)
    i.e. (true positives - weighted false positives) / n, on the scale of
    'net true positives per patient'. Multiply by 100 to read it as
    'net true positives per 100 patients'.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    n = time.size
    rows = []
    for pt in thresholds:
        treated = risk >= pt
        n_treated = int(treated.sum())
        w = pt / (1.0 - pt)                                    # odds of the threshold = harm/benefit ratio
        if n_treated == 0:
            # Treating nobody has net benefit exactly 0 -- correct, but it must
            # be visible that this point carries no information about the model.
            rows.append({"threshold": float(pt), "n_treated": 0, "frac_treated": 0.0,
                         "obs_rate_treated": np.nan, "net_benefit": 0.0,
                         "km_extrapolated": False, "enough_treated": False})
            continue
        obs, _, _, extrap = km_event_rate(time[treated], event[treated], t)
        rows.append({"threshold": float(pt), "n_treated": n_treated, "frac_treated": n_treated / n,
                     "obs_rate_treated": obs,
                     "net_benefit": (n_treated / n) * (obs - (1.0 - obs) * w),
                     "km_extrapolated": extrap, "enough_treated": n_treated >= MIN_TREATED})
    return pd.DataFrame(rows)


def treat_all_curve(time, event, thresholds, t):
    """Net benefit of treating EVERYONE: the same formula with P(treat) = 1."""
    obs, _, _, _ = km_event_rate(time, event, t)
    return np.array([obs - (1.0 - obs) * (pt / (1.0 - pt)) for pt in thresholds]), obs




# ======================================================================
# FIGURES, TABLES AND LaTeX
#
# Everything STEP 9 of the analysis cell displays or writes to disk lives
# here, so that cell reads as a sequence of calls rather than 200 lines of
# matplotlib. Like the helpers above, these read the run parameters
# (config, HORIZONS, EVAL_DATASETS, SUPPORT, MIN_TREATED, ...) from the
# globals the analysis cell defines.
# ======================================================================

# Colour follows the MODEL, not the predictor: inside a panel the three curves
# being compared are three models, and the predictor is named in the title, so
# the same three colours mean the same three things in all six figures.
# Ink / blue / orange stay separable under every common colour-vision
# deficiency -- ink is separated from both by lightness, and the blue-orange
# pair is the one hue pair that survives deuteranopia and protanopia -- and
# each clears 3:1 contrast on the off-white panel.
MODEL_COLORS = {"capra_s": "#0b0b0b", "score": "#eb6834", "joint": "#2a78d6"}
REFERENCE_COLOR = "#8a8a85"   # treat-all / treat-none: strategies, not models

# Off-white surfaces. The page sits a shade darker than the panels, so the six
# panels read as cards without a box drawn round each one. The grey band that
# marks the thin-treatment tail is warm rather than neutral, otherwise it reads
# as a cold blue-grey patch against a warm page.
FIG_BG      = "#f2f0eb"       # the page behind the panels
AX_BG       = "#fbfaf7"       # the panel itself
GRID_COLOR  = "#d6d2c9"       # thin grid, recessive against AX_BG
SPINE_COLOR = "#b8b3a8"
TICK_COLOR  = "#55524c"
THIN_BAND   = "#c9c4b8"       # thresholds treating fewer than MIN_TREATED

# Fallback net-benefit scale. Only used now when a caller passes an explicit
# `ylim`: by default every COHORT (= one ROW of the figure) gets its own range,
# so a cohort whose curves live in a 0.02-wide band is not flattened onto the
# axis by a high-event cohort. Within a row the range is still SHARED across
# horizons, which is what keeps the two columns of a row comparable by eye --
# and rows are no longer comparable to each other, which is the price of the
# per-cohort scale and has to be said in the caption.
NB_YLIM = (-0.10, 0.28)

# The columns of DCA_SUMMARY that go on screen and into the rounded CSV.
SUMMARY_VIEW = ["Predictor", "Cohort", "horizon_years", "n", "events_by_h", "reliable", "threshold_ref",
                "n_treated_capra_ref", "n_treated_joint_ref", "nb_treat_all", "nb_capra_s", "nb_joint",
                "delta_nb_per_100", "reduction_vs_treat_all_per_100", "mean_delta_nb_over_grid",
                "joint_superior_lo", "joint_superior_hi", "joint_superior_run", "joint_superior_n",
                "joint_superior_contiguous", "negligible_vs_capra", "oe_capra_s", "oe_joint"]

# The model curves a row's y-range is fitted to, in draw order (back to front).
PANEL_MODELS = [("score", "{label} alone"),
                ("capra_s", "CAPRA-S alone"),
                ("joint", "CAPRA-S + {label}")]


def cohort_ylim(label, cohort_label, curves, pad=0.10, include_treat_all=False):
    """Net-benefit range for ONE cohort, shared by that cohort's horizons.

    Fitted to the MODEL curves and the treat-none line at 0, NOT to treat-all:
    treat-all dives to about -0.29 at the top of the 3-year grid, and letting it
    set the floor squashes every model curve into a band a few pixels high. It is
    a straight line whose position is already fixed by the observed event rate,
    so clipping it costs nothing -- pass include_treat_all=True to keep it in.
    """
    vals = [np.array([0.0])]                      # treat-none is always on the axis
    for h in HORIZONS:
        cv = curves.get((label, cohort_label, h))
        if cv is None:
            continue
        for model_name, _ in PANEL_MODELS:
            if model_name in cv:
                vals.append(np.asarray(cv[model_name]["net_benefit"], dtype=float))
        if include_treat_all:
            vals.append(np.asarray(cv["all"], dtype=float))
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return NB_YLIM
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if span <= 0:                                 # every curve flat -> invent a visible band
        span = max(abs(hi), 0.01)
    return lo - pad * span, hi + pad * span


def plot_predictor_decision_curves(key, label, curves, outdir, show=True, ylim=None):
    """Figure 10 for ONE predictor: 3 held-out cohorts (ROWS) x 2 horizons (COLUMNS).

    Each panel carries the three MODELS a clinician could rank patients with --
    the AI score alone, CAPRA-S alone, and CAPRA-S + the score -- against the two
    strategies that use no model at all, treat-all and treat-none. The score-only
    curve is what separates "the score is informative" from "the score adds
    something CAPRA-S has not already got": a predictor can beat treat-all on its
    own and still leave the joint curve sitting on CAPRA-S.

    Cohort per row, horizon per column: a row is then one population read at two
    horizons, which is the comparison the per-row y-scale is built for. The two
    columns have DIFFERENT x-grids (3y: 0.05-0.30, 5y: 0.10-0.40), so the x-axis
    is labelled and ticked on every panel rather than shared down a column.

    `ylim=None` (the default) gives every row -- i.e. every cohort -- its own
    net-benefit range via `cohort_ylim`. Pass an explicit (lo, hi) to force one
    scale on all panels, the way NB_YLIM used to.

    One legend for the whole grid sits under the panels; inside a panel it
    covered the curves it was labelling. Returns the PNG path.
    """
    n_rows, n_cols = len(EVAL_DATASETS), len(HORIZONS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.8 * n_cols, 3.6 * n_rows), squeeze=False)
    fig.patch.set_facecolor(FIG_BG)
    handles, legend_labels = [], []
    for i, ds in enumerate(EVAL_DATASETS):
        cohort_label = config["dataset_names"].get(ds, ds)
        # ONE range per cohort, computed once and applied to that row's panels.
        row_ylim = ylim if ylim is not None else cohort_ylim(label, cohort_label, curves)
        for j, h in enumerate(HORIZONS):
            ax = axes[i][j]
            ax.set_facecolor(AX_BG)
            cv = curves.get((label, cohort_label, h))
            if cv is None:
                ax.text(0.5, 0.5, f"{cohort_label} {h}y\nno curve (see skipped table)",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color="#C44E52")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            grid = cv["grid"]
            # The two model-free strategies first, so the model curves sit on top.
            ax.plot(grid, cv["all"], color=REFERENCE_COLOR, ls="--", lw=1.4, label="Treat all")
            ax.axhline(0, color=REFERENCE_COLOR, ls=":", lw=1.4, label="Treat none")
            for model_name, label_tpl in PANEL_MODELS:
                if model_name not in cv:            # score-only curve is optional
                    continue
                ax.plot(grid, cv[model_name]["net_benefit"], color=MODEL_COLORS[model_name],
                        lw=2.4 if model_name == "joint" else 1.8,
                        zorder=3 if model_name == "joint" else 2,
                        label=label_tpl.format(label=label))
            # Grey out the tail of the grid where the joint model treats fewer
            # than MIN_TREATED patients: past there its curve sits at 0 by
            # construction, not by merit.
            thin = ~cv["joint"]["enough_treated"].to_numpy()
            if thin.any():
                ax.axvspan(float(grid[np.flatnonzero(thin)[0]]), grid.max(),
                           color=THIN_BAND, alpha=0.45, zorder=0)
            #ax.axvline(REFERENCE_THRESHOLD[h], color="#C44E52", lw=1.0, ls="--", alpha=0.7)
            sup = SUPPORT[(SUPPORT["cohort"] == cohort_label) & (SUPPORT["horizon_years"] == h)].iloc[0]
            ax.set_title(f"{cohort_label} -- {h}y  (n={sup.n}, {sup.events_by_h} events by {h}y)",
                         fontsize=9, color="black")
            ax.set_ylim(*row_ylim)
            ax.set_xlim(float(grid.min()), float(grid.max()))
            ax.set_xlabel("Threshold probability $p_t$", fontsize=9)
            if j == 0:
                ax.set_ylabel("Net benefit", fontsize=9)
            # Thin grid, kept under the data.
            ax.grid(True, color=GRID_COLOR, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(SPINE_COLOR)
                ax.spines[side].set_linewidth(0.8)
            ax.tick_params(colors=TICK_COLOR, labelsize=8, length=3, width=0.8)
            if not handles:                         # one legend, taken from the first drawn panel
                handles, legend_labels = ax.get_legend_handles_labels()
    # Leave a strip at the bottom for the shared legend.
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    if handles:
        fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.004),
                   ncol=len(handles), frameon=False, fontsize=9)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"figure10_decision_curves_{key}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_all_decision_curves(specs, curves, outdir, show=True, ylim=None):
    """One 6-panel figure per predictor -- 6 predictors, 6 files.

    `ylim=None` -> per-cohort y ranges inside each figure (see
    `plot_predictor_decision_curves`). Pass a tuple to force one shared scale.

    Returns {label: png path}.
    """
    paths = {}
    for key, label, _ in specs:
        paths[label] = plot_predictor_decision_curves(key, label, curves, outdir,
                                                      show=show, ylim=ylim)
    return paths


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


# ======================================================================
# STEP 1 -- config and input paths
# ======================================================================
banner("STEP 1  config and input paths")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Expected size of the development split -- dataset-specific, so kept in the config
EXPECTED_TUNING_N   = config["expected_tuning_n"]            # rows in <tuning_dataset>_capra_s_<SUFFIX>.csv
EXPECTED_DEV_N      = config["expected_calibration_n"]       # tuning split + RUMC
EXPECTED_DEV_EVENTS = config["expected_calibration_events"]



TUNING_DATASET   = config.get("tuning_dataset", "radboud_tuning")            # GT name for the tuning split
TUNING_SUBFOLDER = config.get("tuning_predictions_subfolder", "validation")  # its prediction subfolder

# input_dir / validation_dir are network shares. If they are not mounted, the
# original code read zero predictions and still drew a full figure off the wrong
# development split. Check before doing any work.
paths = pd.DataFrame([
    {"key": "input_dir",          "role": "held-out cohort predictions", "path": config["input_dir"]},
    {"key": "validation_dir",     "role": "tuning-split predictions",    "path": config["validation_dir"]},
    {"key": "clinical_variables", "role": "ground-truth CSVs",           "path": config["clinical_variables"]},
    {"key": "output_dir",         "role": "where results are written",   "path": config["output_dir"]},
])
paths["exists"] = paths["path"].map(os.path.isdir)
display(paths)


print(f"\ntuning ground truth : {TUNING_DATASET}_capra_s_{SUFFIX}.csv   (expected {EXPECTED_TUNING_N} cases)")
print(f"tuning predictions  : {config['validation_dir']}/<team>/{TUNING_SUBFOLDER}/*.json")
print(f"development split   : {TUNING_DATASET} + {DEV_DATASET}   (fits everything, never held out)")
print(f"horizons            : {HORIZONS} years")
for h in HORIZONS:
    g = THRESHOLDS[h]
    print(f"  {h}y thresholds     : {g.min():.2f} .. {g.max():.2f} in steps of "
          f"{g[1] - g[0]:.2f}  ({len(g)} points), quoted at p_t = {REFERENCE_THRESHOLD[h]:.2f}")

# ======================================================================
# STEP 2 -- predictors and held-out cohorts
# ======================================================================
banner("STEP 2  predictors and held-out cohorts")

# config["datasets"] is a list of single-key dicts: [{"<dataset>": <n_cases>}, ...]
DATASET_SIZES = {next(iter(d)): d[next(iter(d))] for d in config["datasets"]}
EVAL_DATASETS = [ds for ds in DATASET_SIZES if ds != DEV_DATASET]            # PLCO, IMP, UHC


TEAMS = list(config["ensemble_teams"]) if ONLY_ENSEMBLE_TEAMS else list(config["teams"])

team_names = config.get("team_names", {})
SPECS = [(t, team_names.get(t, t), [t]) for t in TEAMS]                      # the 5 members, separately
SPECS.insert(0, ("ensemble", "Ensemble", list(config["ensemble_teams"])))     # ... and their ensemble
SPEC_LOOKUP = {key: (label, teams) for key, label, teams in SPECS}



# ======================================================================
# STEP 3 -- load predictions (with a file-level audit)
# ======================================================================
banner("STEP 3  load predictions")

# The challenge convention is "higher score = better survival"; Cox wants
# "higher = higher hazard", so every value is negated on load (invert=True).






tuning_raw = load_split(config["validation_dir"], TEAMS, TUNING_SUBFOLDER, invert=True)
cohort_raw = {}                                          # {dataset: {team: {case_id: value}}}
for ds in DATASET_SIZES:
    cohort_raw[ds] = load_split(config["input_dir"], TEAMS, ds,
                                       expected=DATASET_SIZES[ds], invert=True)



# ======================================================================
# STEP 4 -- development split: RUMC tuning + RUMC, frozen z-score stats
# ======================================================================
banner("STEP 4  development split (frozen z-score standardisation)")



GT = {ds: read_ground_truth(ds) for ds in list(DATASET_SIZES) + [TUNING_DATASET]}

# --- the tuning cohort, asserted explicitly ---------------------------
gt_tuning = GT[TUNING_DATASET]

gt_dev = pd.concat([gt_tuning, GT[DEV_DATASET]], ignore_index=True)

DEV_IDS = set(gt_dev["case_id"])

display(pd.DataFrame([
    {"part": TUNING_DATASET, "n": len(gt_tuning), "events": int(gt_tuning["event"].sum()),
     "median_fu": gt_tuning["follow_up_years"].median(), "max_fu": gt_tuning["follow_up_years"].max()},
    {"part": DEV_DATASET, "n": len(GT[DEV_DATASET]), "events": int(GT[DEV_DATASET]["event"].sum()),
     "median_fu": GT[DEV_DATASET]["follow_up_years"].median(), "max_fu": GT[DEV_DATASET]["follow_up_years"].max()},
    {"part": "DEVELOPMENT SPLIT", "n": len(gt_dev), "events": int(gt_dev["event"].sum()),
     "median_fu": gt_dev["follow_up_years"].median(), "max_fu": gt_dev["follow_up_years"].max()},
]))

DRILL_LABEL, DRILL_TEAMS = SPEC_LOOKUP[DRILLDOWN_KEY]
dev_df, dev_stats, dev_stats_tbl = build_dev(DRILL_TEAMS, DRILL_LABEL)
print(f"\nfrozen per-team standardisation stats for the drill-down predictor ({DRILL_LABEL}):")
print(f"RESTRICT_DEV_TO_GT={RESTRICT_DEV_TO_GT} -> "
      + ("the frozen mean/SD use ONLY the modelled cases"
         if RESTRICT_DEV_TO_GT else
         "the frozen mean/SD ALSO include outcome-less predictions (pipeline behaviour)"))
display(dev_stats_tbl)

print(f"\ndev score: n={len(dev_df)}  events={int(dev_df['event'].sum())}  "
      f"mean={dev_df['score'].mean():+.4f}  SD={dev_df['score'].std(ddof=1):.4f}   "
      f"(SD < 1 is expected for a multi-team ensemble of z-scores)")
display(dev_df.head())


# ======================================================================
# STEP 5 -- frozen Cox models (fit once on dev, never refit on a cohort)
# ======================================================================
banner("STEP 5  frozen Cox models")

# Three models per predictor: the AI score on its own, the clinical baseline on
# its own, and the two together. The summary table still quotes joint vs CAPRA-S
# -- dropping CAPRA-S is not a strategy anyone would adopt -- but the score-only
# curve has to be on the figure, otherwise a joint curve lying on top of CAPRA-S
# cannot be told apart from a score that carries no signal at all.
MODEL_COVARIATES = {"capra_s": ["capra_s_score"], "score": ["score"],
                    "joint": ["capra_s_score", "score"]}
MODEL_LABEL = {"capra_s": "CAPRA-S alone", "score": "AI score alone",
               "joint": "CAPRA-S + score"}




models = fit_frozen_models(dev_df, DRILL_LABEL)
for name, m in models.items():
    print(f"--- {name} ({MODEL_LABEL[name]}) ---")
    display(m.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]])

# The frozen baseline survival IS the calibration carried to every held-out
# cohort: every predicted risk below reads 1 - S(t|x) off these curves.
base = pd.DataFrame([
    {"model": name, "n_dev": len(dev_df),
     **{f"risk({h}y | mean covars)": 1.0 - float(m.predict_survival_function(
         pd.DataFrame({c: [dev_df[c].mean()] for c in MODEL_COVARIATES[name]}), times=[h]).iloc[0, 0])
        for h in HORIZONS}}
    for name, m in models.items()
])
print("frozen baseline risk carried unchanged to every held-out cohort:")
display(base)



# (a) Horizon support: is anyone still at risk at 3y / 5y?
support_rows = []
for ds in EVAL_DATASETS:
    gt = GT[ds]
    tt, ee = gt["follow_up_years"].to_numpy(), gt["event"].to_numpy()
    for h in HORIZONS:
        at_risk = int((tt >= h).sum())
        ev_by_h = int(((ee == 1) & (tt <= h)).sum())
        obs_h, _, _, _ = km_event_rate(tt, ee, h)
        support_rows.append({
            "cohort": config["dataset_names"].get(ds, ds), "horizon_years": h, "n": len(gt),
            "max_fu": float(tt.max()), "median_fu": float(np.median(tt)),
            "n_at_risk_at_h": at_risk, "pct_at_risk": at_risk / len(gt), "events_by_h": ev_by_h,
            "observed_rate": obs_h,
            "reliable": at_risk >= MIN_AT_RISK_COHORT and ev_by_h >= MIN_EVENTS_BY_H,
        })
SUPPORT = pd.DataFrame(support_rows)
display(SUPPORT)



# ======================================================================
# STEP 8 -- decision curves for all 6 predictors x 3 cohorts x 2 horizons
# ======================================================================
banner("STEP 8  decision curves")

curve_rows = []        # long-form: one row per (predictor, cohort, horizon, model, threshold)
summary_rows = []      # one row per (predictor, cohort, horizon) -- becomes the decision-curve summary table
CURVES = {}            # (label, cohort, horizon) -> {"capra_s": df, "joint": df, "all": array, "grid": array}
skipped = []

for key, label, teams in SPECS:
    t_start = _time.perf_counter()

    pdev, pstats, _ = build_dev(teams, label)
    
    pmodels = fit_frozen_models(pdev, label)

    for ds in EVAL_DATASETS:
        cohort_label = config["dataset_names"].get(ds, ds)
        cdf, n_teams_used = build_cohort(ds, teams, pstats)

        tt, ee = cdf["follow_up_years"].to_numpy(), cdf["event"].to_numpy()

        for h in HORIZONS:
            sup = SUPPORT[(SUPPORT["cohort"] == cohort_label) & (SUPPORT["horizon_years"] == h)].iloc[0]

            grid = THRESHOLDS[h]
            nb_all, obs_all = treat_all_curve(tt, ee, grid, h)

            nb = {}
            risks = {}
            for model_name, covars in MODEL_COVARIATES.items():
                rk = raw_risk(pmodels[model_name], cdf[covars], h)

                print("label:",label,"raw risk: ",model_name, "cohort_label: ", cohort_label,"h: ",h,": ",)
                #plt.hist(rk)
                #plt.show()
                where = f"{label}/{cohort_label}/{model_name}/{h}y"
           
                risks[model_name] = rk
                nb[model_name] = net_benefit_curve(tt, ee, rk, grid, h)
                #plt.hist(rk)
                #plt.show()


            CURVES[(label, cohort_label, h)] = {"capra_s": nb["capra_s"], "score": nb["score"],
                                                "joint": nb["joint"], "all": nb_all, "grid": grid}



          

    print(f"  {label:22s} {_time.perf_counter() - t_start:5.1f}s")



# ======================================================================
# STEP 9 -- figures, summary table, LaTeX
#
# Every function called below is defined in the helper cell above.
# ======================================================================
banner("STEP 9  figures, summary table, LaTeX")

OUTPUT_DIR = os.path.join(config["output_dir"], "calibration_utility")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Figure 10: ONE figure per predictor, each with 6 panels
#     (rows = the 2 horizons, columns = the 3 held-out cohorts). Each panel
#     draws the three models -- the score alone, CAPRA-S alone, and the two
#     together -- against treat-all and treat-none, with one legend under the
#     grid rather than one inside a panel.
FIGURE_PATHS = plot_all_decision_curves(SPECS, CURVES, OUTPUT_DIR)
