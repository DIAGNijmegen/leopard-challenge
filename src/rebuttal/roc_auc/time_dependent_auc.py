# Source: run.ipynb, cell 29
# Time-dependent AUC figure + CSV per ensemble team and the Ensemble.

# ======================================================================
# Standalone setup -- duplicated here so this cell runs on its own
# (Kernel > Restart, then run just this cell -- no need to run anything above)
# ======================================================================

# ======================================================================
# Standalone setup -- duplicated here so this cell runs on its own
# (Kernel > Restart, then run just this cell -- no need to run anything above)
# ======================================================================

import os                                    # filesystem paths (predictions live under input_dir/team/dataset/*.json)
import json                                  # each prediction file is a single JSON number
import logging                               # the module logs warnings on skipped/missing data; mirrored here
import yaml                                  # config/config-mac.yaml is a YAML file

import numpy as np                           # numerical arrays, percentiles, z-scoring
import pandas as pd                          # DataFrames for every table result
import matplotlib.pyplot as plt              # plotting the two figure results
from IPython.display import display          # explicit table display (works regardless of cell position)

from lifelines import CoxPHFitter, KaplanMeierFitter    # Cox regression + Kaplan-Meier estimator
from lifelines.utils import concordance_index            # C-index (discrimination) metric

from sksurv.metrics import brier_score, cumulative_dynamic_auc   # IPCW Brier score + Uno's time-dependent AUC
from sksurv.util import Surv                                      # structured (event, time) array sksurv expects
# ======================================================================
# Helper functions for the time-dependent AUC figure (Figure 2)
# ======================================================================



def get_datasets(config):
    # config["datasets"] is a list of single-key dicts, e.g. [{"radboud": 100}, {"plco": 724}, ...]
    return [list(d.keys())[0] for d in config["datasets"]]      # pull out just the dataset-name keys


def get_predictor_specs(config):
    # one row per individual team, plus one row for the multi-team ensemble
    team_names = config.get("team_names", {})                             # optional id -> pretty-name lookup
    specs = [(team, team_names.get(team, team), [team]) for team in config["teams"]]   # (key, label, [team])
    specs.append(("ensemble", "Ensemble", list(config["ensemble_teams"])))             # ensemble row appended last
    return specs


def get_eval_datasets(config):
    # every configured dataset except RUMC, which is folded into the development split instead
    return [d for d in get_datasets(config) if d != DEV_DATASET]


def cohort_label(dataset):
    # pretty cohort name; RUMC is flagged because it is part of the development split (in-sample)
    name = config["dataset_names"].get(dataset, dataset)
    return name


def load_predictions_for_dataset(input_dir, teams, dataset, invert=False):
    # reads validation_dir/<team>/<dataset>/*.json for a split with no expected-count check
    predictions = {}                                           # result: {team: {case_id: value}}
    for team in teams:                                         # one subfolder per team
        dataset_path = os.path.join(input_dir, team, dataset)  # e.g. validation_dir/mevis_updated/validation
        if not os.path.isdir(dataset_path):                    # team may not have predictions for this split
            continue                                           # skip silently, matches the pipeline's behaviour
        team_preds = {}                                        # this team's {case_id: value}
        for fn in os.listdir(dataset_path):                    # every file in the team's folder
            if fn.endswith(".json"):                           # only JSON prediction files
                with open(os.path.join(dataset_path, fn)) as f:   # one file per case
                    team_preds[fn[:-5]] = json.load(f)          # filename minus ".json" is the case_id
        if team_preds:                                         # only keep teams that had at least one file
            if invert:                                         # challenge convention: higher = better survival
                team_preds = {k: -v for k, v in team_preds.items()}   # negate so higher = higher hazard (Cox convention)
            predictions[team] = team_preds                     # store this team's (possibly inverted) predictions
    return predictions

def load_predictions(input_dir, teams, datasets, invert=False):
    # reads input_dir/<team>/<dataset>/*.json for every dataset in config["datasets"], with a count check
    predictions = {}                                             # result: {team: {dataset: {case_id: value}}}
    for team in teams:                                           # one subfolder per team
        predictions[team] = {}                                   # this team's per-dataset dict
        for dataset_dict in datasets:                            # each entry is like {"plco": 724}
            dataset = next(iter(dataset_dict))                   # the dataset name (the dict's only key)
            dataset_path = os.path.join(input_dir, team, dataset)   # e.g. input_dir/mevis_updated/plco
            if not os.path.isdir(dataset_path):                  # team may be missing this dataset entirely
                continue                                         # skip silently
            files = [f for f in os.listdir(dataset_path) if f.endswith(".json")]   # every prediction file
            expected = dataset_dict[dataset]                     # the case count the config says to expect
            if len(files) != expected:                           # flag a mismatch (missing/extra predictions)
                logging.warning(f"{team}/{dataset} expected {expected} preds, found {len(files)}")
            team_preds = {}                                      # this (team, dataset)'s {case_id: value}
            for fn in files:                                     # one file per case
                cid = fn[:-5]                                    # filename minus ".json" is the case_id
                with open(os.path.join(dataset_path, fn)) as f:  # read the single JSON number
                    team_preds[cid] = json.load(f)
            if team_preds:                                       # only keep it if something was actually read
                if invert:                                       # same sign flip as load_predictions_for_dataset
                    team_preds = {k: -v for k, v in team_preds.items()}
                predictions[team][dataset] = team_preds           # store under predictions[team][dataset]
    return predictions


def gt_path(clinical_dir, dataset, suffix):
    # ground-truth CSV path, e.g. clinical_variables/plco_capra_s_median.csv
    return os.path.join(clinical_dir, f"{dataset}_capra_s_{suffix}.csv")    # just a filename join, no I/O here


def tuning_stats(raw_dict):
    # per-team (mean, std) of a {case_id: value} dict of raw (already-inverted) predictions
    values = np.array(list(raw_dict.values()), dtype=float)     # just the values, in dict order
    return float(values.mean()), float(values.std(ddof=1))      # sample SD (ddof=1), matches lifelines' convention


def zscore_apply(raw_dict, mean, std):
    # (value - mean) / std, case by case; guards against a degenerate zero SD
    return {k: (v - mean) / std for k, v in raw_dict.items()}    # standard z-score


def combine_scores(case_ids, per_team_scaled):
    # average each case's available per-team z-scores into one predictor score
    combined = {}                                                 # result: {case_id: averaged score}
    for cid in case_ids:                                          # every case in the ground truth
        vals = [d[cid] for d in per_team_scaled.values() if cid in d]   # this case's z-score from each team that has one
        
                                                         # only count teams with a prediction for this case
        combined[cid] = float(np.mean(vals))                  # simple mean across available teams
    return combined

def load_dev_score(config, teams, suffix, all_tuning_raw, all_cohort_raw):
    tuning_name = config.get("tuning_dataset", "radboud_tuning")       # ground-truth name for the tuning split

    combined_raw = {}                                                   # {team: {case_id: raw inverted value}}
    for team in teams:                                                  # merge tuning-split + RUMC predictions
        merged = {**all_tuning_raw.get(team, {}), **all_cohort_raw.get(team, {}).get(DEV_DATASET, {})}
        # only keep teams with any prediction
        combined_raw[team] = merged

    dev_stats = {team: tuning_stats(preds) for team, preds in combined_raw.items()}            # frozen (mean, std) per team
    scaled = {team: zscore_apply(preds, *dev_stats[team]) for team, preds in combined_raw.items()}   # z-score each team

    gt_val = pd.read_csv(gt_path(config["clinical_variables"], tuning_name, suffix), dtype={"case_id": str})    # tuning GT
    gt_rumc = pd.read_csv(gt_path(config["clinical_variables"], DEV_DATASET, suffix), dtype={"case_id": str})   # RUMC GT
    gt = pd.concat([gt_val, gt_rumc], ignore_index=True)                # development split = tuning + RUMC, stacked
    assert gt["case_id"].is_unique, "tuning and RUMC ground truth case_ids overlap"    # sanity: no shared patients

    score = combine_scores(set(gt["case_id"]), scaled)                  # one averaged score per case_id
    df = gt[gt["case_id"].isin(score)].copy()                           # keep only cases that got a score
    df["score"] = df["case_id"].map(score)                              # attach the predictor score column
    keep = [c for c in ["case_id", "event", "follow_up_years", "capra_s_score", "ISUP", "score"] if c in df.columns]
    return df[keep].reset_index(drop=True), dev_stats                   # (dev_df, dev_stats)


def compute_td_auc(cohort_df, models):
    # Uno's time-dependent AUC of CAPRA-S alone, AI alone and the joint model on one cohort, on one shared time grid
    t_i = cohort_df["follow_up_years"].values
    e_i = cohort_df["event"].values.astype(bool)
    y = Surv.from_arrays(event=e_i, time=t_i)                                  # sksurv structured array

    grid = np.linspace(t_i[e_i].min() if e_i.any() else t_i.min(), t_i.max() * 0.9, 15)   # 15 evaluation times
    grid = grid[(grid > 0) & (grid < t_i.max())]                                # keep the grid inside the follow-up range

    covariates = {"auc_capra_s": ("capra", ["capra_s_score"]),                 # CSV column -> (frozen model, its covariates)
                  "auc_ai_alone": ("score", ["score"]),
                  "auc_joint": ("joint", ["capra_s_score", "score"])}
    aucs = {}
    for column, (model_name, cols) in covariates.items():
        risk = models[model_name].predict_partial_hazard(cohort_df[cols]).values    # partial hazard from the frozen Cox model
        aucs[column], _ = cumulative_dynamic_auc(y, y, risk, grid)                  # Uno's time-dependent AUC
    return grid, aucs

def load_cohort_score(config, dataset, teams, dev_stats, suffix, all_cohort_raw):
    gt = pd.read_csv(gt_path(config["clinical_variables"], dataset, suffix), dtype={"case_id": str})   # cohort GT

    scaled = {}                                                          # {team: {case_id: z-scored value}}
    for team in teams:                                                   # for every team in this predictor
        team_preds = all_cohort_raw.get(team, {}).get(dataset)           # this team's raw predictions on this cohort
        if team_preds and team in dev_stats:                             # need both predictions and frozen stats
            scaled[team] = zscore_apply(team_preds, *dev_stats[team])    # z-score with the DEV split's mean/std (frozen!)
    score = combine_scores(set(gt["case_id"]), scaled)                   # average across teams, same as load_dev_score

    df = gt[gt["case_id"].isin(score)].copy()                            # keep only cases that got a score
    df["score"] = df["case_id"].map(score)                               # attach the predictor score column
    keep = [c for c in ["case_id", "event", "follow_up_years", "capra_s_score", "ISUP", "score"] if c in df.columns]
    return df[keep].reset_index(drop=True)

def fit_frozen_models(dev_df):
    # three Cox models, fit once on the development split -- never refit on a held-out cohort
    capra = CoxPHFitter().fit(                                            # CAPRA-S alone
        dev_df[["capra_s_score", "follow_up_years", "event"]], "follow_up_years", "event"
    )
    score_only = CoxPHFitter().fit(                                       # predictor score alone (unused downstream)
        dev_df[["score", "follow_up_years", "event"]], "follow_up_years", "event"
    )
    joint = CoxPHFitter().fit(                                            # CAPRA-S + predictor score together
        dev_df[["capra_s_score", "score", "follow_up_years", "event"]], "follow_up_years", "event"
    )
    return {"capra": capra, "score": score_only, "joint": joint}          # keep all three; only capra/joint reused below


def compute_td_auc_table(predictor_keys, cohorts):
    # long-format table: one row per (predictor, cohort, grid time point)
    auc_rows = []
    for key in predictor_keys:
        label, teams = spec_lookup[key]
        p_dev_df, p_dev_stats = load_dev_score(config, teams, SUFFIX, all_tuning_raw, all_cohort_raw)   # this predictor's dev split
        p_models = fit_frozen_models(p_dev_df)                                                          # frozen, fit once per predictor
        for dataset in cohorts:
            cohort_df_i = load_cohort_score(config, dataset, teams, p_dev_stats, SUFFIX, all_cohort_raw)   # frozen scoring
            grid, aucs = compute_td_auc(cohort_df_i, p_models)
            for i, t_grid in enumerate(grid):                                   # one row per grid time point
                auc_rows.append({"Cohort": cohort_label(dataset), "Predictor": label, "time_years": t_grid,
                                 **{column: values[i] for column, values in aucs.items()}})
    return pd.DataFrame(auc_rows)


def plot_td_auc_grid(auc_curves, predictor_labels, cohort_labels):
    # rows = predictors (top to bottom), columns = cohorts; reads only the long-format table, so it can be re-run from the CSV
    n_rows, n_cols = len(predictor_labels), len(cohort_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                             sharex="col", sharey=True, squeeze=False, layout="constrained")
    fig.get_layout_engine().set(hspace=0.1)                                     # extra vertical space between rows

    for row, predictor in enumerate(predictor_labels):
        for col, cohort in enumerate(cohort_labels):
            ax = axes[row, col]
            sub = auc_curves[(auc_curves["Predictor"] == predictor) & (auc_curves["Cohort"] == cohort)]
            for column, legend_label, color in AUC_CURVES:
                ax.plot(sub["time_years"], sub[column], label=legend_label, color=color,
                        marker="o", markersize=3.5)                              # dot at every grid time point
            ax.axhline(0.5, color="gray", linestyle=":")                         # chance-level reference
                                             
            ax.set_facecolor("#FFFFFF")                                           # off-white panel background
            ax.grid(True, color="#F5F5F5", linewidth=1)                             # white grid lines
            ax.set_axisbelow(True)                                                # grid behind the curves
            ax.set_ylim(0.65, 1.0)                          # chance-level reference
            if row == 0:
                ax.set_title(cohort)
            if col == 0:
                ax.set_ylabel(f"{predictor}\nTime-dependent AUC")
            if row == n_rows - 1:
                ax.set_xlabel("Time (years)")

    handles, labels = axes[0, 0].get_legend_handles_labels()                     # one shared legend for the whole figure
    fig.legend(handles, labels, loc="outside upper center", ncol=len(AUC_CURVES), frameon=False)
    #fig.suptitle("Time-dependent AUC")
    return fig



# ======================================================================
# ======================================================================

%matplotlib inline

CONFIG_PATH = "/Users/khrystynafaryna/Documents/leopard-rebuttal/config.yaml"   # same config file the real pipeline (main.py) uses
with open(CONFIG_PATH, "r") as f:            # open the YAML file for reading
    config = yaml.safe_load(f)               # parse it into a plain dict -- replaces src.utils.load_config

print("input_dir      :", config["input_dir"])          # where held-out cohort predictions live
print("validation_dir :", config["validation_dir"])     # where tuning/validation-split predictions live
print("output_dir     :", config["output_dir"])         # not written to by this notebook -- read-only walkthrough
print("teams          :", config["teams"])               # every team scored individually
print("ensemble_teams :", config["ensemble_teams"])      # subset averaged into the "Ensemble" predictor

DEV_DATASET = "radboud"                      # RUMC: merged into the development split, never held out

specs = get_predictor_specs(config)          # compute once, reused by every result below
eval_datasets = get_eval_datasets(config)    # the 3 held-out cohorts: plco, brazil, cologne

for key, label, teams in specs:              # print them so you can see what "ensemble" expands to
    print(f"{key:20s} label={label:20s} teams={teams}")
print("\nheld-out cohorts:", eval_datasets)



pred_subfolder = config.get("tuning_predictions_subfolder", "validation")    # subfolder name under validation_dir
all_tuning_raw = load_predictions_for_dataset(                               # {team: {case_id: value}}
    config["validation_dir"], config["teams"], pred_subfolder, invert=True
)

all_cohort_raw = load_predictions(                                # {team: {dataset: {case_id: value}}}
    config["input_dir"], config["teams"], config["datasets"], invert=True
)

PREDICTOR_KEY = "ensemble"                    # any key printed above -- a team id, or "ensemble"
SUFFIX = "median"                             # which *_capra_s_<suffix>.csv ground-truth file to read

spec_lookup = {key: (label, teams) for key, label, teams in specs}   # key -> (pretty label, team list)
PREDICTOR_LABEL, PREDICTOR_TEAMS = spec_lookup[PREDICTOR_KEY]        # unpack the chosen predictor's spec

dev_df, dev_stats = load_dev_score(config, PREDICTOR_TEAMS, SUFFIX, all_tuning_raw, all_cohort_raw)   # run it

# ----------------------------------------------------------------------
# RESULT 5 -- Figure: time-dependent AUC
# one row per ensemble team + one for the Ensemble; one column per cohort (RUMC + held-out)
# ----------------------------------------------------------------------

OUTPUT_DIR = os.path.join(config["output_dir"], "calibration_utility")        # same layout src/calibration_utility.py uses
os.makedirs(OUTPUT_DIR, exist_ok=True)

AUC_PREDICTOR_KEYS = ["ensemble"]  + list(config["ensemble_teams"])        # each ensemble member, then the Ensemble itself
AUC_COHORTS = [DEV_DATASET] + eval_datasets                                   # RUMC first, then plco, brazil, cologne
AUC_CURVES = [    
    ("auc_capra_s", "CAPRA-S alone", "#009AD2"),                                                            # (CSV column, legend label, colour), drawn in order
    ("auc_joint", "CAPRA-S + AI","#DC497F"),
    #("auc_ai_alone", "AI alone", "#EDAE49"),
    
]

auc_curves = compute_td_auc_table(AUC_PREDICTOR_KEYS, AUC_COHORTS)

fig = plot_td_auc_grid(
    auc_curves,
    predictor_labels=[spec_lookup[key][0] for key in AUC_PREDICTOR_KEYS],
    cohort_labels=[cohort_label(d) for d in AUC_COHORTS],
)

auc_csv_path = os.path.join(OUTPUT_DIR, "suppfig3_time_dependent_auc_ensemble_teams.csv")
auc_curves.to_csv(auc_csv_path, index=False)
auc_png_path = os.path.join(OUTPUT_DIR, "td_auc_ensemble_teams.png")
fig.savefig(auc_png_path, dpi=150)
print(f"Saved {auc_csv_path} ({len(auc_curves)} rows) and {auc_png_path}")

plt.show()


