# Rebuttal analysis scripts

Scripts for the rebuttal analyses. Each script is standalone and starts with a `# Source:` line naming the
analysis notebook cell it was extracted from.

## Setup

```
pip install -r requirements.txt
```

Every script reads a YAML config whose path is hardcoded in the script (`config_path`, `CONFIG_PATH`,
or the argument to `main(...)`). Point it at your config before running. The config uses the same
format as [`config/config.yaml`](../../config/config.yaml), plus these keys:

| Key | Used by |
|---|---|
| `tuning_dataset`, `tuning_predictions_subfolder` | scripts that fit frozen models on the RUMC calibration set (RUMC Tuning + RUMC Internal Validation) |
| `expected_tuning_n`, `expected_calibration_n`, `expected_calibration_events` | `calibration/` and `decision_curve/`, as sanity checks on the calibration set |

No data, predictions or ground truth are included. Run the scripts against your own copies of those.

## Layout

Outputs go to the config's `output_dir`, except where noted.

### `c_index/`

| Script | Writes |
|---|---|
| `capra_s_c_index.py` | `c_index_capra_results_rebuttal_median.csv` |
| `isup_c_index.py` | `c_index_isup_results_rebuttal_median.csv` |
| `team_c_index.py` | `c_index_results_median.{csv,tex}` |
| `ensemble_c_index.py` | `c_index_ensemble_results_median.{csv,tex}` |
| `team_plus_capra_s_c_index.py` | `c_index_model_capra_s_combined_results_median.{csv,tex}` |
| `team_plus_isup_c_index.py` | `c_index_model_isup_combined_results_median.{csv,tex}` |
| `team_plus_isup_c_index_hr.py` | `c_index_hr_model_isup_combined_results_median.*` |
| `team_plus_capra_s_c_index_hr.py` | `c_index_hr_model_capra_s_combined_results_median.*` |
| `team_plus_capra_s_c_index_calib.py` | `c_index_model_capra_s_combined_results_median_calib.*` |
| `team_plus_isup_c_index_calib.py` | `c_index_model_isup_combined_results_median_calib.*` |
| `ensemble_c_index_isup_subgroups.py` | `c_index_isup_subgroups_capra_s_rebuttal_median_nat_com_lh.*` |

### `cox/`

| Script | Writes |
|---|---|
| `ensemble_global_cox_capra_s.py` | `total_cox_metrics_capra_s_global_rebuttal_median_nat_com.{csv,tex}` |
| `ensemble_global_cox_isup.py` | `total_cox_metrics_isup_s_global.{csv,tex}` |
| `ensemble_per_dataset_cox_capra_s.py` | `per_dataset_cox_metrics_capra_s_rebuttal_median_nat_com.*` |
| `ensemble_per_dataset_cox_isup.py` | `per_dataset_cox_metrics_isup_rebuttal_median_nat_com.*` |
| `ensemble_metrics_by_risk_and_isup_group.py` | `combined_metrics_capra_s_median_isup_group.{csv,tex}` |
| `team_cox_metrics_capra_s.py` | `cox_metrics_capra_s_median.{csv,tex}` |
| `team_cox_metrics_isup.py` | `cox_metrics_isup_median.{csv,tex}` |

### `roc_auc/`

| Script | Writes |
|---|---|
| `roc_pr_youden_ipcw.py` | ROC/PR figure, `youden_threshold_metrics_*_calib_ipcw.{csv,tex}`, `auc_ap_summary_*_calib_ipcw.csv` |
| `time_dependent_auc.py` | `calibration_utility/td_auc_ensemble_teams.png`, `suppfig3_time_dependent_auc_ensemble_teams.csv` |

### `calibration/`, `decision_curve/`

| Script | Writes (to `calibration_utility/`) |
|---|---|
| `calibration/calibration_table6.py` | `table6_calibration*`, `calibration_summary.{csv,tex}`, calibration-curve figures |
| `decision_curve/decision_curves.py` | `figure10_decision_curves_<predictor>.png`, `figure10_delta_net_benefit.png`, `table_decision_curves*`, `table_dca_horizon_support.csv` |
| `decision_curve/decision_curves_figures.py` | `figure10_decision_curves_<predictor>.png` only (restyled, per-cohort y-limits) |

The two decision-curve scripts write figures with the same names, so whichever runs last wins.

### `figures/`

| Script | Writes |
|---|---|
| `isup_distribution.py` | `isup_distribution_datasets_median.png` (path hardcoded in the script) |
| `kaplan_meier_risk_groups.py` | `kaplan_meier_<team>_median.png` |
| `c_index_comparison_capra_s.py` | `detailed_c_index_capra_plots_median.png` |
| `c_index_comparison_isup.py` | `detailed_c_index_plots_median_calib.png` |
| `methodologies_table.py` | `methodologies.png` (path hardcoded in the script) |

## Run order

Only the two C-index comparison figures depend on other scripts. They read these CSVs, so run the
listed scripts first:

- `figures/c_index_comparison_capra_s.py`: `c_index/team_c_index.py`, `c_index/ensemble_c_index.py`,
  `c_index/capra_s_c_index.py`, `c_index/team_plus_capra_s_c_index_calib.py`
- `figures/c_index_comparison_isup.py`: `c_index/team_c_index.py`, `c_index/ensemble_c_index.py`,
  `c_index/isup_c_index.py`, `c_index/team_plus_isup_c_index_calib.py`
