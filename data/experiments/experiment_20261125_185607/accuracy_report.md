# Gaze Tracking Experiment Report: `experiment_20261125_185607`

## Experiment Overview

| Field | Value |
|---|---|
| **Experiment ID** | `experiment_20261125_185607` |
| **Date and Time** | 2026-11-25 18:56:45 |
| **Screen Resolution** | 1920x1080 (Diagonal: 2202.9 px) |
| **Calibration Model** | DECOUPLED_LINEAR |
| **Normalization Method** | StandardScaler |
| **Number of Training Samples** | 9 targets (249 accepted frames) |
| **Validation Status** | `COMPLETED` |
| **Number of Validation Trials** | 9 valid / 9 planned (0 rejected, 0.0% rej rate) |

## Main Results

> **Accuracy percentages** below measure how often gaze predictions landed successfully within strict screen tolerances or inside target boundaries (higher is better). **Average Error** measures distance from the true target (lower is better).

| Result | Value |
|---|---:|
| **Accuracy within 2%** | **22.2%** (2/9 trials, within 44.1 px) |
| **Accuracy within 5%** | **55.6%** (5/9 trials, within 110.1 px) |
| **Accuracy within 10%** | **100.0%** (9/9 trials, within 220.3 px) |
| **Target Selection Accuracy** | **55.6%** (5/9 hits) |
| **Average Error (pixels)** | 93.6 px |
| **Average Error (% of screen diagonal)** | 4.2% |
| **Rejected Trial Rate** | 0.0% (0/9) |

## Detailed Results

### Prediction Error Breakdown (Independent Validation Trials)

| Error Measurement | Pixels | Screen Diagonal / Normalized |
|---|---:|---:|
| **X-Axis MAE** | 59.9 px | 0.0312 norm |
| **Y-Axis MAE** | 66.7 px | 0.0617 norm |
| **X-Axis RMSE** | 80.5 px | 0.0419 norm |
| **Y-Axis RMSE** | 82.9 px | 0.0767 norm |
| **Median Error** | 74.7 px | 3.4% diag (0.0555 norm) |
| **Maximum Error** | 204.6 px | 9.3% diag (0.1527 norm) |
| **90th Percentile Error** | 192.5 px | — |

### Target Selection & Trial Counts

- **Target Boundary Evaluated:** Button Box (200x150 px)
- **Successful Selections (Hits):** 5
- **Failed Selections (Misses):** 4
- **Total Valid Trials:** 9
- **Rejected / Invalid Trials:** 0 (0.0%)

### Direction Accuracy Breakdown (LEFT / RIGHT / CENTER / UP / DOWN)

| Direction | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `LEFT` | 3 / 0 | **33.3%** | 33.3% | 112.4 px | (+84.2, +6.2) |
| `RIGHT` | 3 / 0 | **66.7%** | 66.7% | 86.6 px | (-12.6, +43.1) |
| `CENTER` | 1 / 0 | **100.0%** | 100.0% | 48.0 px | (+32.1, +35.7) |
| `UP` | 3 / 0 | **0.0%** | 0.0% | 146.5 px | (+71.4, +126.8) |
| `DOWN` | 3 / 0 | **66.7%** | 66.7% | 88.9 px | (+49.5, -41.6) |

### Screen Region Accuracy Breakdown (Rows, Columns & Edges)

| Screen Region | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `top_row` | 3 / 0 | **0.0%** | 0.0% | 146.5 px | (+71.4, +126.8) |
| `middle_row` | 3 / 0 | **100.0%** | 100.0% | 45.4 px | (-3.5, +26.8) |
| `bottom_row` | 3 / 0 | **66.7%** | 66.7% | 88.9 px | (+49.5, -41.6) |
| `left_column` | 3 / 0 | **33.3%** | 33.3% | 112.4 px | (+84.2, +6.2) |
| `center_column` | 3 / 0 | **66.7%** | 66.7% | 81.8 px | (+45.8, +62.7) |
| `right_column` | 3 / 0 | **66.7%** | 66.7% | 86.6 px | (-12.6, +43.1) |
| `edge_regions` | 8 / 0 | **50.0%** | 50.0% | 99.3 px | (+40.0, +37.5) |
| `center_region` | 1 / 0 | **100.0%** | 100.0% | 48.0 px | (+32.1, +35.7) |

### Per-Target Performance

| Target Point | Valid / Rejected | Hits / Valid | Target Selection Accuracy | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `val_bottom_center` | 1 / 0 | 1 / 1 | **100.0%** | 7.7 px | (-7.7, +0.3) |
| `val_bottom_left` | 1 / 0 | 0 / 1 | **0.0%** | 204.6 px | (+187.9, -80.9) |
| `val_bottom_right` | 1 / 0 | 1 / 1 | **100.0%** | 54.4 px | (-31.8, -44.1) |
| `val_center` | 1 / 0 | 1 / 1 | **100.0%** | 48.0 px | (+32.1, +35.7) |
| `val_middle_left` | 1 / 0 | 1 / 1 | **100.0%** | 13.3 px | (+11.4, -6.9) |
| `val_middle_right` | 1 / 0 | 1 / 1 | **100.0%** | 74.7 px | (-53.9, +51.7) |
| `val_top_center` | 1 / 0 | 0 / 1 | **0.0%** | 189.5 px | (+113.1, +152.1) |
| `val_top_left` | 1 / 0 | 0 / 1 | **0.0%** | 119.2 px | (+53.4, +106.6) |
| `val_top_right` | 1 / 0 | 0 / 1 | **0.0%** | 130.7 px | (+47.9, +121.6) |

### Training Performance vs. Independent Validation Performance

> **Note:** Training error only measures how tightly the model fits the 9 calibration points and must NOT be interpreted as real-world validation accuracy.

| Evaluation Split | Purpose | Mean Error (px) | Validation Accuracy |
|---|---|---:|---:|
| **Calibration Training Set** | Model fitting residual (training fit only) | 84.3 px | N/A (training split) |
| **Independent Validation Set** | Unseen test targets (real accuracy) | 93.6 px (4.2% diag) | 55.6% hits / 55.6% within 5% |

## Simple Explanation

Status: [COMPLETED]. The system successfully predicted gaze positions inside the target boundaries in 5 out of 9 valid trials (55.6%). Its average prediction error was 93.6 pixels (median: 74.7 px, 4.2% of the screen diagonal), and 55.6% of valid trials fell within 5% of the screen diagonal (5 out of 9). Weakest direction observed: UP (mean error 146.5 px); weakest screen region: top_row (mean error 146.5 px). This experiment performed better than the previous validated experiment (experiment_20261125_185024) according to the independent validation results (average error changed from 239.2 px to 93.6 px (60.9% error reduction); target selection accuracy changed from 0.0% to 55.6% (+55.6 percentage points)).

## Comparison with Previous Experiments

| Experiment ID | Model | Norm | Sel Acc | Within 2% | Within 5% | Within 10% | Mean Error | Median | Rej % | Weakest Dir / Region | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `experiment_20261123_221716` | **POLYNOMIAL** | None | N/A | N/A | N/A | N/A | N/A | N/A | N/A | -- | Calibrated Only (No Validation) |
| `experiment_20261123_224241` | **POLYNOMIAL** | None | N/A | N/A | N/A | N/A | N/A | N/A | N/A | -- | Calibrated Only (No Validation) |
| `experiment_20261124_220355` | **POLYNOMIAL** | StandardScaler | N/A | N/A | N/A | N/A | N/A | N/A | N/A | -- | Calibrated Only (No Validation) |
| `experiment_20261124_221738` | **POLYNOMIAL** | StandardScaler | N/A | N/A | N/A | N/A | N/A | N/A | N/A | -- | Calibrated Only (No Validation) |
| `experiment_20261124_225435` | **LINEAR** | StandardScaler | N/A | N/A | N/A | N/A | N/A | N/A | N/A | -- | Calibrated Only (No Validation) |
| `experiment_20261124_225544` | **COMPACT_POLYNOMIAL** | StandardScaler | N/A | N/A | N/A | N/A | N/A | N/A | N/A | -- | Calibrated Only (No Validation) |
| `experiment_20261124_232039` | **LINEAR** | StandardScaler | 0.0% | 0.0% | 0.0% | 33.3% | **423.4 px** | 301.0 px | 0.0% | LEFT / left_column | Validated |
| `experiment_20261124_232710` | **LINEAR** | StandardScaler | 0.0% | 0.0% | 0.0% | 33.3% | **314.8 px** | 319.3 px | 0.0% | UP / top_row | Validated |
| `experiment_20261125_141825` | **COMPACT_LINEAR** | StandardScaler | 0.0% | 0.0% | 0.0% | 33.3% | **312.0 px** | 274.1 px | 0.0% | RIGHT / right_column | Validated |
| `experiment_20261125_142358` | **LINEAR** | StandardScaler | 0.0% | 11.1% | 22.2% | 55.6% | **215.9 px** | 216.3 px | 0.0% | DOWN / bottom_row | Validated |
| `experiment_20261125_150006` | **LINEAR** | StandardScaler | 0.0% | 0.0% | 0.0% | 66.7% | **177.0 px** | 163.5 px | 0.0% | LEFT / left_column | Validated |
| `experiment_20261125_150223` | **DECOUPLED_POLYNOMIAL** | StandardScaler | 11.1% | 11.1% | 77.8% | 100.0% | **95.4 px** | 97.3 px | 0.0% | RIGHT / right_column | Validated |
| `experiment_20261125_153718` | **LINEAR** | StandardScaler | 0.0% | 11.1% | 77.8% | 100.0% | **87.8 px** | 83.2 px | 0.0% | LEFT / left_column | Validated |
| `experiment_20261125_162637` | **COMPACT_POLYNOMIAL** | StandardScaler | 0.0% | 0.0% | 22.2% | 77.8% | **175.3 px** | 173.2 px | 0.0% | CENTER / center_column | Validated |
| `experiment_20261125_162740` | **LINEAR** | StandardScaler | 0.0% | 0.0% | 0.0% | 55.6% | **202.9 px** | 197.1 px | 0.0% | UP / top_row | Validated |
| `experiment_20261125_165544` | **COMPACT_LINEAR** | StandardScaler | 0.0% | 0.0% | 22.2% | 44.4% | **253.3 px** | 229.0 px | 0.0% | UP / top_row | Validated |
| `experiment_20261125_184448` | **DECOUPLED_LINEAR** | StandardScaler | 0.0% | 0.0% | 33.3% | 66.7% | **161.6 px** | 146.5 px | 0.0% | LEFT / left_column | Validated |
| `experiment_20261125_184929` | **DECOUPLED_POLYNOMIAL** | StandardScaler | 0.0% | 0.0% | 0.0% | 55.6% | **316.5 px** | 196.9 px | 0.0% | DOWN / bottom_row | Validated |
| `experiment_20261125_185024` | **DECOUPLED_LINEAR** | StandardScaler | 0.0% | 0.0% | 22.2% | 44.4% | **239.2 px** | 222.4 px | 0.0% | RIGHT / right_column | Validated |
| `experiment_20261125_185607` | **DECOUPLED_LINEAR** | StandardScaler | 55.6% | 22.2% | 55.6% | 100.0% | **93.6 px** | 74.7 px | 0.0% | UP / top_row | Validated |

- **Compatibility Warning:** Target button/circle boundaries differ across validated runs.
- **Best Validated Experiment:** `experiment_20261125_185607` (55.6% target selection, 93.6 px mean error) (weakest dir: UP, weakest region: top_row).
- **Lowest Mean Pixel Error:** `experiment_20261125_153718` (87.8 px).
- **Latest vs. Previous Validated (`experiment_20261125_185607` vs `experiment_20261125_185024`):** **IMPROVED** (Mean Error -145.6 px (60.9% error reduction); Target Selection Accuracy +55.6 percentage points; Within 5% Accuracy +33.3 percentage points).

## Diagnostic Plots

![plot_actual_vs_predicted](plot_actual_vs_predicted.png)
![plot_error_distributions](plot_error_distributions.png)
![plot_per_target_breakdown](plot_per_target_breakdown.png)

