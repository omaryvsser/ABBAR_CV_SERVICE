# Gaze Tracking Experiment Report: `experiment_20261125_141825`

## Experiment Overview

| Field | Value |
|---|---|
| **Experiment ID** | `experiment_20261125_141825` |
| **Date and Time** | 2026-11-25 14:19:03 |
| **Screen Resolution** | 1920x1080 (Diagonal: 2202.9 px) |
| **Calibration Model** | COMPACT_LINEAR |
| **Normalization Method** | StandardScaler |
| **Number of Training Samples** | 9 targets (239 accepted frames) |
| **Validation Status** | `COMPLETED` |
| **Number of Validation Trials** | 9 valid / 9 planned (0 rejected, 0.0% rej rate) |

## Main Results

> **Accuracy percentages** below measure how often gaze predictions landed successfully within strict screen tolerances or inside target boundaries (higher is better). **Average Error** measures distance from the true target (lower is better).

| Result | Value |
|---|---:|
| **Accuracy within 2%** | **0.0%** (0/9 trials, within 44.1 px) |
| **Accuracy within 5%** | **0.0%** (0/9 trials, within 110.1 px) |
| **Accuracy within 10%** | **33.3%** (3/9 trials, within 220.3 px) |
| **Target Selection Accuracy** | **0.0%** (0/9 hits) |
| **Average Error (pixels)** | 312.0 px |
| **Average Error (% of screen diagonal)** | 14.2% |
| **Rejected Trial Rate** | 0.0% (0/9) |

## Detailed Results

### Prediction Error Breakdown (Independent Validation Trials)

| Error Measurement | Pixels | Screen Diagonal / Normalized |
|---|---:|---:|
| **X-Axis MAE** | 240.2 px | 0.1251 norm |
| **Y-Axis MAE** | 178.4 px | 0.1652 norm |
| **X-Axis RMSE** | 296.6 px | 0.1545 norm |
| **Y-Axis RMSE** | 188.0 px | 0.1741 norm |
| **Median Error** | 274.1 px | 12.4% diag (0.2064 norm) |
| **Maximum Error** | 673.8 px | 30.6% diag (0.3772 norm) |
| **90th Percentile Error** | 488.3 px | — |

### Target Selection & Trial Counts

- **Target Boundary Evaluated:** Rendered Target Circle (radius 22 display px / 33.0 px @ 1920x1080)
- **Successful Selections (Hits):** 0
- **Failed Selections (Misses):** 9
- **Total Valid Trials:** 9
- **Rejected / Invalid Trials:** 0 (0.0%)

### Direction Accuracy Breakdown (LEFT / RIGHT / CENTER / UP / DOWN)

| Direction | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `LEFT` | 3 / 0 | **0.0%** | 0.0% | 299.3 px | (-234.3, +103.9) |
| `RIGHT` | 3 / 0 | **0.0%** | 0.0% | 375.1 px | (-302.9, -114.1) |
| `CENTER` | 1 / 0 | **0.0%** | 0.0% | 126.0 px | (+44.9, -117.7) |
| `UP` | 3 / 0 | **0.0%** | 0.0% | 337.5 px | (-276.3, +192.9) |
| `DOWN` | 3 / 0 | **0.0%** | 0.0% | 366.4 px | (-309.9, -166.9) |

### Screen Region Accuracy Breakdown (Rows, Columns & Edges)

| Screen Region | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `top_row` | 3 / 0 | **0.0%** | 0.0% | 337.5 px | (-276.3, +192.9) |
| `middle_row` | 3 / 0 | **0.0%** | 0.0% | 232.0 px | (-104.6, -71.3) |
| `bottom_row` | 3 / 0 | **0.0%** | 0.0% | 366.4 px | (-309.9, -166.9) |
| `left_column` | 3 / 0 | **0.0%** | 0.0% | 299.3 px | (-234.3, +103.9) |
| `center_column` | 3 / 0 | **0.0%** | 0.0% | 261.7 px | (-153.7, -35.1) |
| `right_column` | 3 / 0 | **0.0%** | 0.0% | 375.1 px | (-302.9, -114.1) |
| `edge_regions` | 8 / 0 | **0.0%** | 0.0% | 335.3 px | (-264.6, -2.2) |
| `center_region` | 1 / 0 | **0.0%** | 0.0% | 126.0 px | (+44.9, -117.7) |

### Per-Target Performance

| Target Point | Valid / Rejected | Hits / Valid | Target Selection Accuracy | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `val_bottom_center` | 1 / 0 | 0 / 1 | **0.0%** | 265.7 px | (-174.8, -200.1) |
| `val_bottom_left` | 1 / 0 | 0 / 1 | **0.0%** | 159.9 px | (-105.8, -119.9) |
| `val_bottom_right` | 1 / 0 | 0 / 1 | **0.0%** | 673.8 px | (-649.1, -180.6) |
| `val_center` | 1 / 0 | 0 / 1 | **0.0%** | 126.0 px | (+44.9, -117.7) |
| `val_middle_left` | 1 / 0 | 0 / 1 | **0.0%** | 296.0 px | (-251.4, +156.2) |
| `val_middle_right` | 1 / 0 | 0 / 1 | **0.0%** | 274.1 px | (-107.2, -252.3) |
| `val_top_center` | 1 / 0 | 0 / 1 | **0.0%** | 393.4 px | (-331.0, +212.6) |
| `val_top_left` | 1 / 0 | 0 / 1 | **0.0%** | 441.9 px | (-345.6, +275.4) |
| `val_top_right` | 1 / 0 | 0 / 1 | **0.0%** | 177.3 px | (-152.3, +90.7) |

### Training Performance vs. Independent Validation Performance

> **Note:** Training error only measures how tightly the model fits the 9 calibration points and must NOT be interpreted as real-world validation accuracy.

| Evaluation Split | Purpose | Mean Error (px) | Validation Accuracy |
|---|---|---:|---:|
| **Calibration Training Set** | Model fitting residual (training fit only) | 353.9 px | N/A (training split) |
| **Independent Validation Set** | Unseen test targets (real accuracy) | 312.0 px (14.2% diag) | 0.0% hits / 0.0% within 5% |

## Simple Explanation

Status: [COMPLETED]. The system successfully predicted gaze positions inside the target boundaries in 0 out of 9 valid trials (0.0%). Its average prediction error was 312.0 pixels (median: 274.1 px, 14.2% of the screen diagonal), and 0.0% of valid trials fell within 5% of the screen diagonal (0 out of 9). Weakest direction observed: RIGHT (mean error 375.1 px); weakest screen region: right_column (mean error 375.1 px). This experiment performed better than the previous validated experiment (experiment_20261124_232710) according to the independent validation results (average error changed from 314.8 px to 312.0 px (0.9% error reduction); target selection accuracy changed from 0.0% to 0.0% (+0.0 percentage points)).

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

- **Best Validated Experiment:** `experiment_20261125_141825` (0.0% target selection, 312.0 px mean error) (weakest dir: RIGHT, weakest region: right_column).
- **Latest vs. Previous Validated (`experiment_20261125_141825` vs `experiment_20261124_232710`):** **IMPROVED** (Mean Error -2.8 px (0.9% error reduction); Target Selection Accuracy +0.0 percentage points; Within 5% Accuracy +0.0 percentage points).

## Diagnostic Plots

![plot_actual_vs_predicted](plot_actual_vs_predicted.png)
![plot_error_distributions](plot_error_distributions.png)
![plot_per_target_breakdown](plot_per_target_breakdown.png)

