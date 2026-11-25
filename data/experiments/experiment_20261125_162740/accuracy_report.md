# Gaze Tracking Experiment Report: `experiment_20261125_162740`

## Experiment Overview

| Field | Value |
|---|---|
| **Experiment ID** | `experiment_20261125_162740` |
| **Date and Time** | 2026-11-25 16:28:22 |
| **Screen Resolution** | 1920x1080 (Diagonal: 2202.9 px) |
| **Calibration Model** | LINEAR |
| **Normalization Method** | StandardScaler |
| **Number of Training Samples** | 9 targets (226 accepted frames) |
| **Validation Status** | `COMPLETED` |
| **Number of Validation Trials** | 9 valid / 9 planned (0 rejected, 0.0% rej rate) |

## Main Results

> **Accuracy percentages** below measure how often gaze predictions landed successfully within strict screen tolerances or inside target boundaries (higher is better). **Average Error** measures distance from the true target (lower is better).

| Result | Value |
|---|---:|
| **Accuracy within 2%** | **0.0%** (0/9 trials, within 44.1 px) |
| **Accuracy within 5%** | **0.0%** (0/9 trials, within 110.1 px) |
| **Accuracy within 10%** | **55.6%** (5/9 trials, within 220.3 px) |
| **Target Selection Accuracy** | **0.0%** (0/9 hits) |
| **Average Error (pixels)** | 202.9 px |
| **Average Error (% of screen diagonal)** | 9.2% |
| **Rejected Trial Rate** | 0.0% (0/9) |

## Detailed Results

### Prediction Error Breakdown (Independent Validation Trials)

| Error Measurement | Pixels | Screen Diagonal / Normalized |
|---|---:|---:|
| **X-Axis MAE** | 116.9 px | 0.0609 norm |
| **Y-Axis MAE** | 153.3 px | 0.1420 norm |
| **X-Axis RMSE** | 136.0 px | 0.0709 norm |
| **Y-Axis RMSE** | 160.2 px | 0.1484 norm |
| **Median Error** | 197.1 px | 8.9% diag (0.1524 norm) |
| **Maximum Error** | 284.3 px | 12.9% diag (0.2471 norm) |
| **90th Percentile Error** | 271.0 px | — |

### Target Selection & Trial Counts

- **Target Boundary Evaluated:** Rendered Target Circle (radius 22 display px / 33.0 px @ 1920x1080)
- **Successful Selections (Hits):** 0
- **Failed Selections (Misses):** 9
- **Total Valid Trials:** 9
- **Rejected / Invalid Trials:** 0 (0.0%)

### Direction Accuracy Breakdown (LEFT / RIGHT / CENTER / UP / DOWN)

| Direction | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `LEFT` | 3 / 0 | **0.0%** | 0.0% | 199.6 px | (-50.8, +185.4) |
| `RIGHT` | 3 / 0 | **0.0%** | 0.0% | 207.4 px | (-174.8, +110.4) |
| `CENTER` | 1 / 0 | **0.0%** | 0.0% | 197.1 px | (-131.2, +147.1) |
| `UP` | 3 / 0 | **0.0%** | 0.0% | 255.5 px | (-159.6, +190.9) |
| `DOWN` | 3 / 0 | **0.0%** | 0.0% | 142.1 px | (-25.8, +126.5) |

### Screen Region Accuracy Breakdown (Rows, Columns & Edges)

| Screen Region | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `top_row` | 3 / 0 | **0.0%** | 0.0% | 255.5 px | (-159.6, +190.9) |
| `middle_row` | 3 / 0 | **0.0%** | 0.0% | 211.2 px | (-144.4, +142.7) |
| `bottom_row` | 3 / 0 | **0.0%** | 0.0% | 142.1 px | (-25.8, +126.5) |
| `left_column` | 3 / 0 | **0.0%** | 0.0% | 199.6 px | (-50.8, +185.4) |
| `center_column` | 3 / 0 | **0.0%** | 0.0% | 201.8 px | (-104.1, +164.2) |
| `right_column` | 3 / 0 | **0.0%** | 0.0% | 207.4 px | (-174.8, +110.4) |
| `edge_regions` | 8 / 0 | **0.0%** | 0.0% | 203.7 px | (-107.3, +154.1) |
| `center_region` | 1 / 0 | **0.0%** | 0.0% | 197.1 px | (-131.2, +147.1) |

### Per-Target Performance

| Target Point | Valid / Rejected | Hits / Valid | Target Selection Accuracy | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `val_bottom_center` | 1 / 0 | 0 / 1 | **0.0%** | 161.2 px | (-17.7, +160.2) |
| `val_bottom_left` | 1 / 0 | 0 / 1 | **0.0%** | 145.8 px | (+31.6, +142.3) |
| `val_bottom_right` | 1 / 0 | 0 / 1 | **0.0%** | 119.3 px | (-91.3, +76.8) |
| `val_center` | 1 / 0 | 0 / 1 | **0.0%** | 197.1 px | (-131.2, +147.1) |
| `val_middle_left` | 1 / 0 | 0 / 1 | **0.0%** | 168.6 px | (-65.5, +155.4) |
| `val_middle_right` | 1 / 0 | 0 / 1 | **0.0%** | 267.7 px | (-236.5, +125.5) |
| `val_top_center` | 1 / 0 | 0 / 1 | **0.0%** | 247.2 px | (-163.5, +185.3) |
| `val_top_left` | 1 / 0 | 0 / 1 | **0.0%** | 284.3 px | (-118.5, +258.4) |
| `val_top_right` | 1 / 0 | 0 / 1 | **0.0%** | 235.2 px | (-196.7, +128.9) |

### Training Performance vs. Independent Validation Performance

> **Note:** Training error only measures how tightly the model fits the 9 calibration points and must NOT be interpreted as real-world validation accuracy.

| Evaluation Split | Purpose | Mean Error (px) | Validation Accuracy |
|---|---|---:|---:|
| **Calibration Training Set** | Model fitting residual (training fit only) | 32.1 px | N/A (training split) |
| **Independent Validation Set** | Unseen test targets (real accuracy) | 202.9 px (9.2% diag) | 0.0% hits / 0.0% within 5% |

## Simple Explanation

Status: [COMPLETED]. The system successfully predicted gaze positions inside the target boundaries in 0 out of 9 valid trials (0.0%). Its average prediction error was 202.9 pixels (median: 197.1 px, 9.2% of the screen diagonal), and 0.0% of valid trials fell within 5% of the screen diagonal (0 out of 9). Weakest direction observed: UP (mean error 255.5 px); weakest screen region: top_row (mean error 255.5 px). This experiment performed worse than the previous validated experiment (experiment_20261125_162637) according to the independent validation results (average error changed from 175.3 px to 202.9 px (15.7% error increase); target selection accuracy changed from 0.0% to 0.0% (+0.0 percentage points)).

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

- **Compatibility Warning:** Target button/circle boundaries differ across validated runs.
- **Best Validated Experiment:** `experiment_20261125_150223` (11.1% target selection, 95.4 px mean error) (weakest dir: RIGHT, weakest region: right_column).
- **Lowest Mean Pixel Error:** `experiment_20261125_153718` (87.8 px).
- **Latest vs. Previous Validated (`experiment_20261125_162740` vs `experiment_20261125_162637`):** **GOT WORSE** (Mean Error +27.6 px (15.7% error increase); Target Selection Accuracy +0.0 percentage points; Within 5% Accuracy -22.2 percentage points).

## Diagnostic Plots

![plot_actual_vs_predicted](plot_actual_vs_predicted.png)
![plot_error_distributions](plot_error_distributions.png)
![plot_per_target_breakdown](plot_per_target_breakdown.png)

