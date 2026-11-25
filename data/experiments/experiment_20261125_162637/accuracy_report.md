# Gaze Tracking Experiment Report: `experiment_20261125_162637`

## Experiment Overview

| Field | Value |
|---|---|
| **Experiment ID** | `experiment_20261125_162637` |
| **Date and Time** | 2026-11-25 16:27:19 |
| **Screen Resolution** | 1920x1080 (Diagonal: 2202.9 px) |
| **Calibration Model** | COMPACT_POLYNOMIAL |
| **Normalization Method** | StandardScaler |
| **Number of Training Samples** | 9 targets (177 accepted frames) |
| **Validation Status** | `COMPLETED` |
| **Number of Validation Trials** | 9 valid / 9 planned (0 rejected, 0.0% rej rate) |

## Main Results

> **Accuracy percentages** below measure how often gaze predictions landed successfully within strict screen tolerances or inside target boundaries (higher is better). **Average Error** measures distance from the true target (lower is better).

| Result | Value |
|---|---:|
| **Accuracy within 2%** | **0.0%** (0/9 trials, within 44.1 px) |
| **Accuracy within 5%** | **22.2%** (2/9 trials, within 110.1 px) |
| **Accuracy within 10%** | **77.8%** (7/9 trials, within 220.3 px) |
| **Target Selection Accuracy** | **0.0%** (0/9 hits) |
| **Average Error (pixels)** | 175.3 px |
| **Average Error (% of screen diagonal)** | 8.0% |
| **Rejected Trial Rate** | 0.0% (0/9) |

## Detailed Results

### Prediction Error Breakdown (Independent Validation Trials)

| Error Measurement | Pixels | Screen Diagonal / Normalized |
|---|---:|---:|
| **X-Axis MAE** | 104.9 px | 0.0546 norm |
| **Y-Axis MAE** | 128.7 px | 0.1192 norm |
| **X-Axis RMSE** | 125.7 px | 0.0655 norm |
| **Y-Axis RMSE** | 140.0 px | 0.1296 norm |
| **Median Error** | 173.2 px | 7.9% diag (0.1533 norm) |
| **Maximum Error** | 304.4 px | 13.8% diag (0.2076 norm) |
| **90th Percentile Error** | 237.6 px | — |

### Target Selection & Trial Counts

- **Target Boundary Evaluated:** Rendered Target Circle (radius 22 display px / 33.0 px @ 1920x1080)
- **Successful Selections (Hits):** 0
- **Failed Selections (Misses):** 9
- **Total Valid Trials:** 9
- **Rejected / Invalid Trials:** 0 (0.0%)

### Direction Accuracy Breakdown (LEFT / RIGHT / CENTER / UP / DOWN)

| Direction | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `LEFT` | 3 / 0 | **0.0%** | 0.0% | 170.8 px | (-9.4, +163.9) |
| `RIGHT` | 3 / 0 | **0.0%** | 66.7% | 107.7 px | (-79.0, +67.4) |
| `CENTER` | 1 / 0 | **0.0%** | 0.0% | 217.2 px | (+170.0, +135.2) |
| `UP` | 3 / 0 | **0.0%** | 33.3% | 155.6 px | (+12.6, +112.8) |
| `DOWN` | 3 / 0 | **0.0%** | 0.0% | 190.4 px | (+63.3, +126.9) |

### Screen Region Accuracy Breakdown (Rows, Columns & Edges)

| Screen Region | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `top_row` | 3 / 0 | **0.0%** | 33.3% | 155.6 px | (+12.6, +112.8) |
| `middle_row` | 3 / 0 | **0.0%** | 33.3% | 179.9 px | (+28.2, +146.4) |
| `bottom_row` | 3 / 0 | **0.0%** | 0.0% | 190.4 px | (+63.3, +126.9) |
| `left_column` | 3 / 0 | **0.0%** | 0.0% | 170.8 px | (-9.4, +163.9) |
| `center_column` | 3 / 0 | **0.0%** | 0.0% | 247.5 px | (+192.4, +154.8) |
| `right_column` | 3 / 0 | **0.0%** | 66.7% | 107.7 px | (-79.0, +67.4) |
| `edge_regions` | 8 / 0 | **0.0%** | 25.0% | 170.1 px | (+17.8, +127.9) |
| `center_region` | 1 / 0 | **0.0%** | 0.0% | 217.2 px | (+170.0, +135.2) |

### Per-Target Performance

| Target Point | Valid / Rejected | Hits / Valid | Target Selection Accuracy | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `val_bottom_center` | 1 / 0 | 0 / 1 | **0.0%** | 304.4 px | (+249.0, +175.1) |
| `val_bottom_left` | 1 / 0 | 0 / 1 | **0.0%** | 120.5 px | (+51.0, +109.1) |
| `val_bottom_right` | 1 / 0 | 0 / 1 | **0.0%** | 146.4 px | (-110.1, +96.5) |
| `val_center` | 1 / 0 | 0 / 1 | **0.0%** | 217.2 px | (+170.0, +135.2) |
| `val_middle_left` | 1 / 0 | 0 / 1 | **0.0%** | 218.8 px | (-28.9, +216.8) |
| `val_middle_right` | 1 / 0 | 0 / 1 | **0.0%** | 103.9 px | (-56.3, +87.2) |
| `val_top_center` | 1 / 0 | 0 / 1 | **0.0%** | 220.8 px | (+158.2, +154.1) |
| `val_top_left` | 1 / 0 | 0 / 1 | **0.0%** | 173.2 px | (-50.1, +165.8) |
| `val_top_right` | 1 / 0 | 0 / 1 | **0.0%** | 72.8 px | (-70.4, +18.4) |

### Training Performance vs. Independent Validation Performance

> **Note:** Training error only measures how tightly the model fits the 9 calibration points and must NOT be interpreted as real-world validation accuracy.

| Evaluation Split | Purpose | Mean Error (px) | Validation Accuracy |
|---|---|---:|---:|
| **Calibration Training Set** | Model fitting residual (training fit only) | 85.5 px | N/A (training split) |
| **Independent Validation Set** | Unseen test targets (real accuracy) | 175.3 px (8.0% diag) | 0.0% hits / 22.2% within 5% |

## Simple Explanation

Status: [COMPLETED]. The system successfully predicted gaze positions inside the target boundaries in 0 out of 9 valid trials (0.0%). Its average prediction error was 175.3 pixels (median: 173.2 px, 8.0% of the screen diagonal), and 22.2% of valid trials fell within 5% of the screen diagonal (2 out of 9). Weakest direction observed: CENTER (mean error 217.2 px); weakest screen region: center_column (mean error 247.5 px). This experiment performed worse than the previous validated experiment (experiment_20261125_153718) according to the independent validation results (average error changed from 87.8 px to 175.3 px (99.6% error increase); target selection accuracy changed from 0.0% to 0.0% (+0.0 percentage points)).

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

- **Compatibility Warning:** Target button/circle boundaries differ across validated runs.
- **Best Validated Experiment:** `experiment_20261125_150223` (11.1% target selection, 95.4 px mean error) (weakest dir: RIGHT, weakest region: right_column).
- **Lowest Mean Pixel Error:** `experiment_20261125_153718` (87.8 px).
- **Latest vs. Previous Validated (`experiment_20261125_162637` vs `experiment_20261125_153718`):** **GOT WORSE** (Mean Error +87.5 px (99.6% error increase); Target Selection Accuracy +0.0 percentage points; Within 5% Accuracy -55.6 percentage points).

## Diagnostic Plots

![plot_actual_vs_predicted](plot_actual_vs_predicted.png)
![plot_error_distributions](plot_error_distributions.png)
![plot_per_target_breakdown](plot_per_target_breakdown.png)

