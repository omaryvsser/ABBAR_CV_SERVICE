# Gaze Tracking Experiment Report: `experiment_20261125_192928`

## Experiment Overview

| Field | Value |
|---|---|
| **Experiment ID** | `experiment_20261125_192928` |
| **Date and Time** | 2026-11-25 19:30:06 |
| **Screen Resolution** | 1920x1080 (Diagonal: 2202.9 px) |
| **Calibration Model** | COMPACT_POLYNOMIAL |
| **Normalization Method** | StandardScaler |
| **Number of Training Samples** | 9 targets (268 accepted frames) |
| **Validation Status** | `COMPLETED` |
| **Number of Validation Trials** | 9 valid / 9 planned (0 rejected, 0.0% rej rate) |

## Main Results

> **Accuracy percentages** below measure how often gaze predictions landed successfully within strict screen tolerances or inside target boundaries (higher is better). **Average Error** measures distance from the true target (lower is better).

| Result | Value |
|---|---:|
| **Accuracy within 2%** | **0.0%** (0/9 trials, within 44.1 px) |
| **Accuracy within 5%** | **11.1%** (1/9 trials, within 110.1 px) |
| **Accuracy within 10%** | **44.4%** (4/9 trials, within 220.3 px) |
| **Target Selection Accuracy** | **0.0%** (0/9 hits) |
| **Average Error (pixels)** | 253.0 px |
| **Average Error (% of screen diagonal)** | 11.5% |
| **Rejected Trial Rate** | 0.0% (0/9) |

## Detailed Results

### Prediction Error Breakdown (Independent Validation Trials)

| Error Measurement | Pixels | Screen Diagonal / Normalized |
|---|---:|---:|
| **X-Axis MAE** | 185.3 px | 0.0965 norm |
| **Y-Axis MAE** | 153.8 px | 0.1424 norm |
| **X-Axis RMSE** | 209.2 px | 0.1090 norm |
| **Y-Axis RMSE** | 174.0 px | 0.1611 norm |
| **Median Error** | 304.1 px | 13.8% diag (0.1912 norm) |
| **Maximum Error** | 389.7 px | 17.7% diag (0.2981 norm) |
| **90th Percentile Error** | 353.2 px | — |

### Target Selection & Trial Counts

- **Target Boundary Evaluated:** Rendered Target Circle (radius 22 display px / 33.0 px @ 1920x1080)
- **Successful Selections (Hits):** 0
- **Failed Selections (Misses):** 9
- **Total Valid Trials:** 9
- **Rejected / Invalid Trials:** 0 (0.0%)

### Direction Accuracy Breakdown (LEFT / RIGHT / CENTER / UP / DOWN)

| Direction | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `LEFT` | 3 / 0 | **0.0%** | 0.0% | 317.6 px | (+125.4, -231.9) |
| `RIGHT` | 3 / 0 | **0.0%** | 33.3% | 189.6 px | (+139.7, -22.1) |
| `CENTER` | 1 / 0 | **0.0%** | 0.0% | 311.9 px | (-282.7, +131.8) |
| `UP` | 3 / 0 | **0.0%** | 0.0% | 288.9 px | (+216.7, -180.9) |
| `DOWN` | 3 / 0 | **0.0%** | 0.0% | 227.0 px | (+14.1, -21.0) |

### Screen Region Accuracy Breakdown (Rows, Columns & Edges)

| Screen Region | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `top_row` | 3 / 0 | **0.0%** | 0.0% | 288.9 px | (+216.7, -180.9) |
| `middle_row` | 3 / 0 | **0.0%** | 33.3% | 243.3 px | (-28.3, -77.6) |
| `bottom_row` | 3 / 0 | **0.0%** | 0.0% | 227.0 px | (+14.1, -21.0) |
| `left_column` | 3 / 0 | **0.0%** | 0.0% | 317.6 px | (+125.4, -231.9) |
| `center_column` | 3 / 0 | **0.0%** | 0.0% | 251.9 px | (-62.6, -25.6) |
| `right_column` | 3 / 0 | **0.0%** | 33.3% | 189.6 px | (+139.7, -22.1) |
| `edge_regions` | 8 / 0 | **0.0%** | 12.5% | 245.7 px | (+111.2, -121.3) |
| `center_region` | 1 / 0 | **0.0%** | 0.0% | 311.9 px | (-282.7, +131.8) |

### Per-Target Performance

| Target Point | Valid / Rejected | Hits / Valid | Target Selection Accuracy | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `val_bottom_center` | 1 / 0 | 0 / 1 | **0.0%** | 139.8 px | (-139.1, -14.1) |
| `val_bottom_left` | 1 / 0 | 0 / 1 | **0.0%** | 218.9 px | (-108.5, -190.2) |
| `val_bottom_right` | 1 / 0 | 0 / 1 | **0.0%** | 322.3 px | (+289.8, +141.1) |
| `val_center` | 1 / 0 | 0 / 1 | **0.0%** | 311.9 px | (-282.7, +131.8) |
| `val_middle_left` | 1 / 0 | 0 / 1 | **0.0%** | 344.1 px | (+146.8, -311.1) |
| `val_middle_right` | 1 / 0 | 0 / 1 | **0.0%** | 73.8 px | (+50.9, -53.5) |
| `val_top_center` | 1 / 0 | 0 / 1 | **0.0%** | 304.1 px | (+233.8, -194.4) |
| `val_top_left` | 1 / 0 | 0 / 1 | **0.0%** | 389.7 px | (+337.8, -194.4) |
| `val_top_right` | 1 / 0 | 0 / 1 | **0.0%** | 172.7 px | (+78.4, -153.9) |

### Training Performance vs. Independent Validation Performance

> **Note:** Training error only measures how tightly the model fits the 9 calibration points and must NOT be interpreted as real-world validation accuracy.

| Evaluation Split | Purpose | Mean Error (px) | Validation Accuracy |
|---|---|---:|---:|
| **Calibration Training Set** | Model fitting residual (training fit only) | 68.0 px | N/A (training split) |
| **Independent Validation Set** | Unseen test targets (real accuracy) | 253.0 px (11.5% diag) | 0.0% hits / 11.1% within 5% |

## Simple Explanation

Status: [COMPLETED]. The system successfully predicted gaze positions inside the target boundaries in 0 out of 9 valid trials (0.0%). Its average prediction error was 253.0 pixels (median: 304.1 px, 11.5% of the screen diagonal), and 11.1% of valid trials fell within 5% of the screen diagonal (1 out of 9). Weakest direction observed: LEFT (mean error 317.6 px); weakest screen region: left_column (mean error 317.6 px). This experiment performed worse than the previous validated experiment (experiment_20261125_191422) according to the independent validation results (average error changed from 106.8 px to 253.0 px (137.0% error increase); target selection accuracy changed from 11.1% to 0.0% (-11.1 percentage points)).

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
| `experiment_20261125_185711` | **DECOUPLED_LINEAR** | StandardScaler | N/A | N/A | N/A | N/A | N/A | N/A | N/A | -- | Incomplete (Interrupted) |
| `experiment_20261125_191422` | **DECOUPLED_LINEAR** | StandardScaler | 11.1% | 11.1% | 55.6% | 100.0% | **106.8 px** | 102.7 px | 0.0% | DOWN / bottom_row | Validated |
| `experiment_20261125_192928` | **COMPACT_POLYNOMIAL** | StandardScaler | 0.0% | 0.0% | 11.1% | 44.4% | **253.0 px** | 304.1 px | 0.0% | LEFT / left_column | Validated |

- **Compatibility Warning:** Target button/circle boundaries differ across validated runs.
- **Best Validated Experiment:** `experiment_20261125_185607` (55.6% target selection, 93.6 px mean error) (weakest dir: UP, weakest region: top_row).
- **Lowest Mean Pixel Error:** `experiment_20261125_153718` (87.8 px).
- **Latest vs. Previous Validated (`experiment_20261125_192928` vs `experiment_20261125_191422`):** **GOT WORSE** (Mean Error +146.3 px (137.0% error increase); Target Selection Accuracy -11.1 percentage points; Within 5% Accuracy -44.5 percentage points).

## Diagnostic Plots

![plot_actual_vs_predicted](plot_actual_vs_predicted.png)
![plot_error_distributions](plot_error_distributions.png)
![plot_per_target_breakdown](plot_per_target_breakdown.png)

