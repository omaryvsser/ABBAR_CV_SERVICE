# Gaze Tracking Experiment Report: `experiment_20261125_195326`

## Experiment Overview

| Field | Value |
|---|---|
| **Experiment ID** | `experiment_20261125_195326` |
| **Date and Time** | 2026-11-25 19:54:04 |
| **Screen Resolution** | 1920x1080 (Diagonal: 2202.9 px) |
| **Calibration Model** | PURE_DECOUPLED_LINEAR |
| **Normalization Method** | StandardScaler |
| **Number of Training Samples** | 9 targets (246 accepted frames) |
| **Validation Status** | `COMPLETED` |
| **Number of Validation Trials** | 9 valid / 9 planned (0 rejected, 0.0% rej rate) |

## Main Results

> **Accuracy percentages** below measure how often gaze predictions landed successfully within strict screen tolerances or inside target boundaries (higher is better). **Average Error** measures distance from the true target (lower is better).

| Result | Value |
|---|---:|
| **Accuracy within 2%** | **0.0%** (0/9 trials, within 44.1 px) |
| **Accuracy within 5%** | **33.3%** (3/9 trials, within 110.1 px) |
| **Accuracy within 10%** | **88.9%** (8/9 trials, within 220.3 px) |
| **Target Selection Accuracy** | **22.2%** (2/9 hits) |
| **Average Error (pixels)** | 159.4 px |
| **Average Error (% of screen diagonal)** | 7.2% |
| **Rejected Trial Rate** | 0.0% (0/9) |

## Detailed Results

### Prediction Error Breakdown (Independent Validation Trials)

| Error Measurement | Pixels | Screen Diagonal / Normalized |
|---|---:|---:|
| **X-Axis MAE** | 108.2 px | 0.0563 norm |
| **Y-Axis MAE** | 106.0 px | 0.0982 norm |
| **X-Axis RMSE** | 126.0 px | 0.0657 norm |
| **Y-Axis RMSE** | 115.7 px | 0.1071 norm |
| **Median Error** | 181.6 px | 8.2% diag (0.1360 norm) |
| **Maximum Error** | 251.0 px | 11.4% diag (0.1836 norm) |
| **90th Percentile Error** | 215.3 px | — |

### Target Selection & Trial Counts

- **Target Boundary Evaluated:** Button Box (200x150 px)
- **Successful Selections (Hits):** 2
- **Failed Selections (Misses):** 7
- **Total Valid Trials:** 9
- **Rejected / Invalid Trials:** 0 (0.0%)

### Direction Accuracy Breakdown (LEFT / RIGHT / CENTER / UP / DOWN)

| Direction | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `LEFT` | 3 / 0 | **33.3%** | 33.3% | 144.4 px | (-89.6, +112.6) |
| `RIGHT` | 3 / 0 | **33.3%** | 66.7% | 123.0 px | (+3.1, +110.0) |
| `CENTER` | 1 / 0 | **0.0%** | 0.0% | 190.1 px | (+171.7, +81.7) |
| `UP` | 3 / 0 | **66.7%** | 66.7% | 126.8 px | (+79.1, +58.7) |
| `DOWN` | 3 / 0 | **0.0%** | 0.0% | 193.1 px | (-6.3, +155.7) |

### Screen Region Accuracy Breakdown (Rows, Columns & Edges)

| Screen Region | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `top_row` | 3 / 0 | **66.7%** | 66.7% | 126.8 px | (+79.1, +58.7) |
| `middle_row` | 3 / 0 | **0.0%** | 33.3% | 158.4 px | (+26.4, +103.7) |
| `bottom_row` | 3 / 0 | **0.0%** | 0.0% | 193.1 px | (-6.3, +155.7) |
| `left_column` | 3 / 0 | **33.3%** | 33.3% | 144.4 px | (-89.6, +112.6) |
| `center_column` | 3 / 0 | **0.0%** | 0.0% | 210.9 px | (+185.7, +95.6) |
| `right_column` | 3 / 0 | **33.3%** | 66.7% | 123.0 px | (+3.1, +110.0) |
| `edge_regions` | 8 / 0 | **25.0%** | 37.5% | 155.6 px | (+15.7, +109.1) |
| `center_region` | 1 / 0 | **0.0%** | 0.0% | 190.1 px | (+171.7, +81.7) |

### Per-Target Performance

| Target Point | Valid / Rejected | Hits / Valid | Target Selection Accuracy | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `val_bottom_center` | 1 / 0 | 0 / 1 | **0.0%** | 191.7 px | (+149.0, +120.7) |
| `val_bottom_left` | 1 / 0 | 0 / 1 | **0.0%** | 181.2 px | (-98.6, +152.1) |
| `val_bottom_right` | 1 / 0 | 0 / 1 | **0.0%** | 206.4 px | (-69.3, +194.4) |
| `val_center` | 1 / 0 | 0 / 1 | **0.0%** | 190.1 px | (+171.7, +81.7) |
| `val_middle_left` | 1 / 0 | 0 / 1 | **0.0%** | 181.6 px | (-125.5, +131.3) |
| `val_middle_right` | 1 / 0 | 0 / 1 | **0.0%** | 103.6 px | (+32.9, +98.2) |
| `val_top_center` | 1 / 0 | 0 / 1 | **0.0%** | 251.0 px | (+236.4, +84.3) |
| `val_top_left` | 1 / 0 | 1 / 1 | **100.0%** | 70.4 px | (-44.6, +54.4) |
| `val_top_right` | 1 / 0 | 1 / 1 | **100.0%** | 59.0 px | (+45.6, +37.4) |

### Training Performance vs. Independent Validation Performance

> **Note:** Training error only measures how tightly the model fits the 9 calibration points and must NOT be interpreted as real-world validation accuracy.

| Evaluation Split | Purpose | Mean Error (px) | Validation Accuracy |
|---|---|---:|---:|
| **Calibration Training Set** | Model fitting residual (training fit only) | 124.5 px | N/A (training split) |
| **Independent Validation Set** | Unseen test targets (real accuracy) | 159.4 px (7.2% diag) | 22.2% hits / 33.3% within 5% |

## Simple Explanation

Status: [COMPLETED]. The system successfully predicted gaze positions inside the target boundaries in 2 out of 9 valid trials (22.2%). Its average prediction error was 159.4 pixels (median: 181.6 px, 7.2% of the screen diagonal), and 33.3% of valid trials fell within 5% of the screen diagonal (3 out of 9). Weakest direction observed: DOWN (mean error 193.1 px); weakest screen region: center_column (mean error 210.9 px). This experiment performed better than the previous validated experiment (experiment_20261125_195146) according to the independent validation results (average error changed from 345.7 px to 159.4 px (53.9% error reduction); target selection accuracy changed from 0.0% to 22.2% (+22.2 percentage points)).

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
| `experiment_20261125_195146` | **DECOUPLED_LINEAR** | StandardScaler | 0.0% | 0.0% | 0.0% | 44.4% | **345.7 px** | 297.0 px | 0.0% | CENTER / middle_row | Validated |
| `experiment_20261125_195326` | **PURE_DECOUPLED_LINEAR** | StandardScaler | 22.2% | 0.0% | 33.3% | 88.9% | **159.4 px** | 181.6 px | 0.0% | DOWN / center_column | Validated |

- **Compatibility Warning:** Target button/circle boundaries differ across validated runs.
- **Best Validated Experiment:** `experiment_20261125_185607` (55.6% target selection, 93.6 px mean error) (weakest dir: UP, weakest region: top_row).
- **Lowest Mean Pixel Error:** `experiment_20261125_153718` (87.8 px).
- **Latest vs. Previous Validated (`experiment_20261125_195326` vs `experiment_20261125_195146`):** **IMPROVED** (Mean Error -186.2 px (53.9% error reduction); Target Selection Accuracy +22.2 percentage points; Within 5% Accuracy +33.3 percentage points).

## Diagnostic Plots

![plot_actual_vs_predicted](plot_actual_vs_predicted.png)
![plot_error_distributions](plot_error_distributions.png)
![plot_per_target_breakdown](plot_per_target_breakdown.png)

