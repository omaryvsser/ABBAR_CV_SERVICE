# Gaze Tracking Experiment Report: `experiment_20261126_121736`

## Experiment Overview

| Field | Value |
|---|---|
| **Experiment ID** | `experiment_20261126_121736` |
| **Date and Time** | 2026-11-26 12:18:15 |
| **Screen Resolution** | 1920x1080 (Diagonal: 2202.9 px) |
| **Calibration Model** | DECOUPLED_LINEAR |
| **Normalization Method** | StandardScaler |
| **Number of Training Samples** | 9 targets (240 accepted frames) |
| **Validation Status** | `COMPLETED` |
| **Number of Validation Trials** | 9 valid / 9 planned (0 rejected, 0.0% rej rate) |

## Main Results

> **Accuracy percentages** below measure how often gaze predictions landed successfully within strict screen tolerances or inside target boundaries (higher is better). **Average Error** measures distance from the true target (lower is better).

| Result | Value |
|---|---:|
| **Accuracy within 2%** | **0.0%** (0/9 trials, within 44.1 px) |
| **Accuracy within 5%** | **0.0%** (0/9 trials, within 110.1 px) |
| **Accuracy within 10%** | **77.8%** (7/9 trials, within 220.3 px) |
| **Target Selection Accuracy** | **0.0%** (0/9 hits) |
| **Average Error (pixels)** | 195.8 px |
| **Average Error (% of screen diagonal)** | 8.9% |
| **Rejected Trial Rate** | 0.0% (0/9) |

## Detailed Results

### Prediction Error Breakdown (Independent Validation Trials)

| Error Measurement | Pixels | Screen Diagonal / Normalized |
|---|---:|---:|
| **X-Axis MAE** | 137.3 px | 0.0715 norm |
| **Y-Axis MAE** | 114.6 px | 0.1061 norm |
| **X-Axis RMSE** | 153.0 px | 0.0797 norm |
| **Y-Axis RMSE** | 136.3 px | 0.1262 norm |
| **Median Error** | 200.9 px | 9.1% diag (0.1253 norm) |
| **Maximum Error** | 328.6 px | 14.9% diag (0.2698 norm) |
| **90th Percentile Error** | 267.7 px | — |

### Target Selection & Trial Counts

- **Target Boundary Evaluated:** Button Box (200x150 px)
- **Successful Selections (Hits):** 0
- **Failed Selections (Misses):** 9
- **Total Valid Trials:** 9
- **Rejected / Invalid Trials:** 0 (0.0%)

### Direction Accuracy Breakdown (LEFT / RIGHT / CENTER / UP / DOWN)

| Direction | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `LEFT` | 3 / 0 | **0.0%** | 0.0% | 174.4 px | (-92.6, +73.2) |
| `RIGHT` | 3 / 0 | **0.0%** | 0.0% | 191.3 px | (-138.7, -123.4) |
| `CENTER` | 1 / 0 | **0.0%** | 0.0% | 328.6 px | (-183.5, -272.5) |
| `UP` | 3 / 0 | **0.0%** | 0.0% | 218.8 px | (-192.5, -56.8) |
| `DOWN` | 3 / 0 | **0.0%** | 0.0% | 150.1 px | (-24.7, +67.4) |

### Screen Region Accuracy Breakdown (Rows, Columns & Edges)

| Screen Region | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `top_row` | 3 / 0 | **0.0%** | 0.0% | 218.8 px | (-192.5, -56.8) |
| `middle_row` | 3 / 0 | **0.0%** | 0.0% | 218.5 px | (-138.1, -117.9) |
| `bottom_row` | 3 / 0 | **0.0%** | 0.0% | 150.1 px | (-24.7, +67.4) |
| `left_column` | 3 / 0 | **0.0%** | 0.0% | 174.4 px | (-92.6, +73.2) |
| `center_column` | 3 / 0 | **0.0%** | 0.0% | 221.7 px | (-123.9, -57.0) |
| `right_column` | 3 / 0 | **0.0%** | 0.0% | 191.3 px | (-138.7, -123.4) |
| `edge_regions` | 8 / 0 | **0.0%** | 0.0% | 179.2 px | (-110.3, -6.2) |
| `center_region` | 1 / 0 | **0.0%** | 0.0% | 328.6 px | (-183.5, -272.5) |

### Per-Target Performance

| Target Point | Valid / Rejected | Hits / Valid | Target Selection Accuracy | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `val_bottom_center` | 1 / 0 | 0 / 1 | **0.0%** | 135.5 px | (+9.8, +135.2) |
| `val_bottom_left` | 1 / 0 | 0 / 1 | **0.0%** | 145.5 px | (+75.0, +124.7) |
| `val_bottom_right` | 1 / 0 | 0 / 1 | **0.0%** | 169.1 px | (-159.0, -57.7) |
| `val_center` | 1 / 0 | 0 / 1 | **0.0%** | 328.6 px | (-183.5, -272.5) |
| `val_middle_left` | 1 / 0 | 0 / 1 | **0.0%** | 125.2 px | (-101.3, +73.6) |
| `val_middle_right` | 1 / 0 | 0 / 1 | **0.0%** | 201.8 px | (-129.5, -154.8) |
| `val_top_center` | 1 / 0 | 0 / 1 | **0.0%** | 200.9 px | (-198.1, -33.6) |
| `val_top_left` | 1 / 0 | 0 / 1 | **0.0%** | 252.5 px | (-251.6, +21.2) |
| `val_top_right` | 1 / 0 | 0 / 1 | **0.0%** | 203.1 px | (-127.8, -157.8) |

### Training Performance vs. Independent Validation Performance

> **Note:** Training error only measures how tightly the model fits the 9 calibration points and must NOT be interpreted as real-world validation accuracy.

| Evaluation Split | Purpose | Mean Error (px) | Validation Accuracy |
|---|---|---:|---:|
| **Calibration Training Set** | Model fitting residual (training fit only) | 73.9 px | N/A (training split) |
| **Independent Validation Set** | Unseen test targets (real accuracy) | 195.8 px (8.9% diag) | 0.0% hits / 0.0% within 5% |

## Simple Explanation

Status: [COMPLETED]. The system successfully predicted gaze positions inside the target boundaries in 0 out of 9 valid trials (0.0%). Its average prediction error was 195.8 pixels (median: 200.9 px, 8.9% of the screen diagonal), and 0.0% of valid trials fell within 5% of the screen diagonal (0 out of 9). Weakest direction observed: CENTER (mean error 328.6 px); weakest screen region: center_region (mean error 328.6 px). This experiment performed worse than the previous validated experiment (experiment_20261126_121528) according to the independent validation results (average error changed from 193.5 px to 195.8 px (1.2% error increase); target selection accuracy changed from 22.2% to 0.0% (-22.2 percentage points)).

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
| `experiment_20261126_121528` | **DECOUPLED_LINEAR** | StandardScaler | 22.2% | 11.1% | 22.2% | 66.7% | **193.5 px** | 187.2 px | 0.0% | CENTER / center_region | Validated |
| `experiment_20261126_121736` | **DECOUPLED_LINEAR** | StandardScaler | 0.0% | 0.0% | 0.0% | 77.8% | **195.8 px** | 200.9 px | 0.0% | CENTER / center_region | Validated |

- **Compatibility Warning:** Target button/circle boundaries differ across validated runs.
- **Best Validated Experiment:** `experiment_20261125_185607` (55.6% target selection, 93.6 px mean error) (weakest dir: UP, weakest region: top_row).
- **Lowest Mean Pixel Error:** `experiment_20261125_153718` (87.8 px).
- **Latest vs. Previous Validated (`experiment_20261126_121736` vs `experiment_20261126_121528`):** **GOT WORSE** (Mean Error +2.3 px (1.2% error increase); Target Selection Accuracy -22.2 percentage points; Within 5% Accuracy -22.2 percentage points).

## Diagnostic Plots

![plot_actual_vs_predicted](plot_actual_vs_predicted.png)
![plot_error_distributions](plot_error_distributions.png)
![plot_per_target_breakdown](plot_per_target_breakdown.png)

