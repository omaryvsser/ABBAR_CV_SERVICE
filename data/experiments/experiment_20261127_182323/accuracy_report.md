# Gaze Tracking Experiment Report: `experiment_20261127_182323`

## Experiment Overview

| Field | Value |
|---|---|
| **Experiment ID** | `experiment_20261127_182323` |
| **Date and Time** | 2026-11-27 18:24:02 |
| **Screen Resolution** | 1920x1080 (Diagonal: 2202.9 px) |
| **Calibration Model** | COMPACT_LINEAR |
| **Normalization Method** | StandardScaler |
| **Number of Training Samples** | 9 targets (234 accepted frames) |
| **Validation Status** | `COMPLETED` |
| **Number of Validation Trials** | 9 valid / 9 planned (0 rejected, 0.0% rej rate) |

## Main Results

> **Accuracy percentages** below measure how often gaze predictions landed successfully within strict screen tolerances or inside target boundaries (higher is better). **Average Error** measures distance from the true target (lower is better).

| Result | Value |
|---|---:|
| **Accuracy within 2%** | **0.0%** (0/9 trials, within 44.1 px) |
| **Accuracy within 5%** | **0.0%** (0/9 trials, within 110.1 px) |
| **Accuracy within 10%** | **0.0%** (0/9 trials, within 220.3 px) |
| **Target Selection Accuracy** | **0.0%** (0/9 hits) |
| **Average Error (pixels)** | 676.7 px |
| **Average Error (% of screen diagonal)** | 30.7% |
| **Rejected Trial Rate** | 0.0% (0/9) |

## Detailed Results

### Prediction Error Breakdown (Independent Validation Trials)

| Error Measurement | Pixels | Screen Diagonal / Normalized |
|---|---:|---:|
| **X-Axis MAE** | 661.0 px | 0.3443 norm |
| **Y-Axis MAE** | 129.5 px | 0.1199 norm |
| **X-Axis RMSE** | 708.8 px | 0.3692 norm |
| **Y-Axis RMSE** | 148.8 px | 0.1377 norm |
| **Median Error** | 702.5 px | 31.9% diag (0.3715 norm) |
| **Maximum Error** | 1073.9 px | 48.7% diag (0.5706 norm) |
| **90th Percentile Error** | 973.3 px | — |

### Target Selection & Trial Counts

- **Target Boundary Evaluated:** Button Box (200x150 px)
- **Successful Selections (Hits):** 0
- **Failed Selections (Misses):** 9
- **Total Valid Trials:** 9
- **Rejected / Invalid Trials:** 0 (0.0%)

### Direction Accuracy Breakdown (LEFT / RIGHT / CENTER / UP / DOWN)

| Direction | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `LEFT` | 3 / 0 | **0.0%** | 0.0% | 361.8 px | (-345.6, +61.6) |
| `RIGHT` | 3 / 0 | **0.0%** | 0.0% | 982.5 px | (-957.2, +213.0) |
| `CENTER` | 1 / 0 | **0.0%** | 0.0% | 738.5 px | (-737.3, -40.6) |
| `UP` | 3 / 0 | **0.0%** | 0.0% | 660.4 px | (-644.2, +81.7) |
| `DOWN` | 3 / 0 | **0.0%** | 0.0% | 690.8 px | (-671.6, +141.4) |

### Screen Region Accuracy Breakdown (Rows, Columns & Edges)

| Screen Region | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `top_row` | 3 / 0 | **0.0%** | 0.0% | 660.4 px | (-644.2, +81.7) |
| `middle_row` | 3 / 0 | **0.0%** | 0.0% | 678.7 px | (-667.1, +47.4) |
| `bottom_row` | 3 / 0 | **0.0%** | 0.0% | 690.8 px | (-671.6, +141.4) |
| `left_column` | 3 / 0 | **0.0%** | 0.0% | 361.8 px | (-345.6, +61.6) |
| `center_column` | 3 / 0 | **0.0%** | 0.0% | 685.6 px | (-680.1, -4.0) |
| `right_column` | 3 / 0 | **0.0%** | 0.0% | 982.5 px | (-957.2, +213.0) |
| `edge_regions` | 8 / 0 | **0.0%** | 0.0% | 668.9 px | (-651.4, +106.5) |
| `center_region` | 1 / 0 | **0.0%** | 0.0% | 738.5 px | (-737.3, -40.6) |

### Per-Target Performance

| Target Point | Valid / Rejected | Hits / Valid | Target Selection Accuracy | Mean Error (px) | Bias (X, Y) px |
|---|---:|---:|---:|---:|---:|
| `val_bottom_center` | 1 / 0 | 0 / 1 | **0.0%** | 616.0 px | (-605.6, +112.7) |
| `val_bottom_left` | 1 / 0 | 0 / 1 | **0.0%** | 382.6 px | (-345.6, +164.2) |
| `val_bottom_right` | 1 / 0 | 0 / 1 | **0.0%** | 1073.9 px | (-1063.7, +147.4) |
| `val_center` | 1 / 0 | 0 / 1 | **0.0%** | 738.5 px | (-737.3, -40.6) |
| `val_middle_left` | 1 / 0 | 0 / 1 | **0.0%** | 349.6 px | (-345.6, -52.5) |
| `val_middle_right` | 1 / 0 | 0 / 1 | **0.0%** | 948.1 px | (-918.4, +235.4) |
| `val_top_center` | 1 / 0 | 0 / 1 | **0.0%** | 702.5 px | (-697.5, -84.0) |
| `val_top_left` | 1 / 0 | 0 / 1 | **0.0%** | 353.2 px | (-345.6, +73.0) |
| `val_top_right` | 1 / 0 | 0 / 1 | **0.0%** | 925.6 px | (-889.4, +256.1) |

### Training Performance vs. Independent Validation Performance

> **Note:** Training error only measures how tightly the model fits the 9 calibration points and must NOT be interpreted as real-world validation accuracy.

| Evaluation Split | Purpose | Mean Error (px) | Validation Accuracy |
|---|---|---:|---:|
| **Calibration Training Set** | Model fitting residual (training fit only) | 88.7 px | N/A (training split) |
| **Independent Validation Set** | Unseen test targets (real accuracy) | 676.7 px (30.7% diag) | 0.0% hits / 0.0% within 5% |

## Simple Explanation

Status: [COMPLETED]. The system successfully predicted gaze positions inside the target boundaries in 0 out of 9 valid trials (0.0%). Its average prediction error was 676.7 pixels (median: 702.5 px, 30.7% of the screen diagonal), and 0.0% of valid trials fell within 5% of the screen diagonal (0 out of 9). Weakest direction observed: RIGHT (mean error 982.5 px); weakest screen region: right_column (mean error 982.5 px). This experiment performed worse than the previous validated experiment (experiment_20261127_173522) according to the independent validation results (average error changed from 142.6 px to 676.7 px (374.6% error increase); target selection accuracy changed from 11.1% to 0.0% (-11.1 percentage points)).

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
| `experiment_20261126_130647` | **PURE_DECOUPLED_LINEAR** | StandardScaler | 0.0% | 0.0% | 0.0% | 0.0% | **561.3 px** | 616.5 px | 0.0% | RIGHT / right_column | Validated |
| `experiment_20261126_130819` | **COMPACT_LINEAR** | StandardScaler | 0.0% | 0.0% | 0.0% | 0.0% | **412.2 px** | 412.0 px | 0.0% | UP / top_row | Validated |
| `experiment_20261126_131018` | **COMPACT_LINEAR** | StandardScaler | 11.1% | 11.1% | 22.2% | 77.8% | **156.2 px** | 161.3 px | 0.0% | RIGHT / right_column | Validated |
| `experiment_20261127_173255` | **PURE_DECOUPLED_LINEAR** | StandardScaler | 0.0% | 0.0% | 0.0% | 22.2% | **304.2 px** | 327.5 px | 0.0% | CENTER / center_column | Validated |
| `experiment_20261127_173522` | **PURE_DECOUPLED_LINEAR** | StandardScaler | 11.1% | 0.0% | 22.2% | 100.0% | **142.6 px** | 136.1 px | 0.0% | CENTER / center_column | Validated |
| `experiment_20261127_182323` | **COMPACT_LINEAR** | StandardScaler | 0.0% | 0.0% | 0.0% | 0.0% | **676.7 px** | 702.5 px | 0.0% | RIGHT / right_column | Validated |

- **Compatibility Warning:** Target button/circle boundaries differ across validated runs.
- **Best Validated Experiment:** `experiment_20261125_185607` (55.6% target selection, 93.6 px mean error) (weakest dir: UP, weakest region: top_row).
- **Lowest Mean Pixel Error:** `experiment_20261125_153718` (87.8 px).
- **Latest vs. Previous Validated (`experiment_20261127_182323` vs `experiment_20261127_173522`):** **GOT WORSE** (Mean Error +534.1 px (374.6% error increase); Target Selection Accuracy -11.1 percentage points; Within 5% Accuracy -22.2 percentage points).

## Diagnostic Plots

![plot_actual_vs_predicted](plot_actual_vs_predicted.png)
![plot_error_distributions](plot_error_distributions.png)
![plot_per_target_breakdown](plot_per_target_breakdown.png)

