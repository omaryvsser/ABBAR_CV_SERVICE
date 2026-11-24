# Gaze Tracking Experiment Report: `experiment_20261124_221738`

## Experiment Overview

| Field | Value |
|---|---|
| **Experiment ID** | `experiment_20261124_221738` |
| **Date and Time** | 2026-11-24 22:17:56 |
| **Screen Resolution** | 1920x1080 (Diagonal: 2202.9 px) |
| **Calibration Model** | POLYNOMIAL |
| **Normalization Method** | StandardScaler |
| **Number of Training Samples** | 9 targets (258 accepted frames) |
| **Number of Validation Trials** | 0 (Validation not completed) |

## Main Results

> **Status:** Validation not completed — Accuracy is not available yet.

| Result | Value |
|---|---:|
| Accuracy within 2% | Not available (Validation not completed) |
| Accuracy within 5% | Not available (Validation not completed) |
| Accuracy within 10% | Not available (Validation not completed) |
| Target Selection Accuracy | Not available (Validation not completed) |
| Average Error (pixels) | Not available (Validation not completed) |
| Average Error (% of screen diagonal) | Not available (Validation not completed) |

## Detailed Results

### Training Performance vs. Independent Validation Performance

> **Note:** Training error only measures how tightly the model fits the 9 calibration points and must NOT be interpreted as real-world validation accuracy.

| Evaluation Split | Purpose | Mean Error (px) | Validation Accuracy |
|---|---|---:|---:|
| **Calibration Training Set** | Model fitting residual (training fit only) | 0.9 px | N/A (training split) |
| **Independent Validation Set** | Unseen test targets | N/A | Validation not completed — Accuracy is not available yet. |

## Simple Explanation

Validation not completed — Accuracy is not available yet. Calibration training fit completed (average residual on calibration points: 0.9 pixels), which reflects model fitting only and does not represent real-world validation accuracy. Run validation on independent targets to measure accuracy percentages and target selection success.

## Comparison with Previous Experiments

| Experiment ID | Model | Norm | Sel Acc | Within 2% | Within 5% | Within 10% | Mean Error | Median | Max Error | Valid / Rej | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `experiment_20261123_221716` | **POLYNOMIAL** | None | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 / 0 | Calibrated Only (No Validation) |
| `experiment_20261123_224241` | **POLYNOMIAL** | None | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 / 0 | Calibrated Only (No Validation) |
| `experiment_20261124_220355` | **POLYNOMIAL** | StandardScaler | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 / 0 | Calibrated Only (No Validation) |
| `experiment_20261124_221738` | **POLYNOMIAL** | StandardScaler | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 / 0 | Calibrated Only (No Validation) |

- **Best Validated Experiment:** Not available — no completed validation experiments found.

