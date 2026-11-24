# Gaze Tracking Experiment Report: `experiment_20261124_225435`

## Experiment Overview

| Field | Value |
|---|---|
| **Experiment ID** | `experiment_20261124_225435` |
| **Date and Time** | 2026-11-24 22:54:54 |
| **Screen Resolution** | 1920x1080 (Diagonal: 2202.9 px) |
| **Calibration Model** | LINEAR |
| **Normalization Method** | StandardScaler |
| **Number of Training Samples** | 9 targets (216 accepted frames) |
| **Validation Status** | `NOT_STARTED` |
| **Number of Validation Trials** | 0 (NOT_STARTED: Validation not started) |

## Main Results

> **Status (`NOT_STARTED`):** Validation not completed — Accuracy is not available yet.
> **Reason:** Calibration finished, but independent validation has not been started yet (press 't' in GUI or run --validate-screen).

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
| **Calibration Training Set** | Model fitting residual (training fit only) | 120.5 px | N/A (training split) |
| **Independent Validation Set** | Unseen test targets | N/A | Validation not completed — Accuracy is not available yet. |

## Simple Explanation

Validation not completed — Accuracy is not available yet. Status: [NOT_STARTED] — Calibration finished, but independent validation has not been started yet (press 't' in GUI or run --validate-screen). Calibration training fit residual was 120.5 pixels on the calibration grid only, which does NOT represent real-world validation accuracy.

## Comparison with Previous Experiments

| Experiment ID | Model | Norm | Sel Acc | Within 2% | Within 5% | Within 10% | Mean Error | Median | Rej % | Weakest Dir / Region | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `experiment_20261123_221716` | **POLYNOMIAL** | None | N/A | N/A | N/A | N/A | N/A | N/A | N/A | -- | Calibrated Only (No Validation) |
| `experiment_20261123_224241` | **POLYNOMIAL** | None | N/A | N/A | N/A | N/A | N/A | N/A | N/A | -- | Calibrated Only (No Validation) |
| `experiment_20261124_220355` | **POLYNOMIAL** | StandardScaler | N/A | N/A | N/A | N/A | N/A | N/A | N/A | -- | Calibrated Only (No Validation) |
| `experiment_20261124_221738` | **POLYNOMIAL** | StandardScaler | N/A | N/A | N/A | N/A | N/A | N/A | N/A | -- | Calibrated Only (No Validation) |
| `experiment_20261124_225435` | **LINEAR** | StandardScaler | N/A | N/A | N/A | N/A | N/A | N/A | N/A | -- | Calibrated Only (No Validation) |

- **Best Validated Experiment:** Not available — no completed validation experiments found.

