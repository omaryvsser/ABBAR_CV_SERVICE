# AABAR — Computer Vision Service (Phases 1, 2 & 3: Calibration, Gaze Mapping & Evaluation)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.x-green.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/opencv-4.11-orange.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/tests-110%20passed-brightgreen.svg)]()

> **AABAR (عَبْر)** is an Arabic-first Augmentative and Alternative Communication (AAC) graduation platform designed to empower individuals with Amyotrophic Lateral Sclerosis (ALS) and severe motor impairments to communicate via eye gaze.

The **Computer Vision Service** is a standalone, real-time Python service responsible for capturing camera frames, detecting facial and ocular landmarks, extracting normalized eye features, estimating 3D head pose, calibrating personalized neutral gaze baselines, classifying relative gaze direction with hysteresis, conducting 9-point screen calibration, training regularized regression models for continuous gaze-to-screen coordinate mapping, evaluating spatial accuracy on independent test targets, and logging structured experiment datasets (CSV, JSON, Excel).

---

## Table of Contents

- [Core Concepts & Conceptual Distinctions](#core-concepts--conceptual-distinctions)
- [System Architecture (Complete Phase 3 Pipeline)](#system-architecture-complete-phase-3-pipeline)
- [Directory Structure](#directory-structure)
- [Mathematical Formulas & Feature Engineering](#mathematical-formulas--feature-engineering)
  - [1. Normalized Iris Ratios](#1-normalized-iris-ratios)
  - [2. Continuous Eye Openness & States](#2-continuous-eye-openness--states)
  - [3. Binocular Feature Fusion & Occlusion Fallback](#3-binocular-feature-fusion--occlusion-fallback)
  - [4. Saccade-Adaptive Gaze Smoothing (EMA)](#4-saccade-adaptive-gaze-smoothing-ema)
  - [5. Relative Gaze Direction Classification](#5-relative-gaze-direction-classification)
  - [6. Lightweight 3D Head Pose Estimation (solvePnP)](#6-lightweight-3d-head-pose-estimation-solvepnp)
  - [7. 10-Dimensional Continuous Feature Vector](#7-10-dimensional-continuous-feature-vector)
  - [8. Ridge-Regularized Mapping Models (Linear & Polynomial)](#8-ridge-regularized-mapping-models-linear--polynomial)
  - [9. Accuracy Evaluation & Error Metrics](#9-accuracy-evaluation--error-metrics)
- [9-Point Screen Calibration Workflow](#9-point-screen-calibration-workflow)
- [Independent Screen Accuracy Validation](#independent-screen-accuracy-validation)
- [Experiment Datasets & Logging Schema](#experiment-datasets--logging-schema)
- [Calibration Profile Persistence & Compatibility](#calibration-profile-persistence--compatibility)
  - [Coordinate Convention & Legacy Profile Compatibility](#coordinate-convention--legacy-profile-compatibility)
- [Simulated Screen Canvas & Real-Time Visualization](#simulated-screen-canvas--real-time-visualization)
- [Structured Output Schema (JSON)](#structured-output-schema-json)
- [Installation & Setup](#installation--setup)
- [Running the Real-Time Demo](#running-the-real-time-demo)
- [Running the Automated Test Suite](#running-the-automated-test-suite)
- [Performance & Latency Characteristics](#performance--latency-characteristics)
- [Known Limitations & Real-World Accuracy Notice](#known-limitations--real-world-accuracy-notice)
- [Phase 4 Roadmap: AAC Integration & Dwell Selection](#phase-4-roadmap-aac-integration--dwell-selection)
- [License & Attributions](#license--attributions)

---

## Core Concepts & Conceptual Distinctions

To ensure sound engineering and research methodology, this project strictly separates four developmental tiers:

```text
+---------------------------------------------------------------------------------------+
| 1. Eye Landmark Detection (Phase 1)                                                   |
|    Extracting raw (x, y, z) 3D coordinate vectors for eyelids and irises via MediaPipe. |
+---------------------------------------------------------------------------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------------+
| 2. Relative Gaze Estimation (Phase 2)                                                 |
|    Normalizing ocular ratios within canthi bounds and classifying relative direction   |
|    (CENTER, LEFT, RIGHT, UP, DOWN). Independent of screen resolution. NOT screen gaze! |
+---------------------------------------------------------------------------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------------+
| 3. 9-Point Screen Calibration (Phase 3)                                               |
|    A guided target acquisition procedure recording 10-D eye/head features paired with  |
|    known ground-truth screen coordinates (X_target, Y_target) across a 3x3 grid.      |
+---------------------------------------------------------------------------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------------+
| 4. Continuous Gaze-to-Screen Mapping & Validation (Phase 3)                           |
|    Fitting Ridge-regularized regressors to predict continuous screen coordinates      |
|    (gaze_x, gaze_y) in [0.0, 1.0], validated on 9 independent regional test targets.  |
+---------------------------------------------------------------------------------------+
```

---

## System Architecture (Complete Phase 3 Pipeline)

```text
                                Webcam Feed / Synthetic Mock Feed
                                                |
                                                v
                              CameraManager (Low-latency Thread)
                                - Buffer size = 1 (Zero latency)
                                                |
                                                v
                          FaceLandmarkerDetector (MediaPipe Tasks API)
                            - 478 3D Normalized Facial Landmarks
                            - Monotonic timestamps for smoothing
                                                |
                        +-----------------------+-----------------------+
                        |                                               |
                        v                                               v
              EyeFeatureExtractor                              HeadPoseEstimator
          - Normalized Iris Pos (x, y)                   - Canonical 3D anthropometric model
          - EAR & Continuous Openness                    - cv2.solvePnP + RQDecomp3x3
          - Reliability & Single-eye Fallback            - Yaw, Pitch, Roll (in degrees)
                        |                                               |
                        v                                               |
               Binocular Feature Fusion                                 |
          - Reliability-weighted ratio averages                         |
          - VOR Yaw/Pitch Compensation & Offset Alignment <-------------+
                        |
                        +-----------------------+-----------------------+
                        |                                               |
                        v                                               v
                   GazeSmoother                             Continuous Feature Vector
          - Saccade-Adaptive EMA Filter                     [rx, ry, yaw, pitch, roll, ...] (10D)
          - Blink / Invalid Frame Hold                                  |
                        |                               +---------------+---------------+
                        v                               |                               |
             GazeDirectionClassifier                    v                               v
        - Asymmetric dual-threshold hysteresis   ScreenCalibrator               ScreenValidator
        - CENTER, LEFT, RIGHT, UP, DOWN          - 9-Point Targets (3x3)        - 9-Region (full_9) / 5-Pt
                        |                        - Fixation Stability Gate      - MAE, RMSE, Hit Rate
                        |                        - Timeout / Retry ('y')        - Dir & Region Breakdown
                        |                               |                               |
                        |                               v                               v
                        |                        BaseGazeMapper                 ExperimentLogger
                        |                        - CompactLinear / CompactPoly  - calibration_samples.csv
                        |                        - Linear / Degree-2 Poly Ridge - validation_results.csv
                        |                        - One-Euro ScreenGazeStabilizer- Explicit Report Statuses
                        |                        - JSON Profile + Validation    - experiment_report.xlsx
                        |                               |                               |
                        +-----------------------+-------+-------------------------------+
                                                |
                                                v
                                         GazeFrameResult
                                - relative_direction (CENTER, LEFT, ...)
                                - screen_gaze (gaze_x, gaze_y, pixel_x, pixel_y)
                                - tracking_quality, confidence, latency
                                                |
                                                v
                                       CVVisualizer Overlay
                                - Selfie mirror display (anatomically correct)
                                - 16:9 Simulated Screen Canvas & Stabilized Cursor
                                - Target i/N Bullseye + Fixation Quality Feedback
                                - HUD: Direction badge, Radar, Validation Summary
```

---

## Directory Structure

```text
cv-service/
├── .env.example                               # Environment template with Phase 3 parameters
├── .gitignore                                 # Git exclusions (.venv, models, data, caches)
├── requirements.txt                           # Pinned dependencies (MediaPipe, OpenCV, openpyxl)
├── README.md                                  # Complete service documentation
├── download_model.py                          # MediaPipe model downloader
├── main.py                                    # CLI runner with --calibrate-screen and --validate-screen
├── models/
│   └── face_landmarker.task                   # Pretrained MediaPipe Face Landmarker model (3.6 MB)
├── data/
│   ├── calibration/                           # Stored user calibration profiles (JSON)
│   │   └── profile_user_default_*.json
│   └── experiments/                           # Output benchmark experiment datasets
│       ├── AABAR_Experiments_Tracker.xlsx     # Central 6-sheet workbook aggregating all runs
│       └── experiment_YYYYMMDD_HHMMSS/
│           ├── calibration_samples.csv        # Raw frame telemetry per calibration target
│           ├── validation_results.csv         # Individual validation trial measurements
│           ├── accuracy_summary.csv           # Summary accuracy % and prediction error metrics
│           ├── experiment_metadata.json       # Hardware, scaler, calibration & validation metadata
│           ├── accuracy_report.json           # Detailed machine-readable accuracy report
│           ├── accuracy_report.md             # Simple beginner-friendly Markdown summary
│           ├── accuracy_report.html           # Interactive standalone HTML report
│           ├── accuracy_report.txt            # Formatted plain-text summary report
│           ├── plot_actual_vs_predicted.png   # Diagnostic scatter plot with error vectors
│           ├── plot_error_distributions.png   # Euclidean error histogram & signed bias scatter
│           ├── plot_per_target_breakdown.png  # Per-target bar chart with threshold lines
│           └── experiment_report.xlsx         # Per-session Excel report workbook
├── src/
│   ├── __init__.py
│   ├── config.py                              # Master typed config (Camera, Model, Gaze, Screen, Viz)
│   ├── pipeline.py                            # End-to-end pipeline coordinator
│   ├── camera/
│   │   ├── __init__.py
│   │   └── camera_manager.py                  # Thread-safe webcam capture
│   ├── core/
│   │   ├── __init__.py
│   │   ├── accuracy_metrics.py                # Single-source-of-truth metric & region/direction engine
│   │   ├── landmark_indices.py                # 478 MediaPipe indices & 3D head pose points
│   │   ├── face_landmarker.py                 # MediaPipe live video detector
│   │   ├── eye_features.py                    # Normalized ratios, EAR, openness, VOR compensation
│   │   ├── head_pose.py                       # solvePnP 3D Head Pose (Yaw, Pitch, Roll)
│   │   ├── gaze_direction.py                  # Relative gaze classification (CENTER, LEFT, ...)
│   │   ├── gaze_smoother.py                   # Saccade EMA & adaptive One-Euro ScreenGazeStabilizer
│   │   ├── gaze_mapper.py                     # Feature extraction, Compact/Linear/Poly Ridge regressors
│   │   ├── calibration_profile.py             # CalibrationProfile dataclass and ProfileManager
│   │   ├── screen_calibration.py              # 9-point calibration state machine with fixation gating
│   │   ├── screen_validator.py                # 9-region & 5-point independent validation workflow
│   │   ├── experiment_logger.py               # CSV, JSON, HTML/MD/TXT, plots & openpyxl Excel exporter
│   │   ├── experiment_tracker.py              # Central historical importer & AABAR_Experiments_Tracker.xlsx
│   │   └── gaze_output.py                     # Structured output data models (ScreenGazeOutput)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── fps_counter.py                     # FPS, frame cycle, and latency tracker
│   └── visualization/
│       ├── __init__.py
│       └── visualizer.py                      # Multi-row HUD, Screen Canvas, Target Overlays
└── tests/
    ├── conftest.py                            # Synthetic face fixtures (478 landmarks)
    ├── test_accuracy_enhancements.py          # Baseline calibration, hysteresis, VOR compensation
    ├── test_accuracy_evaluation.py            # Phase 3 accuracy benchmark, error vectors, thresholds
    ├── test_accuracy_optimization_and_validation.py # VOR yaw sign, One-Euro stabilizer, LOOCV compact models, 9-region retry/skip
    ├── test_calibration_profile.py            # Profile serialization & resolution checks
    ├── test_camera.py                         # CameraManager lifecycle & mock reader
    ├── test_experiment_logger.py              # CSV/JSON logger & file generation
    ├── test_experiment_tracker.py             # Central Excel tracker, historical recovery, dedupe & consistency
    ├── test_eye_features.py                   # Normalized iris ratios, openness, EAR, single-eye fallback
    ├── test_face_landmarker.py                # MediaPipe tasks detector & empty frames
    ├── test_feature_norm_and_reports.py       # StandardScaler leakage prevention, reports & queued validation
    ├── test_gaze_diagnostics.py               # Anatomical LEFT/RIGHT verification & mirror separation
    ├── test_gaze_direction.py                 # Relative direction thresholds & compound diagonals
    ├── test_gaze_mapper.py                    # Ridge regressors, degree-2 polynomial, clamping, selection
    ├── test_gaze_output.py                    # Output schemas, JSON serialization
    ├── test_gaze_smoother.py                  # Saccade acceleration & blink hold
    ├── test_head_pose.py                      # solvePnP Euler angles & safety bounds
    ├── test_horizontal_consistency.py         # Left/Right coordinate consistency, mirror modes & legacy rejection
    ├── test_landmark_indices.py               # MediaPipe topology bounds & uniqueness
    ├── test_pipeline.py                       # Integration tests for end-to-end pipeline
    ├── test_screen_calibration.py             # 9-point calibration workflow & outlier rejection
    └── test_screen_validator.py               # Independent accuracy validation & metrics
```

---

## Mathematical Formulas & Feature Engineering

### 1. Normalized Iris Ratios

To decouple iris position from face distance, head rotation, and camera resolution, iris coordinates are projected onto a local eye coordinate frame defined by the anatomical inner and outer canthi:

$$\hat{\mathbf{u}} = \frac{\mathbf{P}_{\text{inner}} - \mathbf{P}_{\text{outer}}}{\|\mathbf{P}_{\text{inner}} - \mathbf{P}_{\text{outer}}\|}, \quad \hat{\mathbf{v}} = [-\hat{u}_y, \hat{u}_x]^T$$

The horizontal ratio measures the projection along the canthal axis:

$$r_{\text{horizontal}} = \frac{(\mathbf{P}_{\text{iris}} - \mathbf{P}_{\text{outer}}) \cdot \hat{\mathbf{u}}}{\|\mathbf{P}_{\text{inner}} - \mathbf{P}_{\text{outer}}\|}$$

The vertical ratio measures palpebral aperture traversal relative to the superior and inferior eyelid midpoints:

$$r_{\text{vertical}} = \frac{(\mathbf{P}_{\text{iris}} - \mathbf{P}_{\text{top}}) \cdot \hat{\mathbf{w}}_{\text{down}}}{\|(\mathbf{P}_{\text{bottom}} - \mathbf{P}_{\text{top}}) \cdot \hat{\mathbf{w}}_{\text{down}}\|}$$

Both ratios strictly satisfy $r \in [0.0, 1.0]$ via numerical clipping.

### 2. Continuous Eye Openness & States

Eye Aspect Ratio (EAR) is computed using two vertical palpebral chords and the horizontal canthal diameter:

$$\text{EAR} = \frac{\|\mathbf{p}_2 - \mathbf{p}_6\| + \|\mathbf{p}_3 - \mathbf{p}_5\|}{2 \|\mathbf{p}_1 - \mathbf{p}_4\|}$$

Continuous openness maps EAR into $[0.0, 1.0]$:

$$\text{openness} = \text{clip}\left(\frac{\text{EAR} - \text{EAR}_{\text{min}}}{\text{EAR}_{\text{max}} - \text{EAR}_{\text{min}}}, 0.0, 1.0\right)$$

- $\text{openness} < 0.20$: **`CLOSED`** (Blink detected; frozen filter state)
- $0.20 \le \text{openness} < 0.50$: **`PARTIAL`** (Lower tracking confidence)
- $\text{openness} \ge 0.50$: **`OPEN`** (Nominal tracking fidelity)

### 3. Binocular Feature Fusion & Occlusion Fallback

When both eyes are detected, features are combined using ocular reliability weights $w_{\text{left}}, w_{\text{right}} \in [0.0, 1.0]$:

$$r_{\text{combined}} = \frac{w_{\text{right}} \cdot r_{\text{right}} + w_{\text{left}} \cdot r_{\text{left}}}{w_{\text{right}} + w_{\text{left}}}$$

If one eye is occluded (e.g., due to extreme head yaw or asymmetric lighting), the system falls back seamlessly to the remaining reliable eye without crashing or reporting missing data.

### 4. Saccade-Adaptive EMA & One-Euro Screen Cursor Stabilization

Relative gaze ratios use a Saccade-Adaptive Exponential Moving Average (EMA) filter:

$$\Delta = \|\mathbf{x}_t - \hat{\mathbf{x}}_{t-1}\|$$

$$\alpha_t = \text{clip}\left(\alpha_{\text{base}} + (\alpha_{\text{saccade}} - \alpha_{\text{base}}) \cdot \left(\frac{\Delta - \tau_{\text{fixation}}}{\tau_{\text{saccade}} - \tau_{\text{fixation}}}\right), \alpha_{\text{base}}, \alpha_{\text{saccade}}\right)$$

$$\hat{\mathbf{x}}_t = \alpha_t \mathbf{x}_t + (1 - \alpha_t) \hat{\mathbf{x}}_{t-1}$$

During blinks ($\text{openness} < 0.20$), the filter holds its previous position for up to 5 consecutive frames before resetting.

Calibrated continuous screen coordinates `(gaze_x, gaze_y)` additionally pass through `ScreenGazeStabilizer` (an adaptive **One-Euro Filter** with low-confidence spike rejection), damping steady-fixation jitter while responding rapidly during intentional saccades.

### 5. VOR Compensation, Single-Eye Alignment & Relative Direction Classification

`compensate_eye_features()` stabilizes raw iris ratios before Phase 2 classification and Phase 3 calibration/mapping:
- **Corrected VOR Subtraction**: Turning the head left ($+\psi$) counter-rotates the eyes right ($+r_x$); subtracting $0.0065 \cdot \Delta\psi$ horizontally and $0.0020 \cdot \Delta\theta$ vertically removes spurious head-turn drift instead of amplifying it.
- **Single-Eye Resting Offset Alignment**: If one eye blinks or is momentarily occluded, the surviving eye is shifted using calibrated resting offsets (`neutral_r_x`, `neutral_l_x`) so single-eye fallback does not inject step jumps.
- **Hysteresis Classifier**: Relative classification applies personalized baseline zeroing and asymmetric dual-threshold hysteresis (`enter` vs. `exit`).

### 6. Lightweight 3D Head Pose Estimation (solvePnP)

Using 6 canonical 3D facial landmarks (nose tip, chin, eye corners, mouth corners) and `cv2.solvePnP` with Levenberg-Marquardt optimization, 3D rotation vectors are extracted and decomposed via `cv2.RQDecomp3x3` into Tait-Bryan Euler angles:

$$\mathbf{R} = \text{Rodrigues}(\mathbf{r}_{\text{vec}}), \quad [\text{Yaw } (\psi), \text{Pitch } (\theta), \text{Roll } (\phi)] = \text{RQDecomp3x3}(\mathbf{R})$$

### 7. 10-Dimensional Continuous Feature Vector & Normalization

Screen coordinate mapping uses a 10-dimensional continuous numerical feature vector:

$$\mathbf{f} = [r_{\text{combined},x}, r_{\text{combined},y}, r_{R,x}, r_{R,y}, r_{L,x}, r_{L,y}, o_R, o_L, \psi / 30^\circ, \theta / 30^\circ] \in \mathbb{R}^{10}$$

Before regression fitting, `StandardScaler` (pure NumPy, JSON-serialized) is fitted **strictly on calibration training features**:

$$\mathbf{z} = \text{clip}\left(\frac{\mathbf{f} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}}, -8.0, 8.0\right)$$

Constant (zero-variance) features receive $\sigma = 1.0$, and any missing/`NaN`/`Inf` values are safely imputed to $\boldsymbol{\mu}_{\text{train}}$ ($\mathbf{z} = \mathbf{0}$). The exact fitted scaler parameters are saved inside the calibration profile and reused identically during validation and real-time inference. Target coordinates `(gaze_x, gaze_y)` remain normalized to $[0.0, 1.0]$.

### 8. Ridge-Regularized Mapping Models & Generalization Selection

Given $N$ aggregated calibration targets with standardized feature matrix $\mathbf{Z} \in \mathbb{R}^{N \times 10}$ and screen coordinates $\mathbf{Y} \in \mathbb{R}^{N \times 2}$, `compare_and_select_mapper()` evaluates four regularized model families across swept Ridge alphas ($\lambda$):

- **Compact Linear Ridge (`compact_linear`)**: Low-variance model over the 6 core binocular + VOR head features (`raw_x`, `raw_y`, yaw, pitch, and eye symmetry/difference), preventing collinear overfitting on 9-point training sets.
- **Compact Polynomial Ridge (`compact_polynomial`)**: Degree-2 diagonal/interaction terms on the compact subspace (`raw_x`, `raw_y`, yaw, pitch).
- **Full Linear Ridge (`linear`)**: All 10 standardized features $\mathbf{\Phi} = [\mathbf{1}, \mathbf{Z}] \in \mathbb{R}^{N \times 11}$.
- **Full Degree-2 Polynomial Ridge (`polynomial`)**: Interaction basis $\mathbf{\Phi} = [\mathbf{1}, \mathbf{Z}, \mathbf{Z}^2, \{z_i z_j\}] \in \mathbb{R}^{N \times 36}$.
- **Auto Model Selection**: Uses Leave-One-Out Cross-Validation (refitting scaler inside each fold) or held-out validation data to pick the model family and Ridge penalty minimizing out-of-sample generalization error rather than in-sample training error.

### 9. Accuracy Percentages vs. Error Metrics

Accuracy percentages and prediction errors are strictly distinguished:

- **Accuracy within Threshold** (configurable default $2\%$, $5\%$, $10\%$ of screen diagonal $D = \sqrt{W^2 + H^2}$):
  $$\text{Success Rate (\%)} = \frac{\text{Successful Valid Trials}}{\text{Total Valid Trials}} \times 100$$
- **Target Selection Accuracy** (actual rendered target/button boundary hits):
  $$\text{Target Selection Accuracy (\%)} = \frac{\text{Successful Target Hits}}{\text{Total Valid Trials}} \times 100$$
- **Direction & Screen Region Breakdown**: Reports separate hit rates, within-5%/10% diagonal percentages, and mean pixel errors across all directions (`LEFT`, `RIGHT`, `CENTER`, `UP`, `DOWN`) and 9 regions (rows, columns, edges vs. center).
- **Prediction Error**:
  $$\text{error}_x = \hat{x} - x, \quad \text{error}_y = \hat{y} - y, \quad E_{\text{px}} = \sqrt{(\text{error}_x \cdot W)^2 + (\text{error}_y \cdot H)^2}, \quad E_{\%} = \frac{E_{\text{px}}}{D} \times 100$$

---

## 9-Point Screen Calibration Workflow

The calibration process presents 9 targets distributed across a $3 \times 3$ normalized grid $(x, y) \in \{0.10, 0.50, 0.90\}^2$:

```text
[0] Top-Left      (0.10, 0.10)    [1] Top-Center    (0.50, 0.10)    [2] Top-Right      (0.90, 0.10)
[3] Middle-Left   (0.10, 0.50)    [4] Center        (0.50, 0.50)    [5] Middle-Right   (0.90, 0.50)
[6] Bottom-Left   (0.10, 0.90)    [7] Bottom-Center (0.50, 0.90)    [8] Bottom-Right   (0.90, 0.90)
```

### Two-Phase Target Acquisition & Fixation Stability Protocol

For every target:
1. **Settling Phase (0.8s)**:
   The user shifts gaze and establishes steady fixation. A yellow countdown ring contracts toward the bullseye. Samples during this transition are discarded to eliminate saccadic noise.
2. **Collecting Phase (1.2s, max timeout 5.0s)**:
   The target pulses green while steady fixation frames ($\ge 15$ samples) are accumulated. If gaze wanders or blinks persist until timeout, the target enters `FAILED` with clear visual guidance; pressing **`y`** retries that single target cleanly without restarting the whole calibration.

### Outlier Rejection & Robust Aggregation
- **Blink & Closure Rejection**: Frames with $\text{openness} < 0.25$ or $\text{is\_blink} = \text{True}$ are rejected.
- **Head Rotation & Velocity Guard**: Frames with $|\text{yaw}| > 25^\circ$, $|\text{pitch}| > 20^\circ$, sudden head velocity steps ($>4^\circ$/frame), or large inter-frame saccade jumps are discarded.
- **Rolling Fixation Dispersion Gate**: Frames whose recent 4–6 frame rolling window exceeds `fixation_max_std` are excluded so wandering gaze never corrupts target centroids.
- **Trimmed-Median Filter**: For each target, outliers outside the 10th–90th percentiles are filtered before computing the target's median feature vector, and per-target statistics are saved to `calibration_metadata`.

---

## Independent Screen Accuracy Validation

To benchmark mapping accuracy objectively without data leakage across the entire screen, default validation (`validation_grid_mode="full_9"`) tests **9 non-training targets spanning all 9 screen regions** (`0.18`, `0.50`, `0.82`), while an optional 5-target interior mode (`interior_5`, `0.30/0.50/0.70`) is also supported:

```text
[V0] Top-Left      (0.18, 0.18)    [V1] Top-Center    (0.50, 0.18)    [V2] Top-Right      (0.82, 0.18)
[V3] Middle-Left   (0.18, 0.50)    [V4] Center        (0.50, 0.50)    [V5] Middle-Right   (0.82, 0.50)
[V6] Bottom-Left   (0.18, 0.82)    [V7] Bottom-Center (0.50, 0.82)    [V8] Bottom-Right   (0.82, 0.82)
```

For each validation target:
- Settling (0.8s) allows eye arrival without scoring saccades.
- Collecting (1.2s, max timeout 4.5s) accumulates steady fixation features and aggregates via median. If a target fails due to blinking or unstable gaze, press **`y`** to retry or **`k`** to skip and mark that trial `REJECTED`.
- Validation reports track explicit lifecycle states (**`COMPLETED`**, **`FAILED`**, **`INCOMPLETE`**, **`IN_PROGRESS`**, **`NOT_STARTED`**) with plain-language explanations, update both the saved calibration profile and the same experiment directory when run immediately after calibration (`'s'` -> `'t'`), and compute MAE, RMSE, Euclidean pixel error, Target Selection Accuracy, Direction Accuracy (`LEFT/RIGHT/CENTER/UP/DOWN`), and Screen Region Accuracy.

---

## Experiment Datasets & Logging Schema

Every calibration and validation run writes a structured experiment directory:

`data/experiments/experiment_YYYYMMDD_HHMMSS/`

### 1. `calibration_samples.csv`
Contains raw frame-by-frame telemetry for all 9 targets:
- `experiment_id`, `session_id`, `target_id`, `target_x`, `target_y`, `timestamp`
- `raw_x`, `raw_y`, `r_iris_x`, `r_iris_y`, `l_iris_x`, `l_iris_y`, `r_openness`, `l_openness`
- `yaw`, `pitch`, `roll`, `confidence`, `tracking_quality`
- `accepted` (`True`/`False`), `rejection_reason`

### 2. `validation_results.csv`
Logs trial-level predictions and errors for independent validation targets:
- `target_id`, `target_x`, `target_y`, `predicted_x`, `predicted_y`
- `error_x`, `error_y`, `euclidean_error`
- `target_x_px`, `target_y_px`, `predicted_x_px`, `predicted_y_px`, `euclidean_error_px`
- `sample_count`, `duration_sec`, `status`

### 3. `accuracy_summary.csv`
Summary benchmark metrics:
- `total_trials`, `successful_trials`, `mae_x`, `mae_y`, `rmse_x`, `rmse_y`
- `mean_euclidean_error`, `median_euclidean_error`, `max_euclidean_error`
- `mae_x_px`, `mae_y_px`, `mean_euclidean_error_px`, `median_euclidean_error_px`, `max_euclidean_error_px`

### 4. `experiment_metadata.json` & `accuracy_report.json`
Machine-readable JSON records containing model hyperparameters, screen dimensions, camera index, selected model type, and target-by-target breakdowns.

### 5. `AABAR_Experiments_Tracker.xlsx` (Central Multi-Experiment Tracker)
Maintained automatically inside `data/experiments/AABAR_Experiments_Tracker.xlsx` whenever an experiment or validation finishes (or on demand via `python main.py --sync-tracker`). Aggregates all historical and future experiments across 6 formatted sheets:
- **Sheet 1 (`Experiments Summary`)**: Main dashboard showing model, normalization, samples, trials, Target Selection Accuracy, 2%/5%/10% diagonal accuracy, Mean/Median/Max error, Status, Best/Weakest Direction, and Comparison Chart.
- **Sheet 2 (`All Test Trials`)**: Every validation trial across all experiments with Actual/Predicted coordinates, Actual/Predicted directions, pixel/diagonal errors, boundary hit, and rejection reason.
- **Sheet 3 (`Direction Accuracy`)**: Per-experiment `LEFT`, `RIGHT`, `CENTER`, `UP`, `DOWN` performance and Direction Chart.
- **Sheet 4 (`Screen Region Accuracy`)**: Per-experiment 9-region metrics (`Top Left` through `Bottom Right`) plus 3x3 visual grid matrix.
- **Sheet 5 (`Experiment Comparison`)**: Objective validation-only benchmark ranking highlighting the #1 validated experiment (unvalidated sessions excluded honestly).
- **Sheet 6 (`Data Quality & Validation`)**: Diagnostics showing validation start/completion state, missing targets, invalid trials, failure causes, and next actions.

### 6. `experiment_report.xlsx` (Per-Session Excel Report)
Generated alongside each experiment session via `openpyxl`:
- **Sheet 1**: Experiment Overview & Hardware Metadata
- **Sheet 2**: Accuracy Benchmark Summary
- **Sheet 3**: Validation Trial Results
- **Sheet 4**: Direction & Region Accuracy Breakdown

---

## Calibration Profile Persistence & Compatibility

Trained mapping models are serialized as pure JSON dictionaries without insecure binary pickles:

`data/calibration/profile_{user_id}_{timestamp}.json`

### Profile Structure
```json
{
  "profile_id": "user_default_1718000000",
  "user_id": "user_default",
  "created_at": 1718000000.45,
  "screen_width": 1920,
  "screen_height": 1080,
  "camera_index": 0,
  "model_type": "polynomial",
  "coordinate_convention": "interface_left0_right1_v1",
  "model_data": {
    "model_type": "polynomial",
    "degree": 2,
    "ridge_alpha": 0.01,
    "weights": [[...], [...]]
  },
  "training_metrics": {
    "mae_x": 0.018,
    "mae_y": 0.022,
    "mean_euclidean_px": 38.4
  },
  "model_version": "3.1"
}
```

### Coordinate Convention & Legacy Profile Compatibility
All screen-space targets, gaze predictions (`gaze_x`, `gaze_y`), and Simulated Screen coordinates follow one canonical interface convention:
- `gaze_x = 0.0` → **Left** side of the patient interface (`1.0` → **Right** side)
- `gaze_y = 0.0` → **Top** edge (`1.0` → **Bottom** edge)

Camera selfie mirroring (`visualization.mirror_display = True`) flips **only** the webcam image and drawn camera-space facial landmarks for visual preview ergonomics. MediaPipe landmark extraction, ocular features, `GazeDirection`, `ScreenCalibrator`, target overlays, and the Simulated Screen widget never flip screen coordinates.

Profiles saved without `"coordinate_convention": "interface_left0_right1_v1"` (legacy `"3.0"` profiles calibrated under the old mirrored target overlay) raise `IncompatibleCalibrationProfileError` and are explicitly rejected at load time instead of silently predicting inverted horizontal coordinates.

### Display Resolution Validation
When loading a profile via `--profile <path>`, the system verifies both `coordinate_convention` and screen resolution compatibility. Legacy profiles are refused cleanly with a clear recalibration notice; resolution mismatches log a prominent warning.

---

## Simulated Screen Canvas & Real-Time Visualization

The GUI visualization window includes a high-fidelity diagnostic overlay:

```text
+-----------------------------------------------------------------------------------------+
| STATE: [ACTIVE] | FPS: 30.2 | CAM: MIRRORED | HEAD: Y:+2.1deg P:-1.4deg R:+0.3deg       |
| GAZE: [ CENTER ] | Openness: L:95% | R:94%                                              |
| SCREEN GAZE: (0.521, 0.483) | Px: (1001, 522) | SimCursor: ( 89/170, 46/96) | [GOOD]    |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|                 Selfie Mirror Camera View (video & landmarks only)                      |
|         Targets, gaze_x predictions & Simulated Screen ALWAYS use L=0 -> R=1           |
|                                                                                         |
|       (Target i/N Bullseye & Fixation Feedback: rendered at interface (target.x, y))  |
|                                                                                         |
| +-------------------------+                                     +---------------------+ |
| | Simulated Screen   L..R |                                     | Gaze Radar          | |
| |        o (cursor)       |                                     |       [x]           | |
| +-------------------------+                                     +---------------------+ |
+-----------------------------------------------------------------------------------------+
| Controls: 's'=Calib | 't'=Validate | 'y'=Retry | 'k'=Skip | 'c'=Center | 'r'=Rec | 'q' |
+-----------------------------------------------------------------------------------------+
```

1. **Simulated Screen Widget (Bottom-Left)**:
   A semi-transparent 16:9 monitor box (`170x96`) with explicit `L` / `R` orientation markers, center axes, live stabilized cursor `(gaze_x, gaze_y)`, and widget cursor pixel coordinates `c:(x, y)`.
2. **Active Target Bullseye & Fixation Feedback**:
   During calibration (`'s'`) or validation (`'t'`), an animated bullseye shows exact progress `Target i/N`, progress arc, and live color-coded fixation stability feedback directly at interface `(target.x, target.y)` without horizontal camera-mirror flipping. Failed targets pause with clear retry (`'y'`) / skip (`'k'`) prompts.
3. **HUD Rows 1–4**:
   Displays camera preview mode (`CAM: MIRRORED` / `NON-MIRRORED`), `CALIB: Target i/N` / `VALID: Target i/N` / `VALIDATED` summary metrics, relative direction (`GAZE: [ ... ]`), live screen coordinates, Simulated Screen offset `SimCursor`, and tracking quality.

---

## Structured Output Schema (JSON)

Every processed frame produces a structured data record via `GazeFrameResult.to_dict()`:

```json
{
  "frame_id": 142,
  "timestamp": 1718000000.123,
  "processing_latency_ms": 4.62,
  "face_detected": true,
  "is_tracking": true,
  "gaze_direction": "CENTER",
  "gaze_direction_smoothed": "CENTER",
  "raw_ratios": {
    "combined_horizontal": 0.502,
    "combined_vertical": 0.498,
    "right_eye": {
      "horizontal": 0.504,
      "vertical": 0.496,
      "openness": 0.88,
      "openness_state": "OPEN",
      "is_blink": false,
      "reliability": 0.92
    },
    "left_eye": {
      "horizontal": 0.500,
      "vertical": 0.500,
      "openness": 0.86,
      "openness_state": "OPEN",
      "is_blink": false,
      "reliability": 0.90
    }
  },
  "smoothed_ratios": {
    "horizontal": 0.501,
    "vertical": 0.499
  },
  "head_pose": {
    "yaw": 1.25,
    "pitch": -0.80,
    "roll": 0.15,
    "is_valid": true
  },
  "screen_gaze": {
    "gaze_x": 0.5214,
    "gaze_y": 0.4832,
    "pixel_x": 1001,
    "pixel_y": 522,
    "screen_width": 1920,
    "screen_height": 1080,
    "is_calibrated": true,
    "model_type": "polynomial",
    "tracking_quality": "EXCELLENT",
    "confidence": 0.94
  },
  "tracking_confidence": 0.91,
  "is_blink": false
}
```

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- Virtual environment (`venv`)
- Standard laptop webcam (or `--mock-camera` for synthetic testing)

### Setup Instructions
```bash
# Navigate to the CV service directory
cd cv-service

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download MediaPipe Face Landmarker model (3.6 MB)
python download_model.py
```

---

## Running the Real-Time Demo

### 1. Default Interactive Mode
```bash
python main.py
```

### 2. Run Calibration + Validation (Auto-Queued)
```bash
# Start 9-point calibration and automatically run 9-region validation afterward
python main.py --calibrate-screen

# Calibrate only without queuing automatic validation
python main.py --calibrate-screen --calibration-only
```

### 3. Load an Existing Calibration Profile & Validate
```bash
python main.py --profile data/calibration/profile_user_default_latest.json --validate-screen
```

### 4. Multi-Trial and Randomized Accuracy Testing
```bash
# Run 2 repetitions per target with randomized presentation order
python main.py --profile data/calibration/profile_user_default_latest.json --validate-screen --validation-repeats 2 --randomize-val-targets
```

### 5. Central Excel Tracker & Prominent Results Viewing
```bash
# Refresh data/experiments/AABAR_Experiments_Tracker.xlsx from all historical & recent runs
python main.py --sync-tracker

# Display the prominent results summary directly in terminal (no raw JSON needed)
python main.py --show-results latest
python main.py --show-results experiment_20261124_225544

# Compare all sessions side-by-side
python main.py --compare-all-experiments
python main.py --compare-experiments data/experiments/experiment_20261123_221716 data/experiments/experiment_20261123_224241
```

### 6. Synthetic / Mock Camera Mode (No Physical Camera Needed)
```bash
python main.py --mock-camera --max-frames 300
```

### 7. Interactive Keyboard Controls
| Key | Action | Description |
| :---: | :---: | :--- |
| **`s`** / **`S`** | **Calibrate & Validate** | Starts 9-point calibration (and auto-queues 9-region validation upon completion) or cancels active calibration |
| **`t`** / **`T`** | **Screen Validation** | Starts/cancels independent validation (links to active calibration experiment) |
| **`y`** / **`Y`** | **Retry Target** | Retries the current failed calibration or validation target |
| **`k`** / **`K`** | **Skip Target** | Skips a failed validation target (records as `REJECTED`) and advances |
| **`p`** / **`P`** | **Save Profile** | Persists current calibration profile to `data/calibration/` |
| **`c`** / **`C`** | **Recalibrate Baseline**| Re-zeros 1.5s resting center baseline offsets |
| **`v`** / **`V`** | **Recalibrate Vertical**| Re-zeros only the vertical resting baseline |
| **`r`** / **`R`** | **Record Telemetry** | Toggles diagnostic CSV telemetry logging |
| **`q`** / **ESC** | **Quit** | Gracefully cleans up and terminates the pipeline |

---

## Running the Automated Test Suite

The test suite contains **110 automated unit, integration, and diagnostic tests**:

```bash
cd cv-service
MPLCONFIGDIR=/tmp/mpl_test .venv/bin/pytest -v
```

### Breakdown of Test Suites (110 Tests Total)
- **`test_experiment_tracker.py` (4 tests)**: Historical experiment recovery without fabricating missing accuracy, empty startup folder filtering, `(Experiment ID, Trial ID)` deduplication, 6-sheet `AABAR_Experiments_Tracker.xlsx` formatting/charts, automatic tracker updates after new sessions, direction/9-region math, and single-source consistency across JSON, HTML, Markdown, CLI, Excel, and UI HUD.
- **`test_accuracy_optimization_and_validation.py` (8 tests)**: Corrected VOR yaw/pitch subtraction, single-eye resting offset alignment, `ScreenGazeStabilizer` One-Euro saccade/jitter/outlier control, compact Ridge LOOCV generalization selection, calibration fixation gating & retry metadata, 9-region validation direction/region breakdown + retry/skip, explicit report statuses (`NOT_STARTED`/`INCOMPLETE`/`FAILED`/`COMPLETED`) & comparison warnings, and end-to-end queued pipeline execution.
- **`test_horizontal_consistency.py` (6 tests)**: Canonical `gaze_x=0 Left / gaze_x=1 Right` horizontal pipeline consistency across anatomical eye features, 9-point calibration, target overlay & Simulated Screen widget under both `mirror_display=True` and `False`, debug UI indicators, and explicit `IncompatibleCalibrationProfileError` rejection of legacy profiles.
- **`test_feature_norm_and_reports.py` (9 tests)**: `StandardScaler` training-only fitting, constant/missing/extreme input safety, scaler JSON serialization & `ProfileManager` reload, legacy unscaled profile backwards compatibility, LOOCV selection leak prevention, actual circular/rectangular target hit-testing, missing-validation honesty (`Validation not completed`), Markdown/HTML reports, experiment comparison trends, visualizer validation rendering, and pipeline real-time inference / queued validation transitions.
- **`test_accuracy_evaluation.py` (9 tests)**: Exact error formulas (signed, absolute, Euclidean, pixel), 2%/5%/10% diagonal threshold compliance, trial rejection and denominator separation, train/test target independence, multi-trial repeats & randomization, per-target bias patterns, 90th percentile error, live dashboard feed, Matplotlib PNG plot generation, experiment comparison table.
- **`test_screen_calibration.py` (5 tests)**: 9-point grid layout, settling duration, target progression, blink/head turn rejection, cancellation.
- **`test_screen_validator.py` (2 tests)**: Non-training targets geometry, validation workflow, trial recording, accuracy metrics calculation.
- **`test_gaze_mapper.py` (7 tests)**: Linear, CompactLinear, CompactPolynomial & Polynomial Ridge regressors, 10-D feature extraction, boundary clamping, JSON serialization, model comparison.
- **`test_calibration_profile.py` (3 tests)**: Profile serialization roundtrip, ProfileManager save/load, screen resolution compatibility validation.
- **`test_experiment_logger.py` (1 test)**: CSV generation, JSON reports, metadata logging, optional openpyxl Excel export.
- **`test_accuracy_enhancements.py` (13 tests)**: Baseline calibration progress, trimmed-median outlier rejection, dual-threshold hysteresis, persistence window, VOR pitch compensation.
- **`test_gaze_diagnostics.py` (3 tests)**: Anatomical LEFT/RIGHT verification, landmark canthi topology, display frame separation & immutability.
- **`test_gaze_direction.py` (6 tests)**: Relative direction classification, deadzone bounds, compound diagonals.
- **`test_gaze_smoother.py` (6 tests)**: Immediate first frame response, steady-state EMA math, saccade acceleration, blink holding.
- **`test_head_pose.py` (3 tests)**: solvePnP Euler angles on frontal and turned faces, missing/truncated landmark safety.
- **`test_eye_features.py` (10 tests)**: Openness normalization, single-eye occlusion fallback, ratio clamping, degenerate boundaries, blink detection.
- **`test_camera.py` (4 tests)**: Camera failure recovery, mock reader, threaded capture.
- **`test_face_landmarker.py` (4 tests)**: Model loading, empty frames, zero-face frames, landmark subset slicing.
- **`test_landmark_indices.py` (5 tests)**: Internal topology validation, count constants, index uniqueness, eye boundary separation.
- **`test_gaze_output.py` (1 test)**: Output schemas, JSON serialization, ScreenGazeOutput fields.
- **`test_pipeline.py` (1 test)**: Complete end-to-end integration test with mock frames and calibration calls.

---

## Performance & Latency Characteristics

Tested on a standard laptop CPU:
- **MediaPipe Face Landmarker Latency**: ~**2.7 ms** per frame via TensorFlow Lite XNNPACK delegate.
- **Head Pose Estimation Latency**: ~**0.15 ms** via OpenCV solvePnP.
- **Feature Extraction & Ridge Gaze Mapping**: ~**0.08 ms**.
- **Visualizer Overlay & HUD Rendering**: ~**1.4 ms**.
- **Total Pipeline Cycle**: ~**4.8 ms** (sustaining **30 FPS** with low CPU utilization).

---

## Known Limitations & Real-World Accuracy Notice

> [!IMPORTANT]
> **Real-World User Accuracy Notice**:
> All automated tests pass against deterministic synthetic facial landmark fixtures to prove numerical correctness, regularization stability, outlier filtering, and serialization. However, **real-world spatial gaze accuracy (MAE / Euclidean pixel error) remains unmeasured until live 9-point calibration and 9-region validation are conducted with an end user in front of a physical webcam and monitor**.

### Factors Affecting Live Accuracy:
1. **User Distance & Screen Size**: Variations in distance from the screen alter the angular subtense of targets. A standard viewing distance of 50–70 cm is recommended.
2. **Ambient Lighting**: Asymmetric directional shadows can degrade iris contour boundary precision.
3. **Head Motion during Fixation**: While head pose is included in the 10-D feature vector, large voluntary head movements during calibration can introduce non-linear shifts.
4. **Spectacle Glare**: Strong reflections on eyeglasses may occasionally reduce MediaPipe iris landmark confidence.

---

## Phase 4 Roadmap: AAC Integration & Dwell Selection

With Phase 3 complete (screen calibration, continuous gaze mapping, accuracy validation, and profile management), the service is prepared for Phase 4 integration:
1. **Fixation & Dwell-Based Selection**: Dwell timers (e.g., 800ms) with animated progress rings for activating AAC cards without physical motor input.
2. **WebSocket Streaming**: Streaming `GazeFrameResult` JSON over local WebSockets (`ws://localhost:8765`) to the Angular AAC interface.
3. **Virtual Arabic Keyboard**: Linking screen gaze coordinates to dwell-selectable Arabic letter matrices.
4. **Text-to-Speech (TTS) Pipeline**: Synthesizing Arabic speech from selected phrases.

---

## License & Attributions

- **AABAR Graduation Project**: Developed as an assistive communication system for individuals with ALS and motor impairments.
- **MediaPipe**: Licensed under Apache License 2.0 by Google LLC.
- **OpenCV**: Licensed under Apache License 2.0.
