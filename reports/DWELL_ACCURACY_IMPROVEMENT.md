# AABAR — Phase 4 Dwell Selection Accuracy & Reliability Improvement Report

**Date:** November 27, 2026  
**System:** AABAR Eye-Tracking Computer Vision Service (Phase 4 AAC Dwell Engine)  
**Status:** Verification Completed — All 156 Automated Tests Passing (100% Pass Rate)  
**Evaluation Scope:** Deterministic Synthetic Evaluation Suite & Gaze Dynamics Benchmark  

---

## 1. Existing Implementation Audit

A comprehensive code and architectural audit of the existing Phase 4 Dwell Selection Engine was conducted across:
- `dwell/dwell_engine.py`
- `dwell/dwell_config.py`
- `dwell/dwell_state.py`
- `dwell/target_manager.py`
- `docs/DWELL_SELECTION.md`

### Audit Findings by Subsystem:

1. **State Machine Transitions:**
   - The basic state machine correctly established transitions between `IDLE`, `DWELLING`, `SELECTED`, `COOLDOWN`, and `LOST_TRACKING`.
   - **Flaw:** State transitions relied solely on instantaneous target containment without temporal settling or entry confirmation.

2. **Target Entry & Exit Handling:**
   - Single static bounding box (`x <= px <= x + width` and `y <= py <= y + height`).
   - Gaze hovering near a card boundary (e.g. within 5–15 pixels) chattered in and out of containment on alternating frames due to natural ocular micro-tremor.
   - Saccade leave grace (`leave_target_tolerance_ms = 150ms`) only triggered when gaze moved to empty space (`hit_target is None`). If gaze jittered into an adjacent target for even a single frame (33ms), the dwell progress was immediately aborted and wiped to zero.

3. **Gaze Stability & Smoothing:**
   - `stability_filter_enabled: bool = True` and `max_jump_distance: float = 0.25` were declared in `DwellConfig`, but were **completely unreferenced** in `dwell_engine.py`.
   - Raw, unfiltered coordinates were passed directly into target containment checks, exposing target selection to full sensor noise and high-frequency tremor (peak-to-peak jitter exceeding 126 px on 1080p).

4. **Tracking Loss Handling:**
   - Non-finite coordinates (`NaN`, `Inf`) were safely intercepted by `_validate_gaze_input()` and routed into `_handle_lost_tracking()`.
   - `pause_on_lost_tracking: bool = True` held accumulated dwell time for up to `lost_tracking_tolerance_ms = 250ms`, successfully withstanding normal blinks.

5. **Fixation vs. Saccade Discrimination:**
   - No velocity calculation existed. A fast ballistic saccade sweeping across the screen through an AAC button would immediately trigger target entry and accumulate false dwell time.

6. **Delta-Time Robustness:**
   - No safety cap on frame delta time (`dt_ms`). In the event of thread stalls or camera frame drops (e.g., 400ms lag), `dt_ms` would credit the entire elapsed duration in a single frame update.

---

## 2. Baseline Test Results

A 10-scenario deterministic benchmark suite (`dwell/benchmark.py`) was constructed using an injected controllable clock (`DeterministicClock`), exercising the baseline unmodified engine under identical conditions at 30 FPS:

| # | Benchmark Scenario | Expected Behavior | Baseline Outcome | Result |
|---|---|---|---|:---:|
| 1 | **Stable Gaze Inside Target** | 1500ms steady fixation triggers 1 selection at 1200ms | 1 selection at 1266.5ms | **PASS** |
| 2 | **Boundary Jitter (±11 px)** | Fixation near border with ±0.006 noise selects target | 0 selections (chatter reset dwell continuously) | **FAIL** |
| 3 | **Short Saccade (<150ms)** | 100ms saccade outside target preserves progress | 1 selection at 1366.5ms | **PASS** |
| 4 | **Long Exit (>150ms)** | 300ms exit outside target resets dwell progress | 0 selections, reset verified | **PASS** |
| 5 | **Blinks / Tracking Loss** | 150ms blink (None coords) freezes and resumes dwell | 1 selection at 1416.5ms | **PASS** |
| 6 | **Adjacent Target Jitter** | 1-frame jitter across adjacent target preserves dwell | 0 selections (1-frame jitter reset progress) | **FAIL** |
| 7 | **Overlapping Targets** | Priority resolution selects child target (pri=5) | Child selected cleanly | **PASS** |
| 8 | **Repeated Selection Lockout**| Continuous stare locked until looking away | 1st select ok, lockout held, 2nd select after lookaway | **PASS** |
| 9 | **NaN / Inf Coordinates** | Non-finite coordinates handled without exceptions | Zero crashes, safe LOST_TRACKING transition | **PASS** |
| 10 | **Saccade Sweep (90ms)** | Fast sweep (4.15 units/s) does not trigger selection | 0 selections (sweep completed before 1200ms) | **PASS** |

### Baseline Aggregate Metrics:
- **Scenarios Passed:** 8 / 10 (80.0%)
- **Correct Selections:** 6
- **False Selections:** 0
- **Missed Selections:** 2 (Scenarios 2 & 6)
- **Target Switching Accuracy:** **0.0%** under boundary cross-talk
- **Recovery Rate after Tracking Drops:** 100.0%

---

## 3. Identified Problems & Root Causes

1. **Boundary Chattering (Root Cause of Scenario 2 Failure):**  
   Without an expanded exit boundary (hysteresis), when gaze rests within 10–20 pixels of an interactive card's edge, alternating frames fluctuate across the mathematical border. Even though the user's foveal attention is centered on the card, the dwell timer resets repeatedly, resulting in an impossible selection.

2. **Adjacent Target Cross-Talk (Root Cause of Scenario 6 Failure):**  
   In dense AAC layouts (e.g. 2×2 or 3×3 grids with small inter-button margins), looking at Target 1 with slight ocular dispersion causes momentary single-frame excursions across the boundary into Target 2. The engine's `hit_target.target_id != self.current_target.target_id` branch instantly reset accumulated dwell time with **zero grace**, dropping selection accuracy to 0%.

3. **High Fixation Tremor from Lack of Smoothing:**  
   Raw gaze inputs from webcam landmark estimators exhibit typical standard deviations of 25–30 pixels during steady fixation, causing visual cursor jitter and boundary instability.

---

## 4. Implemented Changes

The following architectural enhancements were designed and implemented in `cv-service/dwell/`:

### 4.1 Modular Gaze Smoothing Architecture (`dwell/gaze_filter.py`)
Implemented three pure-Python filter algorithms implementing `BaseGazeFilter`:
- **`PassThroughGazeFilter`**: Zero-latency raw pass-through for deterministic benchmarking and backward compatibility.
- **`EMAGazeFilter`**: Saccade-adaptive Exponential Moving Average with accelerated alpha during sudden steps.
- **`OneEuroGazeFilter`**: 1-Euro adaptive low-pass filter (Casiez et al., CHI 2012). Dynamically modulates cutoff frequency:
  $$f_c = f_{c,min} + \beta |\hat{v}_t|, \quad \alpha = \frac{1}{1 + \frac{\tau}{\Delta t}}, \quad \tau = \frac{1}{2\pi f_c}$$
  At low speeds (fixations), $f_c \to f_{c,min} = 1.2$ Hz to eliminate tremor. At high speeds (saccades), $f_c$ scales with $\beta = 12.0$ to eliminate tracking lag.

### 4.2 Target Boundary Hysteresis (`dwell/target_manager.py`)
Added separate entry and exit boundary evaluation to `DwellTarget`:
- **`contains_entry(px, py, margin=0.0)`**: Requires gaze to enter the true target envelope.
- **`contains_exit(px, py, margin=0.04)`**: Expands the active target's boundary by 4% of screen dimensions while dwelling, forming an attractor zone that prevents edge-chatter resets.
- **`find_target_at()`**: Updated to prioritize holding the active target within its exit boundary if gaze has not entered another target.

### 4.3 Adjacent Target Switch Confirmation (`dwell/dwell_engine.py`)
- Introduced `switch_confirmation_ms = 80.0` ms.
- If gaze momentarily enters an adjacent target while dwelling on the current target, the current target holds its accumulated dwell time in grace.
- Only if gaze resides continuously in the new target for $\ge 80$ ms does the engine execute an intentional target transition, preventing 1-frame boundary noise from wiping dwell progress.

### 4.4 Saccade & Fixation Detection (`dwell/dwell_engine.py`)
- Computes instantaneous foveal speed: $v = \frac{\Delta d}{\Delta t}$ (normalized units / second).
- During ballistic saccades ($v > 1.8$ units/sec), target entry is suppressed until gaze settles, preventing visual button flicker during eye sweeps across the screen.

### 4.5 Dual Telemetry in State Snapshots (`dwell/dwell_state.py`)
- `DwellSnapshot` now serializes both `raw_gaze_x, raw_gaze_y` (unfiltered) and `gaze_x, gaze_y` (stabilized), alongside `is_fixating` and `filter_type`.

### 4.6 Adaptive User Presets (`dwell/dwell_config.py`)
Added `DwellConfig.create_preset(DwellUserSkill)` providing safe, validated parameter sets:
- **`STANDARD`**: 1200ms dwell, 800ms cooldown, One-Euro filter, 4% exit margin.
- **`NOVICE`**: 1500ms dwell, 1000ms cooldown, 6% exit margin, 120ms switch confirmation.
- **`EXPERT`**: 800ms dwell, 500ms cooldown, 3% exit margin, 50ms switch confirmation.
- **`TREMOR_PRONE`**: 1400ms dwell, 900ms cooldown, heavy One-Euro filtering ($f_{c,min}=0.7$ Hz), 8% exit margin.

---

## 5. Quantitative Before / After Comparison

### 5.1 Gaze Filter Benchmark (Identical Synthetic Input Sequence)

Evaluated on 150 frames of stationary fixation with Gaussian noise ($\sigma = 0.015$, ~29 px) and a 0.50 screen-width step saccade on a 1920×1080 display:

| Filter Algorithm | Fixation StdDev (px) | Peak-to-Peak Jitter (px) | 90% Step Latency (ms) | Overshoot (px) | Jitter Reduction |
|---|---:|---:|---:|---:|---:|
| **PassThrough (None)** | 27.70 px | 126.41 px | **0.0 ms** | 0.00 px | Baseline (0%) |
| **Adaptive EMA** | 12.00 px | 60.00 px | 33.3 ms (1 frame) | 0.00 px | **56.7% reduction** |
| **One-Euro (Adaptive)** | **13.33 px** | **64.55 px** | **33.3 ms (1 frame)** | 0.00 px | **51.9% reduction** |

**Selection:** The **One-Euro Filter** was selected as the default filter algorithm because its cutoff frequency scales continuously with velocity, delivering heavy smoothing at rest while dynamically eliminating visual lag during fast saccades without hard threshold switching.

### 5.2 Deterministic Engine Benchmark Comparison (10 Standard Scenarios)

| Metric | Baseline Engine | Enhanced Engine | Improvement |
|---|:---:|:---:|:---:|
| **Scenario Pass Rate** | 8 / 10 (80.0%) | **10 / 10 (100.0%)** | **+20.0 percentage points** |
| **Correct Selections** | 6 | **8** | **+33.3%** |
| **False Selections** | 0 | **0** | Maintained 0 false triggers |
| **Missed Selections** | 2 | **0** | **100% elimination of missed targets** |
| **Boundary Jitter Selection** | FAILED (0 hits) | **PASSED (1 hit)** | Resilient to edge tremor |
| **Adjacent Target Switch Accuracy** | 0.0% | **100.0%** | **+100.0 percentage points** |
| **Tracking Recovery Rate** | 100.0% | **100.0%** | Maintained full recovery |
| **Average Selection Time** | 1349.9 ms | **1309.9 ms** | Faster, more predictable trigger |

---

## 6. Automated Unit & Regression Test Results

The full test suite was executed across all components:
```bash
MPLCONFIGDIR=/tmp/mpl_test .venv/bin/pytest tests/
```
**Results:**
- **Collected Tests:** 156 items
- **Passed Tests:** **156 passed in 16.00s** (100% pass rate)
- **Failures / Errors:** **0**
- **Dedicated Phase 4 Dwell Tests:** 34 tests passing:
  - `tests/test_dwell_engine.py`: 17 passed
  - `tests/test_dwell_enhancements.py`: 7 passed
  - `tests/test_target_manager.py`: 6 passed
  - `tests/test_dwell_state.py`: 4 passed

---

## 7. Simulated vs. Real Webcam Validation

> [!IMPORTANT]
> **Simulated Validation vs. Real-World Webcam Performance:**  
> All metrics and improvements reported above were derived strictly using a **deterministic synthetic test harness** (`DeterministicClock`) and identical recorded mathematical trajectories.  
> **No real-world webcam trials were executed in this phase.**  
> While the synthetic test harness guarantees mathematical correctness, zero regressions, and resilience to simulated Gaussian noise and saccades, real-world webcam performance remains subject to biological variations (eye dryness, pupil dilation, head shifts) and illumination changes.

---

## 8. Remaining Limitations

1. **Spatial Offset Sensitivity:**  
   If the upstream gaze estimation pipeline suffers from a static calibration DC offset (e.g., +150 px vertical shift), the gaze point may land outside the expanded exit margin of small buttons. Hysteresis stabilizes the boundary but does not correct upstream calibration bias.
2. **Dense Multi-Target Grid Spacing:**  
   If interactive buttons are spaced with less than 2% screen margin, the exit hysteresis margin (4%) will overlap into adjacent targets. In such dense layouts, `target_switch_confirmation_ms` is critical, but button spacing $\ge 4\%$ is recommended for AAC UI design.
3. **Head Posture Drift Over Long Sessions:**  
   Postural shifts in wheelchair users over prolonged sessions require an upstream re-centering mechanism (such as the recommended 1-point trim key) to maintain gaze calibration within AAC card boundaries.

---

## 9. Recommended Next Experiments

1. **Live Webcam Dwell Verification (`manual_demo.py`):**
   Execute interactive validation using:
   ```bash
   cd cv-service
   source .venv/bin/activate
   python -m dwell.manual_demo --camera
   ```
   Record dwell selection comfort, false trigger rate, and perceived latency across 20 intentional card activations.

2. **Empirical Hysteresis Tuning with Actual Users:**
   Test `exit_margin` values in the range $[0.03, 0.06]$ on an 8-button Arabic AAC grid to identify the optimal comfort balance between boundary stability and intentional disengagement ease.

3. **Combined Upstream Re-Center Hotkey Integration:**
   Integrate a one-touch re-center trim hotkey into `src/pipeline.py` to eliminate session-level DC drift prior to dwell selection.
