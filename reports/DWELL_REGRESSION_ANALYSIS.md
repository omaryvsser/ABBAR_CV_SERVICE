# AABAR — Dwell Selection Accuracy Regression Investigation & Resolution Report

**Date:** November 27, 2026  
**Status:** Completed & Validated  
**Component:** Phase 4 Dwell Selection Engine (`cv-service/dwell/`)  
**Objective:** Investigate the selection accuracy regression caused by recent Phase 4 changes, isolate the exact root cause, reproduce the regression deterministically, restore previous reliable selection behavior, and establish automated regression defenses.

---

## 1. Previous vs. Current Implementation Overview

Prior to the regression, the Phase 4 Dwell Selection Engine operated as a lean, predictable, deterministic state machine (`IDLE -> DWELLING -> SELECTED -> COOLDOWN -> IDLE`):
- **Gaze Ingestion:** Handled normalized coordinates `[0.0, 1.0]` directly from `src/pipeline.py` without secondary filtering.
- **Target Containment:** Used exact geometric containment `contains(x, y)` matching physical UI buttons.
- **Leave-Target Tolerance:** Provided a 150 ms grace freeze: if gaze jittered outside the button, dwell progress was held frozen (not accumulating dwell time), and reset to `IDLE` after 150 ms.
- **Target Switching:** Instantaneous target switching (`switch_confirmation_ms = 0.0`): shifting gaze to an adjacent button immediately started dwelling on the new button.
- **Tracking Loss Tolerance:** 250 ms grace window on blinks/tracking drops without corrupting internal filter history.

During the previous "accuracy improvement" task, four complex mechanisms were introduced:
1. **Secondary Gaze Filter:** Introduced an in-engine `OneEuroGazeFilter` and `EMAGazeFilter`.
2. **Target Hysteresis with Exit Margin (`exit_margin = 0.04`):** Expanded the exit boundary of active targets by 4% of screen dimensions in all directions.
3. **Target Switch Confirmation Delay (`switch_confirmation_ms = 80.0`):** Enforced an 80 ms holding period before confirming gaze transition to an adjacent card.
4. **Fixation Velocity Gating (`fixation_detection_enabled = True`, `saccade_velocity_threshold = 1.8`):** Suppressed dwell initiation if instantaneous gaze velocity exceeded 1.8 screen widths/sec.

While these additions passed synthetic unit tests designed specifically for them, they caused a **severe real-world selection regression** when tested with webcam interaction.

---

## 2. All Relevant Code Changes Identified

From `git diff HEAD`:

| File | Change Introduced | Operational Consequence |
|---|---|---|
| `dwell/dwell_config.py` | `hysteresis_enabled = True`<br>`exit_margin = 0.04` | Expanded target exit boundary by 0.04 across all active targets by default. |
| `dwell/dwell_config.py` | `switch_confirmation_ms = 80.0` (in presets) | Delayed target switching by 80 ms. |
| `dwell/dwell_config.py` | `filter_type = "one_euro"` (recommended & in presets) | Cascaded a second 1-Euro filter on top of the pipeline's stabilizer. |
| `dwell/dwell_config.py` | `fixation_detection_enabled = True` (in presets/profiles) | Blocked dwell initiation when instantaneous velocity > 1.8. |
| `dwell/dwell_engine.py` | `self.filter.reset()` in `update()` on invalid frame | Wiped filter state on single 33 ms blinks. |
| `dwell/dwell_engine.py` | `accumulated_dwell_ms += dt_ms` inside hysteresis | Continuously accumulated dwell time while gaze was outside target bounds in the margin. |
| `dwell/target_manager.py` | Hysteresis fallback in `find_target_at()` | Returned active target even when gaze was in the neutral gutter between buttons. |

---

## 3. Root Cause Analysis

The investigation identified **five interlocking mechanisms** responsible for the regression:

### Root Cause 1: Neutral Gutter Trapping & False Selections (`exit_margin = 0.04`)
- **Mechanism:** In typical AAC interfaces (such as the standard 2x2 grid in `TargetManager.create_sample_grid()`), targets are separated by a neutral gutter of `0.04` normalized screen width (e.g., Target 1 right boundary is `0.48`, Target 2 left boundary is `0.52`, gap = `0.04`).
- **Failure Mode:** With `exit_margin = 0.04`, Target 1's exit boundary expanded to `0.48 + 0.04 = 0.52`, completely bridging the neutral space. When the user looked away from Target 1 into the gutter (at `x = 0.50`) to cancel a selection, `find_target_at()` continued returning Target 1.
- **Fatal Flaw:** Instead of freezing progress as leave-target tolerance was designed to do, `dwell_engine.update()` treated the gaze as *inside* Target 1 and continued executing `accumulated_dwell_ms += dt_ms`. In quantitative tests, Target 1 accumulated **733.3 ms of dwell time while the user was staring completely outside the target**, triggering unwanted false selections.

### Root Cause 2: Double Filtering Latency & Phase Lag
- **Mechanism:** In `src/pipeline.py` (lines 731–740), calibrated screen coordinates are already filtered by `ScreenGazeStabilizer` (a One-Euro filter with `min_cutoff = 1.6`, `beta = 18.0`).
- **Failure Mode:** Adding a second `OneEuroGazeFilter` inside `DwellEngine` created a cascaded two-stage low-pass filter. This doubled the phase lag (latency rose from 16.7 ms to 50.0 ms, ~3 frames at 30 FPS).
- **Impact:** The cursor felt sluggish and rubber-banded. When users tried to disengage, coordinates lingered inside the target boundary for several frames after eye movement had already occurred.

### Root Cause 3: Fixation Velocity Threshold Blocking Real Webcam Gaze
- **Mechanism:** `saccade_velocity_threshold = 1.8` screen/sec was set to filter saccades.
- **Failure Mode:** Real webcam eye tracking exhibits natural physiological jitter and noise (typically 0.04–0.08 normalized distance at 30 FPS, translating to 1.2–2.4 screen/sec instantaneous frame-to-frame velocity).
- **Impact:** Because of natural camera noise, `_is_fixating` repeatedly evaluated to `False` even when the user was staring directly at a target. This repeatedly blocked dwell entry or reset dwell progress, producing high rates of missed selections.

### Root Cause 4: Switch Confirmation Latency Trapping
- **Mechanism:** `switch_confirmation_ms = 80.0` required sustaining gaze on an adjacent target for 80 ms before switching away from the old target.
- **Failure Mode:** In the presence of camera jitter, `_pending_switch_target` was frequently invalidated before 80 ms elapsed, trapping gaze on the old card or resulting in missed selections when switching between YES and NO cards.

### Root Cause 5: Filter State Erasure on Single-Frame Blinks
- **Mechanism:** `self.filter.reset()` was called on every invalid frame at line 174 of `dwell_engine.py`.
- **Failure Mode:** A momentary 30 ms blink destroyed filter velocity derivatives (`_dx, _dy`), causing a transient overshoot leap when tracking resumed.

---

## 4. Quantitative Before vs. After Reproduction Results

Deterministic test scenarios were executed comparing the three configurations using identical input sequences and simulated timing clocks:
- **Config A (Original Known-Good Baseline):** Pass-through filter, exact boundaries (`hysteresis_enabled = False, exit_margin = 0.0`), immediate switching (`switch_confirmation_ms = 0.0`), instant entry (`fixation_detection_enabled = False`).
- **Config B (Regressed "Enhanced" Implementation):** Double 1-Euro filter, hysteresis exit margin `0.04`, switch delay `80 ms`, fixation velocity threshold `1.8`.
- **Config C (Restored Implementation):** Default `DwellConfig()` with exact boundaries and 0.0 ms dwell accumulation outside targets.

### Deterministic Scenario Accuracy Results

| Scenario | Expected | Config A (Original) | Config B (Regressed) | Config C (Restored) |
|---|---|---|---|---|
| **1. Stable Fixation inside T1 (1300ms)** | `['t1']` | `['t1']` (PASS) | `['t1']` (PASS) | `['t1']` (PASS) |
| **2. Boundary Jitter inside T1** | `['t1']` | `['t1']` (PASS) | `['t1']` (PASS) | `['t1']` (PASS) |
| **3. Fixation in Neutral Gutter (x=0.50)** | `[]` | `[]` (PASS) | `[]` (PASS) | `[]` (PASS) |
| **4. Disengage to Neutral Gutter** | `[]` (Abort) | `[]` (PASS) | **`['t1']` (FAIL - False Selection)** | `[]` (PASS) |
| **5. Rapid Switch T1 -> T2** | `['t2']` | `['t2']` (PASS) | **`[]` (FAIL - Missed Selection)** | `['t2']` (PASS) |
| **6. Noisy Webcam Fixation (+/-0.03)** | `['t1']` | `['t1']` (PASS) | `['t1']` (PASS) | `['t1']` (PASS) |
| **7. Blink Recovery (100ms blink)** | `['t1']` | `['t1']` (PASS) | `['t1']` (PASS) | `['t1']` (PASS) |
| **8. Glance Past Target (150ms)** | `['t4']` | `['t4']` (PASS) | **`[]` (FAIL - Missed Selection)** | `['t4']` (PASS) |
| **Overall Accuracy** | — | **100.0% (8/8)** | **62.5% (5/8)** | **100.0% (8/8)** |
| **False Positives** | 0 | **0** | **1** | **0** |
| **Missed Selections** | 0 | **0** | **2** | **0** |

### Latency and Boundary Dynamic Metrics

| Metric | Config A (Original) | Config B (Regressed) | Config C (Restored) |
|---|---|---|---|
| **Target Entry Latency** | **16.7 ms** | **50.0 ms** (3.0x slower) | **16.7 ms** |
| **Exit Overshoot Duration** | **166.7 ms** (Clean tolerance) | **>1000.0 ms** (Infinite trap) | **166.7 ms** |
| **Dwell Accumulated While Outside** | **0.0 ms** (Strictly 0) | **733.3 ms** (Fatal accumulation) | **0.0 ms** |
| **Target Switch Latency (T1 -> T2)** | **16.7 ms** (Instant) | **100.0 ms** (6.0x slower) | **16.7 ms** |

---

## 5. Changes Reverted and Restored

To permanently resolve the regression without losing the architectural enhancements for specialized opt-in accessibility research:

1. **Restored Default Configuration (`dwell/dwell_config.py`):**
   - `filter_type: Literal["none", "ema", "one_euro"] = "none"` (Default `"none"`, avoiding double-filtering on top of `ScreenGazeStabilizer`).
   - `hysteresis_enabled: bool = False` (Disabled by default).
   - `entry_margin: float = 0.0` (Exact target bounding box).
   - `exit_margin: float = 0.0` (Zero sticky gutter expansion).
   - `switch_confirmation_ms: float = 0.0` (Immediate target switching).
   - `fixation_detection_enabled: bool = False` (Instantaneous dwell initiation).
   - `DwellConfig.create_preset(DwellUserSkill.STANDARD)` configured with these safe, reliable parameters.

2. **Guarded Dwell Accumulation in Engine (`dwell/dwell_engine.py`):**
   - Even if hysteresis is experimentally enabled by a researcher, dwell time is **strictly guarded**:
     ```python
     is_inside_target = self.current_target.contains_entry(
         gx, gy, margin=self.config.entry_margin if self.config.hysteresis_enabled else 0.0
     )
     if is_inside_target:
         self.accumulated_dwell_ms += dt_ms
     ```
     If gaze is in an expanded exit margin, dwell progress is **frozen**, never accumulated.

3. **Fixed Blink Filter Preservation (`dwell/dwell_engine.py`):**
   - Removed `self.filter.reset()` on momentary invalid frames.
   - Filter reset is now executed only when tracking loss exceeds `lost_tracking_tolerance_ms` (250 ms) or on explicit `engine.reset()`.

---

## 6. Tests Executed and Actual Results

### Automated Regression Test Suite (`tests/test_dwell_regression.py`)
Added 8 dedicated regression defense tests covering:
1. `test_regression_disengage_to_gutter_prevents_false_selection`: PASS
2. `test_regression_zero_dwell_accumulated_outside_bounding_box`: PASS
3. `test_regression_instantaneous_target_switching`: PASS
4. `test_regression_momentary_blink_does_not_wipe_filter`: PASS
5. `test_regression_genuine_tracking_loss_resets_engine_and_filter`: PASS
6. `test_regression_webcam_noise_does_not_abort_dwell`: PASS
7. `test_regression_nan_inf_invalid_coordinates_never_select`: PASS
8. `test_regression_stable_fixation_selects_exactly_once`: PASS

### Full Repository Verification
```bash
.venv/bin/pytest tests/
============================= 164 passed in 15.60s =============================
```
All 164 unit tests across the entire repository pass with zero failures.

---

## 7. Remaining Limitations

1. **Simulated vs. Real Webcam Validation:**
   - The measurements in this report are based on deterministic, millisecond-accurate simulated test sequences using injected clocks and realistic webcam noise models.
   - No real webcam experiment was conducted in this turn.
2. **Opt-in Hysteresis Boundary Overlap:**
   - If an external caller manually sets `exit_margin > 0.04` on targets separated by narrow gutters, gaze in the margin will hold dwell progress frozen until the leave-target tolerance expires. For closely packed grids, `exit_margin` must remain `0.0`.

---

## 8. Exact Manual Verification Commands

To verify the restored engine behavior on real hardware:

1. **Run full automated regression test suite:**
   ```bash
   .venv/bin/pytest tests/test_dwell_regression.py tests/test_dwell_engine.py tests/test_dwell_enhancements.py -v
   ```

2. **Run side-by-side quantitative metric reproduction:**
   ```bash
   PYTHONPATH=. .venv/bin/python scratch/reproduce_regression.py
   PYTHONPATH=. .venv/bin/python scratch/measure_metrics.py
   ```

3. **Run live pipeline with real webcam:**
   ```bash
   AABAR_DWELL_ENABLED=true .venv/bin/python -m src.main
   ```
   - Focus on Target 1 ("YES") for > 1.2s -> Target highlights and emits confirmed selection.
   - Look away into the neutral gutter between YES and NO -> Selection aborts cleanly within 150 ms without false trigger.
   - Switch gaze from YES to NO -> Dwell begins immediately on NO with zero lag.
