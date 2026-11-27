"""
Regression Test Suite for Phase 4 Dwell Selection Engine.

Validates that recent accuracy regressions (sticky boundaries, false selections in gutters,
filter phase lag, switch delays, blink filter resets) are fully resolved and that the
original reliable selection behavior is strictly preserved.
"""

import math
import pytest
from typing import List

from dwell.dwell_config import DwellConfig, DwellUserSkill
from dwell.dwell_engine import DwellEngine
from dwell.dwell_state import DwellState, SelectionEvent
from dwell.target_manager import DwellTarget, TargetManager


class MockClock:
    """Controllable clock for deterministic millisecond-accurate timing tests."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.time = start_time

    def __call__(self) -> float:
        return self.time

    def advance_ms(self, milliseconds: float) -> None:
        self.time += milliseconds / 1000.0


def create_standard_test_grid() -> TargetManager:
    """Create standard 2x2 grid with neutral 0.04-wide gutter between targets."""
    tm = TargetManager()
    # Target 1: [0.08, 0.08, 0.40, 0.40] -> right=0.48, bottom=0.48
    # Target 2: [0.52, 0.08, 0.40, 0.40] -> left=0.52, bottom=0.48
    # Gutter between T1 and T2 is [0.48, 0.52] (width 0.04)
    tm.add_target(DwellTarget("t1", "YES", 0.08, 0.08, 0.40, 0.40, priority=1))
    tm.add_target(DwellTarget("t2", "NO", 0.52, 0.08, 0.40, 0.40, priority=1))
    tm.add_target(DwellTarget("t3", "HELP", 0.08, 0.52, 0.40, 0.40, priority=1))
    tm.add_target(DwellTarget("t4", "REST", 0.52, 0.52, 0.40, 0.40, priority=1))
    return tm


# --- Regression Test 1: Disengage to Gutter Prevents False Selection ---
def test_regression_disengage_to_gutter_prevents_false_selection() -> None:
    """
    REGRESSION: Previously, exit_margin=0.04 caused gaze in the neutral gutter (x=0.50)
    to continue accumulating dwell time on Target 1, triggering a false selection.
    VERIFY: Looking away to the neutral gutter stops dwell accumulation and cleanly
    resets to IDLE with 0 selections.
    """
    clk = MockClock(1000.0)
    cfg = DwellConfig()  # Default restored configuration
    tm = create_standard_test_grid()
    eng = DwellEngine(config=cfg, target_manager=tm, clock=clk)

    selections: List[SelectionEvent] = []
    eng.on_selection(lambda ev: selections.append(ev))

    # 1. User fixates inside Target 1 for 600ms (accumulating 600ms of 1200ms dwell)
    dt_ms = 33.33  # 30 FPS
    for _ in range(18):
        clk.advance_ms(dt_ms)
        snap = eng.update(0.28, 0.28)
        assert snap.state == DwellState.DWELLING
        assert snap.current_target_id == "t1"

    assert eng.accumulated_dwell_ms >= 500.0

    # 2. User deliberately shifts gaze to neutral gutter (x=0.50, y=0.28) for 800ms
    for _ in range(24):
        clk.advance_ms(dt_ms)
        snap = eng.update(0.50, 0.28)

    # Must cleanly reset to IDLE after 150ms tolerance with ZERO selections
    assert snap.state == DwellState.IDLE
    assert snap.current_target_id is None
    assert len(selections) == 0, f"False selection occurred! Selected targets: {[s.target_id for s in selections]}"


# --- Regression Test 2: Zero Dwell Accumulated Outside Physical Target ---
def test_regression_zero_dwell_accumulated_outside_bounding_box() -> None:
    """
    REGRESSION: Even with hysteresis enabled, dwell time must NEVER accumulate
    when gaze is outside the physical button entry boundary.
    VERIFY: Accumulated dwell time remains strictly 0.0 ms while in margin/outside.
    """
    clk = MockClock(1000.0)
    # Even if hysteresis is experimentally enabled:
    cfg = DwellConfig(hysteresis_enabled=True, exit_margin=0.04)
    tm = create_standard_test_grid()
    eng = DwellEngine(config=cfg, target_manager=tm, clock=clk)

    dt_ms = 16.67  # 60 FPS
    # Dwell inside T1 for 400ms
    for _ in range(24):
        clk.advance_ms(dt_ms)
        eng.update(0.28, 0.28)
    initial_dwell = eng.accumulated_dwell_ms
    assert initial_dwell >= 350.0

    # Step into margin (x=0.50, in exit margin of T1)
    for _ in range(30):
        clk.advance_ms(dt_ms)
        eng.update(0.50, 0.28)
        # Dwell progress must be FROZEN: never accumulate while outside!
        assert eng.accumulated_dwell_ms == pytest.approx(initial_dwell)


# --- Regression Test 3: Instantaneous Target Switching Without Lag ---
def test_regression_instantaneous_target_switching() -> None:
    """
    REGRESSION: switch_confirmation_ms previously introduced multi-frame lag (80ms),
    holding onto the old target and causing missed selections.
    VERIFY: Target switching is instantaneous by default on the very first frame.
    """
    clk = MockClock(1000.0)
    cfg = DwellConfig()  # Default switch_confirmation_ms=0.0
    tm = create_standard_test_grid()
    eng = DwellEngine(config=cfg, target_manager=tm, clock=clk)

    # 1. Dwell in T1 for 400ms
    clk.advance_ms(33.33)
    eng.update(0.28, 0.28)
    clk.advance_ms(400.0)
    snap1 = eng.update(0.28, 0.28)
    assert snap1.current_target_id == "t1"
    assert snap1.dwell_time_ms >= 399.0

    # 2. Shift directly to T2 (x=0.72)
    clk.advance_ms(33.33)
    snap2 = eng.update(0.72, 0.28)

    # Must immediately switch to T2 on this exact frame!
    assert snap2.current_target_id == "t2"
    assert snap2.state == DwellState.DWELLING
    assert snap2.dwell_time_ms == pytest.approx(0.0)


# --- Regression Test 4: Momentary Blink Does Not Corrupt Filter State ---
def test_regression_momentary_blink_does_not_wipe_filter() -> None:
    """
    REGRESSION: Single invalid frame called self.filter.reset(), wiping filter history
    and causing an overshoot spike on tracking reacquisition.
    VERIFY: Brief blinks (< 250ms tolerance) pause progress without wiping filter state.
    """
    clk = MockClock(1000.0)
    cfg = DwellConfig(filter_type="one_euro")
    tm = create_standard_test_grid()
    eng = DwellEngine(config=cfg, target_manager=tm, clock=clk)

    # 1. Dwell in T1 for 500ms
    for _ in range(15):
        clk.advance_ms(33.33)
        eng.update(0.28, 0.28)

    # Filter is initialized and has historical state
    assert eng.filter._x is not None

    # 2. Momentary blink (1 invalid frame = 33.33ms < 250ms tolerance)
    clk.advance_ms(33.33)
    snap_blink = eng.update(None, None)
    assert snap_blink.is_tracking_valid is False
    assert snap_blink.state == DwellState.DWELLING  # Paused, not aborted

    # Filter state MUST NOT be wiped on a single-frame blink!
    assert eng.filter._x is not None

    # 3. Gaze returns on next frame: smoothly resumes without reset
    clk.advance_ms(33.33)
    snap_resume = eng.update(0.28, 0.28)
    assert snap_resume.is_tracking_valid is True
    assert snap_resume.current_target_id == "t1"
    assert snap_resume.dwell_time_ms >= 490.0


# --- Regression Test 5: Genuine Tracking Loss Cleanses State ---
def test_regression_genuine_tracking_loss_resets_engine_and_filter() -> None:
    """
    VERIFY: When tracking is lost for > 250ms, dwell aborts to LOST_TRACKING
    and the filter is properly reset.
    """
    clk = MockClock(1000.0)
    cfg = DwellConfig(filter_type="one_euro", lost_tracking_tolerance_ms=250.0)
    tm = create_standard_test_grid()
    eng = DwellEngine(config=cfg, target_manager=tm, clock=clk)

    # Dwell in T1 for 300ms
    clk.advance_ms(300.0)
    eng.update(0.28, 0.28)
    assert eng.filter._x is not None

    # Lost tracking for 300ms (> 250ms tolerance)
    clk.advance_ms(300.0)
    snap_lost = eng.update(None, None)
    assert snap_lost.state == DwellState.LOST_TRACKING
    assert snap_lost.current_target_id is None
    assert eng.accumulated_dwell_ms == 0.0
    assert eng.filter._x is None  # Filter reset on genuine tracking loss


# --- Regression Test 6: Webcam Gaze Jitter Does Not Block Selection ---
def test_regression_webcam_noise_does_not_abort_dwell() -> None:
    """
    REGRESSION: fixation_detection_enabled flagged webcam jitter as saccades,
    repeatedly blocking dwell start.
    VERIFY: Default configuration handles realistic webcam noise (+/- 0.03 normalized)
    and successfully triggers selection after 1200ms.
    """
    clk = MockClock(1000.0)
    cfg = DwellConfig()  # Default restored configuration
    tm = create_standard_test_grid()
    eng = DwellEngine(config=cfg, target_manager=tm, clock=clk)

    selections: List[SelectionEvent] = []
    eng.on_selection(lambda ev: selections.append(ev))

    # Rapid entry followed by noisy fixation inside T1 for 1300ms at 30 FPS
    dt_ms = 33.33
    total_frames = int(1.3 * 30)
    for i in range(total_frames):
        clk.advance_ms(dt_ms)
        # Jitter of +/- 0.03 (approx 57px on 1920x1080)
        noise_x = 0.03 * math.sin(i * 1.7)
        noise_y = 0.03 * math.cos(i * 1.3)
        eng.update(0.28 + noise_x, 0.28 + noise_y)

    assert len(selections) == 1
    assert selections[0].target_id == "t1"


# --- Regression Test 7: Invalid NaN / Inf Coordinates Never Trigger Selection ---
def test_regression_nan_inf_invalid_coordinates_never_select() -> None:
    """
    VERIFY: NaN, Inf, and out-of-bounds coordinates never cause crashes,
    never trigger selections, and immediately transition to LOST_TRACKING.
    """
    clk = MockClock(1000.0)
    cfg = DwellConfig()
    tm = create_standard_test_grid()
    eng = DwellEngine(config=cfg, target_manager=tm, clock=clk)

    selections: List[SelectionEvent] = []
    eng.on_selection(lambda ev: selections.append(ev))

    # Send NaN
    clk.advance_ms(33.33)
    snap_nan = eng.update(float("nan"), 0.5)
    assert snap_nan.is_tracking_valid is False
    assert snap_nan.state == DwellState.LOST_TRACKING

    # Send Inf
    clk.advance_ms(33.33)
    snap_inf = eng.update(0.5, float("inf"))
    assert snap_inf.is_tracking_valid is False
    assert snap_inf.state == DwellState.LOST_TRACKING

    # Send extreme out-of-bounds
    clk.advance_ms(33.33)
    snap_oob = eng.update(5.0, -10.0)
    assert snap_oob.is_tracking_valid is False
    assert snap_oob.state == DwellState.LOST_TRACKING

    assert len(selections) == 0


# --- Regression Test 8: Stable Fixation Selects Exactly Once ---
def test_regression_stable_fixation_selects_exactly_once() -> None:
    """
    VERIFY: Sustained fixation on a target selects exactly once,
    transitions through SELECTED into COOLDOWN, and prevents repeated selection
    without looking away first.
    """
    clk = MockClock(1000.0)
    cfg = DwellConfig(dwell_duration_ms=1200.0, cooldown_duration_ms=800.0)
    tm = create_standard_test_grid()
    eng = DwellEngine(config=cfg, target_manager=tm, clock=clk)

    selections: List[SelectionEvent] = []
    eng.on_selection(lambda ev: selections.append(ev))

    # Stare at Target 1 for 2500ms continuously
    dt_ms = 33.33
    states_observed = set()
    for _ in range(int(2.5 * 30)):
        clk.advance_ms(dt_ms)
        snap = eng.update(0.28, 0.28)
        states_observed.add(snap.state)

    # Must have transitioned through DWELLING, SELECTED, COOLDOWN, and settled into IDLE
    assert DwellState.DWELLING in states_observed
    assert DwellState.SELECTED in states_observed
    assert DwellState.COOLDOWN in states_observed
    assert DwellState.IDLE in states_observed

    # Must select EXACTLY ONCE
    assert len(selections) == 1
    assert selections[0].target_id == "t1"
    assert selections[0].dwell_duration_ms >= 1200.0
