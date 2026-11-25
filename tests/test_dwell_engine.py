"""
Comprehensive automated tests for Phase 4 DwellEngine.

Covers all 20 required test scenarios using a controllable deterministic clock
to guarantee instant, deterministic test execution with zero sleep delays.
"""

import math
import pytest
from dwell.dwell_config import DwellConfig
from dwell.dwell_engine import DwellEngine
from dwell.dwell_state import DwellState
from dwell.target_manager import DwellTarget, TargetManager


class MockClock:
    """Controllable clock for deterministic time advancing."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.current_time = start_time

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds

    def advance_ms(self, milliseconds: float) -> None:
        self.current_time += milliseconds / 1000.0


@pytest.fixture
def test_setup():
    """Create a DwellEngine with standard 1200ms dwell and 800ms cooldown."""
    clock = MockClock(1000.0)
    config = DwellConfig(
        dwell_duration_ms=1200.0,
        cooldown_duration_ms=800.0,
        selection_display_duration_ms=200.0,
        leave_target_tolerance_ms=150.0,
        lost_tracking_tolerance_ms=200.0,
        allow_repeat_without_leaving=False,
    )
    manager = TargetManager()
    t1 = DwellTarget("t1", "YES", 0.1, 0.1, 0.3, 0.3, priority=1)
    t2 = DwellTarget("t2", "NO", 0.5, 0.1, 0.3, 0.3, priority=1)
    manager.add_target(t1)
    manager.add_target(t2)
    engine = DwellEngine(config=config, target_manager=manager, clock=clock)
    return engine, manager, clock, t1, t2


# --- Scenario 1 & 2: Gaze enters target and dwell starts ---
def test_scenario_01_and_02_gaze_enters_target_and_dwell_starts(test_setup) -> None:
    engine, _, clock, t1, _ = test_setup
    # Frame 1: gaze in empty space
    snap = engine.update(0.05, 0.05)
    assert snap.state == DwellState.IDLE
    assert snap.current_target_id is None

    # Frame 2: gaze enters t1 (center 0.25, 0.25)
    clock.advance_ms(50)
    snap = engine.update(0.25, 0.25)
    assert snap.state == DwellState.DWELLING
    assert snap.current_target_id == "t1"
    assert snap.current_target_label == "YES"
    assert snap.dwell_progress == 0.0


# --- Scenario 3: Progress increases over time ---
def test_scenario_03_progress_increases_over_time(test_setup) -> None:
    engine, _, clock, _, _ = test_setup
    engine.update(0.25, 0.25)

    # 300 ms later -> 25% progress (300 / 1200)
    clock.advance_ms(300)
    snap = engine.update(0.25, 0.25)
    assert snap.state == DwellState.DWELLING
    assert snap.dwell_progress == pytest.approx(0.25, abs=0.01)
    assert snap.dwell_time_ms == pytest.approx(300.0, abs=1.0)
    assert snap.remaining_time_ms == pytest.approx(900.0, abs=1.0)

    # Another 300 ms later -> 50% progress (600 / 1200)
    clock.advance_ms(300)
    snap = engine.update(0.25, 0.25)
    assert snap.dwell_progress == pytest.approx(0.50, abs=0.01)


# --- Scenario 4: Selection occurs after 1200 ms ---
def test_scenario_04_selection_occurs_after_1200ms(test_setup) -> None:
    engine, _, clock, _, _ = test_setup
    engine.update(0.25, 0.25)

    # Advance 1200 ms
    clock.advance_ms(1200)
    snap = engine.update(0.25, 0.25)
    assert snap.state == DwellState.SELECTED
    assert snap.dwell_progress == 1.0
    assert snap.remaining_time_ms == 0.0
    assert snap.selection_event is not None
    assert snap.selection_event.target_id == "t1"
    assert snap.selection_event.target_label == "YES"
    assert snap.selection_event.selection_state == "CONFIRMED"


# --- Scenario 5: Selection occurs only once (one-shot event) ---
def test_scenario_05_selection_occurs_only_once(test_setup) -> None:
    engine, _, clock, _, _ = test_setup
    engine.update(0.25, 0.25)
    clock.advance_ms(1200)
    snap1 = engine.update(0.25, 0.25)
    assert snap1.selection_event is not None  # Emitted here

    # Next frame immediately after: event must be None
    clock.advance_ms(30)
    snap2 = engine.update(0.25, 0.25)
    assert snap2.selection_event is None
    assert engine.total_selections_count == 1


# --- Scenario 6: Cooldown prevents repeated selection ---
def test_scenario_06_cooldown_prevents_repeated_selection(test_setup) -> None:
    engine, _, clock, _, _ = test_setup
    # Trigger selection
    engine.update(0.25, 0.25)
    clock.advance_ms(1200)
    engine.update(0.25, 0.25)

    # After selection display duration (200ms), transitions to COOLDOWN
    clock.advance_ms(210)
    snap = engine.update(0.25, 0.25)
    assert snap.state == DwellState.COOLDOWN
    assert snap.cooldown_progress > 0.0

    # User continues staring at same target through entire 800ms cooldown
    clock.advance_ms(400)
    snap = engine.update(0.25, 0.25)
    assert snap.state == DwellState.COOLDOWN

    clock.advance_ms(450)  # Total > 800ms cooldown expired
    snap = engine.update(0.25, 0.25)
    # Because allow_repeat_without_leaving=False, user must look away before re-dwelling
    assert snap.state == DwellState.IDLE
    assert snap.selection_event is None
    assert engine.total_selections_count == 1


# --- Scenario 7: Gaze leaves target before completion ---
def test_scenario_07_gaze_leaves_target_before_completion(test_setup) -> None:
    engine, _, clock, _, _ = test_setup
    engine.update(0.25, 0.25)
    clock.advance_ms(500)  # 500ms accumulated
    snap = engine.update(0.25, 0.25)
    assert snap.dwell_time_ms == pytest.approx(500.0)

    # Gaze leaves target to empty space beyond saccade grace window (150ms)
    clock.advance_ms(200)
    snap = engine.update(0.05, 0.05)
    assert snap.state == DwellState.IDLE
    assert snap.current_target_id is None
    assert snap.dwell_progress == 0.0


# --- Scenario 8: Gaze moves from one target to another ---
def test_scenario_08_gaze_moves_between_targets(test_setup) -> None:
    engine, _, clock, _, _ = test_setup
    # Dwelling on t1 (YES)
    engine.update(0.25, 0.25)
    clock.advance_ms(600)
    snap1 = engine.update(0.25, 0.25)
    assert snap1.current_target_id == "t1"
    assert snap1.dwell_progress == pytest.approx(0.50, abs=0.01)

    # Gaze jumps directly to t2 (NO, center 0.65, 0.25)
    clock.advance_ms(50)
    snap2 = engine.update(0.65, 0.25)
    assert snap2.state == DwellState.DWELLING
    assert snap2.current_target_id == "t2"
    assert snap2.current_target_label == "NO"
    # Dwell progress on t2 starts fresh
    assert snap2.dwell_progress == pytest.approx(0.0, abs=0.01)


# --- Scenario 9 & 10: Invalid gaze coordinates & NaN/Inf ---
def test_scenario_09_and_10_invalid_and_nan_inf_gaze(test_setup) -> None:
    engine, _, clock, _, _ = test_setup
    # None coordinates
    snap = engine.update(None, None)
    assert snap.state == DwellState.LOST_TRACKING
    assert snap.is_tracking_valid is False

    # NaN coordinates
    clock.advance_ms(50)
    snap = engine.update(float("nan"), 0.5)
    assert snap.state == DwellState.LOST_TRACKING
    assert snap.is_tracking_valid is False

    # Infinite coordinates
    clock.advance_ms(50)
    snap = engine.update(0.5, float("inf"))
    assert snap.state == DwellState.LOST_TRACKING
    assert snap.is_tracking_valid is False

    # Wild out-of-bounds coordinates (e.g. 50.0 in normalized space)
    clock.advance_ms(50)
    snap = engine.update(50.0, 50.0)
    assert snap.state == DwellState.LOST_TRACKING


# --- Scenario 11 & 12: Tracking loss during dwell and recovery ---
def test_scenario_11_and_12_tracking_loss_and_recovery(test_setup) -> None:
    engine, _, clock, _, _ = test_setup
    # Dwell on t1 for 600ms
    engine.update(0.25, 0.25)
    clock.advance_ms(600)
    engine.update(0.25, 0.25)

    # Brief tracking dropout of 100ms (within tolerance 200ms)
    clock.advance_ms(100)
    snap_lost = engine.update(None, None)
    assert snap_lost.is_tracking_valid is False
    # Dwell progress is frozen/paused, not discarded yet
    assert snap_lost.dwell_time_ms == pytest.approx(600.0)

    # Tracking recovers at 150ms total
    clock.advance_ms(50)
    snap_recovered = engine.update(0.25, 0.25)
    assert snap_recovered.state == DwellState.DWELLING
    assert snap_recovered.is_tracking_valid is True
    assert snap_recovered.current_target_id == "t1"

    # Now simulate tracking lost longer than tolerance (250ms > 200ms)
    clock.advance_ms(250)
    snap_aborted = engine.update(None, None)
    assert snap_aborted.state == DwellState.LOST_TRACKING
    assert snap_aborted.dwell_time_ms == 0.0


# --- Scenario 13: Disabled targets ---
def test_scenario_13_disabled_targets(test_setup) -> None:
    engine, manager, clock, t1, _ = test_setup
    manager.set_target_enabled("t1", False)

    # Gaze is inside t1 bounding box
    snap = engine.update(0.25, 0.25)
    assert snap.state == DwellState.IDLE
    assert snap.current_target_id is None

    # Re-enable t1
    manager.set_target_enabled("t1", True)
    clock.advance_ms(50)
    snap = engine.update(0.25, 0.25)
    assert snap.state == DwellState.DWELLING
    assert snap.current_target_id == "t1"


# --- Scenario 14: Overlapping targets resolution ---
def test_scenario_14_overlapping_targets(test_setup) -> None:
    engine, manager, clock, _, _ = test_setup
    manager.clear()
    t_parent = DwellTarget("parent", "PARENT", 0.1, 0.1, 0.8, 0.8, priority=0)
    t_child = DwellTarget("child", "CHILD", 0.3, 0.3, 0.2, 0.2, priority=5)
    manager.add_target(t_parent)
    manager.add_target(t_child)

    # Gaze inside child (and parent)
    snap = engine.update(0.35, 0.35)
    assert snap.current_target_id == "child"  # Higher priority wins


# --- Scenario 15: Invalid target dimensions rejection ---
def test_scenario_15_invalid_target_dimensions() -> None:
    with pytest.raises(ValueError, match="width must be > 0"):
        DwellTarget("err", "ERR", 0.1, 0.1, -0.5, 0.5)


# --- Scenario 16: Dwell duration configuration ---
def test_scenario_16_dwell_duration_configuration() -> None:
    clock = MockClock(1000.0)
    config = DwellConfig(dwell_duration_ms=600.0)  # Fast 600ms dwell
    manager = TargetManager([DwellTarget("t_fast", "FAST", 0.1, 0.1, 0.3, 0.3)])
    engine = DwellEngine(config=config, target_manager=manager, clock=clock)

    engine.update(0.2, 0.2)
    clock.advance_ms(600)
    snap = engine.update(0.2, 0.2)
    assert snap.state == DwellState.SELECTED
    assert snap.selection_event is not None
    assert snap.selection_event.target_id == "t_fast"


# --- Scenario 17: Stability & saccade grace tolerance ---
def test_scenario_17_stability_saccade_grace_tolerance(test_setup) -> None:
    engine, _, clock, _, _ = test_setup
    # Dwell for 500ms
    engine.update(0.25, 0.25)
    clock.advance_ms(500)
    engine.update(0.25, 0.25)

    # Gaze momentarily darts outside target for 100ms (< 150ms grace)
    clock.advance_ms(100)
    snap_grace = engine.update(0.05, 0.05)
    # State remains DWELLING and progress held
    assert snap_grace.state == DwellState.DWELLING
    assert snap_grace.dwell_time_ms == pytest.approx(500.0)

    # Gaze returns inside target: continues from 500ms
    clock.advance_ms(50)
    snap_return = engine.update(0.25, 0.25)
    assert snap_return.state == DwellState.DWELLING
    clock.advance_ms(700)  # 500 + 700 = 1200ms
    snap_finish = engine.update(0.25, 0.25)
    assert snap_finish.state == DwellState.SELECTED


# --- Scenario 18: Long continuous gaze does not re-trigger ---
def test_scenario_18_long_continuous_gaze_does_not_retrigger(test_setup) -> None:
    engine, _, clock, _, _ = test_setup
    # Trigger selection
    engine.update(0.25, 0.25)
    clock.advance_ms(1200)
    engine.update(0.25, 0.25)
    assert engine.total_selections_count == 1

    # User continues staring at same target for 5 full seconds
    for _ in range(50):
        clock.advance_ms(100)
        snap = engine.update(0.25, 0.25)
        assert snap.selection_event is None

    assert engine.total_selections_count == 1


# --- Scenario 19: Repeated selection after a fresh dwell cycle ---
def test_scenario_19_repeated_selection_after_fresh_cycle(test_setup) -> None:
    engine, _, clock, _, _ = test_setup
    # 1. First selection on t1
    engine.update(0.25, 0.25)
    clock.advance_ms(1200)
    engine.update(0.25, 0.25)
    assert engine.total_selections_count == 1

    # 2. Complete cooldown (200ms selected display + 800ms cooldown)
    clock.advance_ms(1100)
    engine.update(0.25, 0.25)

    # 3. User looks away to empty space (exiting t1)
    clock.advance_ms(200)
    snap_away = engine.update(0.05, 0.05)
    assert snap_away.state == DwellState.IDLE

    # 4. User looks BACK at t1 and dwells for 1200ms
    clock.advance_ms(50)
    engine.update(0.25, 0.25)
    clock.advance_ms(1200)
    snap_second = engine.update(0.25, 0.25)
    assert snap_second.state == DwellState.SELECTED
    assert snap_second.selection_event is not None
    assert engine.total_selections_count == 2


# --- Scenario 20: Dwell disabled mode ---
def test_scenario_20_dwell_disabled_mode(test_setup) -> None:
    engine, _, clock, _, _ = test_setup
    engine.enabled = False

    snap = engine.update(0.25, 0.25)
    assert snap.state == DwellState.IDLE
    assert snap.current_target_id is None

    clock.advance_ms(2000)
    snap2 = engine.update(0.25, 0.25)
    assert snap2.state == DwellState.IDLE
    assert snap2.selection_event is None
    assert engine.total_selections_count == 0
