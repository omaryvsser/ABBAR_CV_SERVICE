"""
Automated Unit Tests for Enhanced Phase 4 Dwell Selection Engine.

Tests:
1. Filter latency, stability, and NaN handling (None, EMA, One-Euro).
2. Entry and exit target hysteresis boundaries.
3. Adjacent target switching with jitter confirmation.
4. Fixation detection and ballistic saccade suppression.
5. Adaptive dwell user presets and configuration bounds.
6. Telemetry snapshots containing raw vs filtered coordinates.
"""

import math
import pytest

from dwell.dwell_config import DwellConfig, DwellUserSkill
from dwell.dwell_engine import DwellEngine
from dwell.dwell_state import DwellState
from dwell.gaze_filter import (
    EMAGazeFilter,
    GazeFilterType,
    OneEuroGazeFilter,
    PassThroughGazeFilter,
    create_gaze_filter,
)
from dwell.target_manager import DwellTarget, TargetManager


class MockClock:
    """Controllable clock for deterministic time advancing."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.time = start_time

    def __call__(self) -> float:
        return self.time

    def advance_ms(self, milliseconds: float) -> None:
        self.time += milliseconds / 1000.0


# --- Test 1: Filter Algorithms & NaN Safety ---
def test_filter_algorithms_initialization_and_nan_safety() -> None:
    # 1. Pass-through filter
    pt = create_gaze_filter("none")
    assert pt.filter_type == "none"
    fx, fy = pt.update(0.5, 0.5, 1000.0, is_valid=True)
    assert fx == 0.5 and fy == 0.5

    # NaN safety
    fx, fy = pt.update(float("nan"), 0.5, 1000.033, is_valid=True)
    assert fx is None and fy is None

    # 2. EMA Filter
    ema = create_gaze_filter("ema", ema_alpha=0.35)
    assert ema.filter_type == "ema"
    fx, fy = ema.update(0.5, 0.5, 1000.0, is_valid=True)
    assert fx == 0.5 and fy == 0.5
    # Small step: smoothed
    fx2, _ = ema.update(0.52, 0.5, 1000.033, is_valid=True)
    assert 0.50 < fx2 < 0.52

    # 3. One-Euro Filter
    one = create_gaze_filter("one_euro", min_cutoff=1.2, beta=12.0)
    assert one.filter_type == "one_euro"
    fx, fy = one.update(0.40, 0.40, 1000.0, is_valid=True)
    assert fx == 0.40 and fy == 0.40
    # Step movement
    fx2, fy2 = one.update(0.42, 0.40, 1000.033, is_valid=True)
    assert 0.40 < fx2 < 0.42
    assert fy2 == 0.40

    # Test reset
    one.reset()
    assert one._x is None


# --- Test 2: Target Hysteresis Entry and Exit Boundaries ---
def test_target_hysteresis_boundaries() -> None:
    target = DwellTarget("card1", "YES", 0.20, 0.20, 0.30, 0.30)  # spans x in [0.20, 0.50], y in [0.20, 0.50]

    # Inside standard bounding box
    assert target.contains(0.35, 0.35) is True
    assert target.contains_entry(0.35, 0.35) is True
    assert target.contains_exit(0.35, 0.35) is True

    # 10 px outside right border: x = 0.52 (within 0.04 exit margin, but outside entry)
    assert target.contains(0.52, 0.35) is False
    assert target.contains_entry(0.52, 0.35, margin=0.0) is False
    assert target.contains_exit(0.52, 0.35, margin=0.04) is True

    # Far outside: x = 0.60 (outside both entry and exit)
    assert target.contains(0.60, 0.35) is False
    assert target.contains_exit(0.60, 0.35, margin=0.04) is False


def test_target_manager_hysteresis_lookup() -> None:
    tm = TargetManager()
    t1 = DwellTarget("t1", "YES", 0.10, 0.10, 0.30, 0.30)  # [0.10, 0.40]
    t2 = DwellTarget("t2", "NO", 0.45, 0.10, 0.30, 0.30)   # [0.45, 0.75]
    tm.add_target(t1)
    tm.add_target(t2)

    # Point at 0.42: in gap between t1 and t2, but within t1's 0.04 exit margin
    hit_no_active = tm.find_target_at(0.42, 0.25)
    assert hit_no_active is None

    hit_with_active_t1 = tm.find_target_at(0.42, 0.25, current_target_id="t1", exit_margin=0.04)
    assert hit_with_active_t1 is not None
    assert hit_with_active_t1.target_id == "t1"  # Hysteresis holds t1!

    # Point inside t2 entry (0.50, 0.25): t2 wins even if active is t1
    hit_t2 = tm.find_target_at(0.50, 0.25, current_target_id="t1", exit_margin=0.04)
    assert hit_t2 is not None
    assert hit_t2.target_id == "t2"


# --- Test 3: Adjacent Target Switching Confirmation ---
def test_adjacent_target_switch_confirmation_protection() -> None:
    clk = MockClock(1000.0)
    cfg = DwellConfig(
        dwell_duration_ms=1000.0,
        hysteresis_enabled=True,
        exit_margin=0.04,
        switch_confirmation_ms=80.0,
    )
    tm = TargetManager()
    t1 = DwellTarget("t1", "YES", 0.10, 0.10, 0.30, 0.30)
    t2 = DwellTarget("t2", "NO", 0.42, 0.10, 0.30, 0.30)
    tm.add_target(t1)
    tm.add_target(t2)
    eng = DwellEngine(config=cfg, target_manager=tm, clock=clk)

    # 1. Dwell on t1 for 500ms
    eng.update(0.25, 0.25)
    clk.advance_ms(500)
    snap = eng.update(0.25, 0.25)
    assert snap.current_target_id == "t1"
    assert snap.dwell_time_ms == pytest.approx(500.0)

    # 2. Brief 33ms jitter into t2 (inside t2 entry: 0.45, 0.25)
    clk.advance_ms(33.33)
    snap_jitter = eng.update(0.45, 0.25)
    # Under switch confirmation window (33ms < 80ms), t1 dwell is NOT destroyed!
    assert snap_jitter.current_target_id == "t1"
    assert snap_jitter.dwell_time_ms == pytest.approx(500.0)

    # 3. Return to t1: resumes accumulation smoothly
    clk.advance_ms(33.33)
    snap_resumed = eng.update(0.25, 0.25)
    assert snap_resumed.current_target_id == "t1"
    assert snap_resumed.dwell_time_ms > 500.0

    # 4. Now intentional sustained gaze into t2 for > 80ms confirmation
    clk.advance_ms(50.0)
    eng.update(0.45, 0.25)  # Pending switch initiated (t = 0ms)
    clk.advance_ms(50.0)
    eng.update(0.45, 0.25)  # 50ms elapsed (< 80ms confirmation, t1 held)
    clk.advance_ms(50.0)
    snap_switched = eng.update(0.45, 0.25)  # 100ms elapsed (> 80ms confirmation, switch confirmed!)
    assert snap_switched.current_target_id == "t2"
    assert snap_switched.dwell_time_ms == pytest.approx(0.0)


# --- Test 4: Fixation Detection & Saccade Suppression ---
def test_fixation_detection_saccade_suppression() -> None:
    clk = MockClock(1000.0)
    cfg = DwellConfig(
        dwell_duration_ms=1000.0,
        fixation_detection_enabled=True,
        saccade_velocity_threshold=1.5,
    )
    tm = TargetManager([DwellTarget("t1", "YES", 0.20, 0.20, 0.40, 0.40)])
    eng = DwellEngine(config=cfg, target_manager=tm, clock=clk)

    # Frame 1: empty space
    eng.update(0.05, 0.25)

    # Frame 2 (30ms later): fast saccade sweeping into t1
    # distance = 0.30 in 0.030s => velocity = 10.0 units/sec (> 1.5 threshold)
    clk.advance_ms(30.0)
    snap_saccade = eng.update(0.35, 0.25)
    assert snap_saccade.is_fixating is False
    assert snap_saccade.state == DwellState.IDLE  # Saccade traversal rejected!

    # Frame 3 (30ms later): gaze stabilizes at (0.35, 0.25) (zero distance, low velocity)
    clk.advance_ms(30.0)
    snap_fixated = eng.update(0.35, 0.25)
    assert snap_fixated.is_fixating is True
    assert snap_fixated.state == DwellState.DWELLING  # Dwell begins smoothly!
    assert snap_fixated.current_target_id == "t1"


# --- Test 5: Adaptive Dwell Presets and User Profiles ---
def test_adaptive_dwell_presets() -> None:
    # Standard preset (default 1200ms - restored reliable baseline)
    std = DwellConfig.create_preset(DwellUserSkill.STANDARD)
    assert std.dwell_duration_ms == 1200.0
    assert std.cooldown_duration_ms == 800.0
    assert std.filter_type == "none"
    assert std.exit_margin == 0.0

    # Novice preset (slower 1500ms, wider margins)
    nov = DwellConfig.create_preset(DwellUserSkill.NOVICE)
    assert nov.dwell_duration_ms == 1500.0
    assert nov.cooldown_duration_ms == 1000.0
    assert nov.exit_margin == 0.06

    # Expert preset (faster 800ms)
    exp = DwellConfig.create_preset(DwellUserSkill.EXPERT)
    assert exp.dwell_duration_ms == 800.0
    assert exp.cooldown_duration_ms == 500.0
    assert exp.exit_margin == 0.03

    # Tremor-prone preset (heavy filtering, large exit hysteresis)
    tremor = DwellConfig.create_preset(DwellUserSkill.TREMOR_PRONE)
    assert tremor.dwell_duration_ms == 1400.0
    assert tremor.exit_margin == 0.08
    assert tremor.filter_min_cutoff == 0.7


# --- Test 6: Snapshot Telemetry with Raw vs Filtered Coordinates ---
def test_snapshot_telemetry_coordinates() -> None:
    clk = MockClock(1000.0)
    cfg = DwellConfig(filter_type="one_euro")
    tm = TargetManager([DwellTarget("t1", "YES", 0.1, 0.1, 0.3, 0.3)])
    eng = DwellEngine(config=cfg, target_manager=tm, clock=clk)

    snap = eng.update(0.25, 0.25)
    assert snap.gaze_x == pytest.approx(0.25)
    assert snap.gaze_y == pytest.approx(0.25)
    assert snap.raw_gaze_x == pytest.approx(0.25)
    assert snap.raw_gaze_y == pytest.approx(0.25)
    assert snap.filter_type == "one_euro"
    assert snap.is_fixating is True

    d = snap.to_dict()
    assert "raw_gaze_x" in d
    assert "raw_gaze_y" in d
    assert "is_fixating" in d
    assert "filter_type" in d
