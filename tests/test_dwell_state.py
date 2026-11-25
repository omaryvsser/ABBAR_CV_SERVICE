"""
Tests for Phase 4 DwellState, SelectionEvent, DwellSnapshot, and DwellConfig.
"""

import pytest
from dwell.dwell_config import DwellConfig
from dwell.dwell_state import DwellSnapshot, DwellState, SelectionEvent


def test_dwell_state_enum() -> None:
    """Verify all 5 core state machine states are defined."""
    assert DwellState.IDLE.value == "IDLE"
    assert DwellState.DWELLING.value == "DWELLING"
    assert DwellState.SELECTED.value == "SELECTED"
    assert DwellState.COOLDOWN.value == "COOLDOWN"
    assert DwellState.LOST_TRACKING.value == "LOST_TRACKING"


def test_selection_event_serialization() -> None:
    """Verify structured selection event generation and to_dict()."""
    evt = SelectionEvent(
        target_id="btn_help",
        target_label="HELP",
        timestamp=1700000000.1234,
        gaze_x=0.45,
        gaze_y=0.65,
        dwell_duration_ms=1200.0,
        metadata={"priority": 5, "sound": "alert.wav"},
    )
    assert len(evt.event_id) > 10  # auto-generated UUID
    d = evt.to_dict()
    assert d["target_id"] == "btn_help"
    assert d["target_label"] == "HELP"
    assert d["gaze_x"] == 0.45
    assert d["gaze_y"] == 0.65
    assert d["dwell_duration_ms"] == 1200.0
    assert d["selection_state"] == "CONFIRMED"
    assert d["metadata"]["sound"] == "alert.wav"


def test_dwell_snapshot_serialization() -> None:
    """Verify live snapshot fields and serialization."""
    snap = DwellSnapshot(
        state=DwellState.DWELLING,
        current_target_id="target_1",
        current_target_label="YES",
        dwell_time_ms=600.0,
        dwell_progress=0.50,
        remaining_time_ms=600.0,
        is_tracking_valid=True,
        gaze_x=0.25,
        gaze_y=0.35,
        cooldown_progress=0.0,
    )
    d = snap.to_dict()
    assert d["state"] == "DWELLING"
    assert d["current_target_id"] == "target_1"
    assert d["dwell_progress"] == 0.50
    assert d["has_selection"] is False
    assert d["selection_event"] is None


def test_dwell_config_defaults_and_validation() -> None:
    """Verify default accessibility settings and bounds checking."""
    cfg = DwellConfig()
    assert cfg.dwell_duration_ms == 1200.0
    assert cfg.cooldown_duration_ms == 800.0
    assert cfg.stability_filter_enabled is True
    assert cfg.allow_repeat_without_leaving is False
    assert cfg.pause_on_lost_tracking is True

    # Invalid dwell durations
    with pytest.raises(ValueError, match="dwell_duration_ms must be positive"):
        DwellConfig(dwell_duration_ms=0.0)

    with pytest.raises(ValueError, match="cooldown_duration_ms cannot be negative"):
        DwellConfig(cooldown_duration_ms=-100.0)
