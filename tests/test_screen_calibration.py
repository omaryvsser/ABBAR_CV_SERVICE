"""
Unit tests for 9-Point Screen Calibration Workflow (Phase 3).

Covers:
- Target geometry & coordinate bounds
- Settling vs Collecting state machine transitions
- Sample filtering, blink rejection, head turn filtering
- Trimmed median feature aggregation across 9 targets
- Model training & profile creation upon calibration completion
- Calibration cancellation and incomplete target handling
"""

import time
import numpy as np
import pytest

from src.core.eye_features import extract_eye_features
from src.core.gaze_baseline import GazeBaseline
from src.core.gaze_mapper import BaseGazeMapper
from src.core.head_pose import HeadPose
from src.core.screen_calibration import (
    CalibrationState,
    CalibrationTarget,
    DEFAULT_9_TARGETS,
    ScreenCalibrator,
    TargetPhase,
)
from tests.conftest import create_synthetic_landmarks


def test_calibration_targets_layout_and_pixel_conversion():
    """Verify 9-point grid layout and screen pixel conversion."""
    assert len(DEFAULT_9_TARGETS) == 9

    # Check bounds: All targets should be within [0.05, 0.95]
    for t in DEFAULT_9_TARGETS:
        assert 0.05 <= t.x <= 0.95
        assert 0.05 <= t.y <= 0.95

    # Center target must be (0.50, 0.50)
    center_target = DEFAULT_9_TARGETS[4]
    assert center_target.name == "center"
    assert center_target.x == 0.50
    assert center_target.y == 0.50

    # Test pixel conversion
    px, py = center_target.to_pixel(screen_width=1920, screen_height=1080)
    assert px == 960
    assert py == 540


def test_calibrator_lifecycle_and_settling_delay():
    """Verify state machine: SETTLING delay rejects samples, then transitions to COLLECTING."""
    calibrator = ScreenCalibrator(
        settling_duration=0.2,
        collection_duration=0.4,
        min_samples_per_target=5,
    )
    assert calibrator.state == CalibrationState.IDLE
    assert not calibrator.is_active

    calibrator.start()
    assert calibrator.is_active
    assert calibrator.current_target_index == 0
    assert calibrator.target_phase == TargetPhase.SETTLING

    landmarks = create_synthetic_landmarks(center_offset_x=0.0, center_offset_y=0.0)
    feats = extract_eye_features(landmarks)

    # Frame during settling phase: should NOT accumulate samples
    res = calibrator.update(feats)
    assert res is None
    assert len(calibrator._target_samples[0]) == 0

    # Wait out settling duration
    time.sleep(0.25)
    # Next update triggers transition to COLLECTING
    calibrator.update(feats)
    assert calibrator.target_phase == TargetPhase.COLLECTING

    # Now samples should accumulate
    calibrator.update(feats)
    assert len(calibrator._target_samples[0]) >= 1


def test_calibrator_rejection_of_blinks_and_poor_tracking():
    """Verify that blinking frames or excessive head turns are recorded as rejected."""
    calibrator = ScreenCalibrator(
        settling_duration=0.05,
        collection_duration=0.5,
        min_samples_per_target=5,
    )
    calibrator.start()
    time.sleep(0.06)

    # Feed normal frame to enter COLLECTING
    normal_lm = create_synthetic_landmarks()
    calibrator.update(extract_eye_features(normal_lm))
    assert calibrator.target_phase == TargetPhase.COLLECTING

    # 1. Closed eyes (blink)
    blink_lm = create_synthetic_landmarks(eye_open_height=0.002)
    blink_feats = extract_eye_features(blink_lm)
    calibrator.update(blink_feats)

    # 2. Excessive head yaw (+35 deg)
    turned_pose = HeadPose(yaw=35.0, pitch=0.0, roll=0.0, is_valid=True)
    calibrator.update(extract_eye_features(normal_lm), head_pose=turned_pose)

    samples = calibrator.get_all_samples()
    # At least 2 rejected samples should be logged with reasons
    rejected = [s for s in samples if not s.accepted]
    assert len(rejected) >= 2
    reasons = [r.rejection_reason for r in rejected]
    assert any("BLINK" in r for r in reasons)
    assert any("EXCESSIVE_HEAD" in r for r in reasons)


def test_full_9_point_calibration_flow():
    """Verify full progression through all 9 targets, producing a fitted mapper."""
    calibrator = ScreenCalibrator(
        settling_duration=0.05,
        collection_duration=0.10,
        min_samples_per_target=5,
        screen_width=1920,
        screen_height=1080,
    )
    calibrator.start()

    # Step through all 9 targets
    for target_idx in range(9):
        assert calibrator.current_target_index == target_idx
        # Settle
        time.sleep(0.06)
        target = calibrator.current_target
        assert target is not None

        # Synthesize target-correlated eye landmarks
        # Screen targets [0.1..0.9] map to offset [-0.03..+0.03]
        off_x = (target.x - 0.50) * 0.06
        off_y = (target.y - 0.50) * 0.06
        lm = create_synthetic_landmarks(center_offset_x=off_x, center_offset_y=off_y)
        feats = extract_eye_features(lm)

        # Feed min required samples
        fitted_model = None
        for _ in range(calibrator.min_samples_per_target + 2):
            time.sleep(0.02)
            res = calibrator.update(feats)
            if res is not None:
                fitted_model = res

        if target_idx < 8:
            assert fitted_model is None
            assert not calibrator.is_completed
        else:
            # 9th target completion should return fitted model
            assert fitted_model is not None
            assert isinstance(fitted_model, BaseGazeMapper)
            assert fitted_model.is_fitted
            assert calibrator.is_completed
            assert calibrator.profile is not None
            assert calibrator.profile.screen_width == 1920


def test_calibrator_cancellation():
    """Verify calibration can be cleanly cancelled and reset."""
    calibrator = ScreenCalibrator()
    calibrator.start()
    assert calibrator.is_active

    calibrator.cancel()
    assert not calibrator.is_active
    assert calibrator.state == CalibrationState.CANCELLED
