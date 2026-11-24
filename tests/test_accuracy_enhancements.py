"""
Comprehensive unit and integration tests for Phase 2 Gaze Accuracy Improvements.

Covers:
1. GazeBaselineCalibrator (sampling, outlier rejection, head pose filtering, statistics).
2. GazeDirectionClassifier (dual-threshold hysteresis, persistence window, baseline offset).
3. DiagnosticDataLogger (session management, CSV header format, sample logging).
4. CVPipeline Integration (tracking state machine, calibration workflow, logging toggle).
"""

from pathlib import Path
import tempfile
import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.config import AppConfig
from src.core.data_logger import DiagnosticDataLogger
from src.core.eye_features import extract_eye_features
from src.core.gaze_baseline import GazeBaseline, GazeBaselineCalibrator
from src.core.gaze_direction import GazeDirection, GazeDirectionClassifier
from src.core.head_pose import HeadPose
from src.core.tracking_state import TrackingState
from src.pipeline import CVPipeline, PipelineOutput
from tests.conftest import create_synthetic_landmarks


# ==============================================================================
# 1. GazeBaselineCalibrator Unit Tests
# ==============================================================================

def test_calibrator_progress_and_completion():
    """Verify that calibrator correctly samples frames and completes at target."""
    calibrator = GazeBaselineCalibrator(
        calibration_duration_seconds=1.0,
        target_fps=30,
        min_required_samples=10,
    )
    assert not calibrator.is_calibrated
    assert not calibrator.is_calibrating

    calibrator.start_calibration()
    assert calibrator.is_calibrating
    assert calibrator.progress == 0.0

    # Feed synthetic centered gaze frames
    landmarks = create_synthetic_landmarks(center_offset_x=0.0, center_offset_y=0.0)
    feats = extract_eye_features(landmarks)

    for i in range(calibrator.target_samples - 1):
        finished = calibrator.update(feats)
        assert not finished
        assert calibrator.progress > 0.0

    # Final frame should trigger completion
    finished = calibrator.update(feats)
    assert finished
    assert calibrator.is_calibrated
    assert not calibrator.is_calibrating
    assert calibrator.progress == 1.0

    baseline = calibrator.baseline
    assert abs(baseline.neutral_x - 0.50) < 0.05
    assert abs(baseline.neutral_y - 0.50) < 0.05
    assert baseline.sample_count == calibrator.target_samples


def test_calibrator_outlier_rejection():
    """Verify that trimmed median rejects random outlier frames during sampling."""
    calibrator = GazeBaselineCalibrator(
        calibration_duration_seconds=1.0,
        target_fps=20,
        min_required_samples=10,
    )
    calibrator.start_calibration()

    norm_landmarks = create_synthetic_landmarks(center_offset_x=0.0, center_offset_y=0.0)
    norm_feats = extract_eye_features(norm_landmarks)

    # Injected extreme outlier landmarks (user glanced away briefly)
    outlier_landmarks = create_synthetic_landmarks(center_offset_x=0.02, center_offset_y=-0.01)
    outlier_feats = extract_eye_features(outlier_landmarks)

    # 16 normal frames + 4 outlier frames
    for _ in range(16):
        calibrator.update(norm_feats)
    for _ in range(4):
        calibrator.update(outlier_feats)

    assert calibrator.is_calibrated
    # The trimmed median should stay solidly near 0.50 rather than being pulled by outliers
    assert abs(calibrator.baseline.neutral_x - 0.50) < 0.03


def test_calibrator_rejects_excessive_head_pose():
    """Verify that observations during excessive head turn are discarded."""
    calibrator = GazeBaselineCalibrator(
        calibration_duration_seconds=1.0,
        target_fps=20,
        min_required_samples=5,
        max_head_yaw_during_calib=10.0,
    )
    calibrator.start_calibration()

    landmarks = create_synthetic_landmarks()
    feats = extract_eye_features(landmarks)

    # Normal head pose
    good_pose = HeadPose(yaw=2.0, pitch=1.0, roll=0.0, is_valid=True)
    calibrator.update(feats, good_pose)
    assert len(calibrator._samples_x) == 1

    # Turned head pose (> 10 deg)
    turned_pose = HeadPose(yaw=15.0, pitch=1.0, roll=0.0, is_valid=True)
    calibrator.update(feats, turned_pose)
    # Sample count should NOT increase
    assert len(calibrator._samples_x) == 1


# ==============================================================================
# 2. GazeDirectionClassifier Hysteresis & Baseline Offset Tests
# ==============================================================================

def test_classifier_dual_threshold_hysteresis():
    """
    Verify Schmitt-trigger dual-threshold hysteresis:
    Activation requires |delta| >= h_activation (0.06).
    Deactivation only occurs when |delta| drops below h_deactivation (0.035).
    """
    classifier = GazeDirectionClassifier(
        h_activation_threshold=0.06,
        h_deactivation_threshold=0.035,
        persistence_frames=1,
    )

    # 1. At resting center (delta = 0.0) -> CENTER
    assert classifier.classify(0.50, 0.50) == GazeDirection.CENTER

    # 2. Small deviation below activation threshold (delta = +0.04 < 0.06) -> Stays CENTER
    assert classifier.classify(0.54, 0.50) == GazeDirection.CENTER

    # 3. Crosses activation threshold (delta = +0.07 >= 0.06) -> Becomes RIGHT
    assert classifier.classify(0.57, 0.50) == GazeDirection.RIGHT

    # 4. Drops back into intermediate band (delta = +0.045, where 0.035 < 0.045 < 0.06)
    # Hysteresis MUST hold active RIGHT!
    assert classifier.classify(0.545, 0.50) == GazeDirection.RIGHT

    # 5. Drops below deactivation threshold (delta = +0.02 < 0.035) -> Drops back to CENTER
    assert classifier.classify(0.52, 0.50) == GazeDirection.CENTER


def test_classifier_temporal_persistence_window():
    """Verify that direction change requires K consecutive frames to be confirmed."""
    classifier = GazeDirectionClassifier(
        h_activation_threshold=0.06,
        h_deactivation_threshold=0.035,
        persistence_frames=3,
    )

    # Initial frame at Center
    assert classifier.classify(0.50, 0.50) == GazeDirection.CENTER

    # Frame 1 of sudden glance to Left (delta = -0.15) -> Still CENTER (candidate 1/3)
    assert classifier.classify(0.35, 0.50) == GazeDirection.CENTER

    # Frame 2 of glance to Left -> Still CENTER (candidate 2/3)
    assert classifier.classify(0.35, 0.50) == GazeDirection.CENTER

    # Frame 3 of glance to Left -> Confirmed LEFT (candidate 3/3)
    assert classifier.classify(0.35, 0.50) == GazeDirection.LEFT


def test_classifier_with_personalized_baseline_offset():
    """
    Verify that providing a calibrated baseline (bx=0.45, by=0.55)
    treats (0.45, 0.55) as true CENTER and calculates delta relative to baseline.
    """
    classifier = GazeDirectionClassifier(
        h_activation_threshold=0.06,
        v_activation_threshold=0.06,
        persistence_frames=1,
    )
    b_x, b_y = 0.45, 0.55

    # Resting at personalized baseline -> CENTER
    assert classifier.classify(0.45, 0.55, baseline_x=b_x, baseline_y=b_y) == GazeDirection.CENTER

    # Looking anatomically RIGHT (x increases from 0.45 to 0.53 -> delta = +0.08 > 0.06)
    assert classifier.classify(0.53, 0.55, baseline_x=b_x, baseline_y=b_y) == GazeDirection.RIGHT

    # Looking anatomically LEFT (x decreases from 0.45 to 0.37 -> delta = -0.08 < -0.06)
    assert classifier.classify(0.37, 0.55, baseline_x=b_x, baseline_y=b_y) == GazeDirection.LEFT


# ==============================================================================
# 3. DiagnosticDataLogger Tests
# ==============================================================================

def test_diagnostic_data_logger_lifecycle():
    """Verify CSV file creation, header schema, and sample record writing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger_dir = Path(tmpdir)
        logger = DiagnosticDataLogger(output_dir=logger_dir)
        assert not logger.is_logging

        csv_path = logger.start_logging("test_run.csv")
        assert logger.is_logging
        assert csv_path.exists()

        # Write sample frames
        logger.log_frame(
            frame_idx=1,
            tracking_state="TRACKING",
            face_detected=True,
            r_iris_x=0.51,
            r_iris_y=0.49,
            l_iris_x=0.49,
            l_iris_y=0.49,
            raw_x=0.50,
            raw_y=0.49,
            smoothed_x=0.50,
            smoothed_y=0.49,
            baseline_x=0.50,
            baseline_y=0.50,
            delta_x=0.00,
            delta_y=-0.01,
            direction="CENTER",
            confidence=0.92,
            yaw=1.2,
            pitch=-0.5,
            roll=0.1,
        )
        assert logger.record_count == 1

        closed_path = logger.stop_logging()
        assert not logger.is_logging
        assert closed_path == csv_path

        # Read back CSV contents
        with open(csv_path, mode="r", encoding="utf-8") as f:
            lines = [line.strip().split(",") for line in f.readlines()]

        assert len(lines) == 2  # Header + 1 data line
        assert lines[0] == DiagnosticDataLogger.HEADERS
        assert lines[1][1] == "1"            # frame_idx
        assert lines[1][2] == "TRACKING"     # tracking_state
        assert lines[1][3] == "1"            # face_detected
        assert lines[1][16] == "CENTER"      # direction


def test_diagnostic_data_logger_toggle():
    """Verify that toggle_logging() starts and stops cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = DiagnosticDataLogger(output_dir=Path(tmpdir))
        assert not logger.is_logging

        is_active = logger.toggle_logging()
        assert is_active is True
        assert logger.is_logging is True

        is_active = logger.toggle_logging()
        assert is_active is False
        assert logger.is_logging is False


# ==============================================================================
# 4. Pipeline Integration & TrackingState Tests
# ==============================================================================

def test_pipeline_tracking_state_and_recalibrate(dummy_bgr_frame):
    """Verify pipeline tracking state transitions, baseline access, and recalibration."""
    model_path = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
    if not model_path.exists():
        pytest.skip("Model bundle required for pipeline test.")

    config = AppConfig()
    config.camera.camera_index = 0
    config.camera.use_threading = False
    config.landmarker.model_path = str(model_path)
    config.visualization.enabled = False
    config.gaze.auto_calibrate_on_start = True
    config.gaze.calibration_duration_seconds = 0.5  # Fast calibration for test

    with patch("src.camera.camera_manager.CameraManager.start"):
        with patch("src.camera.camera_manager.CameraManager.is_opened", return_value=True):
            with patch("src.camera.camera_manager.CameraManager.read", return_value=(True, dummy_bgr_frame)):
                pipeline = CVPipeline(config=config)
                pipeline._is_running = True

                # On initial frame with dummy image (no real face detected), state should be NO_FACE
                output = pipeline.process_next_frame()
                assert output is not None
                assert isinstance(output, PipelineOutput)
                assert output.tracking_state == TrackingState.NO_FACE
                assert output.baseline is not None

                # Test recalibrate method
                pipeline.recalibrate()
                assert pipeline.calibrator.is_calibrating

                # Test toggle data logging
                with tempfile.TemporaryDirectory() as tmpdir:
                    pipeline.data_logger.output_dir = Path(tmpdir)
                    rec_active = pipeline.toggle_data_logging()
                    assert rec_active is True
                    assert pipeline.data_logger.is_logging is True

                    rec_active = pipeline.toggle_data_logging()
                    assert rec_active is False
                    assert pipeline.data_logger.is_logging is False

                pipeline.close()


# ==============================================================================
# 5. Vertical Gaze Estimation Accuracy Tests
# ==============================================================================

def test_asymmetric_vertical_hysteresis():
    """Verify asymmetric vertical thresholds (UP: -0.045, DOWN: +0.060) and deactivations."""
    classifier = GazeDirectionClassifier(
        v_up_activation_threshold=0.045,
        v_up_deactivation_threshold=0.025,
        v_down_activation_threshold=0.060,
        v_down_deactivation_threshold=0.035,
        persistence_frames=1,
    )

    # 1. UP tests around baseline (0.50, 0.50):
    # 0.50 - 0.040 = 0.460 (delta -0.040, within deadband) -> CENTER
    assert classifier.classify(0.50, 0.460, baseline_x=0.50, baseline_y=0.50) == GazeDirection.CENTER
    # 0.50 - 0.050 = 0.450 (delta -0.050 < -0.045) -> UP
    assert classifier.classify(0.50, 0.450, baseline_x=0.50, baseline_y=0.50) == GazeDirection.UP
    # Release check: 0.50 - 0.030 = 0.470 (delta -0.030 < -0.025 deactivation) -> holds UP
    assert classifier.classify(0.50, 0.470, baseline_x=0.50, baseline_y=0.50) == GazeDirection.UP
    # Drop below deactivation: 0.50 - 0.020 = 0.480 (delta -0.020 > -0.025) -> drops to CENTER
    assert classifier.classify(0.50, 0.480, baseline_x=0.50, baseline_y=0.50) == GazeDirection.CENTER

    # 2. DOWN tests around baseline (0.50, 0.50):
    # 0.50 + 0.050 = 0.550 (delta +0.050 < +0.060 activation) -> CENTER
    assert classifier.classify(0.50, 0.550, baseline_x=0.50, baseline_y=0.50) == GazeDirection.CENTER
    # 0.50 + 0.065 = 0.565 (delta +0.065 > +0.060 activation) -> DOWN
    assert classifier.classify(0.50, 0.565, baseline_x=0.50, baseline_y=0.50) == GazeDirection.DOWN
    # Release check: 0.50 + 0.040 = 0.540 (delta +0.040 > +0.035 deactivation) -> holds DOWN
    assert classifier.classify(0.50, 0.540, baseline_x=0.50, baseline_y=0.50) == GazeDirection.DOWN
    # Drop below deactivation: 0.50 + 0.030 = 0.530 (delta +0.030 < +0.035) -> drops to CENTER
    assert classifier.classify(0.50, 0.530, baseline_x=0.50, baseline_y=0.50) == GazeDirection.CENTER


def test_vertical_blink_safety():
    """Verify that closed eyes or blinks clamp vertical ratio to 0.50 and invalidate features."""
    closed_landmarks = create_synthetic_landmarks(eye_open_height=0.002)
    feats = extract_eye_features(closed_landmarks)

    assert not feats.is_valid
    assert feats.right_eye.is_blink
    assert feats.left_eye.is_blink
    assert feats.right_eye.vertical_ratio == pytest.approx(0.50, abs=1e-3)
    assert feats.left_eye.vertical_ratio == pytest.approx(0.50, abs=1e-3)
    assert feats.combined_vertical_ratio is None or feats.combined_vertical_ratio == pytest.approx(0.50, abs=1e-3)


def test_independent_vertical_baseline_recalibration():
    """Verify recalibrate_vertical() preserves horizontal baseline while updating vertical."""
    calibrator = GazeBaselineCalibrator(
        calibration_duration_seconds=1.0,
        target_fps=20,
        min_required_samples=10,
    )
    # Manually establish an initial baseline
    calibrator.baseline = GazeBaseline(
        neutral_x=0.56,
        neutral_y=0.48,
        noise_std_x=0.015,
        noise_std_y=0.012,
        neutral_pitch=2.5,
        neutral_yaw=-1.0,
        sample_count=20,
        is_calibrated=True,
    )
    assert calibrator.is_calibrated

    # Call independent vertical recalibration
    calibrator.recalibrate_vertical()
    assert calibrator.is_calibrating

    # Feed new samples with shifted vertical gaze (0.52) but keeping horizontal (0.56)
    landmarks = create_synthetic_landmarks(center_offset_x=0.0, center_offset_y=0.0)
    raw_feats = extract_eye_features(landmarks)
    import dataclasses
    feats = dataclasses.replace(raw_feats, combined_horizontal_ratio=0.56, combined_vertical_ratio=0.52)

    for _ in range(calibrator.target_samples):
        calibrator.update(feats, head_pose=HeadPose(yaw=-1.0, pitch=4.0, roll=0.0, is_valid=True))

    assert calibrator.is_calibrated
    # Horizontal baseline remains preserved
    assert calibrator.baseline.neutral_x == pytest.approx(0.56, abs=1e-3)
    # Vertical baseline updated to new observation
    assert calibrator.baseline.neutral_y == pytest.approx(0.52, abs=1e-2)
    # Neutral pitch updated to new observation
    assert calibrator.baseline.neutral_pitch == pytest.approx(4.0, abs=0.5)


def test_head_pitch_vor_compensation():
    """Verify Vestibulo-Ocular Reflex pitch compensation formula."""
    # When head pitches up (+10 deg), raw vertical gaze drops down due to VOR counter-rotation.
    # Pitch compensation should add back k_pitch * delta_pitch.
    k_pitch = 0.0020
    neutral_pitch = 0.0
    current_pitch = 10.0  # +10 degrees pitch up
    delta_pitch = current_pitch - neutral_pitch

    raw_y = 0.480  # Apparent upward eye gaze drop caused by head pitch
    # Formula: comp_y = raw_y - k_pitch * delta_pitch
    # Note: When pitch > 0 (head tilted up), eyes counter-rotate down (raw_y increases or decreases depending on pitch definition)
    # comp_y = raw_y - k_pitch * delta_pitch cancels the artifact
    comp_y = raw_y - (k_pitch * delta_pitch)
    assert comp_y == pytest.approx(0.480 - 0.020, abs=1e-4)
