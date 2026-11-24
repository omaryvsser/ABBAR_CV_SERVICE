"""
Tests for Accuracy Optimization, VOR Sign Compensation, Stabilization,
Compact Regularized Model Selection, and Full 9-Region Validation Lifecycle.
"""

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

from src.config import AppConfig, CameraConfig, GazeConfig, ScreenConfig, VisualizationConfig
from src.core.calibration_profile import CalibrationProfile
from src.core.experiment_logger import (
    VALIDATION_NOT_COMPLETED_MSG,
    ExperimentLogger,
    compare_experiments,
)
from src.core.eye_features import (
    BinocularEyeFeatures,
    BoundingBox,
    EyeOpennessState,
    Point2D,
    SingleEyeFeatures,
    compensate_eye_features,
    extract_eye_features,
)
from src.core.face_landmarker import FaceLandmarkerDetectionResult
from src.core.gaze_baseline import GazeBaseline
from src.core.gaze_mapper import (
    CompactLinearGazeMapper,
    CompactPolynomialGazeMapper,
    compare_and_select_mapper,
)
from src.core.gaze_smoother import ScreenGazeStabilizer
from src.core.head_pose import HeadPose
from src.core.screen_calibration import (
    DEFAULT_9_TARGETS,
    CalibrationTarget,
    ScreenCalibrator,
    TargetPhase,
)
from src.core.screen_validator import (
    FULL_9_REGION_VALIDATION_TARGETS,
    ScreenValidator,
    ValidationPhase,
)
from src.pipeline import CVPipeline
from tests.conftest import create_synthetic_landmarks


def _make_binocular_eye(
    h_ratio: float,
    v_ratio: float,
    confidence: float = 0.92,
    r_valid: bool = True,
    l_valid: bool = True,
    r_h: Optional[float] = None,
    l_h: Optional[float] = None,
    r_v: Optional[float] = None,
    l_v: Optional[float] = None,
) -> BinocularEyeFeatures:
    """Create a realistic BinocularEyeFeatures instance for deterministic tests."""
    rh = h_ratio if r_h is None else r_h
    lh = h_ratio if l_h is None else l_h
    rv = v_ratio if r_v is None else r_v
    lv = v_ratio if l_v is None else l_v

    right_eye = SingleEyeFeatures(
        side="right",
        iris_center=Point2D(0.42, 0.35),
        bounding_box=BoundingBox(0.39, 0.33, 0.45, 0.37),
        horizontal_ratio=rh,
        vertical_ratio=rv,
        eye_aspect_ratio=0.28 if r_valid else 0.08,
        openness=0.85 if r_valid else 0.0,
        openness_state=EyeOpennessState.OPEN if r_valid else EyeOpennessState.CLOSED,
        is_blink=not r_valid,
        reliability=0.92 if r_valid else 0.0,
        is_valid=r_valid,
    )
    left_eye = SingleEyeFeatures(
        side="left",
        iris_center=Point2D(0.58, 0.35),
        bounding_box=BoundingBox(0.55, 0.33, 0.61, 0.37),
        horizontal_ratio=lh,
        vertical_ratio=lv,
        eye_aspect_ratio=0.28 if l_valid else 0.08,
        openness=0.85 if l_valid else 0.0,
        openness_state=EyeOpennessState.OPEN if l_valid else EyeOpennessState.CLOSED,
        is_blink=not l_valid,
        reliability=0.92 if l_valid else 0.0,
        is_valid=l_valid,
    )

    is_valid = r_valid or l_valid
    comb_h = float(h_ratio) if is_valid else None
    comb_v = float(v_ratio) if is_valid else None
    return BinocularEyeFeatures(
        left_eye=left_eye,
        right_eye=right_eye,
        combined_horizontal_ratio=comb_h,
        combined_vertical_ratio=comb_v,
        inter_iris_distance=0.16 if (r_valid and l_valid) else None,
        confidence=confidence if is_valid else 0.0,
        is_valid=is_valid,
    )


def _kinematic_features(
    target_x: float,
    target_y: float,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    noise_x: float = 0.0,
    noise_y: float = 0.0,
) -> np.ndarray:
    """Generate realistic 10-D eye-feature vector for a target screen point."""
    h_base = 0.50 + 0.14 * (target_x - 0.50) + noise_x
    v_base = 0.50 + 0.14 * (target_y - 0.50) + noise_y
    r_h = h_base + 0.015
    l_h = h_base - 0.015
    r_v = v_base + 0.005
    l_v = v_base - 0.005
    return np.array(
        [
            h_base,
            v_base,
            r_h,
            r_v,
            l_h,
            l_v,
            yaw_deg,
            pitch_deg,
            0.0,
            0.85,
        ],
        dtype=np.float64,
    )


def test_vor_yaw_compensation_cancels_head_turn_without_overinversion() -> None:
    """
    Turning head LEFT (+yaw) produces VOR counter-rotation (+raw_x);
    subtracting yaw_factor * delta_yaw restores centered gaze rather than doubling drift.
    """
    baseline = GazeBaseline(
        neutral_x=0.50,
        neutral_y=0.50,
        neutral_yaw=0.0,
        neutral_pitch=0.0,
        is_calibrated=True,
    )

    # Head turned left +10 deg while fixating center -> raw_x shifted +0.065
    eye_left_turn = _make_binocular_eye(0.565, 0.520)
    pose_left_turn = HeadPose(yaw=10.0, pitch=10.0, roll=0.0, is_valid=True)
    comp_left = compensate_eye_features(
        eye_left_turn,
        head_pose=pose_left_turn,
        baseline=baseline,
        enable_head_compensation=True,
        yaw_factor=0.0065,
        pitch_factor=0.0020,
    )
    assert comp_left is not None
    assert comp_left.combined_horizontal_ratio == pytest.approx(0.50, abs=1e-4)
    assert comp_left.combined_vertical_ratio == pytest.approx(0.50, abs=1e-4)

    # Head turned right -10 deg while fixating center -> raw_x shifted -0.065
    eye_right_turn = _make_binocular_eye(0.435, 0.480)
    pose_right_turn = HeadPose(yaw=-10.0, pitch=-10.0, roll=0.0, is_valid=True)
    comp_right = compensate_eye_features(
        eye_right_turn,
        head_pose=pose_right_turn,
        baseline=baseline,
        enable_head_compensation=True,
        yaw_factor=0.0065,
        pitch_factor=0.0020,
    )
    assert comp_right is not None
    assert comp_right.combined_horizontal_ratio == pytest.approx(0.50, abs=1e-4)
    assert comp_right.combined_vertical_ratio == pytest.approx(0.50, abs=1e-4)


def test_single_eye_resting_offset_alignment() -> None:
    """When one eye closes/blinks, the surviving eye aligns to calibrated binocular neutral."""
    baseline = GazeBaseline(
        neutral_x=0.50,
        neutral_y=0.50,
        neutral_yaw=0.0,
        neutral_pitch=0.0,
        is_calibrated=True,
        neutral_r_x=0.56,
        neutral_l_x=0.44,
        neutral_r_y=0.52,
        neutral_l_y=0.48,
    )

    # Left eye occluded/blinking; right eye at its resting neutral (0.56, 0.52)
    single_r = _make_binocular_eye(
        h_ratio=0.56,
        v_ratio=0.52,
        r_valid=True,
        l_valid=False,
        r_h=0.56,
        r_v=0.52,
    )
    aligned = compensate_eye_features(single_r, baseline=baseline, enable_head_compensation=False)
    assert aligned is not None
    assert aligned.combined_horizontal_ratio == pytest.approx(0.50, abs=1e-4)
    assert aligned.combined_vertical_ratio == pytest.approx(0.50, abs=1e-4)


def test_screen_gaze_stabilizer_saccade_jitter_and_outlier_rejection() -> None:
    """ScreenGazeStabilizer damps steady fixation jitter, suppresses low-confidence spikes, and tracks fast saccades."""
    stab = ScreenGazeStabilizer(
        min_cutoff=1.6,
        beta=18.0,
        d_cutoff=1.0,
        max_invalid_frames=3,
        jump_outlier_threshold=0.55,
        jump_confidence_floor=0.65,
    )

    t = 100.0
    sx0, sy0 = stab.update(0.50, 0.50, timestamp=t, is_valid=True, confidence=0.95)
    assert sx0 == pytest.approx(0.50, abs=1e-6)
    assert sy0 == pytest.approx(0.50, abs=1e-6)

    # 1. Steady fixation jitter (+/-0.025 alternating noise)
    raw_xs = []
    filt_xs = []
    for i in range(1, 12):
        t += 0.033
        noise = 0.025 if (i % 2 == 0) else -0.025
        rx = 0.50 + noise
        raw_xs.append(rx)
        sx, _ = stab.update(rx, 0.50 - noise * 0.5, timestamp=t, is_valid=True, confidence=0.90)
        assert sx is not None
        filt_xs.append(sx)
    assert np.std(filt_xs) < 0.60 * np.std(raw_xs)

    # 2. Low-confidence sudden outlier spike across screen (>0.55 jump at confidence=0.40) -> suppressed
    t += 0.033
    sx_spike, sy_spike = stab.update(0.98, 0.02, timestamp=t, is_valid=True, confidence=0.40)
    assert sx_spike is not None and abs(sx_spike - 0.50) < 0.10

    # 3. Genuine high-confidence saccade to (0.85, 0.20) -> converges quickly
    for _ in range(5):
        t += 0.033
        sx_sac, sy_sac = stab.update(0.85, 0.20, timestamp=t, is_valid=True, confidence=0.95)
    assert sx_sac is not None and abs(sx_sac - 0.85) < 0.05
    assert sy_sac is not None and abs(sy_sac - 0.20) < 0.05

    # 4. Blink holds position up to max_invalid_frames, then resets to None
    for _ in range(3):
        t += 0.033
        hx, hy = stab.update(None, None, timestamp=t, is_valid=False, confidence=0.0)
        assert hx is not None and hy is not None
    t += 0.033
    rx_none, ry_none = stab.update(None, None, timestamp=t, is_valid=False, confidence=0.0)
    assert rx_none is None and ry_none is None


def test_compact_models_prevent_overfitting_on_noisy_9_targets() -> None:
    """
    On 9 targets with realistic head pose & iris noise, compare_and_select_mapper sweeps
    compact linear/poly families and chooses a well-generalizing regularized model.
    """
    rng = np.random.default_rng(42)
    targets = [(0.1, 0.1), (0.5, 0.1), (0.9, 0.1), (0.1, 0.5), (0.5, 0.5), (0.9, 0.5), (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)]
    X_list = []
    y_list = []
    for tx, ty in targets:
        yaw_offset = rng.uniform(-5.0, 5.0)
        pitch_offset = rng.uniform(-3.0, 3.0)
        X_list.append(
            _kinematic_features(
                tx,
                ty,
                yaw_deg=yaw_offset,
                pitch_deg=pitch_offset,
                noise_x=float(rng.normal(0, 0.002)),
                noise_y=float(rng.normal(0, 0.002)),
            )
        )
        y_list.append([tx, ty])

    X_train = np.array(X_list)
    y_train = np.array(y_list)

    selected, report = compare_and_select_mapper(
        X_train, y_train, screen_width=1920, screen_height=1080, use_scaler=True
    )

    assert "compact_linear_metrics" in report
    assert "compact_polynomial_metrics" in report
    assert "candidate_benchmarks" in report
    assert report["selection_basis"] == "loocv"
    assert "mean_euclidean_pixels" in report[f"{selected.model_type}_metrics"]
    assert selected.model_type in ("compact_linear", "compact_polynomial", "linear", "polynomial")
    loocv_px = report["selected_eval_metrics"]["mean_euclidean_pixels"]
    assert loocv_px < 120.0


def test_calibrator_fixation_gating_failure_retry_and_metadata() -> None:
    """
    ScreenCalibrator rejects rapid saccade jumps/high dispersion during collection,
    transitions to TargetPhase.FAILED on timeout, supports retry_current_target(),
    and stores detailed per_target_stats in calibration_metadata.
    """
    cal = ScreenCalibrator(
        targets=DEFAULT_9_TARGETS[:2],
        settling_duration=0.02,
        collection_duration=0.04,
        min_samples_per_target=4,
        max_collection_timeout=0.07,
        fixation_max_std=0.015,
        max_interframe_step=0.030,
        allow_target_retry=True,
    )
    cal.start()
    time.sleep(0.03)
    # First update advances SETTLING -> COLLECTING
    cal.update(_make_binocular_eye(0.20, 0.50), HeadPose(0.0, 0.0, 0.0, is_valid=True))

    # Feed unstable oscillating saccades until max_collection_timeout (max(0.04+0.5, 0.07)=0.54s) triggers FAILED
    t_end = time.time() + 0.90
    osc = False
    while time.time() < t_end and cal.target_phase != TargetPhase.FAILED:
        h_val = 0.20 if osc else 0.80
        osc = not osc
        cal.update(_make_binocular_eye(h_val, 0.50), HeadPose(0.0, 0.0, 0.0, is_valid=True))
        time.sleep(0.01)

    assert cal.target_phase == TargetPhase.FAILED
    assert "TIMEOUT" in cal.fixation_status or "FAIL" in cal.fixation_status or "retry" in cal.fixation_feedback.lower()

    # Retry target 0 cleanly
    assert cal.retry_current_target() is True
    assert cal.target_phase == TargetPhase.SETTLING
    assert cal.retried_targets.get(DEFAULT_9_TARGETS[0].target_id, 0) == 1

    # Complete both targets with steady fixations
    for idx, t in enumerate(DEFAULT_9_TARGETS[:2]):
        time.sleep(0.03)
        h_target = 0.50 + 0.14 * (t.x - 0.50)
        v_target = 0.50 + 0.14 * (t.y - 0.50)
        t_dead = time.time() + 0.60
        prev_idx = cal.current_target_index
        while cal.is_active and cal.current_target_index == prev_idx and time.time() < t_dead:
            cal.update(_make_binocular_eye(h_target, v_target), HeadPose(0.0, 0.0, 0.0, is_valid=True))
            time.sleep(0.008)

    assert cal.is_completed
    assert "per_target_stats" in cal.calibration_metadata
    assert len(cal.calibration_metadata["per_target_stats"]) == 2
    first_stat = cal.calibration_metadata["per_target_stats"][0]
    assert first_stat["retried_count"] == 1
    assert first_stat["accepted_samples"] >= 4
    assert "raw_x_std" in first_stat["feature_stats"]


def test_full_9_region_validation_directions_regions_retry_and_skip() -> None:
    """
    Full 9-region independent validation grid covers TOP/MIDDLE/BOTTOM + LEFT/CENTER/RIGHT + UP/DOWN,
    supports retry_current_target() and skip_failed_target(), and populates direction & region accuracy maps.
    """
    assert len(FULL_9_REGION_VALIDATION_TARGETS) == 9

    # Train compact linear mapper cleanly on 9 calibration grid targets
    X_cal = np.array([_kinematic_features(t.x, t.y) for t in DEFAULT_9_TARGETS])
    y_cal = np.array([[t.x, t.y] for t in DEFAULT_9_TARGETS])
    mapper = CompactLinearGazeMapper(ridge_alpha=0.001, use_scaler=True).fit(X_cal, y_cal)

    validator = ScreenValidator(
        targets=FULL_9_REGION_VALIDATION_TARGETS,
        settling_duration=0.015,
        collection_duration=0.035,
        min_samples_per_target=3,
        max_collection_timeout=0.06,
        screen_width=1920,
        screen_height=1080,
        calibration_profile_id="prof_test",
        allow_target_retry=True,
    )
    validator.start(mapper=mapper)

    # Target 0: force failure via blinks -> retry -> succeed
    time.sleep(0.025)
    t_dead = time.time() + 0.40
    while validator.target_phase != ValidationPhase.FAILED and time.time() < t_dead:
        validator.update(_make_binocular_eye(0.50, 0.50, r_valid=False, l_valid=False))
        time.sleep(0.01)
    assert validator.target_phase == ValidationPhase.FAILED
    assert validator.retry_current_target() is True

    # Complete Target 0 cleanly after retry
    t0 = FULL_9_REGION_VALIDATION_TARGETS[0]
    time.sleep(0.02)
    while validator.is_active and validator.current_target_index == 0:
        validator.update(
            _make_binocular_eye(0.50 + 0.14 * (t0.x - 0.50), 0.50 + 0.14 * (t0.y - 0.50)),
            HeadPose(0.0, 0.0, 0.0, is_valid=True),
        )
        time.sleep(0.006)

    # Target 1: force failure via blinks -> skip_failed_target()
    assert validator.current_target_index == 1
    time.sleep(0.025)
    t_dead = time.time() + 0.40
    while validator.target_phase != ValidationPhase.FAILED and time.time() < t_dead:
        validator.update(_make_binocular_eye(0.50, 0.50, r_valid=False, l_valid=False))
        time.sleep(0.01)
    assert validator.target_phase == ValidationPhase.FAILED
    summary_mid = validator.skip_failed_target()
    assert summary_mid is None
    assert validator.current_target_index == 2

    # Targets 2..8: complete with clean fixations matching each region
    summary = None
    while validator.is_active:
        idx = validator.current_target_index
        tgt = FULL_9_REGION_VALIDATION_TARGETS[idx]
        time.sleep(0.02)
        while validator.is_active and validator.current_target_index == idx:
            summary = validator.update(
                _make_binocular_eye(
                    0.50 + 0.14 * (tgt.x - 0.50),
                    0.50 + 0.14 * (tgt.y - 0.50),
                ),
                HeadPose(0.0, 0.0, 0.0, is_valid=True),
            )
            time.sleep(0.006)

    assert summary is not None
    assert summary.validation_status == "COMPLETED"
    assert summary.total_trials == 9
    assert summary.successful_trials == 8
    assert summary.rejected_trials == 1
    assert summary.rejected_trial_pct == pytest.approx(100.0 / 9.0, abs=0.2)
    assert summary.target_selection_accuracy_pct > 90.0

    for d_key in ("LEFT", "RIGHT", "CENTER", "UP", "DOWN"):
        assert d_key in summary.direction_accuracies
        assert summary.direction_accuracies[d_key]["valid_trials"] >= 1

    for r_key in ("top_row", "middle_row", "bottom_row", "left_column", "center_column", "right_column", "edge_regions", "center_region"):
        assert r_key in summary.region_accuracies

    assert "weakest_direction" in summary.pattern_analysis
    assert "weakest_region" in summary.pattern_analysis


def test_experiment_report_statuses_and_comparison_protocol_warnings(tmp_path: Path) -> None:
    """
    Reports explicitly distinguish NOT_STARTED, INCOMPLETE, FAILED, and COMPLETED,
    and compare_experiments flags incompatible screen resolutions / protocols.
    """
    # 1. NOT_STARTED
    exp_ns = ExperimentLogger(base_dir=tmp_path, experiment_id="exp_not_started")
    exp_ns.log_experiment(
        calibration_samples=[],
        accuracy_summary=None,
        validation_results=[],
        metadata_extra={"status": "CALIBRATION_COMPLETED_ONLY"},
        export_excel=False,
        generate_plots=False,
    )
    txt_ns = (tmp_path / "exp_not_started" / "accuracy_report.txt").read_text(encoding="utf-8")
    md_ns = (tmp_path / "exp_not_started" / "accuracy_report.md").read_text(encoding="utf-8")
    assert "NOT_STARTED" in txt_ns
    assert VALIDATION_NOT_COMPLETED_MSG in txt_ns
    assert "NOT_STARTED" in md_ns

    # 2. INCOMPLETE
    exp_inc = ExperimentLogger(base_dir=tmp_path, experiment_id="exp_incomplete")
    exp_inc.log_experiment(
        calibration_samples=[],
        accuracy_summary=None,
        validation_results=[],
        metadata_extra={"status": "INTERRUPTED_BY_USER"},
        export_excel=False,
        generate_plots=False,
    )
    txt_inc = (tmp_path / "exp_incomplete" / "accuracy_report.txt").read_text(encoding="utf-8")
    assert "INCOMPLETE" in txt_inc

    # 3. FAILED (zero valid trials)
    exp_fail = ExperimentLogger(base_dir=tmp_path, experiment_id="exp_failed")
    exp_fail.log_experiment(
        calibration_samples=[],
        accuracy_summary=None,
        validation_results=[],
        metadata_extra={"status": "NO_VALID_TRIALS", "failure_reason": "Blinks on all 9 targets."},
        export_excel=False,
        generate_plots=False,
    )
    txt_fail = (tmp_path / "exp_failed" / "accuracy_report.txt").read_text(encoding="utf-8")
    assert "FAILED" in txt_fail

    # 4. Two completed experiments with different resolutions -> Compatibility Warning
    X_cal = np.array([_kinematic_features(t.x, t.y) for t in DEFAULT_9_TARGETS])
    y_cal = np.array([[t.x, t.y] for t in DEFAULT_9_TARGETS])
    mapper = CompactLinearGazeMapper(ridge_alpha=0.001, use_scaler=True).fit(X_cal, y_cal)

    def _run_completed(exp_id: str, w: int, h: int) -> None:
        val = ScreenValidator(
            targets=FULL_9_REGION_VALIDATION_TARGETS,
            settling_duration=0.01,
            collection_duration=0.025,
            min_samples_per_target=2,
            screen_width=w,
            screen_height=h,
        )
        val.start(mapper=mapper)
        s = None
        while val.is_active:
            tgt = val.current_target
            assert tgt is not None
            time.sleep(0.015)
            idx = val.current_target_index
            while val.is_active and val.current_target_index == idx:
                s = val.update(_make_binocular_eye(0.50 + 0.14 * (tgt.x - 0.50), 0.50 + 0.14 * (tgt.y - 0.50)))
                time.sleep(0.005)
        assert s is not None
        lg = ExperimentLogger(base_dir=tmp_path, experiment_id=exp_id)
        lg.log_experiment(
            calibration_samples=[],
            validation_results=val.get_trial_records(),
            accuracy_summary=s,
            metadata_extra={"screen_width": w, "screen_height": h, "status": s.validation_status, "validation_grid_mode": "full_9"},
            export_excel=False,
            generate_plots=False,
        )

    _run_completed("exp_1080p", 1920, 1080)
    _run_completed("exp_720p", 1280, 720)

    comp_md = compare_experiments([tmp_path / "exp_1080p", tmp_path / "exp_720p"], output_format="markdown")
    assert "Rej %" in comp_md
    assert "Weakest Dir / Region" in comp_md
    assert "Compatibility Warning" in comp_md


def test_pipeline_end_to_end_calibration_auto_queues_full_9_validation(tmp_path: Path) -> None:
    """
    Drive real CVPipeline.process_next_frame() through 9-point calibration followed
    automatically by 9-region independent validation, saving completed profile & reports
    into the SAME experiment directory without any mocked accuracy figures.
    """
    prof_dir = tmp_path / "profiles"
    exp_dir = tmp_path / "experiments"
    cfg = AppConfig(
        camera=CameraConfig(frame_width=640, frame_height=480, target_fps=120),
        gaze=GazeConfig(auto_calibrate_on_start=False),
        visualization=VisualizationConfig(enabled=False),
        screen=ScreenConfig(
            screen_width=1920,
            screen_height=1080,
            settling_duration=0.01,
            collection_duration=0.04,
            min_samples_per_target=3,
            calib_max_collection_timeout=1.5,
            validation_max_timeout=1.5,
            validation_grid_mode="full_9",
            profiles_dir=str(prof_dir),
            experiments_dir=str(exp_dir),
            export_excel=False,
        ),
    )

    pipeline = CVPipeline(config=cfg, use_mock_camera=True)
    pipeline.screen_validator.min_samples_per_target = 3

    def _fake_process_frame(frame, timestamp_ms: int = 0) -> FaceLandmarkerDetectionResult:
        tgt = None
        if pipeline.screen_calibrator.is_active:
            tgt = pipeline.screen_calibrator.current_target
        elif pipeline.screen_validator.is_active:
            tgt = pipeline.screen_validator.current_target

        tx = tgt.x if tgt is not None else 0.50
        ty = tgt.y if tgt is not None else 0.50
        # Camera coordinate frame: looking RIGHT (+tx) shifts iris toward viewer LEFT (-offset_x)
        offset_x = (0.50 - tx) * 0.036
        offset_y = (ty - 0.50) * 0.020
        lms = create_synthetic_landmarks(center_offset_x=offset_x, center_offset_y=offset_y, eye_open_height=0.030)
        return FaceLandmarkerDetectionResult(
            has_face=True,
            all_landmarks=lms,
            left_eye_landmarks=None,
            right_eye_landmarks=None,
            left_iris_landmarks=None,
            right_iris_landmarks=None,
            left_iris_center=None,
            right_iris_center=None,
            timestamp_ms=timestamp_ms,
            inference_time_ms=1.0,
        )

    pipeline.detector.process_frame = _fake_process_frame  # type: ignore[method-assign]
    pipeline.start()
    pipeline.start_screen_calibration()
    pipeline.queue_screen_validation()
    calib_exp_dir = pipeline.experiment_logger.experiment_dir

    # Run frames until calibration + auto-queued 9-region validation both complete
    t_limit = time.time() + 12.0
    while time.time() < t_limit:
        pipeline.process_next_frame()
        if pipeline.screen_validator.summary is not None:
            break
        time.sleep(0.008)

    try:
        summary = pipeline.screen_validator.summary
        assert summary is not None
        assert summary.validation_status == "COMPLETED"
        assert summary.total_trials == 9
        assert summary.successful_trials == 9
        assert summary.target_selection_accuracy_pct >= 85.0
        assert summary.mean_error_pixels < 120.0
        assert pipeline.last_experiment_dir == calib_exp_dir

        # Check saved files inside the unified experiment directory
        assert (calib_exp_dir / "calibration_samples.csv").exists()
        assert (calib_exp_dir / "validation_results.csv").exists()
        assert (calib_exp_dir / "accuracy_summary.csv").exists()
        report_txt = (calib_exp_dir / "accuracy_report.txt").read_text(encoding="utf-8")
        assert "COMPLETED" in report_txt
        assert "DIRECTION ACCURACY" in report_txt
        assert "SCREEN REGION ACCURACY" in report_txt

        # Verify persisted profile was updated with validation_metrics
        prof_files = list(prof_dir.glob("profile_*.json"))
        assert len(prof_files) == 1
        prof_json = json.loads(prof_files[0].read_text(encoding="utf-8"))
        assert prof_json["validation_metrics"] is not None
        assert prof_json["validation_metrics"]["validation_status"] == "COMPLETED"
        assert prof_json["calibration_metadata"] is not None
    finally:
        pipeline.close()
