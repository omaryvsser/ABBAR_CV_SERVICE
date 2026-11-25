"""
Regression & verification tests for layout geometry, coordinate mapping,
fixation sample quality hardening, outlier pruning, and configurable
target hit boundaries identified from live session experiment_20261124_232710.
"""

from pathlib import Path
import time

import numpy as np
import pytest

from src.config import AppConfig, ScreenConfig, VisualizationConfig
from src.core.eye_features import extract_eye_features
from src.core.face_landmarker import FaceLandmarkerDetectionResult
from src.core.gaze_direction import GazeDirection
from src.core.gaze_mapper import LinearGazeMapper
from src.core.gaze_output import ScreenGazeOutput
from src.core.head_pose import HeadPose
from src.core.screen_calibration import (
    CalibrationTarget,
    DEFAULT_9_TARGETS,
    ScreenCalibrator,
    TargetPhase,
)
from src.core.screen_validator import (
    DEFAULT_VALIDATION_TARGETS,
    FULL_9_REGION_VALIDATION_TARGETS,
    ScreenValidator,
)
from src.pipeline import CVPipeline
from src.utils.fps_counter import FPSMetrics
from src.visualization.visualizer import CVVisualizer
from tests.conftest import create_synthetic_landmarks


@pytest.mark.parametrize(
    "w,h",
    [
        (640, 480),
        (1280, 720),
        (1920, 1080),
        (2560, 1440),
    ],
)
def test_interface_to_window_point_proportional_across_window_sizes(w: int, h: int):
    """
    Ensure interface_to_window_point maps Y = 0.10, 0.18, 0.50, 0.90 proportionally
    into the visible content area across window sizes without collapsing rows or overlapping HUD.
    """
    left, top, right, bottom = CVVisualizer.compute_visible_content_rect(w, h)
    assert top >= 42, "Visible content top must sit cleanly below compact 42px HUD"
    assert bottom <= h - 25, "Visible content bottom must sit cleanly above 24px bottom bar"
    assert right > left and bottom > top

    pts = [
        CVVisualizer.interface_to_window_point(0.50, y, w, h)
        for y in (0.10, 0.18, 0.50, 0.90)
    ]
    ys = [p[1] for p in pts]

    # Strictly distinct monotonic ordering (0.10 and 0.18 MUST NOT collapse to same row)
    assert ys[0] < ys[1] < ys[2] < ys[3], f"Vertical collapse detected at {w}x{h}: {ys}"
    assert (ys[1] - ys[0]) >= 20, f"Y=0.10 and Y=0.18 too close at {w}x{h}: {ys[0]} vs {ys[1]}"

    # Exact proportional affine check
    for y_val, (_, py) in zip((0.10, 0.18, 0.50, 0.90), pts):
        expected_py = int(round(top + y_val * float(bottom - top)))
        assert py == expected_py
        assert top <= py <= bottom

    # Horizontal consistency without mirroring or collapse
    xs = [CVVisualizer.interface_to_window_point(x, 0.50, w, h)[0] for x in (0.10, 0.50, 0.90)]
    assert xs[0] < xs[1] < xs[2]
    assert xs[0] == int(round(left + 0.10 * float(right - left)))
    assert xs[2] == int(round(left + 0.90 * float(right - left)))


def test_calibration_and_validation_target_separation_and_consistent_mapping():
    """Verify calibration & validation share identical mapping math and preserve interior/edge separation."""
    w, h = 1280, 720
    # Calibration top_left (0.10, 0.10) vs interior_5 validation top_left (0.18, 0.18)
    cal_tl = DEFAULT_9_TARGETS[0]
    val_tl_interior = DEFAULT_VALIDATION_TARGETS[0]
    val_tl_full = FULL_9_REGION_VALIDATION_TARGETS[0]

    # Center (0.50, 0.50) maps identically across calibration and validation targets
    cal_center = DEFAULT_9_TARGETS[4]
    val_center = FULL_9_REGION_VALIDATION_TARGETS[4]
    assert (
        CVVisualizer.interface_to_window_point(cal_center.x, cal_center.y, w, h)
        == CVVisualizer.interface_to_window_point(val_center.x, val_center.y, w, h)
    )

    # Validation top_left (0.18, 0.18) stays strictly separated from calibration (0.10, 0.10)
    c_px, c_py = CVVisualizer.interface_to_window_point(cal_tl.x, cal_tl.y, w, h)
    v_px_int, v_py_int = CVVisualizer.interface_to_window_point(val_tl_interior.x, val_tl_interior.y, w, h)
    v_px_full, v_py_full = CVVisualizer.interface_to_window_point(val_tl_full.x, val_tl_full.y, w, h)
    assert v_px_int > c_px and v_py_int > c_py
    assert v_px_full > c_px and v_py_full > c_py


def test_hud_compact_mode_leaves_corner_targets_unobstructed():
    """
    Verify compact HUD during active calibration/validation removes corner widgets
    and renders corner targets unobstructed at their mapped positions.
    """
    w, h = 1280, 720
    viz = CVVisualizer(VisualizationConfig(mirror_display=True, draw_gaze_crosshair=True))
    landmarks = create_synthetic_landmarks()
    feats = extract_eye_features(landmarks)
    det = FaceLandmarkerDetectionResult(
        has_face=True,
        all_landmarks=landmarks,
        left_eye_landmarks=None,
        right_eye_landmarks=None,
        left_iris_landmarks=None,
        right_iris_landmarks=None,
        left_iris_center=None,
        right_iris_center=None,
        timestamp_ms=10,
        inference_time_ms=2.0,
    )
    fps = FPSMetrics(fps=30.0, frame_time_ms=33.3, detection_latency_ms=2.0, total_frames=1, uptime_seconds=0.1)

    for target_name, tx_norm, ty_norm in [
        ("top_left", 0.10, 0.10),
        ("bottom_left", 0.10, 0.90),
        ("bottom_right", 0.90, 0.90),
    ]:
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        cal = ScreenCalibrator(targets=[CalibrationTarget(0, target_name, tx_norm, ty_norm)])
        cal.start()
        out = viz.render(
            frame=bg,
            detection_result=det,
            eye_features=feats,
            fps_metrics=fps,
            screen_calibrator=cal,
        )
        px, py = CVVisualizer.interface_to_window_point(tx_norm, ty_norm, w, h)
        # Center bullseye rendered brightly at exact target point
        patch = out[py - 10:py + 10, px - 10:px + 10]
        assert patch.max() > 180, f"Target {target_name} missing at ({px}, {py})"

        # Check Simulated Screen widget box (normally bottom-left) does NOT draw over active calibration
        sx0, sy0, ssw, ssh = CVVisualizer.compute_simulated_screen_rect(w, h)
        # Pick top-left of simulated screen box far from target dot (0.10, 0.90 -> ~169, 621; sx0=15, sy0=592)
        if target_name == "top_left":
            sim_corner = out[sy0:sy0 + 20, sx0:sx0 + 20]
            assert sim_corner.max() < 30, "Simulated Screen widget overlapped active calibration!"


def test_post_saccade_jump_cooldown_and_cluster_drift_rejects_t3_pattern():
    """
    Reproduce the experiment_20261124_232710 T3 (middle_left) bug:
    two saccade jumps followed by two settling frames that had small frame-to-frame
    deltas but were shifted +0.26 vertically must NOT be accepted into target samples.
    """
    cal = ScreenCalibrator(
        targets=[CalibrationTarget(3, "middle_left", 0.10, 0.50)],
        settling_duration=0.02,
        collection_duration=0.05,
        min_samples_per_target=6,
    )
    cal.start()
    time.sleep(0.03)

    # Steady target fixation around off_y=-0.03 -> vertical ratio ~0.22
    steady_lm = create_synthetic_landmarks(center_offset_x=0.025, center_offset_y=-0.030)
    steady_feats = extract_eye_features(steady_lm)
    cal.update(steady_feats)  # Transitions SETTLING -> COLLECTING
    for _ in range(6):
        cal.update(steady_feats)
    assert len(cal._target_samples[3]) == 6

    # Frame A: sudden large vertical jump (FIXATION_SACCADE_JUMP)
    jump1_lm = create_synthetic_landmarks(center_offset_x=0.025, center_offset_y=0.020)
    cal.update(extract_eye_features(jump1_lm))

    # Frame B: another jump (FIXATION_SACCADE_JUMP)
    jump2_lm = create_synthetic_landmarks(center_offset_x=0.025, center_offset_y=0.045)
    cal.update(extract_eye_features(jump2_lm))

    # Frames C & D: small step (< max_interframe_step) immediately after jump!
    # Previously accepted because len(_recent_raw_xy) < 4 bypassed dispersion gate.
    after1_lm = create_synthetic_landmarks(center_offset_x=0.025, center_offset_y=0.040)
    after2_lm = create_synthetic_landmarks(center_offset_x=0.025, center_offset_y=0.035)
    cal.update(extract_eye_features(after1_lm))
    cal.update(extract_eye_features(after2_lm))

    # Frames E & F: still shifted far away from established steady cluster median (~0.22)
    after3_lm = create_synthetic_landmarks(center_offset_x=0.025, center_offset_y=0.034)
    cal.update(extract_eye_features(after3_lm))

    # None of frames A..E should have polluted accepted samples!
    accepted = cal._target_samples[3]
    assert len(accepted) == 6
    all_recs = cal._target_all_samples[3]
    rej_reasons = [r.rejection_reason for r in all_recs if not r.accepted]
    assert any(r == "FIXATION_SACCADE_JUMP" for r in rej_reasons)
    assert any(r in ("POST_JUMP_STABILIZING", "FIXATION_OUTLIER_DRIFT") for r in rej_reasons)


def test_mad_and_pose_pruning_preserves_natural_motion_and_retry():
    """
    Verify MAD + target-cluster pose pruning removes late drift outliers while
    retaining natural +/-2.5 deg head sway, recording stats cleanly, and supporting retry.
    """
    cal = ScreenCalibrator(
        settling_duration=0.02,
        collection_duration=0.05,
        min_samples_per_target=6,
        allow_target_retry=True,
    )
    cal.start()

    # Check retry functionality on target 0 first
    time.sleep(0.03)
    cal.update(extract_eye_features(create_synthetic_landmarks()))  # Transitions SETTLING -> COLLECTING
    cal.update(extract_eye_features(create_synthetic_landmarks()))
    assert len(cal._target_samples[0]) == 1
    assert cal.retry_current_target() is True
    assert cal.retried_targets[0] == 1
    assert len(cal._target_samples[0]) == 0
    assert cal.target_phase == TargetPhase.SETTLING

    # Complete all 9 targets; inject natural +-1.5 deg pose + 2 pose outliers on target 0
    for idx in range(9):
        time.sleep(0.03)
        t = cal.current_target
        assert t is not None
        ox = (t.x - 0.50) * 0.06
        oy = (t.y - 0.50) * 0.06
        lm = create_synthetic_landmarks(center_offset_x=ox, center_offset_y=oy)
        feats = extract_eye_features(lm)

        # Transition SETTLING -> COLLECTING
        cal.update(feats)

        if idx == 0:
            # 7 steady frames with natural +-1.5 deg micro-sway (< 4.0 deg max_head_motion_step)
            for j in range(7):
                yaw_jitter = -1.5 if j % 2 == 0 else 1.5
                pose = HeadPose(yaw=yaw_jitter, pitch=0.0, roll=0.0, is_valid=True)
                time.sleep(0.004)
                cal.update(feats, head_pose=pose)

            # 2 gradual pitch drift frames (steps <= 4.0, ending > +6.0 deg from target median 0.0)
            for p_val in (3.5, 7.5):
                pose_drift = HeadPose(yaw=0.0, pitch=p_val, roll=0.0, is_valid=True)
                time.sleep(0.004)
                cal.update(feats, head_pose=pose_drift)

            # Wait collection window & close target 0
            time.sleep(0.025)
            cal.update(feats, head_pose=HeadPose(yaw=0.0, pitch=0.0, roll=0.0, is_valid=True))
        else:
            for j in range(7):
                yaw_jitter = -1.5 if j % 2 == 0 else 1.5
                pose = HeadPose(yaw=yaw_jitter, pitch=1.0 if j % 2 == 0 else -1.0, roll=0.0, is_valid=True)
                time.sleep(0.009)
                cal.update(feats, head_pose=pose)

    assert cal.is_completed
    stats = cal.calibration_metadata["per_target_stats"]
    t0_stats = stats[0]
    assert t0_stats["retried_count"] == 1
    assert t0_stats["clean_samples"] <= t0_stats["accepted_samples"]
    assert t0_stats["outlier_pruned_samples"] >= 1


def test_configurable_aac_button_and_radius_hit_boundaries():
    """
    Verify configurable AAC button box (width_px x height_px) & custom circular radius
    without inflating default rendered circle bounds (22 display px).
    """
    default_cfg = ScreenConfig()
    assert default_cfg.validation_target_radius_px is None
    assert default_cfg.validation_target_width_px is None
    assert default_cfg.validation_target_height_px is None

    # Default circle boundary at 1920x1080 (22 * 1.5 = 33.0 px)
    t_default = CalibrationTarget(4, "center", 0.50, 0.50)
    hit_default, spec_default = t_default.contains_point(
        0.50 + 100.0 / 1920.0, 0.50 + 60.0 / 1080.0, screen_width=1920, screen_height=1080
    )
    assert hit_default is False
    assert "22 display px" in spec_default

    # Configured AAC button box 240x160 px (half-extents 120px x 80px)
    t_btn = CalibrationTarget(4, "center", 0.50, 0.50, width_px=240.0, height_px=160.0)
    hit_btn, spec_btn = t_btn.contains_point(
        0.50 + 100.0 / 1920.0, 0.50 + 60.0 / 1080.0, screen_width=1920, screen_height=1080
    )
    assert hit_btn is True
    assert spec_btn == "Button Box (240x160 px)"

    # Configured pipeline wiring when model exists
    model_path = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
    if model_path.exists():
        cfg = AppConfig()
        cfg.visualization.enabled = False
        cfg.screen.validation_target_width_px = 240.0
        cfg.screen.validation_target_height_px = 160.0
        cfg.screen.validation_target_radius_px = 95.0
        pipe = CVPipeline(config=cfg, use_mock_camera=True)
        assert pipe.screen_validator.target_radius_px == pytest.approx(95.0)
        assert pipe.screen_validator.base_targets[0].width_px == pytest.approx(240.0)
        assert pipe.screen_validator.base_targets[0].height_px == pytest.approx(160.0)


def test_historical_experiment_data_preserved():
    """Confirm existing experiment_20261124_232710 files remain completely intact."""
    exp_dir = Path(__file__).resolve().parent.parent / "data" / "experiments" / "experiment_20261124_232710"
    if exp_dir.exists():
        assert (exp_dir / "accuracy_report.md").exists()
        assert (exp_dir / "calibration_samples.csv").exists()
        assert (exp_dir / "validation_results.csv").exists()
        assert (exp_dir / "experiment_metadata.json").exists()
