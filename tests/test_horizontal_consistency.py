"""
Tests for end-to-end horizontal direction consistency across webcam preview,
ocular feature extraction, gaze direction classification, 9-point calibration,
screen prediction, Simulated Screen rendering, and legacy profile safety.

Canonical Interface & Screen Coordinate Convention:
- gaze_x = 0.0 -> Left side of the patient interface
- gaze_x = 1.0 -> Right side of the patient interface
- gaze_y = 0.0 -> Top
- gaze_y = 1.0 -> Bottom

Display mirroring (`VisualizationConfig.mirror_display`) applies ONLY to
camera frames and facial landmark overlays; it NEVER flips target coordinates,
gaze predictions, or the Simulated Screen cursor.
"""

from pathlib import Path
import time

import numpy as np
import pytest

from src.config import AppConfig, VisualizationConfig
from src.core.calibration_profile import (
    COORDINATE_CONVENTION_VERSION,
    LEGACY_MIRRORED_CONVENTION,
    CalibrationProfile,
    IncompatibleCalibrationProfileError,
    ProfileManager,
)
from src.core.eye_features import extract_eye_features
from src.core.face_landmarker import FaceLandmarkerDetectionResult
from src.core.gaze_baseline import GazeBaseline
from src.core.gaze_direction import GazeDirection, GazeDirectionClassifier
from src.core.gaze_mapper import extract_gaze_features
from src.core.gaze_output import ScreenGazeOutput
from src.core.screen_calibration import CalibrationTarget, ScreenCalibrator
from src.core.screen_validator import ScreenValidator
from src.pipeline import CVPipeline
from src.utils.fps_counter import FPSMetrics
from src.visualization.visualizer import CVVisualizer
from tests.conftest import create_synthetic_landmarks


def _make_detection_and_fps(landmarks):
    det = FaceLandmarkerDetectionResult(
        has_face=True,
        all_landmarks=landmarks,
        left_eye_landmarks=None,
        right_eye_landmarks=None,
        left_iris_landmarks=None,
        right_iris_landmarks=None,
        left_iris_center=None,
        right_iris_center=None,
        timestamp_ms=100,
        inference_time_ms=2.0,
    )
    fps = FPSMetrics(
        fps=30.0,
        frame_time_ms=33.3,
        detection_latency_ms=2.0,
        total_frames=1,
        uptime_seconds=0.1,
    )
    return det, fps


def _train_anatomical_9_point_mapper():
    """Train a 9-point calibrator with anatomically consistent iris shifts."""
    calibrator = ScreenCalibrator(
        settling_duration=0.025,
        collection_duration=0.045,
        min_samples_per_target=3,
        screen_width=1920,
        screen_height=1080,
    )
    calibrator.start()

    fitted_mapper = None
    deadline = time.time() + 5.0
    while not calibrator.is_completed and time.time() < deadline:
        target = calibrator.current_target
        assert target is not None

        # Anatomically, looking toward a right-side target (target.x > 0.5) moves
        # the iris toward camera-left (negative offset in raw un-mirrored camera X).
        off_x = -(target.x - 0.50) * 0.05
        off_y = (target.y - 0.50) * 0.03
        lm = create_synthetic_landmarks(center_offset_x=off_x, center_offset_y=off_y)
        feats = extract_eye_features(lm)

        res = calibrator.update(feats)
        if res is not None:
            fitted_mapper = res
            break
        time.sleep(0.012)

    assert calibrator.is_completed, "9-point calibration did not complete within deadline"
    assert fitted_mapper is not None
    assert calibrator.profile is not None
    return calibrator, fitted_mapper


def test_anatomical_left_center_right_raw_to_classification():
    """Verify Left, Center, and Right gaze map consistently from eye features to classification."""
    classifier = GazeDirectionClassifier(persistence_frames=1)

    left_feats = extract_eye_features(create_synthetic_landmarks(center_offset_x=0.025, center_offset_y=0.0))
    center_feats = extract_eye_features(create_synthetic_landmarks(center_offset_x=0.0, center_offset_y=0.0))
    right_feats = extract_eye_features(create_synthetic_landmarks(center_offset_x=-0.025, center_offset_y=0.0))

    assert left_feats.combined_horizontal_ratio < 0.42
    assert 0.42 <= center_feats.combined_horizontal_ratio <= 0.58
    assert right_feats.combined_horizontal_ratio > 0.58

    assert classifier.classify(left_feats.combined_horizontal_ratio, left_feats.combined_vertical_ratio) == GazeDirection.LEFT
    assert classifier.classify(center_feats.combined_horizontal_ratio, center_feats.combined_vertical_ratio) == GazeDirection.CENTER
    assert classifier.classify(right_feats.combined_horizontal_ratio, right_feats.combined_vertical_ratio) == GazeDirection.RIGHT


def test_end_to_end_calibration_left_center_right_to_simulated_screen():
    """
    Verify looking Left/Center/Right yields gaze_x near 0.1/0.5/0.9 and places
    the Simulated Screen cursor on Left/Center/Right accordingly.
    """
    calibrator, mapper = _train_anatomical_9_point_mapper()
    profile = calibrator.profile
    assert profile is not None
    assert profile.coordinate_convention == COORDINATE_CONVENTION_VERSION
    assert profile.model_version == "3.1"

    w, h = 1280, 720
    x0, y0, sw, sh = CVVisualizer.compute_simulated_screen_rect(w, h)
    mid_x = x0 + sw / 2.0
    mid_y = y0 + sh / 2.0

    cases = [
        ("LEFT", 0.020, 0.0, lambda gx, gy: gx < 0.30, lambda cx, cy: cx < mid_x - 25),
        ("CENTER", 0.0, 0.0, lambda gx, gy: 0.40 <= gx <= 0.60, lambda cx, cy: abs(cx - mid_x) <= 18),
        ("RIGHT", -0.020, 0.0, lambda gx, gy: gx > 0.70, lambda cx, cy: cx > mid_x + 25),
        ("TOP", 0.0, -0.011, lambda gx, gy: gy < 0.35, lambda cx, cy: cy < mid_y - 12),
        ("BOTTOM", 0.0, 0.011, lambda gx, gy: gy > 0.65, lambda cx, cy: cy > mid_y + 12),
    ]

    for label, off_x, off_y, pred_check, cursor_check in cases:
        feats = extract_eye_features(create_synthetic_landmarks(center_offset_x=off_x, center_offset_y=off_y))
        vec = extract_gaze_features(feats)
        assert vec is not None
        pred = mapper.predict(vec)
        gx, gy = float(pred[0]), float(pred[1])
        sim_x, sim_y = CVVisualizer.compute_simulated_screen_cursor(gx, gy, w, h)
        assert pred_check(gx, gy), f"{label} prediction mismatch: ({gx:.3f}, {gy:.3f})"
        assert cursor_check(sim_x, sim_y), f"{label} simulated cursor mismatch: ({sim_x}, {sim_y})"


def test_visualizer_mirror_and_non_mirror_preserve_interface_and_simulated_screen():
    """
    Verify mirror_display only mirrors camera image pixels (and camera landmarks),
    while target overlays, gaze markers, and Simulated Screen cursors stay in
    fixed Left=0 -> Right=1 screen space in BOTH display modes.
    """
    w, h = 1280, 720
    landmarks = create_synthetic_landmarks(center_offset_x=-0.02, center_offset_y=0.0)
    feats = extract_eye_features(landmarks)
    det, fps = _make_detection_and_fps(landmarks)

    # Asymmetric camera frame: bright blue block strictly on camera-left [320:420]
    # away from validation dashboard (<280) and eye/target overlays.
    raw_frame = np.zeros((h, w, 3), dtype=np.uint8)
    raw_frame[310:410, 320:420] = (255, 20, 20)
    raw_before = raw_frame.copy()

    # Right-side validation target and right-side prediction (gaze_x ~ 0.90)
    val = ScreenValidator(
        targets=[CalibrationTarget(5, "middle_right", 0.90, 0.50)],
        screen_width=1920,
        screen_height=1080,
    )
    from src.core.gaze_mapper import LinearGazeMapper
    dummy_mapper = LinearGazeMapper()
    dummy_mapper.fit(np.random.randn(4, 10), np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]))
    val.start(dummy_mapper)
    screen_right = ScreenGazeOutput(
        gaze_x=0.90, gaze_y=0.50, pixel_x=1728, pixel_y=540,
        is_calibrated=True, tracking_quality="GOOD", confidence=0.92,
    )

    viz_mirrored = CVVisualizer(VisualizationConfig(mirror_display=True, draw_gaze_crosshair=False))
    viz_unmirrored = CVVisualizer(VisualizationConfig(mirror_display=False, draw_gaze_crosshair=False))

    out_mirrored = viz_mirrored.render(
        frame=raw_frame,
        detection_result=det,
        eye_features=feats,
        fps_metrics=fps,
        gaze_direction=GazeDirection.RIGHT,
        screen_gaze=screen_right,
        screen_validator=val,
    )
    out_unmirrored = viz_unmirrored.render(
        frame=raw_frame,
        detection_result=det,
        eye_features=feats,
        fps_metrics=fps,
        gaze_direction=GazeDirection.RIGHT,
        screen_gaze=screen_right,
        screen_validator=val,
    )

    # Raw input frame must never be mutated
    assert np.array_equal(raw_frame, raw_before)

    # Camera background flips horizontally when mirror_display=True, not when False
    assert out_mirrored[340:380, 880:940, 0].mean() > 200
    assert out_mirrored[340:380, 340:400, 0].mean() < 30
    assert out_unmirrored[340:380, 340:400, 0].mean() > 200
    assert out_unmirrored[340:380, 880:940, 0].mean() < 30

    # Target overlay (0.90, 0.50) must be painted on window RIGHT (~1152, 360) in BOTH modes,
    # NOT on the mirrored wrong LEFT (~128, 360).
    tx_expected, ty_expected = CVVisualizer.interface_to_window_point(
        0.90, 0.50, w, h, margin_x=30, margin_top=130, margin_bottom=50
    )
    assert tx_expected == int(round(0.90 * w))
    tx_wrong = int(round(0.10 * w))

    for out_img, mode in [(out_mirrored, "mirrored"), (out_unmirrored, "non-mirrored")]:
        right_patch = out_img[ty_expected - 15:ty_expected + 15, tx_expected - 15:tx_expected + 15]
        wrong_left_patch = out_img[ty_expected - 15:ty_expected + 15, tx_wrong - 15:tx_wrong + 15]
        assert right_patch.max() > 180, f"Target bullseye missing on RIGHT in {mode} mode"
        assert wrong_left_patch.max() < 20, f"Target bullseye erroneously rendered on LEFT in {mode} mode"

    # Simulated Screen widget pixels must match across mirrored & non-mirrored modes
    # and place cursor pixels strictly in Left half for gaze_x=0.15 and Right half for 0.85.
    uniform_bg = np.full((h, w, 3), 10, dtype=np.uint8)
    x0, y0, sw, sh = CVVisualizer.compute_simulated_screen_rect(w, h)
    half_x = x0 + sw // 2

    for gaze_x, expect_left in [(0.15, True), (0.85, False)]:
        sg = ScreenGazeOutput(
            gaze_x=gaze_x, gaze_y=0.50,
            pixel_x=int(gaze_x * 1920), pixel_y=540,
            is_calibrated=True, tracking_quality="GOOD", confidence=0.95,
        )
        m_img = viz_mirrored.render(uniform_bg, det, feats, fps, screen_gaze=sg)
        u_img = viz_unmirrored.render(uniform_bg, det, feats, fps, screen_gaze=sg)

        widget_m = m_img[y0:y0 + sh, x0:x0 + sw]
        widget_u = u_img[y0:y0 + sh, x0:x0 + sw]
        assert np.array_equal(widget_m, widget_u)

        # Bright cursor green/yellow channel inside active screen region (excluding header/footer/L-R edges)
        left_energy = int(m_img[y0 + 18:y0 + sh - 16, x0 + 14:half_x - 4, 1].max())
        right_energy = int(m_img[y0 + 18:y0 + sh - 16, half_x + 4:x0 + sw - 14, 1].max())
        if expect_left:
            assert left_energy > 200 and right_energy < 100
        else:
            assert right_energy > 200 and left_energy < 100


def test_visualizer_debug_indicators_show_camera_mode_and_coordinates():
    """Verify explicit visual debug indicators render MIRRORED vs NON-MIRRORED and coordinates."""
    w, h = 1280, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    landmarks = create_synthetic_landmarks(center_offset_x=-0.02, center_offset_y=0.0)
    feats = extract_eye_features(landmarks)
    det, fps = _make_detection_and_fps(landmarks)
    sg = ScreenGazeOutput(
        gaze_x=0.82, gaze_y=0.35, pixel_x=1574, pixel_y=378,
        is_calibrated=True, tracking_quality="GOOD", confidence=0.93,
    )

    mirrored_img = CVVisualizer(VisualizationConfig(mirror_display=True, draw_fps=False)).render(
        frame, det, feats, fps, gaze_direction=GazeDirection.RIGHT, screen_gaze=sg,
    )
    unmirrored_img = CVVisualizer(VisualizationConfig(mirror_display=False, draw_fps=False)).render(
        frame, det, feats, fps, gaze_direction=GazeDirection.RIGHT, screen_gaze=sg,
    )

    # Row 1 camera display mode banner at x in [555, 760], y in [12, 34]
    cam_patch_m = mirrored_img[12:34, 555:760]
    cam_patch_u = unmirrored_img[12:34, 555:760]
    assert cam_patch_m.max() > 150
    assert cam_patch_u.max() > 150
    assert not np.array_equal(cam_patch_m, cam_patch_u)

    # Row 4 screen gaze & SimCursor debug text region at y in [94, 116], x in [15, 750]
    assert mirrored_img[94:116, 15:750, 1].max() > 150


def test_legacy_mirrored_profile_explicitly_rejected(tmp_path):
    """
    Verify legacy calibration profiles lacking `coordinate_convention` are detected as
    incompatible and NEVER silently loaded into active gaze prediction.
    """
    from src.core.gaze_mapper import LinearGazeMapper

    mapper = LinearGazeMapper()
    mapper.fit(np.random.randn(5, 10), np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.5], [0.1, 0.9], [0.9, 0.9]]))
    legacy_dict = {
        "profile_id": "legacy_inverted_user",
        "user_id": "default_user",
        "screen_width": 1920,
        "screen_height": 1080,
        "camera_index": 0,
        "model_type": "linear",
        "model_data": mapper.to_dict(),
        "normalization_method": "StandardScaler",
        "normalization_params": mapper.scaler.to_dict(),
        "model_version": "3.0",
    }

    parsed = CalibrationProfile.from_dict(legacy_dict)
    assert parsed.coordinate_convention == LEGACY_MIRRORED_CONVENTION

    pm = ProfileManager(profiles_dir=tmp_path)
    compat, msg = pm.validate_compatibility(parsed, 1920, 1080)
    assert compat is False
    assert "Incompatible coordinate convention" in msg

    with pytest.raises(IncompatibleCalibrationProfileError, match="Recalibration|fresh 9-point"):
        pm.extract_mapper(parsed)

    # Also test against a real legacy workspace profile read-only if present
    workspace_legacy = Path(__file__).resolve().parent.parent / "data" / "calibration" / "profile_user_default_1795466581.json"
    if workspace_legacy.exists():
        loaded_real = pm.load_profile(workspace_legacy)
        assert loaded_real.coordinate_convention == LEGACY_MIRRORED_CONVENTION
        with pytest.raises(IncompatibleCalibrationProfileError):
            pm.extract_mapper(loaded_real)

    # Verify CVPipeline explicitly refuses to activate the legacy profile
    model_path = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
    if not model_path.exists():
        pytest.skip("Model bundle required for CVPipeline check.")

    legacy_file = pm.save_profile(parsed, filepath=tmp_path / "profile_legacy.json")
    cfg = AppConfig()
    cfg.landmarker.model_path = str(model_path)
    cfg.gaze.auto_calibrate_on_start = False

    pipeline = CVPipeline(config=cfg, use_mock_camera=True)
    with pytest.raises(IncompatibleCalibrationProfileError):
        pipeline.load_calibration_profile(legacy_file)
    assert pipeline.active_mapper is None
    assert pipeline.active_profile is None
    pipeline.close()

    # Verify startup config with legacy active_profile_path warns cleanly without activating inverted mapper
    cfg_startup = AppConfig()
    cfg_startup.landmarker.model_path = str(model_path)
    cfg_startup.gaze.auto_calibrate_on_start = False
    cfg_startup.screen.active_profile_path = str(legacy_file)
    pipeline_startup = CVPipeline(config=cfg_startup, use_mock_camera=True)
    assert pipeline_startup.active_mapper is None
    assert pipeline_startup.active_profile is None
    pipeline_startup.close()


def test_pipeline_compatible_profile_load_and_mock_frame_modes(tmp_path):
    """Verify compatible profile loads cleanly and mock pipeline runs in both mirror modes."""
    model_path = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
    if not model_path.exists():
        pytest.skip("Model bundle required for CVPipeline check.")

    calibrator, _ = _train_anatomical_9_point_mapper()
    pm = ProfileManager(profiles_dir=tmp_path)
    saved_path = pm.save_profile(calibrator.profile, filepath=tmp_path / "profile_v31.json")

    for mirror in [True, False]:
        cfg = AppConfig()
        cfg.landmarker.model_path = str(model_path)
        cfg.gaze.auto_calibrate_on_start = False
        cfg.visualization.mirror_display = mirror

        pipeline = CVPipeline(config=cfg, use_mock_camera=True)
        loaded = pipeline.load_calibration_profile(saved_path)
        assert loaded.coordinate_convention == COORDINATE_CONVENTION_VERSION
        assert pipeline.active_mapper is not None

        pipeline._is_running = True
        out = pipeline.process_next_frame()
        assert out is not None
        assert out.annotated_frame is not None
        assert out.screen_gaze is not None
        assert out.screen_gaze.is_calibrated is True
        pipeline.close()
