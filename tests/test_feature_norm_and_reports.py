"""
Automated tests for Phase 3 Improvements:
- Feature Normalization (StandardScaler, zero leakage, missing/extreme handling, persistence)
- Target Selection Accuracy using actual target boundaries (circle & rectangle)
- Error % of screen diagonal & configurable thresholds
- User-friendly Markdown & HTML experiment reports
- Validation integrity (missing validation never produces fake accuracy)
- Experiment comparison (strict separation of training vs validation metrics, trends, best selection)
- Visualizer live validation overlay (without AttributeError)
- Pipeline real-time inference using saved scaler + sequential calibration->validation transition
"""

import json
import math
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from src.config import AppConfig
from src.core.calibration_profile import CalibrationProfile, ProfileManager
from src.core.experiment_logger import (
    ExperimentLogger,
    VALIDATION_NOT_COMPLETED_MSG,
    build_experiment_comparison_summary,
    compare_experiments,
)
from src.core.eye_features import extract_eye_features
from src.core.face_landmarker import FaceLandmarkerDetectionResult
from src.core.gaze_mapper import (
    FEATURE_NAMES,
    LinearGazeMapper,
    PolynomialGazeMapper,
    StandardScaler,
    compare_and_select_mapper,
    create_mapper,
)
from src.core.gaze_output import ScreenGazeOutput
from src.core.screen_calibration import CalibrationSampleRecord, CalibrationTarget
from src.core.screen_validator import (
    AccuracySummary,
    ScreenValidator,
    ValidationTrialRecord,
)
from src.pipeline import CVPipeline
from src.utils.fps_counter import FPSMetrics
from src.visualization.visualizer import CVVisualizer
from tests.conftest import create_synthetic_landmarks


def test_standard_scaler_fit_transform_and_edge_cases():
    """Verify StandardScaler math, constant-feature guard, missing/Inf imputation, and outlier clipping."""
    # Columns: 0 has spread, 1 is constant (zero std), 2 has a NaN in training
    X_train = np.array([
        [0.40, 1.0, 0.10],
        [0.50, 1.0, 0.20],
        [0.60, 1.0, np.nan],
        [0.50, 1.0, 0.30],
    ])
    scaler = StandardScaler(feature_names=["f0", "f1", "f2"], z_clip=6.0)
    Z_train = scaler.fit_transform(X_train)

    assert scaler.is_fitted
    assert np.isclose(scaler.mean_[0], 0.50)
    assert np.isclose(scaler.mean_[1], 1.00)
    assert np.isclose(scaler.mean_[2], 0.20)  # nanmean of [0.1, 0.2, 0.3]
    # Constant column 1 gets safe scale=1.0 and normalizes to 0.0
    assert np.isclose(scaler.scale_[1], 1.0)
    assert np.allclose(Z_train[:, 1], 0.0)
    assert np.isfinite(Z_train).all()

    # Transform with None, NaN, Inf, and extreme values at inference
    X_infer = [
        [0.50, 1.0, None],          # Missing value -> imputed to mean (0.20) -> Z=0.0
        [np.inf, -np.inf, 0.20],    # Non-finite -> imputed to means -> Z=0.0
        [100.0, 1.0, -50.0],        # Extreme outliers -> clipped safely to +/-6.0
    ]
    Z_infer = scaler.transform(X_infer)
    assert Z_infer.shape == (3, 3)
    assert np.isfinite(Z_infer).all()
    assert Z_infer[0, 2] == pytest.approx(0.0)
    assert Z_infer[1, 0] == pytest.approx(0.0)
    assert Z_infer[1, 1] == pytest.approx(0.0)
    assert Z_infer[2, 0] == pytest.approx(6.0)
    assert Z_infer[2, 2] == pytest.approx(-6.0)

    # Validation transform must NOT alter fitted scaler stats
    saved_mean = scaler.mean_.copy()
    saved_scale = scaler.scale_.copy()
    _ = scaler.transform(np.array([[9.9, 5.5, 4.4]]))
    assert np.allclose(scaler.mean_, saved_mean)
    assert np.allclose(scaler.scale_, saved_scale)


def test_mapper_scaler_zero_leakage_and_serialization(tmp_path: Path):
    """
    Verify mapper fits scaler strictly on training data, applies identical normalization
    across training/validation/inference, saves/loads scaler params via ProfileManager,
    and preserves compatibility with legacy unscaled profiles.
    """
    np.random.seed(7)
    X_train = np.column_stack([
        np.linspace(0.40, 0.60, 12),
        np.linspace(0.15, 0.35, 12),
        np.ones((12, 6)),             # Constant features
        np.linspace(-0.2, 0.2, 12),
        np.linspace(-0.1, 0.1, 12),
    ])
    y_train = np.column_stack([
        np.linspace(0.10, 0.90, 12),
        np.linspace(0.10, 0.90, 12),
    ])

    mapper = PolynomialGazeMapper(ridge_alpha=0.01, use_scaler=True).fit(X_train, y_train)
    assert mapper.scaler is not None and mapper.scaler.is_fitted
    assert mapper.normalization_method == "StandardScaler"
    mean_before = mapper.scaler.mean_.copy()
    scale_before = mapper.scaler.scale_.copy()

    # Validation set with drastically different mean must NOT alter scaler parameters
    X_val = np.full((5, 10), 2.5)
    preds_before = mapper.predict(X_val)
    assert np.allclose(mapper.scaler.mean_, mean_before)
    assert np.allclose(mapper.scaler.scale_, scale_before)
    assert np.all((preds_before >= 0.0) & (preds_before <= 1.0))

    # Targets remain strictly unscaled in [0, 1]
    preds_train = mapper.predict(X_train)
    assert np.mean(np.abs(preds_train - y_train)) < 0.03

    # Persist into CalibrationProfile and reload
    pm = ProfileManager(profiles_dir=tmp_path)
    profile = CalibrationProfile(
        profile_id="test_norm_prof",
        model_type=mapper.model_type,
        model_data=mapper.to_dict(),
        normalization_method=mapper.normalization_method,
        normalization_params=mapper.scaler.to_dict(),
        training_sample_count=len(X_train),
    )
    saved_path = pm.save_profile(profile)
    loaded_profile = pm.load_profile(saved_path)
    assert loaded_profile.normalization_method == "StandardScaler"
    assert loaded_profile.normalization_params is not None
    assert loaded_profile.training_sample_count == 12

    loaded_mapper = pm.extract_mapper(loaded_profile)
    assert loaded_mapper.scaler is not None and loaded_mapper.scaler.is_fitted
    assert np.allclose(loaded_mapper.scaler.mean_, mean_before)
    assert np.allclose(loaded_mapper.scaler.scale_, scale_before)

    preds_after = loaded_mapper.predict(X_val)
    assert np.allclose(preds_before, preds_after)

    # Legacy unscaled profile backwards compatibility (missing 'scaler', but using current convention)
    legacy_dict = {
        "profile_id": "legacy_prof",
        "model_type": "linear",
        "coordinate_convention": "interface_left0_right1_v1",
        "model_data": {
            "model_type": "linear",
            "ridge_alpha": 0.01,
            "is_fitted": True,
            "training_sample_count": 2,
            "feature_names": list(FEATURE_NAMES),
            "weights": [[0.5, 0.5]] + [[0.0, 0.0]] * 10,
        },
    }
    legacy_prof = CalibrationProfile.from_dict(legacy_dict)
    assert legacy_prof.normalization_method == "None"
    legacy_mapper = pm.extract_mapper(legacy_prof)
    assert legacy_mapper.use_scaler is False
    assert np.allclose(legacy_mapper.predict(np.ones(10)), [0.5, 0.5])


def test_loocv_selection_prevents_leakage():
    """Verify compare_and_select_mapper without X_val separates CV selection from training fit."""
    np.random.seed(11)
    X_train = np.random.uniform(0.3, 0.7, size=(9, 10))
    y_train = np.column_stack([
        1.5 * (X_train[:, 0] - 0.5) + 0.5,
        1.5 * (X_train[:, 1] - 0.5) + 0.5,
    ])
    y_train = np.clip(y_train, 0.1, 0.9)

    selected, report = compare_and_select_mapper(X_train, y_train, use_scaler=True)
    assert selected.is_fitted
    assert selected.scaler is not None
    assert report["selection_basis"] == "loocv"
    assert "training_metrics" in report
    assert "linear_training_metrics" in report


def test_target_selection_accuracy_and_diagonal_percentages():
    """
    Verify Target Selection Accuracy with actual target boundaries (default rendered circle
    and explicit rectangular AAC button box), plus configurable error thresholds.
    """
    screen_w, screen_h = 1920, 1080
    diag_px = math.hypot(screen_w, screen_h)  # 2202.9 px

    # Default circle on 1280x720 preview has display radius 22px -> 33.0px at 1920x1080
    t_circle = CalibrationTarget(0, "circle_center", 0.50, 0.50)
    # 20px error -> HIT (20 <= 33)
    hit, _ = t_circle.contains_point(0.50 + 20.0 / screen_w, 0.50, screen_w, screen_h)
    assert hit is True
    # 50px error -> MISS (50 > 33)
    miss, _ = t_circle.contains_point(0.50 + 50.0 / screen_w, 0.50, screen_w, screen_h)
    assert miss is False

    # Rectangular AAC button box (300 x 200 px centered at (0.3, 0.3)) -> +/-150 px X, +/-100 px Y
    t_box_hit = CalibrationTarget(0, "aac_btn_1", 0.30, 0.30, width_px=300.0, height_px=200.0)
    t_box_miss = CalibrationTarget(1, "aac_btn_2", 0.70, 0.70, width_px=300.0, height_px=200.0)

    val = ScreenValidator(
        targets=[t_box_hit, t_box_miss],
        screen_width=screen_w,
        screen_height=screen_h,
        threshold_2pct=0.02,
        threshold_5pct=0.05,
        threshold_10pct=0.10,
    )
    val.mapper = LinearGazeMapper()
    val._trial_sequence = [t_box_hit, t_box_miss]
    val._trial_records = [
        # Trial 1: predicted shifted +100 px X, +50 px Y -> inside 300x200 box (<=150, <=100) -> HIT
        ValidationTrialRecord(
            experiment_id="exp_acc", trial_id=1, target_id=0, target_name="aac_btn_1",
            actual_x=0.30, actual_y=0.30,
            predicted_x=0.30 + 100.0 / screen_w, predicted_y=0.30 + 50.0 / screen_h,
            error_x=100.0 / screen_w, error_y=50.0 / screen_h,
            euclidean_error_norm=None, euclidean_error_pixels=None,
            screen_width=screen_w, screen_height=screen_h, tracking_quality="EXCELLENT",
            accepted_sample_count=20, trial_status="VALID",
        ),
        # Trial 2: predicted shifted +200 px X -> outside 300x200 box (>150) -> MISS
        ValidationTrialRecord(
            experiment_id="exp_acc", trial_id=2, target_id=1, target_name="aac_btn_2",
            actual_x=0.70, actual_y=0.70,
            predicted_x=0.70 + 200.0 / screen_w, predicted_y=0.70,
            error_x=200.0 / screen_w, error_y=0.0,
            euclidean_error_norm=None, euclidean_error_pixels=None,
            screen_width=screen_w, screen_height=screen_h, tracking_quality="EXCELLENT",
            accepted_sample_count=20, trial_status="VALID",
        ),
        # Trial 3: REJECTED -> must NOT enter denominator!
        ValidationTrialRecord(
            experiment_id="exp_acc", trial_id=3, target_id=1, target_name="aac_btn_2",
            actual_x=0.70, actual_y=0.70,
            predicted_x=None, predicted_y=None,
            error_x=None, error_y=None,
            euclidean_error_norm=None, euclidean_error_pixels=None,
            screen_width=screen_w, screen_height=screen_h, tracking_quality="POOR",
            accepted_sample_count=2, trial_status="REJECTED", rejection_reason="BLINK_DETECTED",
        ),
    ]
    summary = val._finalize_validation()

    # Verify denominator discipline (2 valid trials only)
    assert summary.total_trials == 2 or summary.successful_trials == 2
    assert summary.successful_trials == 2
    assert summary.rejected_trials == 1
    assert summary.successful_selections == 1
    assert summary.failed_selections == 1
    assert summary.target_selection_accuracy_pct == pytest.approx(50.0)

    # Error calculations verification:
    # Trial 1 px = hypot(100, 50) = 111.803 px (~5.075% diag -> >5%, <=10%)
    # Trial 2 px = 200.0 px (~9.079% diag -> >5%, <=10%)
    e1 = math.hypot(100.0, 50.0)
    e2 = 200.0
    assert summary.mean_error_pixels == pytest.approx((e1 + e2) / 2.0, abs=0.1)
    assert summary.mean_error_pct_diagonal == pytest.approx(((e1 + e2) / 2.0 / diag_px) * 100.0, abs=0.05)
    assert summary.pct_within_2pct == pytest.approx(0.0)
    assert summary.pct_within_5pct == pytest.approx(0.0)
    assert summary.pct_within_10pct == pytest.approx(100.0)

    # Per-target breakdown check
    assert summary.target_accuracies["aac_btn_1"]["target_selection_accuracy_pct"] == pytest.approx(100.0)
    assert summary.target_accuracies["aac_btn_2"]["target_selection_accuracy_pct"] == pytest.approx(0.0)


def test_missing_validation_never_produces_fake_accuracy(tmp_path: Path):
    """Calibration-only sessions must state validation is incomplete and never present training fit as accuracy."""
    logger = ExperimentLogger(base_dir=tmp_path, experiment_id="exp_calib_only")
    profile = CalibrationProfile(
        profile_id="prof_only",
        model_type="linear",
        normalization_method="StandardScaler",
        training_sample_count=9,
        calibration_frame_count=180,
        training_metrics={"mean_euclidean_pixels": 65.0, "mae_x": 0.02, "mae_y": 0.03},
        validation_metrics=None,
    )

    files = logger.log_experiment(
        calibration_samples=[],
        validation_results=None,
        accuracy_summary=None,
        profile=profile,
        export_excel=False,
        generate_plots=False,
    )

    assert files["accuracy_report_md"].exists()
    assert files["accuracy_report_html"].exists()
    assert files["accuracy_report_text"].exists()

    md_text = files["accuracy_report_md"].read_text(encoding="utf-8")
    html_text = files["accuracy_report_html"].read_text(encoding="utf-8")
    txt_text = files["accuracy_report_text"].read_text(encoding="utf-8")

    assert VALIDATION_NOT_COMPLETED_MSG in md_text
    assert "Validation not completed" in html_text
    assert VALIDATION_NOT_COMPLETED_MSG in txt_text
    # Training metric shown explicitly as training fit, NOT validation accuracy
    assert "65.0 px" in md_text
    assert "Calibration Training Set" in md_text
    assert not (logger.experiment_dir / "accuracy_summary.csv").exists()


def test_experiment_reports_and_comparison_strict_integrity(tmp_path: Path):
    """Verify multi-experiment comparison excludes calibration-only training metrics and tracks trends."""
    # Exp 1: Calibration only (training error = 5.0 px - must NOT rank as best validated!)
    log0 = ExperimentLogger(base_dir=tmp_path, experiment_id="experiment_20261124_100000")
    log0.log_experiment(
        calibration_samples=[],
        validation_results=None,
        accuracy_summary=None,
        profile=CalibrationProfile(
            profile_id="p0",
            training_metrics={"mean_euclidean_pixels": 5.0},
        ),
        export_excel=False,
        generate_plots=False,
    )

    # Exp 2: First validated session (Mean Error = 100 px, Selection Acc = 40%)
    log1 = ExperimentLogger(base_dir=tmp_path, experiment_id="experiment_20261124_110000")
    s1 = AccuracySummary(
        experiment_id="experiment_20261124_110000",
        model_name="linear",
        model_configuration={"model_type": "linear"},
        normalization_method="None",
        total_trials=5, successful_trials=5, rejected_trials=0,
        successful_selections=2, failed_selections=3, target_selection_accuracy_pct=40.0,
        mae_x=0.04, mae_y=0.04, rmse_x=0.05, rmse_y=0.05,
        mae_x_px=76.8, mae_y_px=43.2, rmse_x_px=96.0, rmse_y_px=54.0,
        mean_euclidean_error=0.05, median_euclidean_error=0.045, maximum_error=0.08,
        mean_error_pixels=100.0, median_error_pixels=90.0, max_error_pixels=160.0,
        mean_error_pct_diagonal=4.54, median_error_pct_diagonal=4.09, max_error_pct_diagonal=7.26,
        trials_within_2pct=0, pct_within_2pct=0.0,
        trials_within_5pct=3, pct_within_5pct=60.0,
        trials_within_10pct=5, pct_within_10pct=100.0,
        screen_resolution="1920x1080", evaluation_timestamp=100.0,
    )
    log1.log_experiment([], [], s1, export_excel=False, generate_plots=False)

    # Exp 3: Second validated session with StandardScaler (Mean Error = 60 px -> 40% reduction, Sel Acc = 80%)
    log2 = ExperimentLogger(base_dir=tmp_path, experiment_id="experiment_20261124_120000")
    s2 = AccuracySummary(
        experiment_id="experiment_20261124_120000",
        model_name="polynomial",
        model_configuration={"model_type": "polynomial"},
        normalization_method="StandardScaler",
        training_sample_count=9,
        total_trials=6, successful_trials=5, rejected_trials=1,
        successful_selections=4, failed_selections=1, target_selection_accuracy_pct=80.0,
        mae_x=0.02, mae_y=0.02, rmse_x=0.03, rmse_y=0.03,
        mae_x_px=38.4, mae_y_px=21.6, rmse_x_px=57.6, rmse_y_px=32.4,
        mean_euclidean_error=0.03, median_euclidean_error=0.025, maximum_error=0.05,
        mean_error_pixels=60.0, median_error_pixels=50.0, max_error_pixels=95.0,
        mean_error_pct_diagonal=2.72, median_error_pct_diagonal=2.27, max_error_pct_diagonal=4.31,
        trials_within_2pct=2, pct_within_2pct=40.0,
        trials_within_5pct=5, pct_within_5pct=100.0,
        trials_within_10pct=5, pct_within_10pct=100.0,
        screen_resolution="1920x1080", evaluation_timestamp=200.0,
    )
    log2.log_experiment([], [], s2, export_excel=False, generate_plots=False)

    dirs = [log0.experiment_dir, log1.experiment_dir, log2.experiment_dir]
    comp_summary = build_experiment_comparison_summary(dirs)
    assert comp_summary["best_experiment"]["experiment_id"] == "experiment_20261124_120000"
    assert comp_summary["latest_vs_previous"]["verdict"] == "improved"
    assert comp_summary["latest_vs_previous"]["mean_error_improvement_pct"] == pytest.approx(40.0)
    assert comp_summary["latest_vs_previous"]["target_selection_pp_change"] == pytest.approx(40.0)

    table_md = compare_experiments(dirs)
    assert "experiment_20261124_100000" in table_md
    assert "Calibrated Only" in table_md or "Validation not completed" in table_md
    assert "IMPROVED" in table_md
    assert "40.0% error reduction" in table_md
    assert "80.0%" in table_md

    # Verify generated HTML & MD report for Exp 3 contain required structure and explanation
    md3 = (log2.experiment_dir / "accuracy_report.md").read_text(encoding="utf-8")
    for heading in ("## Experiment Overview", "## Main Results", "## Detailed Results", "## Simple Explanation"):
        assert heading in md3
    assert "4 out of 5 valid trials" in md3
    assert "performed better than the previous validated experiment" in md3
    assert "StandardScaler" in md3
    assert "60.0 px" in md3


def test_visualizer_renders_validation_targets_without_attribute_error(dummy_bgr_frame):
    """Verify live validation target overlay & dashboard render without ScreenGazeOutput attribute crashes."""
    viz = CVVisualizer()
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
    screen_gaze = ScreenGazeOutput(
        gaze_x=0.35, gaze_y=0.32, pixel_x=672, pixel_y=346,
        is_calibrated=True, tracking_quality="EXCELLENT", confidence=0.95,
    )

    val = ScreenValidator(
        targets=[CalibrationTarget(0, "aac_btn", 0.30, 0.30, width_px=240.0, height_px=160.0)],
        screen_width=1920,
        screen_height=1080,
    )
    val.start(LinearGazeMapper().fit(np.random.randn(4, 10), np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])))

    rendered = viz.render(
        frame=dummy_bgr_frame,
        detection_result=det,
        eye_features=feats,
        fps_metrics=fps,
        screen_gaze=screen_gaze,
        screen_validator=val,
    )
    assert rendered.shape == dummy_bgr_frame.shape

    # Also test default circular target rendering
    val_circle = ScreenValidator(screen_width=1920, screen_height=1080)
    val_circle.start(val.mapper)
    rendered_circle = viz.render(
        frame=dummy_bgr_frame,
        detection_result=det,
        eye_features=feats,
        fps_metrics=fps,
        screen_gaze=screen_gaze,
        screen_validator=val_circle,
    )
    assert rendered_circle.shape == dummy_bgr_frame.shape


def test_pipeline_inference_uses_loaded_scaler(tmp_path: Path, dummy_bgr_frame):
    """Verify CVPipeline real-time inference uses saved StandardScaler from calibration profile."""
    model_path = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
    if not model_path.exists():
        pytest.skip("Model bundle required for pipeline test.")

    # Create and save a profile with a fitted StandardScaler + Linear mapper
    np.random.seed(5)
    X_tr = np.random.uniform(0.3, 0.7, size=(8, 10))
    y_tr = np.random.uniform(0.1, 0.9, size=(8, 2))
    mapper = LinearGazeMapper(ridge_alpha=0.01, use_scaler=True).fit(X_tr, y_tr)

    pm = ProfileManager(profiles_dir=tmp_path / "profiles")
    prof = CalibrationProfile(
        profile_id="rt_scaler_test",
        model_type="linear",
        model_data=mapper.to_dict(),
        normalization_method=mapper.normalization_method,
        normalization_params=mapper.scaler.to_dict(),
        training_sample_count=8,
    )
    prof_path = pm.save_profile(prof)

    cfg = AppConfig()
    cfg.camera.use_threading = False
    cfg.landmarker.model_path = str(model_path)
    cfg.visualization.enabled = False
    cfg.screen.active_profile_path = str(prof_path)
    cfg.screen.profiles_dir = str(tmp_path / "profiles")
    cfg.screen.experiments_dir = str(tmp_path / "experiments")

    with patch("src.camera.camera_manager.CameraManager.start"):
        with patch("src.camera.camera_manager.CameraManager.is_opened", return_value=True):
            with patch("src.camera.camera_manager.CameraManager.read", return_value=(True, dummy_bgr_frame)):
                pipeline = CVPipeline(config=cfg)
                pipeline._is_running = True
                assert pipeline.active_mapper is not None
                assert pipeline.active_mapper.scaler is not None
                assert np.allclose(pipeline.active_mapper.scaler.mean_, mapper.scaler.mean_)

                # Feed a raw feature vector directly through active mapper vs original fitted mapper
                sample_vec = np.full(10, 0.5)
                assert np.allclose(pipeline.active_mapper.predict(sample_vec), mapper.predict(sample_vec))

                pipeline.close()


def test_pipeline_queued_validation_transition(tmp_path: Path):
    """Verify calibration followed cleanly by queued validation updates reports without errors."""
    model_path = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
    if not model_path.exists():
        pytest.skip("Model bundle required for pipeline test.")

    cfg = AppConfig()
    cfg.camera.use_threading = False
    cfg.landmarker.model_path = str(model_path)
    cfg.visualization.enabled = False
    cfg.screen.profiles_dir = str(tmp_path / "profiles")
    cfg.screen.experiments_dir = str(tmp_path / "experiments")

    pipeline = CVPipeline(config=cfg, use_mock_camera=True)
    pipeline.start()
    pipeline.start_screen_calibration()
    pipeline.queue_screen_validation()
    assert pipeline._queue_validation_after_calibration is True

    # Simulate finishing calibration -> auto-starts validation
    X_tr = np.random.uniform(0.3, 0.7, size=(9, 10))
    y_tr = np.random.uniform(0.1, 0.9, size=(9, 2))
    fitted = LinearGazeMapper().fit(X_tr, y_tr)
    pipeline.screen_calibrator.state = pipeline.screen_calibrator.state.COMPLETED
    pipeline.screen_calibrator.profile = CalibrationProfile(
        profile_id="queued_prof",
        model_type="linear",
        model_data=fitted.to_dict(),
        normalization_method="StandardScaler",
        normalization_params=fitted.scaler.to_dict(),
        training_sample_count=9,
    )
    pipeline.active_mapper = fitted
    pipeline.active_profile = pipeline.screen_calibrator.profile

    # Calling start_screen_validation(reuse_active_experiment=True) attaches scaler metadata cleanly
    pipeline.start_screen_validation(reuse_active_experiment=True)
    assert pipeline.screen_validator.is_active
    assert pipeline.screen_validator.normalization_method == "StandardScaler"
    assert pipeline.screen_validator.training_sample_count == 9
    pipeline.close()
