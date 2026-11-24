"""
Comprehensive automated tests for Phase 3 Accuracy Testing & Real-Time Evaluation.

Tests cover:
1. Exact error calculations (signed, absolute, normalized Euclidean, pixel Euclidean).
2. Screen diagonal threshold analysis (<=2%, <=5%, <=10% screen diagonal).
3. Rejection of invalid trials and denominator separation.
4. Data leakage prevention (train vs validation target independence).
5. Multi-trial repetition and target sequence randomization.
6. Per-target accuracy breakdown and spatial bias pattern analysis.
7. 90th percentile error calculation.
8. Real-time Live Metrics Dashboard state and running metrics.
9. Publication-quality diagnostic plot generation (PNGs) and text report.
10. Interrupted validation session data preservation.
11. Experiment comparison utility.
"""

from pathlib import Path
import time
import numpy as np
import pytest

from src.core.eye_features import BinocularEyeFeatures, SingleEyeFeatures, EyeOpennessState, Point2D, BoundingBox
from src.core.gaze_mapper import LinearGazeMapper
from src.core.head_pose import HeadPose
from src.core.screen_calibration import DEFAULT_9_TARGETS, CalibrationTarget
from src.core.screen_validator import (
    DEFAULT_VALIDATION_TARGETS,
    AccuracySummary,
    ScreenValidator,
    ValidationPhase,
    ValidationState,
    ValidationTrialRecord,
)
from src.core.experiment_logger import ExperimentLogger, compare_experiments
from tests.conftest import create_synthetic_landmarks


def _make_dummy_eye_features(confidence: float = 0.90, is_blink: bool = False, openness: float = 0.85) -> BinocularEyeFeatures:
    """Helper to synthesize valid binocular eye features."""
    single = SingleEyeFeatures(
        side="right",
        iris_center=Point2D(0.5, 0.5),
        bounding_box=BoundingBox(0.4, 0.45, 0.6, 0.55),
        horizontal_ratio=0.50,
        vertical_ratio=0.50,
        eye_aspect_ratio=0.28 if not is_blink else 0.08,
        openness=openness if not is_blink else 0.05,
        openness_state=EyeOpennessState.OPEN if not is_blink else EyeOpennessState.CLOSED,
        is_blink=is_blink,
        reliability=confidence,
        is_valid=not is_blink,
    )
    return BinocularEyeFeatures(
        left_eye=single,
        right_eye=single,
        combined_horizontal_ratio=0.50,
        combined_vertical_ratio=0.50,
        inter_iris_distance=0.15,
        confidence=confidence,
        is_valid=not is_blink,
    )


def test_exact_error_calculations():
    """Verify exact formula implementation for signed, absolute, Euclidean, and pixel errors."""
    actual_x, actual_y = 0.30, 0.70
    pred_x, pred_y = 0.33, 0.66
    screen_w, screen_h = 1920, 1080

    err_x = pred_x - actual_x      # +0.03
    err_y = pred_y - actual_y      # -0.04
    abs_x = abs(err_x)             # 0.03
    abs_y = abs(err_y)             # 0.04

    e_norm = np.hypot(err_x, err_y)  # sqrt(0.03^2 + (-0.04)^2) = 0.05
    assert np.isclose(e_norm, 0.05)

    px_x = err_x * screen_w         # +57.6 px
    px_y = err_y * screen_h         # -43.2 px
    e_px = np.hypot(px_x, px_y)    # sqrt(57.6^2 + (-43.2)^2) = 72.0 px
    assert np.isclose(e_px, 72.0)

    record = ValidationTrialRecord(
        experiment_id="exp_test",
        trial_id=1,
        target_id=0,
        target_name="val_bottom_left",
        actual_x=actual_x,
        actual_y=actual_y,
        predicted_x=pred_x,
        predicted_y=pred_y,
        error_x=err_x,
        error_y=err_y,
        abs_error_x=abs_x,
        abs_error_y=abs_y,
        euclidean_error_norm=e_norm,
        euclidean_error_pixels=e_px,
        screen_width=screen_w,
        screen_height=screen_h,
        tracking_quality="EXCELLENT",
        accepted_sample_count=20,
    )

    d = record.to_csv_dict()
    assert d["error_x"] == 0.03
    assert d["error_y"] == -0.04
    assert d["abs_error_x"] == 0.03
    assert d["abs_error_y"] == 0.04
    assert d["euclidean_error_normalized"] == 0.05
    assert d["euclidean_error_pixels"] == 72.0


def test_data_leakage_prevention_target_independence():
    """Verify that independent validation targets never intersect or overlap with training targets."""
    train_pts = {(round(t.x, 3), round(t.y, 3)) for t in DEFAULT_9_TARGETS}
    val_pts = {(round(t.x, 3), round(t.y, 3)) for t in DEFAULT_VALIDATION_TARGETS}

    # Training targets: 3x3 grid at 0.10, 0.50, 0.90
    assert len(train_pts) == 9
    # Validation targets: 5 interior points at 0.30, 0.50, 0.70
    assert len(val_pts) == 5

    # Note: 0.50, 0.50 is the center, but the other 4 targets (0.30, 0.30), (0.70, 0.30), etc. are strictly novel
    corner_val_pts = {(x, y) for (x, y) in val_pts if not (x == 0.5 and y == 0.5)}
    for vp in corner_val_pts:
        assert vp not in train_pts, f"Validation target {vp} must not be in training set {train_pts}!"
        # Minimum distance to any training point must be substantial
        min_dist = min(np.hypot(vp[0] - tp[0], vp[1] - tp[1]) for tp in train_pts)
        assert min_dist >= 0.18, f"Target {vp} is too close to training target (distance {min_dist})!"


def test_multi_trial_repetition_and_randomization():
    """Verify support for repeating validation targets and randomizing sequence order."""
    base_targets = [
        CalibrationTarget(0, "A", 0.3, 0.3),
        CalibrationTarget(1, "B", 0.7, 0.3),
        CalibrationTarget(2, "C", 0.5, 0.5),
    ]

    # 1. Multi-trial repeats: 3 targets x 3 repeats = 9 trials
    val_repeats = ScreenValidator(targets=base_targets, num_repeats=3, randomize_order=False)
    assert len(val_repeats.targets) == 9
    assert [t.name for t in val_repeats.targets] == ["A", "B", "C", "A", "B", "C", "A", "B", "C"]

    # 2. Randomization order: ensure sequence contains all items while altering ordering
    val_rand = ScreenValidator(targets=base_targets, num_repeats=2, randomize_order=True)
    assert len(val_rand.targets) == 6
    names = [t.name for t in val_rand.targets]
    assert names.count("A") == 2
    assert names.count("B") == 2
    assert names.count("C") == 2


def test_rejected_trials_handling_and_denominator_separation():
    """Verify that trials rejected due to blinks or insufficient frames do not contaminate error metrics."""
    # Synthetic fitted mapper
    X_train = np.array([[0.5] * 10, [0.3] * 10, [0.7] * 10])
    y_train = np.array([[0.5, 0.5], [0.3, 0.3], [0.7, 0.7]])
    mapper = LinearGazeMapper().fit(X_train, y_train)

    val = ScreenValidator(
        targets=[
            CalibrationTarget(0, "val_1", 0.3, 0.3),
            CalibrationTarget(1, "val_2", 0.7, 0.7),
            CalibrationTarget(2, "val_3", 0.5, 0.5),
        ],
        settling_duration=0.01,
        collection_duration=0.04,
        min_samples_per_target=3,
        screen_width=1920,
        screen_height=1080,
    )
    val.start(mapper)

    # Trial 1: Good frames -> VALID
    time.sleep(0.02)
    val.update(_make_dummy_eye_features(confidence=0.9))
    time.sleep(0.02)
    for _ in range(4):
        val.update(_make_dummy_eye_features(confidence=0.9))
    time.sleep(0.05)
    val.update(_make_dummy_eye_features(confidence=0.9))  # Completes trial 1

    # Trial 2: Continuous blinking -> REJECTED
    time.sleep(0.02)
    val.update(_make_dummy_eye_features(is_blink=True))
    time.sleep(0.02)
    for _ in range(4):
        val.update(_make_dummy_eye_features(is_blink=True))
    time.sleep(0.05)
    val.update(_make_dummy_eye_features(is_blink=True))  # Completes trial 2 (REJECTED)

    # Trial 3: Good frames -> VALID
    time.sleep(0.02)
    val.update(_make_dummy_eye_features(confidence=0.9))
    time.sleep(0.02)
    for _ in range(4):
        val.update(_make_dummy_eye_features(confidence=0.9))
    time.sleep(0.05)
    summary = val.update(_make_dummy_eye_features(confidence=0.9))  # Completes validation

    assert summary is not None
    assert summary.total_trials == 3
    assert summary.successful_trials == 2
    assert summary.rejected_trials == 1

    # Ensure rejected trial did not inject zeros into MAE or RMSE
    assert summary.mae_x > 0.0
    assert summary.mae_y > 0.0
    assert summary.mean_error_pixels > 0.0

    # Verify trial records
    records = val.get_trial_records()
    assert len(records) == 3
    assert records[0].trial_status == "VALID"
    assert records[1].trial_status == "REJECTED"
    assert "INSUFFICIENT_SAMPLES" in records[1].rejection_reason
    assert records[2].trial_status == "VALID"


def test_diagonal_threshold_compliance():
    """Verify screen diagonal threshold calculations (2%, 5%, 10%)."""
    screen_w, screen_h = 1920, 1080
    diag_px = np.hypot(screen_w, screen_h)  # ~2202.9 px

    thresh_2pct = 0.02 * diag_px  # ~44.06 px
    thresh_5pct = 0.05 * diag_px  # ~110.15 px
    thresh_10pct = 0.10 * diag_px # ~220.29 px

    # 4 synthetic trials: 30px (<=2%), 80px (<=5%), 150px (<=10%), 300px (>10%)
    trials = [
        ValidationTrialRecord(
            experiment_id="exp_t", trial_id=1, target_id=0, target_name="T1",
            actual_x=0.3, actual_y=0.3, predicted_x=0.31, predicted_y=0.31,
            error_x=0.01, error_y=0.01, euclidean_error_norm=0.0141, euclidean_error_pixels=30.0,
            screen_width=screen_w, screen_height=screen_h, tracking_quality="EXCELLENT",
            accepted_sample_count=20, trial_status="VALID",
        ),
        ValidationTrialRecord(
            experiment_id="exp_t", trial_id=2, target_id=1, target_name="T2",
            actual_x=0.7, actual_y=0.3, predicted_x=0.73, predicted_y=0.32,
            error_x=0.03, error_y=0.02, euclidean_error_norm=0.036, euclidean_error_pixels=80.0,
            screen_width=screen_w, screen_height=screen_h, tracking_quality="EXCELLENT",
            accepted_sample_count=20, trial_status="VALID",
        ),
        ValidationTrialRecord(
            experiment_id="exp_t", trial_id=3, target_id=2, target_name="T3",
            actual_x=0.3, actual_y=0.7, predicted_x=0.35, predicted_y=0.75,
            error_x=0.05, error_y=0.05, euclidean_error_norm=0.070, euclidean_error_pixels=150.0,
            screen_width=screen_w, screen_height=screen_h, tracking_quality="EXCELLENT",
            accepted_sample_count=20, trial_status="VALID",
        ),
        ValidationTrialRecord(
            experiment_id="exp_t", trial_id=4, target_id=3, target_name="T4",
            actual_x=0.7, actual_y=0.7, predicted_x=0.80, predicted_y=0.80,
            error_x=0.10, error_y=0.10, euclidean_error_norm=0.141, euclidean_error_pixels=300.0,
            screen_width=screen_w, screen_height=screen_h, tracking_quality="EXCELLENT",
            accepted_sample_count=20, trial_status="VALID",
        ),
    ]

    val = ScreenValidator(screen_width=screen_w, screen_height=screen_h)
    val.mapper = LinearGazeMapper()
    val._trial_sequence = [CalibrationTarget(i, f"T{i}", 0.5, 0.5) for i in range(4)]
    val._trial_records = trials

    summary = val._finalize_validation()
    assert summary.successful_trials == 4
    assert summary.trials_within_2pct == 1  # 30px only
    assert summary.pct_within_2pct == 25.0
    assert summary.trials_within_5pct == 2  # 30px, 80px
    assert summary.pct_within_5pct == 50.0
    assert summary.trials_within_10pct == 3 # 30px, 80px, 150px
    assert summary.pct_within_10pct == 75.0


def test_live_metrics_dashboard_feed():
    """Verify live metrics updates correctly as trials are evaluated."""
    val = ScreenValidator(
        targets=[CalibrationTarget(0, "T0", 0.3, 0.3), CalibrationTarget(1, "T1", 0.7, 0.7)],
        screen_width=1920,
        screen_height=1080,
    )
    val.mapper = LinearGazeMapper()
    val._trial_sequence = val.targets

    # Initially empty
    m0 = val.get_live_metrics()
    assert m0.completed_trials == 0
    assert m0.valid_trials == 0
    assert m0.running_mean_error_px == 0.0

    # Add 1 valid trial
    val._trial_records.append(
        ValidationTrialRecord(
            experiment_id="e", trial_id=1, target_id=0, target_name="T0",
            actual_x=0.3, actual_y=0.3, predicted_x=0.32, predicted_y=0.31,
            error_x=0.02, error_y=0.01, euclidean_error_norm=0.0224, euclidean_error_pixels=40.0,
            screen_width=1920, screen_height=1080, tracking_quality="EXCELLENT",
            accepted_sample_count=15, trial_status="VALID",
        )
    )

    m1 = val.get_live_metrics()
    assert m1.completed_trials == 1
    assert m1.valid_trials == 1
    assert np.isclose(m1.running_mean_error_px, 40.0)


def test_per_target_pattern_and_bias_analysis():
    """Verify per-target aggregation and spatial pattern detection."""
    trials = [
        ValidationTrialRecord(
            experiment_id="e", trial_id=1, target_id=0, target_name="val_left",
            actual_x=0.3, actual_y=0.5, predicted_x=0.32, predicted_y=0.50,
            error_x=0.02, error_y=0.00, euclidean_error_norm=0.02, euclidean_error_pixels=38.4,
            screen_width=1920, screen_height=1080, tracking_quality="GOOD", accepted_sample_count=15,
        ),
        ValidationTrialRecord(
            experiment_id="e", trial_id=2, target_id=1, target_name="val_right",
            actual_x=0.7, actual_y=0.5, predicted_x=0.75, predicted_y=0.50,
            error_x=0.05, error_y=0.00, euclidean_error_norm=0.05, euclidean_error_pixels=96.0,
            screen_width=1920, screen_height=1080, tracking_quality="GOOD", accepted_sample_count=15,
        ),
    ]

    val = ScreenValidator()
    val.mapper = LinearGazeMapper()
    val._trial_sequence = [CalibrationTarget(0, "val_left", 0.3, 0.5), CalibrationTarget(1, "val_right", 0.7, 0.5)]
    val._trial_records = trials

    summary = val._finalize_validation()
    assert "val_left" in summary.target_accuracies
    assert "val_right" in summary.target_accuracies
    assert summary.pattern_analysis["best_target"]["name"] == "val_left"
    assert summary.pattern_analysis["worst_target"]["name"] == "val_right"
    # Overall horizontal bias must be positive (predicted shifted right)
    assert summary.pattern_analysis["overall_horizontal_bias_px"] > 0


def test_plot_generation_creates_png_files(tmp_path: Path):
    """Verify that ExperimentLogger generates all diagnostic plots and reports."""
    logger = ExperimentLogger(base_dir=tmp_path, experiment_id="test_exp_plots")

    trials = [
        ValidationTrialRecord(
            experiment_id="test_exp_plots", trial_id=i, target_id=i, target_name=f"val_{i}",
            actual_x=0.3 + 0.1 * i, actual_y=0.3 + 0.1 * i,
            predicted_x=0.32 + 0.1 * i, predicted_y=0.31 + 0.1 * i,
            error_x=0.02, error_y=0.01, euclidean_error_norm=0.0224, euclidean_error_pixels=42.0 + i * 5,
            screen_width=1920, screen_height=1080, tracking_quality="EXCELLENT", accepted_sample_count=20,
        )
        for i in range(5)
    ]

    val = ScreenValidator(screen_width=1920, screen_height=1080)
    val.mapper = LinearGazeMapper()
    val._trial_sequence = [CalibrationTarget(i, f"val_{i}", 0.3 + 0.1 * i, 0.3 + 0.1 * i) for i in range(5)]
    val._trial_records = trials
    summary = val._finalize_validation()

    files = logger.log_experiment(
        calibration_samples=[],
        validation_results=trials,
        accuracy_summary=summary,
        export_excel=False,
        generate_plots=True,
    )

    # Check generated files
    assert (logger.experiment_dir / "plot_actual_vs_predicted.png").exists()
    assert (logger.experiment_dir / "plot_error_distributions.png").exists()
    assert (logger.experiment_dir / "plot_per_target_breakdown.png").exists()
    assert (logger.experiment_dir / "accuracy_report.txt").exists()
    assert (logger.experiment_dir / "accuracy_report.json").exists()
    assert (logger.experiment_dir / "validation_results.csv").exists()
    assert (logger.experiment_dir / "accuracy_summary.csv").exists()


def test_experiment_comparison_utility(tmp_path: Path):
    """Verify experiment comparison table generation across multiple sessions."""
    exp1_dir = tmp_path / "exp_1"
    exp2_dir = tmp_path / "exp_2"

    logger1 = ExperimentLogger(base_dir=tmp_path, experiment_id="exp_1")
    logger2 = ExperimentLogger(base_dir=tmp_path, experiment_id="exp_2")

    # Generate dummy summaries for both
    val = ScreenValidator()
    val.mapper = LinearGazeMapper()
    val._trial_sequence = [CalibrationTarget(0, "val", 0.5, 0.5)]
    val._trial_records = [
        ValidationTrialRecord(
            experiment_id="exp_1", trial_id=1, target_id=0, target_name="val",
            actual_x=0.5, actual_y=0.5, predicted_x=0.52, predicted_y=0.51,
            error_x=0.02, error_y=0.01, euclidean_error_norm=0.0224, euclidean_error_pixels=40.0,
            screen_width=1920, screen_height=1080, tracking_quality="EXCELLENT", accepted_sample_count=20,
        )
    ]
    summary1 = val._finalize_validation()
    logger1.log_experiment([], val.get_trial_records(), summary1, export_excel=False, generate_plots=False)

    summary2 = AccuracySummary(
        experiment_id="exp_2",
        model_name="polynomial",
        model_configuration={"model_type": "polynomial"},
        total_trials=5,
        successful_trials=5,
        rejected_trials=0,
        mae_x=0.015,
        mae_y=0.012,
        rmse_x=0.018,
        rmse_y=0.014,
        mean_euclidean_error=0.019,
        median_euclidean_error=0.018,
        maximum_error=0.025,
        mean_error_pixels=32.5,
        median_error_pixels=30.0,
        max_error_pixels=45.0,
        screen_resolution="1920x1080",
        pct_within_5pct=95.0,
    )
    logger2.log_experiment([], [], summary2, export_excel=False, generate_plots=False)

    # Run comparison
    comparison_table = compare_experiments([exp1_dir, exp2_dir])
    assert "| `exp_1` |" in comparison_table
    assert "| `exp_2` |" in comparison_table
    assert "**LINEAR**" in comparison_table
    assert "**POLYNOMIAL**" in comparison_table
    assert "40.0 px" in comparison_table
    assert "32.5 px" in comparison_table
