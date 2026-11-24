"""
Unit tests for Screen Accuracy Validation & Evaluation (Phase 3).

Covers:
- Validation targets layout
- Evaluating trained mapper on independent test points
- Generating per-trial metrics (actual vs predicted, error in norm and pixels)
- Statistical summary metrics calculation (MAE, RMSE, mean/median/max error)
"""

import time
import numpy as np
import pytest

from src.core.eye_features import extract_eye_features
from src.core.gaze_mapper import LinearGazeMapper
from src.core.screen_validator import (
    AccuracySummary,
    DEFAULT_VALIDATION_TARGETS,
    ScreenValidator,
    ValidationPhase,
    ValidationState,
)
from tests.conftest import create_synthetic_landmarks


def test_validation_targets_geometry():
    """Verify validation targets are inside the screen and independent of 9-point calibration edges."""
    assert len(DEFAULT_VALIDATION_TARGETS) == 5

    # Center target must be present
    center = [t for t in DEFAULT_VALIDATION_TARGETS if t.name == "val_center"][0]
    assert center.x == 0.50
    assert center.y == 0.50

    # Corners should be at interior locations (0.30, 0.70)
    for t in DEFAULT_VALIDATION_TARGETS:
        assert 0.25 <= t.x <= 0.75
        assert 0.25 <= t.y <= 0.75


def test_screen_validator_workflow_and_metrics():
    """Verify validation session completes and produces valid numerical accuracy metrics."""
    # Pre-train a synthetic linear mapper
    X_train = np.array([
        [0.45] * 10,
        [0.55] * 10,
    ])
    y_train = np.array([
        [0.30, 0.30],
        [0.70, 0.70],
    ])
    mapper = LinearGazeMapper(ridge_alpha=0.01).fit(X_train, y_train)

    validator = ScreenValidator(
        settling_duration=0.05,
        collection_duration=0.10,
        min_samples_per_target=5,
        screen_width=1920,
        screen_height=1080,
    )
    assert validator.state == ValidationState.IDLE

    validator.start(mapper)
    assert validator.is_active

    # Step through all 5 validation targets
    summary: AccuracySummary = None
    for t_idx in range(5):
        target = validator.current_target
        assert target is not None

        # Settle
        time.sleep(0.06)

        # Correlated landmarks
        off_x = (target.x - 0.50) * 0.05
        off_y = (target.y - 0.50) * 0.05
        lm = create_synthetic_landmarks(center_offset_x=off_x, center_offset_y=off_y)
        feats = extract_eye_features(lm)

        # Transition SETTLING -> COLLECTING cleanly
        validator.update(feats)

        # Feed sufficient valid frames across collection window until trial finishes
        for _ in range(12):
            time.sleep(0.015)
            res = validator.update(feats)
            if res is not None:
                summary = res
            if validator.current_target_index != t_idx or validator.state == ValidationState.COMPLETED:
                break

    assert validator.state == ValidationState.COMPLETED
    assert summary is not None
    assert summary.total_trials == 5
    assert summary.successful_trials == 5
    assert summary.rejected_trials == 0

    # Verify metrics sanity
    assert summary.mae_x >= 0.0
    assert summary.mae_y >= 0.0
    assert summary.rmse_x >= 0.0
    assert summary.rmse_y >= 0.0
    assert summary.mean_euclidean_error >= 0.0
    assert summary.mean_error_pixels >= 0.0
    assert summary.screen_resolution == "1920x1080"
    assert len(summary.target_accuracies) == 5

    # Check individual trial records
    trials = validator.get_trial_records()
    assert len(trials) == 5
    for trial in trials:
        assert 0.0 <= trial.actual_x <= 1.0
        assert 0.0 <= trial.predicted_x <= 1.0
        assert trial.euclidean_error_pixels >= 0.0
        assert trial.accepted_sample_count >= 5
