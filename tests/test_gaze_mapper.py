"""
Unit tests for Gaze-to-Screen Mapping Regression Models (Phase 3).

Covers:
- Feature extraction from eye landmarks and head pose
- Linear Ridge Regression mapper
- Polynomial (Degree-2) Ridge Regression mapper
- Strict screen boundary clamping [0.0, 1.0]
- Serialization, JSON saving, and loading
- Automated model evaluation metrics and model comparison
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.core.eye_features import extract_eye_features
from src.core.gaze_mapper import (
    BaseGazeMapper,
    LinearGazeMapper,
    PolynomialGazeMapper,
    compare_and_select_mapper,
    create_mapper,
    evaluate_mapper_predictions,
    extract_gaze_features,
    load_mapper_auto,
    FEATURE_NAMES,
)
from src.core.head_pose import HeadPose
from tests.conftest import create_synthetic_landmarks


def test_feature_extraction_valid_and_invalid():
    """Verify that extract_gaze_features creates expected 10-D feature vector."""
    landmarks = create_synthetic_landmarks(center_offset_x=0.0, center_offset_y=0.0)
    feats = extract_eye_features(landmarks)
    pose = HeadPose(yaw=6.0, pitch=-3.0, roll=0.0, is_valid=True)

    vec = extract_gaze_features(feats, head_pose=pose)
    assert vec is not None
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (10,)
    assert len(FEATURE_NAMES) == 10

    # Index 0: raw_x (~0.50), Index 1: raw_y (~0.50)
    assert abs(vec[0] - 0.50) < 0.10
    assert abs(vec[1] - 0.50) < 0.10
    # Head pose normalized by 30: 6/30 = 0.20, -3/30 = -0.10
    assert vec[8] == pytest.approx(0.20, abs=1e-3)
    assert vec[9] == pytest.approx(-0.10, abs=1e-3)

    # Invalid eye features should return None
    assert extract_gaze_features(None) is None


def test_linear_gaze_mapper_fitting_and_prediction():
    """Verify LinearGazeMapper learns linear relationship and predicts accurately."""
    # Synthetic linear mapping: Screen X = 0.8 * Eye X + 0.1, Screen Y = 0.8 * Eye Y + 0.1
    np.random.seed(42)
    N = 20
    D = 10
    X = np.random.uniform(0.2, 0.8, size=(N, D))
    y = np.zeros((N, 2))
    y[:, 0] = 0.8 * X[:, 0] + 0.1
    y[:, 1] = 0.8 * X[:, 1] + 0.1

    mapper = LinearGazeMapper(ridge_alpha=1e-4)
    assert not mapper.is_fitted
    mapper.fit(X, y)
    assert mapper.is_fitted
    assert mapper.weights is not None
    assert mapper.weights.shape == (D + 1, 2)

    # Test prediction
    preds = mapper.predict(X)
    assert preds.shape == (N, 2)
    assert np.all((preds >= 0.0) & (preds <= 1.0))
    # Predictions should closely match targets for linear relationship
    mae = np.mean(np.abs(preds - y))
    assert mae < 0.05


def test_polynomial_gaze_mapper_non_linear_fitting():
    """Verify PolynomialGazeMapper fits non-linear quadratic relationship."""
    np.random.seed(42)
    N = 30
    D = 10
    X = np.random.uniform(0.2, 0.8, size=(N, D))
    y = np.zeros((N, 2))
    # Non-linear quadratic transformation
    y[:, 0] = 0.5 * (X[:, 0] ** 2) + 0.3 * X[:, 0] + 0.1
    y[:, 1] = 0.4 * (X[:, 1] ** 2) + 0.4 * X[:, 1] + 0.1

    poly_mapper = PolynomialGazeMapper(ridge_alpha=1e-4).fit(X, y)
    assert poly_mapper.is_fitted
    preds = poly_mapper.predict(X)
    assert preds.shape == (N, 2)
    assert np.all((preds >= 0.0) & (preds <= 1.0))

    mae = np.mean(np.abs(preds - y))
    assert mae < 0.04


def test_mapper_boundary_clamping_invariants():
    """Verify that predictions are strictly clamped to [0.0, 1.0] even with extreme inputs."""
    X_train = np.array([
        [0.2] * 10,
        [0.8] * 10,
    ])
    y_train = np.array([
        [0.1, 0.1],
        [0.9, 0.9],
    ])

    mapper = LinearGazeMapper(ridge_alpha=0.01).fit(X_train, y_train)

    # Extreme off-screen inputs
    extreme_inputs = np.array([
        [-5.0] * 10,
        [+10.0] * 10,
    ])
    preds = mapper.predict(extreme_inputs)

    assert preds[0, 0] == 0.0
    assert preds[0, 1] == 0.0
    assert preds[1, 0] == 1.0
    assert preds[1, 1] == 1.0

    # Single 1D input array prediction
    single_pred = mapper.predict(np.array([+20.0] * 10))
    assert single_pred.shape == (2,)
    assert single_pred[0] == 1.0
    assert single_pred[1] == 1.0


def test_mapper_serialization_and_json_persistence():
    """Verify that mapper weights and configuration serialize and reload identically."""
    np.random.seed(123)
    X = np.random.uniform(0.2, 0.8, size=(10, 10))
    y = np.random.uniform(0.1, 0.9, size=(10, 2))

    for m_type in ["linear", "polynomial"]:
        mapper = create_mapper(m_type, ridge_alpha=0.02).fit(X, y)
        orig_preds = mapper.predict(X)

        # Dictionary roundtrip
        d = mapper.to_dict()
        restored = create_mapper(m_type).from_dict(d)
        assert restored.is_fitted
        assert np.allclose(restored.weights, mapper.weights)
        restored_preds = restored.predict(X)
        assert np.allclose(orig_preds, restored_preds)

        # File saving and loading roundtrip
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / f"test_model_{m_type}.json"
            mapper.save(file_path)
            assert file_path.exists()

            loaded = load_mapper_auto(file_path)
            assert loaded.model_type == m_type
            assert loaded.is_fitted
            loaded_preds = loaded.predict(X)
            assert np.allclose(orig_preds, loaded_preds)


def test_evaluate_mapper_predictions():
    """Verify statistical evaluation metrics calculations."""
    y_true = np.array([
        [0.10, 0.10],
        [0.50, 0.50],
        [0.90, 0.90],
    ])
    # Exact prediction: zero error
    perfect_metrics = evaluate_mapper_predictions(y_true, y_true, screen_width=1920, screen_height=1080)
    assert perfect_metrics["mae_x"] == 0.0
    assert perfect_metrics["mae_y"] == 0.0
    assert perfect_metrics["rmse_x"] == 0.0
    assert perfect_metrics["mean_euclidean_pixels"] == 0.0

    # Prediction with known offset: +0.02 on X, -0.01 on Y
    y_pred = y_true + np.array([0.02, -0.01])
    offset_metrics = evaluate_mapper_predictions(y_pred=y_pred, y_true=y_true, screen_width=1920, screen_height=1080)
    assert offset_metrics["mae_x"] == pytest.approx(0.02, abs=1e-5)
    assert offset_metrics["mae_y"] == pytest.approx(0.01, abs=1e-5)
    expected_norm_e = np.hypot(0.02, 0.01)
    assert offset_metrics["mean_euclidean_norm"] == pytest.approx(expected_norm_e, abs=1e-5)
    expected_px_e = np.hypot(0.02 * 1920, 0.01 * 1080)
    assert offset_metrics["mean_euclidean_pixels"] == pytest.approx(expected_px_e, abs=1e-2)


def test_compare_and_select_mapper():
    """Verify compare_and_select_mapper benchmarks and chooses best performing model."""
    np.random.seed(99)
    N = 25
    X_train = np.random.uniform(0.2, 0.8, size=(N, 10))
    # Quadratic non-linear target
    y_train = np.zeros((N, 2))
    y_train[:, 0] = 0.5 * (X_train[:, 0] ** 2) + 0.2 * X_train[:, 1] + 0.1
    y_train[:, 1] = 0.5 * (X_train[:, 1] ** 2) + 0.2 * X_train[:, 0] + 0.1

    X_val = np.random.uniform(0.2, 0.8, size=(10, 10))
    y_val = np.zeros((10, 2))
    y_val[:, 0] = 0.5 * (X_val[:, 0] ** 2) + 0.2 * X_val[:, 1] + 0.1
    y_val[:, 1] = 0.5 * (X_val[:, 1] ** 2) + 0.2 * X_val[:, 0] + 0.1

    best_mapper, report = compare_and_select_mapper(
        X_train, y_train, X_val, y_val, screen_width=1920, screen_height=1080
    )

    assert isinstance(best_mapper, BaseGazeMapper)
    assert best_mapper.is_fitted
    assert "selected_model" in report
    assert "linear_metrics" in report
    assert "polynomial_metrics" in report
