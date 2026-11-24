"""
Unit tests for Calibration Profile Management (Phase 3).

Covers:
- Profile serialization and deserialization
- File saving and loading
- Hardware and resolution compatibility checking
- Mapper reconstruction from saved profile
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.core.calibration_profile import CalibrationProfile, ProfileManager
from src.core.gaze_mapper import LinearGazeMapper, PolynomialGazeMapper


def test_profile_serialization_roundtrip():
    """Verify profile converts to and from dictionary identically."""
    profile = CalibrationProfile(
        profile_id="prof_test_001",
        user_id="als_user_1",
        screen_width=1920,
        screen_height=1080,
        camera_index=0,
        model_type="polynomial",
        model_data={"weights": [[1.0, 2.0]], "ridge_alpha": 0.01, "is_fitted": True},
        training_metrics={"mae_x": 0.025, "mean_euclidean_pixels": 42.5},
    )

    d = profile.to_dict()
    restored = CalibrationProfile.from_dict(d)

    assert restored.profile_id == profile.profile_id
    assert restored.user_id == profile.user_id
    assert restored.screen_width == 1920
    assert restored.screen_height == 1080
    assert restored.model_type == "polynomial"
    assert restored.training_metrics["mae_x"] == 0.025


def test_profile_manager_save_and_load():
    """Verify ProfileManager saves to and loads from disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ProfileManager(profiles_dir=Path(tmpdir))

        # Train a polynomial mapper
        X = np.random.uniform(0.2, 0.8, size=(10, 10))
        y = np.random.uniform(0.1, 0.9, size=(10, 2))
        mapper = PolynomialGazeMapper(ridge_alpha=0.01).fit(X, y)

        profile = CalibrationProfile(
            profile_id="user_omar_session1",
            user_id="omar",
            screen_width=2560,
            screen_height=1440,
            model_type=mapper.model_type,
            model_data=mapper.to_dict(),
        )

        saved_path = manager.save_profile(profile)
        assert saved_path.exists()

        # Load back
        loaded = manager.load_profile(saved_path)
        assert loaded.profile_id == "user_omar_session1"
        assert loaded.screen_width == 2560

        # Extract mapper and test prediction matches
        extracted_mapper = manager.extract_mapper(loaded)
        assert extracted_mapper.is_fitted
        orig_preds = mapper.predict(X)
        extracted_preds = extracted_mapper.predict(X)
        assert np.allclose(orig_preds, extracted_preds)


def test_profile_compatibility_validation():
    """Verify compatibility check detects display resolution changes."""
    manager = ProfileManager()
    profile = CalibrationProfile(
        profile_id="p1",
        screen_width=1920,
        screen_height=1080,
        camera_index=0,
    )

    # Identical resolution -> Compatible
    compat, msg = manager.validate_compatibility(profile, current_screen_w=1920, current_screen_h=1080)
    assert compat is True

    # Changed resolution -> Incompatible
    compat_bad, msg_bad = manager.validate_compatibility(profile, current_screen_w=1280, current_screen_h=720)
    assert compat_bad is False
    assert "Resolution mismatch" in msg_bad
