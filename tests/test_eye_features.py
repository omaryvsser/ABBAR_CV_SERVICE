"""Tests for eye boundary and feature extraction."""
import pytest
from src.core.eye_features import (
    BinocularEyeFeatures,
    BoundingBox,
    Point2D,
    SingleEyeFeatures,
    extract_eye_features,
)
from tests.conftest import MockNormalizedLandmark, create_synthetic_landmarks


def test_extract_eye_features_centered(synthetic_face_landmarks):
    """Test feature extraction on centered eyes."""
    features = extract_eye_features(synthetic_face_landmarks)

    assert isinstance(features, BinocularEyeFeatures)
    assert features.is_valid is True
    assert features.left_eye is not None
    assert features.right_eye is not None

    # Centered iris should produce horizontal and vertical ratios approximately ~0.5
    assert 0.45 <= features.right_eye.horizontal_ratio <= 0.55
    assert 0.45 <= features.right_eye.vertical_ratio <= 0.55
    assert 0.45 <= features.left_eye.horizontal_ratio <= 0.55
    assert 0.45 <= features.left_eye.vertical_ratio <= 0.55

    assert features.average_horizontal_ratio is not None
    assert 0.45 <= features.average_horizontal_ratio <= 0.55
    assert features.average_vertical_ratio is not None
    assert 0.45 <= features.average_vertical_ratio <= 0.55


def test_iris_horizontal_movement():
    """
    Verify that moving iris horizontally shifts normalized ratio in Anatomical Gaze Space.

    Anatomical Gaze Space Convention:
    - Subject looking to anatomical LEFT  (iris shifts towards image-right / +X) -> horizontal_ratio < 0.40
    - Subject looking to anatomical RIGHT (iris shifts towards image-left / -X)  -> horizontal_ratio > 0.60
    """
    # Subject looking toward anatomical RIGHT (in camera image: iris shifts towards image-left)
    landmarks_right = create_synthetic_landmarks(center_offset_x=-0.02)
    feats_right = extract_eye_features(landmarks_right)
    assert feats_right.right_eye.horizontal_ratio > 0.60
    assert feats_right.left_eye.horizontal_ratio > 0.60

    # Subject looking toward anatomical LEFT (in camera image: iris shifts towards image-right)
    landmarks_left = create_synthetic_landmarks(center_offset_x=0.02)
    feats_left = extract_eye_features(landmarks_left)
    assert feats_left.right_eye.horizontal_ratio < 0.40
    assert feats_left.left_eye.horizontal_ratio < 0.40


def test_iris_vertical_movement():
    """Verify that moving iris vertically shifts normalized ratio accordingly."""
    # Iris looking up (negative Y offset in image coordinates)
    landmarks_up = create_synthetic_landmarks(center_offset_y=-0.01)
    feats_up = extract_eye_features(landmarks_up)
    assert feats_up.right_eye.vertical_ratio < 0.35

    # Iris looking down (positive Y offset)
    landmarks_down = create_synthetic_landmarks(center_offset_y=0.01)
    feats_down = extract_eye_features(landmarks_down)
    assert feats_down.right_eye.vertical_ratio > 0.65


def test_clamping_ratio_bounds():
    """Ensure ratios never exceed [0.0, 1.0] even with extreme off-screen iris displacement."""
    landmarks_extreme = create_synthetic_landmarks(center_offset_x=0.50, center_offset_y=-0.50)
    feats = extract_eye_features(landmarks_extreme)

    assert 0.0 <= feats.right_eye.horizontal_ratio <= 1.0
    assert 0.0 <= feats.right_eye.vertical_ratio <= 1.0
    assert 0.0 <= feats.left_eye.horizontal_ratio <= 1.0
    assert 0.0 <= feats.left_eye.vertical_ratio <= 1.0


def test_missing_landmarks_safety():
    """Ensure missing or None landmarks are handled gracefully without raising exceptions."""
    # None input
    res_none = extract_eye_features(None)
    assert res_none.is_valid is False
    assert res_none.left_eye is None
    assert res_none.right_eye is None
    assert res_none.average_horizontal_ratio is None

    # Empty list
    res_empty = extract_eye_features([])
    assert res_empty.is_valid is False

    # Truncated landmarks list (only 100 points, missing irises)
    res_truncated = extract_eye_features([MockNormalizedLandmark(0.5, 0.5) for _ in range(100)])
    assert res_truncated.is_valid is False


def test_degenerate_zero_size_eye_boundary():
    """Ensure degenerate eye boundaries (all landmarks identical point) do not divide by zero."""
    collapsed = [MockNormalizedLandmark(0.5, 0.5) for _ in range(478)]
    feats = extract_eye_features(collapsed)
    # Should safely return fallback ratios and flag invalid
    assert feats.right_eye.is_valid is False
    assert feats.left_eye.is_valid is False
    assert feats.is_valid is False


def test_blink_detection():
    """Verify that closed eyelid height triggers blink flag."""
    # When eye height is almost zero (0.002)
    closed_landmarks = create_synthetic_landmarks(eye_open_height=0.002)
    feats = extract_eye_features(closed_landmarks)

    assert feats.right_eye.is_blink is True
    assert feats.left_eye.is_blink is True
    # If both eyes are blinking, overall binocular is_valid should be False (unusable gaze)
    assert feats.is_valid is False


def test_point2d_pixel_conversion():
    """Test Point2D conversion to image pixel coordinates."""
    p = Point2D(x=0.5, y=0.25)
    px, py = p.to_pixel(image_width=1000, image_height=800)
    assert px == 500
    assert py == 200

    # Test clamping on boundary extremes
    p_neg = Point2D(x=-0.5, y=1.5)
    px_c, py_c = p_neg.to_pixel(image_width=1000, image_height=800)
    assert px_c == 0
    assert py_c == 799


def test_calculate_eye_openness():
    """Verify EAR mapping to normalized openness score and categorical state."""
    from src.core.eye_features import EyeOpennessState, calculate_eye_openness

    # Closed eye (EAR = 0.14 < 0.16)
    score_closed, state_closed = calculate_eye_openness(0.14, ear_min=0.16, ear_max=0.32)
    assert score_closed == 0.0
    assert state_closed == EyeOpennessState.CLOSED

    # Fully open eye (EAR = 0.32)
    score_open, state_open = calculate_eye_openness(0.32, ear_min=0.16, ear_max=0.32)
    assert score_open == 1.0
    assert state_open == EyeOpennessState.OPEN

    # Partially closed eye (EAR = 0.22 -> ~0.375 openness)
    score_partial, state_partial = calculate_eye_openness(0.22, ear_min=0.16, ear_max=0.32)
    assert 0.25 <= score_partial < 0.65
    assert state_partial == EyeOpennessState.PARTIALLY_CLOSED

    # Invalid / negative EAR
    score_inv, state_inv = calculate_eye_openness(-1.0)
    assert score_inv == 0.0
    assert state_inv == EyeOpennessState.INVALID


def test_single_eye_occlusion_fallback():
    """Verify that when one eye is closed/occluded, gaze falls back to the open eye."""
    # Create landmarks where right eye is open and left eye is closed
    landmarks = create_synthetic_landmarks(eye_open_height=0.03)

    # Collapse left eyelid landmarks to simulate left eye closure
    landmarks[386] = MockNormalizedLandmark(x=0.58, y=0.35, z=0.0)
    landmarks[374] = MockNormalizedLandmark(x=0.58, y=0.35, z=0.0)
    landmarks[385] = MockNormalizedLandmark(x=0.58, y=0.35, z=0.0)
    landmarks[380] = MockNormalizedLandmark(x=0.58, y=0.35, z=0.0)
    landmarks[387] = MockNormalizedLandmark(x=0.58, y=0.35, z=0.0)
    landmarks[373] = MockNormalizedLandmark(x=0.58, y=0.35, z=0.0)

    feats = extract_eye_features(landmarks)

    assert feats.right_eye.is_valid is True
    assert feats.left_eye.is_blink is True

    # Fused gaze should still be valid, falling back to the right eye
    assert feats.is_valid is True
    assert feats.combined_horizontal_ratio == pytest.approx(feats.right_eye.horizontal_ratio)
    assert feats.combined_vertical_ratio == pytest.approx(feats.right_eye.vertical_ratio)
    # Confidence should be capped due to single eye occlusion
    assert feats.confidence <= 0.65
