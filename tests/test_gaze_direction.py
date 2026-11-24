"""Unit tests for Relative Gaze Direction Classification."""
import pytest
from src.core.gaze_direction import GazeDirection, GazeDirectionClassifier


def test_center_gaze_classification():
    """Verify neutral center gaze classification."""
    classifier = GazeDirectionClassifier(
        horizontal_center_min=0.42,
        horizontal_center_max=0.58,
        vertical_center_min=0.38,
        vertical_center_max=0.62,
    )
    assert classifier.classify(0.50, 0.50) == GazeDirection.CENTER
    assert classifier.classify(0.45, 0.55) == GazeDirection.CENTER


def test_cardinal_directions():
    """Verify primary 4 cardinal directions."""
    classifier = GazeDirectionClassifier()

    # Looking Left (subject right / image left)
    assert classifier.classify(0.30, 0.50) == GazeDirection.LEFT

    # Looking Right (subject left / image right)
    assert classifier.classify(0.70, 0.50) == GazeDirection.RIGHT

    # Looking Up
    assert classifier.classify(0.50, 0.25) == GazeDirection.UP

    # Looking Down
    assert classifier.classify(0.50, 0.75) == GazeDirection.DOWN


def test_compound_diagonal_directions():
    """Verify compound diagonal directions when enabled."""
    classifier = GazeDirectionClassifier(allow_compound_directions=True)

    assert classifier.classify(0.30, 0.25) == GazeDirection.UP_LEFT
    assert classifier.classify(0.70, 0.25) == GazeDirection.UP_RIGHT
    assert classifier.classify(0.30, 0.75) == GazeDirection.DOWN_LEFT
    assert classifier.classify(0.70, 0.75) == GazeDirection.DOWN_RIGHT


def test_dominant_axis_when_compound_disabled():
    """When compound is disabled, the axis with larger deviation from neutral should dominate."""
    classifier = GazeDirectionClassifier(allow_compound_directions=False)

    # h_dev = |0.20 - 0.50| = 0.30, v_dev = |0.35 - 0.50| = 0.15 -> Horizontal dominates
    assert classifier.classify(0.20, 0.35) == GazeDirection.LEFT

    # v_dev = |0.15 - 0.50| = 0.35, h_dev = |0.40 - 0.50| = 0.10 -> Vertical dominates
    assert classifier.classify(0.40, 0.15) == GazeDirection.UP


def test_missing_or_invalid_inputs():
    """Verify handling of None inputs."""
    classifier = GazeDirectionClassifier()
    assert classifier.classify(None, 0.50) == GazeDirection.UNKNOWN
    assert classifier.classify(0.50, None) == GazeDirection.UNKNOWN
    assert classifier.classify(None, None) == GazeDirection.UNKNOWN


def test_configurable_custom_thresholds():
    """Verify custom user thresholds alter classification bounds."""
    # Stricter deadzone: 0.48 to 0.52
    strict_classifier = GazeDirectionClassifier(
        horizontal_center_min=0.48,
        horizontal_center_max=0.52,
    )
    # 0.45 was CENTER under default (0.42..0.58), but is LEFT under strict
    assert strict_classifier.classify(0.45, 0.50) == GazeDirection.LEFT
