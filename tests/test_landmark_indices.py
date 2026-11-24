"""Tests for centralized landmark topology and index definitions."""
import pytest
from src.core.landmark_indices import (
    ALL_EYE_LANDMARKS,
    ALL_IRIS_LANDMARKS,
    LEFT_EYE_CONTOUR,
    LEFT_IRIS_ALL,
    LEFT_IRIS_CENTER,
    NUM_FACE_MESH_LANDMARKS,
    NUM_IRIS_LANDMARKS,
    RIGHT_EYE_CONTOUR,
    RIGHT_IRIS_ALL,
    RIGHT_IRIS_CENTER,
    TOTAL_LANDMARKS,
    validate_topology,
)


def test_topology_validation_passes():
    """Verify that internal topology validation succeeds with zero errors."""
    assert validate_topology() is True


def test_landmark_count_constants():
    """Ensure total landmark counts match MediaPipe specification."""
    assert TOTAL_LANDMARKS == 478
    assert NUM_FACE_MESH_LANDMARKS == 468
    assert NUM_IRIS_LANDMARKS == 10
    assert NUM_FACE_MESH_LANDMARKS + NUM_IRIS_LANDMARKS == TOTAL_LANDMARKS


def test_iris_indices_range_and_uniqueness():
    """Validate that iris landmark indices are distinct and bounded within 468..477."""
    assert len(RIGHT_IRIS_ALL) == 5
    assert len(LEFT_IRIS_ALL) == 5
    assert len(ALL_IRIS_LANDMARKS) == 10

    for idx in ALL_IRIS_LANDMARKS:
        assert 468 <= idx < 478

    assert RIGHT_IRIS_CENTER == 468
    assert LEFT_IRIS_CENTER == 473


def test_no_overlap_between_eyes():
    """Ensure left and right eye contours share no overlapping landmark indices."""
    r_set = set(RIGHT_EYE_CONTOUR)
    l_set = set(LEFT_EYE_CONTOUR)
    assert len(r_set.intersection(l_set)) == 0


def test_all_indices_within_valid_bounds():
    """Verify every index in ALL_EYE_LANDMARKS is within [0, TOTAL_LANDMARKS - 1]."""
    for idx in ALL_EYE_LANDMARKS:
        assert 0 <= idx < TOTAL_LANDMARKS
