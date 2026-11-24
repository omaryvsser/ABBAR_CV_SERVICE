"""Pytest configuration and synthetic fixtures for AABAR CV tests."""
from dataclasses import dataclass
from typing import List
import numpy as np
import pytest

from src.core.landmark_indices import (
    LEFT_EYE_CONTOUR,
    LEFT_IRIS_ALL,
    LEFT_IRIS_CENTER,
    RIGHT_EYE_CONTOUR,
    RIGHT_IRIS_ALL,
    RIGHT_IRIS_CENTER,
    TOTAL_LANDMARKS,
)


@dataclass
class MockNormalizedLandmark:
    """Mock MediaPipe NormalizedLandmark object."""
    x: float
    y: float
    z: float = 0.0


def create_synthetic_landmarks(
    center_offset_x: float = 0.0,
    center_offset_y: float = 0.0,
    eye_open_height: float = 0.03,
) -> List[MockNormalizedLandmark]:
    """
    Generate synthetic 478 face landmarks with realistic facial geometry.

    Right Eye: centered around (x=0.42, y=0.35)
    Left Eye:  centered around (x=0.58, y=0.35)
    Nose tip:  centered at (x=0.50, y=0.48)
    Chin:      at (x=0.50, y=0.74)
    """
    landmarks = [MockNormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(TOTAL_LANDMARKS)]

    # 1. Populate Right Eye (subject right, viewer left)
    r_cx, r_cy = 0.42, 0.35
    r_w = 0.06
    r_h = eye_open_height

    for idx in RIGHT_EYE_CONTOUR:
        landmarks[idx] = MockNormalizedLandmark(x=r_cx, y=r_cy, z=0.0)

    landmarks[33] = MockNormalizedLandmark(x=r_cx - r_w / 2.0, y=r_cy, z=0.0)    # Lateral corner
    landmarks[133] = MockNormalizedLandmark(x=r_cx + r_w / 2.0, y=r_cy, z=0.0)   # Medial corner
    landmarks[159] = MockNormalizedLandmark(x=r_cx, y=r_cy - r_h / 2.0, z=0.0)   # Top margin
    landmarks[145] = MockNormalizedLandmark(x=r_cx, y=r_cy + r_h / 2.0, z=0.0)   # Bottom margin

    # Vertical EAR landmarks
    landmarks[160] = MockNormalizedLandmark(x=r_cx - 0.01, y=r_cy - r_h / 2.0, z=0.0)
    landmarks[144] = MockNormalizedLandmark(x=r_cx - 0.01, y=r_cy + r_h / 2.0, z=0.0)
    landmarks[158] = MockNormalizedLandmark(x=r_cx + 0.01, y=r_cy - r_h / 2.0, z=0.0)
    landmarks[153] = MockNormalizedLandmark(x=r_cx + 0.01, y=r_cy + r_h / 2.0, z=0.0)

    # Right Iris
    r_iris_x = r_cx + center_offset_x
    r_iris_y = r_cy + center_offset_y
    landmarks[RIGHT_IRIS_CENTER] = MockNormalizedLandmark(x=r_iris_x, y=r_iris_y, z=0.0)
    for idx in RIGHT_IRIS_ALL:
        landmarks[idx] = MockNormalizedLandmark(x=r_iris_x, y=r_iris_y, z=0.0)

    # 2. Populate Left Eye (subject left, viewer right)
    l_cx, l_cy = 0.58, 0.35
    l_w = 0.06
    l_h = eye_open_height

    for idx in LEFT_EYE_CONTOUR:
        landmarks[idx] = MockNormalizedLandmark(x=l_cx, y=l_cy, z=0.0)

    landmarks[362] = MockNormalizedLandmark(x=l_cx - l_w / 2.0, y=l_cy, z=0.0)   # Medial corner
    landmarks[263] = MockNormalizedLandmark(x=l_cx + l_w / 2.0, y=l_cy, z=0.0)   # Lateral corner
    landmarks[386] = MockNormalizedLandmark(x=l_cx, y=l_cy - l_h / 2.0, z=0.0)   # Top margin
    landmarks[374] = MockNormalizedLandmark(x=l_cx, y=l_cy + l_h / 2.0, z=0.0)   # Bottom margin

    landmarks[385] = MockNormalizedLandmark(x=l_cx - 0.01, y=l_cy - l_h / 2.0, z=0.0)
    landmarks[380] = MockNormalizedLandmark(x=l_cx - 0.01, y=l_cy + l_h / 2.0, z=0.0)
    landmarks[387] = MockNormalizedLandmark(x=l_cx + 0.01, y=l_cy - l_h / 2.0, z=0.0)
    landmarks[373] = MockNormalizedLandmark(x=l_cx + 0.01, y=l_cy + l_h / 2.0, z=0.0)

    # Left Iris
    l_iris_x = l_cx + center_offset_x
    l_iris_y = l_cy + center_offset_y
    landmarks[LEFT_IRIS_CENTER] = MockNormalizedLandmark(x=l_iris_x, y=l_iris_y, z=0.0)
    for idx in LEFT_IRIS_ALL:
        landmarks[idx] = MockNormalizedLandmark(x=l_iris_x, y=l_iris_y, z=0.0)

    # 3. Canonical Head Pose Landmarks (solvePnP)
    landmarks[1] = MockNormalizedLandmark(x=0.50, y=0.48, z=0.0)    # Nose tip
    landmarks[152] = MockNormalizedLandmark(x=0.50, y=0.74, z=0.0)  # Chin
    landmarks[291] = MockNormalizedLandmark(x=0.56, y=0.60, z=0.0)  # Left mouth corner
    landmarks[61] = MockNormalizedLandmark(x=0.44, y=0.60, z=0.0)   # Right mouth corner

    return landmarks


@pytest.fixture
def synthetic_face_landmarks() -> List[MockNormalizedLandmark]:
    """Provide standard centered synthetic face landmarks fixture."""
    return create_synthetic_landmarks()


@pytest.fixture
def dummy_bgr_frame() -> np.ndarray:
    """Provide a 720x1280 3-channel dummy image frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)
