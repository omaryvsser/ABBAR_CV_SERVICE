"""Unit tests for 3D Head Pose Estimation."""
from dataclasses import dataclass
import pytest
from src.core.head_pose import HeadPose, HeadPoseEstimator
from tests.conftest import MockNormalizedLandmark, create_synthetic_landmarks


def test_head_pose_estimation_centered(synthetic_face_landmarks):
    """Verify head pose estimation on centered frontal face."""
    estimator = HeadPoseEstimator()
    pose = estimator.estimate(synthetic_face_landmarks, image_width=1280, image_height=720)

    assert isinstance(pose, HeadPose)
    assert pose.is_valid is True
    # Frontal synthetic face should be within +/- 15 degrees of neutral
    assert abs(pose.yaw) < 20.0
    assert abs(pose.pitch) < 20.0
    assert abs(pose.roll) < 20.0

    pose_dict = pose.to_dict()
    assert "yaw" in pose_dict
    assert "pitch" in pose_dict
    assert "roll" in pose_dict


def test_head_pose_turn_detection():
    """Verify that shifting facial landmarks horizontally alters estimated yaw."""
    estimator = HeadPoseEstimator()

    # Shift nose tip toward subject's left (positive X in image coordinates)
    landmarks_turned = create_synthetic_landmarks()
    # Nose tip is index 1
    landmarks_turned[1] = MockNormalizedLandmark(x=0.56, y=0.50, z=0.0)

    pose_turned = estimator.estimate(landmarks_turned, image_width=1280, image_height=720)
    assert pose_turned.is_valid is True
    # Turning head causes significant non-zero yaw
    assert abs(pose_turned.yaw) > 2.0


def test_head_pose_missing_landmarks():
    """Verify graceful handling when landmarks are None or truncated."""
    estimator = HeadPoseEstimator()

    pose_none = estimator.estimate(None, image_width=1280, image_height=720)
    assert pose_none.is_valid is False
    assert pose_none.yaw == 0.0

    pose_truncated = estimator.estimate([MockNormalizedLandmark(0.5, 0.5) for _ in range(50)], 1280, 720)
    assert pose_truncated.is_valid is False
