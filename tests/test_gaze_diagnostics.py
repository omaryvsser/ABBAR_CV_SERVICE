"""
Diagnostic Verification Tests for Gaze Direction and Coordinate Conventions.

Validates:
1. Anatomical Gaze Convention:
   - Subject looking to their anatomical RIGHT -> GazeDirection.RIGHT
   - Subject looking to their anatomical LEFT  -> GazeDirection.LEFT
   - Subject looking UP                        -> GazeDirection.UP
   - Subject looking DOWN                      -> GazeDirection.DOWN
   - Subject looking CENTER                    -> GazeDirection.CENTER
2. Separation of raw camera frame from rendered display frame.
3. Landmark anatomical topology verification (left vs right eye canthi).
"""

import numpy as np
import pytest

from src.config import VisualizationConfig
from src.core.eye_features import extract_eye_features
from src.core.face_landmarker import FaceLandmarkerDetectionResult
from src.core.gaze_direction import GazeDirection, GazeDirectionClassifier
from src.core.landmark_indices import (
    LEFT_EYE_INNER_CANTHUS,
    LEFT_EYE_OUTER_CANTHUS,
    LEFT_IRIS_CENTER,
    NOSE_TIP,
    RIGHT_EYE_INNER_CANTHUS,
    RIGHT_EYE_OUTER_CANTHUS,
    RIGHT_IRIS_CENTER,
)
from src.utils.fps_counter import FPSMetrics
from src.visualization.visualizer import CVVisualizer
from tests.conftest import MockNormalizedLandmark, create_synthetic_landmarks


def test_anatomical_gaze_left_right_verification():
    """
    CRITICAL DIAGNOSTIC TEST:
    Verify that horizontal gaze classification matches the user's actual anatomical direction.

    Scenario:
    - User looks to their anatomical RIGHT (toward user's right shoulder):
      In camera image, irises deflect towards image-left (smaller X).
      System must report: GazeDirection.RIGHT.
    - User looks to their anatomical LEFT (toward user's left shoulder):
      In camera image, irises deflect towards image-right (larger X).
      System must report: GazeDirection.LEFT.
    """
    classifier = GazeDirectionClassifier()

    # 1. Neutral Center Gaze
    center_landmarks = create_synthetic_landmarks(center_offset_x=0.0, center_offset_y=0.0)
    feats_center = extract_eye_features(center_landmarks)
    assert feats_center.is_valid is True
    dir_center = classifier.classify(feats_center.combined_horizontal_ratio, feats_center.combined_vertical_ratio)
    assert dir_center == GazeDirection.CENTER

    # 2. Subject looks to their anatomical RIGHT (negative offset in camera X)
    right_landmarks = create_synthetic_landmarks(center_offset_x=-0.02, center_offset_y=0.0)
    feats_right = extract_eye_features(right_landmarks)
    assert feats_right.combined_horizontal_ratio > 0.58
    dir_right = classifier.classify(feats_right.combined_horizontal_ratio, feats_right.combined_vertical_ratio)
    assert dir_right == GazeDirection.RIGHT, f"Expected RIGHT, got {dir_right}"

    # 3. Subject looks to their anatomical LEFT (positive offset in camera X)
    left_landmarks = create_synthetic_landmarks(center_offset_x=0.02, center_offset_y=0.0)
    feats_left = extract_eye_features(left_landmarks)
    assert feats_left.combined_horizontal_ratio < 0.42
    dir_left = classifier.classify(feats_left.combined_horizontal_ratio, feats_left.combined_vertical_ratio)
    assert dir_left == GazeDirection.LEFT, f"Expected LEFT, got {dir_left}"

    # 4. Subject looks UP (negative offset in camera Y)
    up_landmarks = create_synthetic_landmarks(center_offset_x=0.0, center_offset_y=-0.01)
    feats_up = extract_eye_features(up_landmarks)
    assert feats_up.combined_vertical_ratio < 0.38
    dir_up = classifier.classify(feats_up.combined_horizontal_ratio, feats_up.combined_vertical_ratio)
    assert dir_up == GazeDirection.UP, f"Expected UP, got {dir_up}"

    # 5. Subject looks DOWN (positive offset in camera Y)
    down_landmarks = create_synthetic_landmarks(center_offset_x=0.0, center_offset_y=0.01)
    feats_down = extract_eye_features(down_landmarks)
    assert feats_down.combined_vertical_ratio > 0.62
    dir_down = classifier.classify(feats_down.combined_horizontal_ratio, feats_down.combined_vertical_ratio)
    assert dir_down == GazeDirection.DOWN, f"Expected DOWN, got {dir_down}"


def test_landmark_anatomical_canthi_topology():
    """
    Verify landmark canthi positions correspond to the correct anatomical eyes.
    In unmirrored camera coordinates:
    - Subject's right eye is on image left: X_outer_canthus (33) < X_inner_canthus (133) < X_nose (1)
    - Subject's left eye is on image right: X_nose (1) < X_inner_canthus (362) < X_outer_canthus (263)
    """
    landmarks = create_synthetic_landmarks()

    nose_x = landmarks[NOSE_TIP].x
    r_outer_x = landmarks[RIGHT_EYE_OUTER_CANTHUS].x
    r_inner_x = landmarks[RIGHT_EYE_INNER_CANTHUS].x
    r_iris_x = landmarks[RIGHT_IRIS_CENTER].x

    l_inner_x = landmarks[LEFT_EYE_INNER_CANTHUS].x
    l_outer_x = landmarks[LEFT_EYE_OUTER_CANTHUS].x
    l_iris_x = landmarks[LEFT_IRIS_CENTER].x

    # Subject Right Eye: entirely to the left of the nose
    assert r_outer_x < r_inner_x < nose_x
    assert r_outer_x <= r_iris_x <= r_inner_x

    # Subject Left Eye: entirely to the right of the nose
    assert nose_x < l_inner_x < l_outer_x
    assert l_inner_x <= l_iris_x <= l_outer_x


def test_display_frame_separation_and_immutability():
    """
    Verify that rendering the visualizer does not mutate the raw camera frame,
    and correctly respects the mirror_display configuration.
    """
    raw_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    raw_frame_copy = raw_frame.copy()

    landmarks = create_synthetic_landmarks()
    detection = FaceLandmarkerDetectionResult(
        has_face=True,
        all_landmarks=landmarks,
        left_eye_landmarks=[],
        right_eye_landmarks=[],
        left_iris_landmarks=[],
        right_iris_landmarks=[],
        left_iris_center=landmarks[LEFT_IRIS_CENTER],
        right_iris_center=landmarks[RIGHT_IRIS_CENTER],
        timestamp_ms=1000,
        inference_time_ms=10.0,
    )
    eyes = extract_eye_features(landmarks)
    fps_metrics = FPSMetrics(
        fps=30.0,
        frame_time_ms=33.3,
        detection_latency_ms=10.0,
        total_frames=1,
        uptime_seconds=0.033,
    )

    # 1. Render with mirror_display=True
    viz_mirrored = CVVisualizer(config=VisualizationConfig(enabled=True, mirror_display=True))
    annotated_mirrored = viz_mirrored.render(
        frame=raw_frame,
        detection_result=detection,
        eye_features=eyes,
        fps_metrics=fps_metrics,
        gaze_direction=GazeDirection.RIGHT,
    )

    # Raw frame must remain strictly unmutated
    assert np.array_equal(raw_frame, raw_frame_copy), "CVVisualizer must not mutate the input raw frame!"
    assert annotated_mirrored is not None
    assert annotated_mirrored.shape == raw_frame.shape

    # 2. Render with mirror_display=False
    viz_unmirrored = CVVisualizer(config=VisualizationConfig(enabled=True, mirror_display=False))
    annotated_unmirrored = viz_unmirrored.render(
        frame=raw_frame,
        detection_result=detection,
        eye_features=eyes,
        fps_metrics=fps_metrics,
        gaze_direction=GazeDirection.RIGHT,
    )

    # Raw frame still unmutated
    assert np.array_equal(raw_frame, raw_frame_copy)
    assert annotated_unmirrored is not None
