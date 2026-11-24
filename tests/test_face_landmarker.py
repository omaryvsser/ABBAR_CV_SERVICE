"""Unit tests for FaceLandmarkerDetector wrapper."""
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.core.face_landmarker import FaceLandmarkerDetectionResult, FaceLandmarkerDetector
from tests.conftest import MockNormalizedLandmark, create_synthetic_landmarks


def test_missing_model_file_raises_filenotfound():
    """Ensure missing model path raises FileNotFoundError with actionable message."""
    with pytest.raises(FileNotFoundError) as exc_info:
        FaceLandmarkerDetector(model_path="non_existent/path/model.task")
    assert "MediaPipe Face Landmarker model not found" in str(exc_info.value)


def test_empty_frame_handling():
    """Ensure processing empty or None frames returns has_face=False without crashing."""
    model_path = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
    if not model_path.exists():
        pytest.skip("Model file not present in models directory.")

    detector = FaceLandmarkerDetector(model_path=str(model_path), running_mode="IMAGE")
    try:
        # None frame
        res_none = detector.process_frame(None)
        assert res_none.has_face is False
        assert res_none.all_landmarks is None

        # Zero-sized frame
        res_empty = detector.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))
        assert res_empty.has_face is False
    finally:
        detector.close()


def test_no_face_detection_on_black_image():
    """Test detector execution on a blank black frame (no face present)."""
    model_path = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
    if not model_path.exists():
        pytest.skip("Model file not present in models directory.")

    detector = FaceLandmarkerDetector(model_path=str(model_path), running_mode="IMAGE")
    try:
        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.process_frame(black_img)
        assert isinstance(result, FaceLandmarkerDetectionResult)
        assert result.has_face is False
        assert result.all_landmarks is None
        assert result.left_eye_landmarks is None
        assert result.right_eye_landmarks is None
        assert result.inference_time_ms >= 0.0
    finally:
        detector.close()


def test_landmarks_slicing_with_mock():
    """Verify that detected landmarks are properly sliced into eye and iris subsets."""
    synthetic_landmarks = create_synthetic_landmarks()

    with patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options") as mock_create:
        mock_internal = MagicMock()
        mock_result = MagicMock()
        mock_result.face_landmarks = [synthetic_landmarks]
        mock_internal.detect.return_value = mock_result
        mock_create.return_value = mock_internal

        # Use dummy existing file path to satisfy init check
        dummy_task = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
        detector = FaceLandmarkerDetector(model_path=str(dummy_task), running_mode="IMAGE")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.process_frame(frame)

        assert result.has_face is True
        assert len(result.all_landmarks) == 478
        assert len(result.left_eye_landmarks) == 16
        assert len(result.right_eye_landmarks) == 16
        assert len(result.left_iris_landmarks) == 5
        assert len(result.right_iris_landmarks) == 5
        assert result.left_iris_center is not None
        assert result.right_iris_center is not None

        detector.close()
