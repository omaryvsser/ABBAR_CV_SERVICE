"""Unit tests for CameraManager component with mocked video capture."""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from src.camera.camera_manager import CameraInitializationError, CameraManager


def test_camera_init_failure_raises_exception():
    """Verify CameraInitializationError is raised when device fails to open."""
    with patch("cv2.VideoCapture") as mock_vc_cls:
        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = False
        mock_vc_cls.return_value = mock_vc

        cam = CameraManager(camera_index=99, use_threading=False)
        with pytest.raises(CameraInitializationError) as exc_info:
            cam.start()

        assert "Failed to open video capture device" in str(exc_info.value)
        mock_vc.release.assert_called_once()


def test_camera_initial_frame_failure():
    """Verify error raised if camera opens but fails to read initial test frame."""
    with patch("cv2.VideoCapture") as mock_vc_cls:
        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = True
        mock_vc.read.return_value = (False, None)
        mock_vc_cls.return_value = mock_vc

        cam = CameraManager(camera_index=0, use_threading=False)
        with pytest.raises(CameraInitializationError) as exc_info:
            cam.start()

        assert "failed to capture initial test frame" in str(exc_info.value)
        mock_vc.release.assert_called()


def test_camera_successful_synchronous_read():
    """Test successful camera lifecycle in synchronous mode."""
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 120

    with patch("cv2.VideoCapture") as mock_vc_cls:
        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = True
        mock_vc.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)
        mock_vc.read.return_value = (True, dummy_frame)
        mock_vc_cls.return_value = mock_vc

        with CameraManager(camera_index=0, use_threading=False) as cam:
            assert cam.is_opened is True
            ret, frame = cam.read()
            assert ret is True
            assert frame is not None
            assert frame.shape == (480, 640, 3)

        # After with-block, camera should be released
        assert cam.is_opened is False
        mock_vc.release.assert_called()


def test_camera_threaded_read():
    """Test camera lifecycle in threaded worker mode."""
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200

    with patch("cv2.VideoCapture") as mock_vc_cls:
        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = True
        mock_vc.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)
        mock_vc.read.return_value = (True, dummy_frame)
        mock_vc_cls.return_value = mock_vc

        cam = CameraManager(camera_index=0, use_threading=True)
        cam.start()

        assert cam.is_opened is True
        ret, frame = cam.read()
        assert ret is True
        assert frame is not None

        cam.release()
        assert cam.is_opened is False
