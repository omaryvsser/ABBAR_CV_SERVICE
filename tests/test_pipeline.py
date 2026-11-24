"""Integration tests for CVPipeline with mocked camera frames."""
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.config import AppConfig
from src.pipeline import CVPipeline, PipelineOutput


def test_pipeline_initialization_and_frame_cycle(dummy_bgr_frame):
    """Verify complete end-to-end processing of a frame through CVPipeline."""
    model_path = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
    if not model_path.exists():
        pytest.skip("Model bundle required for pipeline test.")

    config = AppConfig()
    config.camera.camera_index = 0
    config.camera.use_threading = False
    config.landmarker.model_path = str(model_path)
    config.visualization.enabled = True

    # Patch CameraManager to return dummy frames without requiring physical camera
    with patch("src.camera.camera_manager.CameraManager.start"):
        with patch("src.camera.camera_manager.CameraManager.is_opened", return_value=True):
            with patch("src.camera.camera_manager.CameraManager.read", return_value=(True, dummy_bgr_frame)):
                pipeline = CVPipeline(config=config)
                pipeline._is_running = True

                output = pipeline.process_next_frame()

                assert output is not None
                assert isinstance(output, PipelineOutput)
                assert output.raw_frame.shape == (720, 1280, 3)
                assert output.detection_result is not None
                assert output.eye_features is not None
                assert output.fps_metrics is not None
                assert output.annotated_frame is not None
                assert output.annotated_frame.shape == (720, 1280, 3)
                assert output.screen_gaze is not None
                assert output.screen_gaze.is_calibrated is False
                assert output.gaze_frame_result.screen_gaze is not None

                # Test screen calibration trigger and cancellation
                pipeline.start_screen_calibration()
                assert pipeline.screen_calibrator.is_active
                pipeline.cancel_screen_calibration()
                assert not pipeline.screen_calibrator.is_active

                # Test activating a mapper produces calibrated screen gaze predictions
                from src.core.gaze_mapper import LinearGazeMapper
                dummy_mapper = LinearGazeMapper()
                dummy_mapper.fit(np.zeros((3, 10)), np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]))
                pipeline.active_mapper = dummy_mapper

                # Next frame should produce calibrated output container
                out_calib = pipeline.process_next_frame()
                assert out_calib is not None
                assert out_calib.screen_gaze.is_calibrated is True

                pipeline.close()
