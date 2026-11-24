"""
Face Landmarker Module using MediaPipe Solutions.

Detects a single user face, extracts 478 3D landmarks (face mesh + both irises),
and handles live webcam video streaming with temporal smoothing.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import Any, List, Optional, Sequence, Tuple
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from .landmark_indices import (
    LEFT_EYE_CONTOUR,
    LEFT_IRIS_ALL,
    LEFT_IRIS_CENTER,
    RIGHT_EYE_CONTOUR,
    RIGHT_IRIS_ALL,
    RIGHT_IRIS_CENTER,
    TOTAL_LANDMARKS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceLandmarkerDetectionResult:
    """
    Structured result from Face Landmarker processing on a single frame.

    Attributes:
        has_face: True if at least one face was detected with high confidence.
        all_landmarks: List of all 478 NormalizedLandmark objects (or None).
        left_eye_landmarks: Sublist of landmarks corresponding to left eye contour.
        right_eye_landmarks: Sublist of landmarks corresponding to right eye contour.
        left_iris_landmarks: Sublist of landmarks corresponding to left iris.
        right_iris_landmarks: Sublist of landmarks corresponding to right iris.
        left_iris_center: Explicit center landmark for left iris.
        right_iris_center: Explicit center landmark for right iris.
        timestamp_ms: Monotonic timestamp in milliseconds for video processing.
        inference_time_ms: Detection latency in milliseconds.
    """
    has_face: bool
    all_landmarks: Optional[List[Any]]
    left_eye_landmarks: Optional[List[Any]]
    right_eye_landmarks: Optional[List[Any]]
    left_iris_landmarks: Optional[List[Any]]
    right_iris_landmarks: Optional[List[Any]]
    left_iris_center: Optional[Any]
    right_iris_center: Optional[Any]
    timestamp_ms: int
    inference_time_ms: float


class FaceLandmarkerDetector:
    """
    Wrapper around MediaPipe FaceLandmarker Task for real-time video feeds.

    Operates in `RunningMode.VIDEO` by default to enable built-in temporal landmark
    smoothing and tracking without frame jitter.
    """

    def __init__(
        self,
        model_path: str = "models/face_landmarker.task",
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: str = "VIDEO",
    ) -> None:
        """
        Initialize the Face Landmarker detector.

        Args:
            model_path: Path to the downloaded face_landmarker.task model.
            num_faces: Maximum number of faces to detect (AABAR uses 1).
            min_face_detection_confidence: Confidence threshold for face detection.
            min_face_presence_confidence: Confidence threshold for landmark presence.
            min_tracking_confidence: Confidence threshold for temporal tracking.
            running_mode: "VIDEO" or "IMAGE".
        """
        self.model_path = Path(model_path).resolve()
        self.num_faces = num_faces
        self.min_face_detection_confidence = min_face_detection_confidence
        self.min_face_presence_confidence = min_face_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.running_mode_str = running_mode.upper()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe Face Landmarker model not found at:\n  {self.model_path}\n"
                f"Run `python download_model.py` to retrieve the official model bundle."
            )

        self._detector: Optional[vision.FaceLandmarker] = None
        self._last_timestamp_ms = -1
        self._init_detector()

    def _init_detector(self) -> None:
        """Instantiate MediaPipe FaceLandmarker with specified options."""
        logger.info(f"Loading FaceLandmarker model from {self.model_path}...")
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))

        mode = (
            vision.RunningMode.VIDEO
            if self.running_mode_str == "VIDEO"
            else vision.RunningMode.IMAGE
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=self.num_faces,
            min_face_detection_confidence=self.min_face_detection_confidence,
            min_face_presence_confidence=self.min_face_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            running_mode=mode,
        )

        self._detector = vision.FaceLandmarker.create_from_options(options)
        logger.info("FaceLandmarker successfully initialized.")

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> FaceLandmarkerDetectionResult:
        """
        Process a single BGR OpenCV frame and extract facial/iris landmarks.

        Args:
            bgr_frame: Input image in OpenCV BGR format.
            timestamp_ms: Monotonic timestamp in milliseconds (required for VIDEO mode).
                          If None, wall-clock time will be computed automatically.

        Returns:
            FaceLandmarkerDetectionResult with extracted landmarks and performance metadata.
        """
        if self._detector is None:
            raise RuntimeError("FaceLandmarker detector has been closed or is not initialized.")

        if bgr_frame is None or bgr_frame.size == 0:
            return FaceLandmarkerDetectionResult(
                has_face=False,
                all_landmarks=None,
                left_eye_landmarks=None,
                right_eye_landmarks=None,
                left_iris_landmarks=None,
                right_iris_landmarks=None,
                left_iris_center=None,
                right_iris_center=None,
                timestamp_ms=timestamp_ms or 0,
                inference_time_ms=0.0,
            )

        # Convert OpenCV BGR to RGB
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Ensure monotonically increasing timestamp for VIDEO mode
        if timestamp_ms is None:
            now_ms = int(time.time() * 1000)
            if now_ms <= self._last_timestamp_ms:
                now_ms = self._last_timestamp_ms + 1
            timestamp_ms = now_ms

        self._last_timestamp_ms = timestamp_ms

        t_start = time.perf_counter()
        try:
            if self.running_mode_str == "VIDEO":
                result = self._detector.detect_for_video(mp_image, timestamp_ms)
            else:
                result = self._detector.detect(mp_image)
        except Exception as e:
            logger.error(f"Error during FaceLandmarker detection: {e}", exc_info=True)
            return FaceLandmarkerDetectionResult(
                has_face=False,
                all_landmarks=None,
                left_eye_landmarks=None,
                right_eye_landmarks=None,
                left_iris_landmarks=None,
                right_iris_landmarks=None,
                left_iris_center=None,
                right_iris_center=None,
                timestamp_ms=timestamp_ms,
                inference_time_ms=0.0,
            )
        t_inference_ms = (time.perf_counter() - t_start) * 1000.0

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return FaceLandmarkerDetectionResult(
                has_face=False,
                all_landmarks=None,
                left_eye_landmarks=None,
                right_eye_landmarks=None,
                left_iris_landmarks=None,
                right_iris_landmarks=None,
                left_iris_center=None,
                right_iris_center=None,
                timestamp_ms=timestamp_ms,
                inference_time_ms=t_inference_ms,
            )

        # Single user face: primary face is index 0
        landmarks = result.face_landmarks[0]

        if len(landmarks) < TOTAL_LANDMARKS:
            logger.warning(
                f"Face detected with only {len(landmarks)} landmarks (expected {TOTAL_LANDMARKS}). "
                "Iris landmarks may be missing."
            )
            return FaceLandmarkerDetectionResult(
                has_face=True,
                all_landmarks=landmarks,
                left_eye_landmarks=[landmarks[i] for i in LEFT_EYE_CONTOUR if i < len(landmarks)],
                right_eye_landmarks=[landmarks[i] for i in RIGHT_EYE_CONTOUR if i < len(landmarks)],
                left_iris_landmarks=None,
                right_iris_landmarks=None,
                left_iris_center=None,
                right_iris_center=None,
                timestamp_ms=timestamp_ms,
                inference_time_ms=t_inference_ms,
            )

        # Slice specific subsets cleanly using centralized topology indices
        left_eye_lms = [landmarks[i] for i in LEFT_EYE_CONTOUR]
        right_eye_lms = [landmarks[i] for i in RIGHT_EYE_CONTOUR]
        left_iris_lms = [landmarks[i] for i in LEFT_IRIS_ALL]
        right_iris_lms = [landmarks[i] for i in RIGHT_IRIS_ALL]
        left_center = landmarks[LEFT_IRIS_CENTER]
        right_center = landmarks[RIGHT_IRIS_CENTER]

        return FaceLandmarkerDetectionResult(
            has_face=True,
            all_landmarks=landmarks,
            left_eye_landmarks=left_eye_lms,
            right_eye_landmarks=right_eye_lms,
            left_iris_landmarks=left_iris_lms,
            right_iris_landmarks=right_iris_lms,
            left_iris_center=left_center,
            right_iris_center=right_center,
            timestamp_ms=timestamp_ms,
            inference_time_ms=t_inference_ms,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._detector is not None:
            logger.info("Closing FaceLandmarker detector...")
            try:
                self._detector.close()
            except Exception as e:
                logger.debug(f"Exception during FaceLandmarker close: {e}")
            finally:
                self._detector = None

    def __enter__(self) -> "FaceLandmarkerDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
