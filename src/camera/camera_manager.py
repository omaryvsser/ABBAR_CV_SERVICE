"""
Camera Manager Module.

Provides low-latency, thread-safe webcam capture with error handling,
automatic buffer management, and graceful shutdown.
"""

import logging
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraInitializationError(RuntimeError):
    """Raised when the camera device cannot be opened or initialized."""
    pass


class CameraManager:
    """
    Manages webcam capture lifecycle, optimized for real-time computer vision.

    Features:
    - Minimized buffer size (`cv2.CAP_PROP_BUFFERSIZE = 1`) to eliminate display lag.
    - Optional background thread for asynchronous frame grabbing.
    - Graceful startup, recovery, and resource cleanup.
    - Context manager support (`with CameraManager() as cam:`).
    """

    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 1280,
        frame_height: int = 720,
        target_fps: int = 30,
        buffer_size: int = 1,
        use_threading: bool = True,
    ) -> None:
        """
        Initialize camera capture parameters.

        Args:
            camera_index: Index of video device (0 is default).
            frame_width: Requested frame width in pixels.
            frame_height: Requested frame height in pixels.
            target_fps: Target frame rate in FPS.
            buffer_size: Driver buffer size (1 minimizes queue lag).
            use_threading: Whether to read frames on a dedicated worker thread.
        """
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        self.use_threading = use_threading

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ret: bool = False
        self._frame_count = 0
        self._start_time = 0.0

    @property
    def is_opened(self) -> bool:
        """Check if the underlying camera capture device is open."""
        return self._cap is not None and self._cap.isOpened()

    def start(self) -> None:
        """
        Open the camera device and start frame acquisition.

        Raises:
            CameraInitializationError: If camera cannot be opened.
        """
        if self.is_opened:
            logger.warning("CameraManager is already started.")
            return

        logger.info(f"Opening camera at index {self.camera_index}...")
        self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            # Release any partial handle
            self._cap.release()
            self._cap = None
            raise CameraInitializationError(
                f"Failed to open video capture device at index {self.camera_index}. "
                f"Verify camera connection, permissions, and device index."
            )

        # Set capture properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Minimize driver buffer latency where supported
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        except Exception as e:
            logger.debug(f"Unable to set CAP_PROP_BUFFERSIZE: {e}")

        # Check actual negotiated dimensions
        try:
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        except (TypeError, ValueError):
            actual_w = self.frame_width
            actual_h = self.frame_height
            actual_fps = float(self.target_fps)

        logger.info(
            f"Camera {self.camera_index} opened: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS "
            f"(requested {self.frame_width}x{self.frame_height} @ {self.target_fps} FPS)"
        )

        self._is_running = True
        self._start_time = time.time()
        self._frame_count = 0

        # Read one initial frame to verify the pipeline
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self.release()
            raise CameraInitializationError(
                f"Camera at index {self.camera_index} opened, but failed to capture initial test frame."
            )

        with self._lock:
            self._latest_ret = ret
            self._latest_frame = frame
            self._frame_count = 1

        if self.use_threading:
            self._thread = threading.Thread(
                target=self._capture_worker,
                name="CameraWorkerThread",
                daemon=True,
            )
            self._thread.start()
            logger.info("Camera capture worker thread started.")

    def _capture_worker(self) -> None:
        """Dedicated thread continuously grabbing the most recent frame."""
        while self._is_running:
            if self._cap is None or not self._cap.isOpened():
                break

            ret, frame = self._cap.read()
            with self._lock:
                self._latest_ret = ret
                if ret and frame is not None:
                    self._latest_frame = frame
                    self._frame_count += 1
                else:
                    logger.warning("Camera frame read failed in worker thread.")

            # Small sleep to yield CPU if target FPS is bounded
            if self.target_fps > 0:
                time.sleep(1.0 / (self.target_fps * 2))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieve the latest frame from the webcam.

        Returns:
            Tuple of (success_flag, bgr_frame_ndarray).
        """
        if not self.is_opened:
            return False, None

        if self.use_threading:
            with self._lock:
                if self._latest_frame is None:
                    return False, None
                return self._latest_ret, self._latest_frame.copy()
        else:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self._frame_count += 1
            return ret, frame

    def release(self) -> None:
        """Gracefully stop background thread and release webcam resources."""
        logger.info(f"Releasing camera at index {self.camera_index}...")
        self._is_running = False

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
            self._thread = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning(f"Error during VideoCapture release: {e}")
            finally:
                self._cap = None

        with self._lock:
            self._latest_frame = None
            self._latest_ret = False

        logger.info("Camera resources released.")

    def __enter__(self) -> "CameraManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
