"""Performance and latency tracking utilities."""
from dataclasses import dataclass
import time
from typing import Deque, Optional
from collections import deque


@dataclass(frozen=True)
class FPSMetrics:
    """
    Performance and timing metrics.

    Attributes:
        fps: Smoothed frames per second.
        frame_time_ms: Total time to capture and process the last frame in milliseconds.
        detection_latency_ms: Time spent in MediaPipe inference in milliseconds.
        total_frames: Total number of frames processed so far.
        uptime_seconds: Total time elapsed since counter start.
    """
    fps: float
    frame_time_ms: float
    detection_latency_ms: float
    total_frames: int
    uptime_seconds: float


class FPSCounter:
    """
    Calculates moving-average FPS, frame cycle duration, and detection latency.

    Optimized for smooth HUD display without jitter.
    """

    def __init__(self, window_size: int = 30) -> None:
        """
        Initialize the FPS counter.

        Args:
            window_size: Number of frames over which to compute moving-average FPS.
        """
        self.window_size = window_size
        self._frame_timestamps: Deque[float] = deque(maxlen=window_size)
        self._start_time = time.perf_counter()
        self._last_frame_time = self._start_time
        self._last_frame_duration_ms = 0.0
        self._last_detection_latency_ms = 0.0
        self._total_frames = 0

    def start_frame(self) -> float:
        """Record the start of a frame processing cycle. Returns timestamp."""
        return time.perf_counter()

    def update(self, frame_start_time: float, detection_latency_ms: float = 0.0) -> FPSMetrics:
        """
        Update the counter with completed frame timing.

        Args:
            frame_start_time: Timestamp from start_frame().
            detection_latency_ms: MediaPipe inference latency in milliseconds.

        Returns:
            FPSMetrics dataclass with current performance statistics.
        """
        now = time.perf_counter()
        self._last_frame_duration_ms = (now - frame_start_time) * 1000.0
        self._last_detection_latency_ms = detection_latency_ms
        self._total_frames += 1

        self._frame_timestamps.append(now)

        # Calculate moving average FPS
        if len(self._frame_timestamps) > 1:
            duration = self._frame_timestamps[-1] - self._frame_timestamps[0]
            if duration > 1e-5:
                fps = (len(self._frame_timestamps) - 1) / duration
            else:
                fps = 0.0
        else:
            fps = 0.0

        uptime = now - self._start_time

        return FPSMetrics(
            fps=fps,
            frame_time_ms=self._last_frame_duration_ms,
            detection_latency_ms=self._last_detection_latency_ms,
            total_frames=self._total_frames,
            uptime_seconds=uptime,
        )
