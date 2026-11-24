"""
Gaze Smoothing Module using Saccade-Adaptive Exponential Moving Average (EMA).

Reduces high-frequency tracking tremor and sensor noise while retaining rapid
responsiveness to intentional saccadic eye movements.
"""

import math
from typing import Optional, Tuple


class GazeSmoother:
    """
    Saccade-Adaptive Exponential Moving Average (EMA) smoother for 2D gaze coordinates.

    Formula:
        S_t = alpha * X_t + (1 - alpha) * S_{t-1}

    Features:
    - Adaptive Alpha: If a rapid eye jump (saccade) is detected, alpha increases
      to snap instantly to the new position without visual drag.
    - Blink / Invalid Frame Tolerance: Retains the previous smoothed position for up to
      N frames of eye closure, avoiding sudden drops during natural blinks.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        saccade_threshold: float = 0.12,
        saccade_alpha: float = 0.85,
        max_invalid_frames: int = 5,
        alpha_y: Optional[float] = None,
        saccade_threshold_y: Optional[float] = None,
        saccade_alpha_y: Optional[float] = None,
    ) -> None:
        """
        Initialize the gaze smoother with independent horizontal and vertical dynamics.

        Args:
            alpha: Steady-state smoothing factor for horizontal gaze.
            saccade_threshold: Horizontal delta triggering saccade acceleration.
            saccade_alpha: Elevated smoothing factor during horizontal saccades.
            max_invalid_frames: Consecutive invalid frames to hold before resetting.
            alpha_y: Independent vertical steady-state smoothing factor (defaults to alpha).
            saccade_threshold_y: Independent vertical delta triggering saccade acceleration (defaults to 0.050).
            saccade_alpha_y: Independent vertical saccade smoothing factor (defaults to saccade_alpha).
        """
        self.alpha_x = alpha
        self.alpha_y = alpha_y if alpha_y is not None else alpha
        self.alpha = alpha

        self.saccade_threshold_x = saccade_threshold
        if saccade_threshold_y is not None:
            self.saccade_threshold_y = saccade_threshold_y
        elif saccade_threshold != 0.12:
            self.saccade_threshold_y = saccade_threshold
        else:
            self.saccade_threshold_y = 0.050
        self.saccade_threshold = saccade_threshold

        self.saccade_alpha_x = saccade_alpha
        self.saccade_alpha_y = saccade_alpha_y if saccade_alpha_y is not None else saccade_alpha
        self.saccade_alpha = saccade_alpha

        self.max_invalid_frames = max_invalid_frames

        self._smoothed_x: Optional[float] = None
        self._smoothed_y: Optional[float] = None
        self._last_raw_x: Optional[float] = None
        self._last_raw_y: Optional[float] = None
        self._consecutive_invalid_frames = 0

    @property
    def smoothed_x(self) -> Optional[float]:
        return self._smoothed_x

    @property
    def smoothed_y(self) -> Optional[float]:
        return self._smoothed_y

    @property
    def raw_x(self) -> Optional[float]:
        return self._last_raw_x

    @property
    def raw_y(self) -> Optional[float]:
        return self._last_raw_y

    def update(
        self,
        raw_x: Optional[float],
        raw_y: Optional[float],
        is_valid: bool = True,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Update the smoother with a new raw observation.

        Args:
            raw_x: Raw normalized horizontal gaze ratio.
            raw_y: Raw normalized vertical gaze ratio.
            is_valid: Whether the current observation is valid (eyes open and tracked).

        Returns:
            Tuple of (smoothed_x, smoothed_y).
        """
        self._last_raw_x = raw_x
        self._last_raw_y = raw_y

        if not is_valid or raw_x is None or raw_y is None:
            self._consecutive_invalid_frames += 1
            if self._consecutive_invalid_frames <= self.max_invalid_frames:
                # Hold previous smoothed gaze during brief blinks
                return self._smoothed_x, self._smoothed_y
            else:
                # Reset if invalid duration exceeds threshold
                self.reset()
                return None, None

        self._consecutive_invalid_frames = 0

        # First observation initialization
        if self._smoothed_x is None or self._smoothed_y is None:
            self._smoothed_x = raw_x
            self._smoothed_y = raw_y
            return self._smoothed_x, self._smoothed_y

        # Detect decoupled horizontal and vertical saccades
        delta_x = abs(raw_x - self._smoothed_x)
        delta_y = abs(raw_y - self._smoothed_y)

        eff_alpha_x = self.saccade_alpha_x if delta_x > self.saccade_threshold_x else self.alpha_x
        eff_alpha_y = self.saccade_alpha_y if delta_y > self.saccade_threshold_y else self.alpha_y

        # Decoupled Exponential Moving Average update
        self._smoothed_x = eff_alpha_x * raw_x + (1.0 - eff_alpha_x) * self._smoothed_x
        self._smoothed_y = eff_alpha_y * raw_y + (1.0 - eff_alpha_y) * self._smoothed_y

        return self._smoothed_x, self._smoothed_y

    def reset(self) -> None:
        """Reset internal filter state."""
        self._smoothed_x = None
        self._smoothed_y = None
        self._consecutive_invalid_frames = 0


class ScreenGazeStabilizer:
    """
    One-Euro adaptive filter for calibrated 2D screen coordinates in [0.0, 1.0].

    Combines low-cutoff jitter suppression during steady fixations with high-bandwidth
    saccade responsiveness and jump/blink outlier protection so cursor tremor is reduced
    without introducing sluggish drag.
    """

    def __init__(
        self,
        min_cutoff: float = 1.6,
        beta: float = 18.0,
        d_cutoff: float = 1.0,
        max_invalid_frames: int = 5,
        jump_outlier_threshold: float = 0.65,
        jump_confidence_floor: float = 0.65,
    ) -> None:
        self.min_cutoff = max(0.05, float(min_cutoff))
        self.beta = max(0.0, float(beta))
        self.d_cutoff = max(0.05, float(d_cutoff))
        self.max_invalid_frames = max(0, int(max_invalid_frames))
        self.jump_outlier_threshold = float(jump_outlier_threshold)
        self.jump_confidence_floor = float(jump_confidence_floor)

        self._x: Optional[float] = None
        self._y: Optional[float] = None
        self._dx: float = 0.0
        self._dy: float = 0.0
        self._last_time: Optional[float] = None
        self._invalid_streak: int = 0
        self._pending_jump_count: int = 0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(1e-4, cutoff))
        return 1.0 / (1.0 + tau / max(1e-4, dt))

    def update(
        self,
        raw_x: Optional[float],
        raw_y: Optional[float],
        timestamp: float,
        is_valid: bool = True,
        confidence: float = 1.0,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Update stabilized screen gaze prediction."""
        if not is_valid or raw_x is None or raw_y is None or not (math.isfinite(raw_x) and math.isfinite(raw_y)):
            self._invalid_streak += 1
            if self._invalid_streak <= self.max_invalid_frames:
                return self._x, self._y
            self.reset()
            return None, None

        self._invalid_streak = 0
        rx = min(1.0, max(0.0, float(raw_x)))
        ry = min(1.0, max(0.0, float(raw_y)))

        if self._x is None or self._y is None or self._last_time is None:
            self._x, self._y = rx, ry
            self._dx, self._dy = 0.0, 0.0
            self._last_time = float(timestamp)
            self._pending_jump_count = 0
            return self._x, self._y

        # Reject isolated low-confidence extreme jump artifacts until confirmed on a second frame
        step_dist = math.hypot(rx - self._x, ry - self._y)
        if step_dist > self.jump_outlier_threshold and confidence < self.jump_confidence_floor:
            self._pending_jump_count += 1
            if self._pending_jump_count < 2:
                return self._x, self._y
        self._pending_jump_count = 0

        dt = max(1e-3, min(0.25, float(timestamp) - self._last_time))
        self._last_time = float(timestamp)

        # Filtered derivative estimate
        a_d = self._alpha(self.d_cutoff, dt)
        vx = (rx - self._x) / dt
        vy = (ry - self._y) / dt
        self._dx = a_d * vx + (1.0 - a_d) * self._dx
        self._dy = a_d * vy + (1.0 - a_d) * self._dy

        # Speed-adaptive cutoff per axis
        cutoff_x = self.min_cutoff + self.beta * abs(self._dx)
        cutoff_y = self.min_cutoff + self.beta * abs(self._dy)
        ax = self._alpha(cutoff_x, dt)
        ay = self._alpha(cutoff_y, dt)

        self._x = min(1.0, max(0.0, ax * rx + (1.0 - ax) * self._x))
        self._y = min(1.0, max(0.0, ay * ry + (1.0 - ay) * self._y))
        return self._x, self._y

    def reset(self) -> None:
        """Reset filter state."""
        self._x = None
        self._y = None
        self._dx = 0.0
        self._dy = 0.0
        self._last_time = None
        self._invalid_streak = 0
        self._pending_jump_count = 0
