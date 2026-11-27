"""
AABAR CV Service - Dwell Gaze Smoothing Filters.

Provides configurable, pure-Python gaze smoothing algorithms designed for
dwell selection in human-computer interaction (AAC):
1. PassThroughGazeFilter: Raw coordinates with zero latency.
2. EMAGazeFilter: Saccade-adaptive Exponential Moving Average.
3. OneEuroGazeFilter: 1-Euro adaptive filter (Casiez et al., CHI 2012)
   providing heavy jitter rejection at low velocity during steady fixation
   and zero-lag tracking at high velocity during saccades.

Preserves the original raw gaze coordinates alongside filtered coordinates
for telemetry, diagnostics, and comparative evaluation.
"""

from abc import ABC, abstractmethod
from enum import Enum
import math
from typing import Optional, Tuple


class GazeFilterType(str, Enum):
    """Supported gaze smoothing filter algorithms."""
    NONE = "none"
    EMA = "ema"
    ONE_EURO = "one_euro"


class BaseGazeFilter(ABC):
    """Abstract base class for 2D screen gaze smoothing filters."""

    @property
    @abstractmethod
    def filter_type(self) -> str:
        """Name of the filtering algorithm."""
        pass

    @abstractmethod
    def update(
        self,
        raw_x: Optional[float],
        raw_y: Optional[float],
        timestamp: float,
        is_valid: bool = True,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Filter the current 2D gaze observation.

        Args:
            raw_x: Raw horizontal coordinate.
            raw_y: Raw vertical coordinate.
            timestamp: Observation timestamp in seconds.
            is_valid: Whether input observation is valid (tracked and non-NaN).

        Returns:
            Tuple of (filtered_x, filtered_y).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal filter state."""
        pass


class PassThroughGazeFilter(BaseGazeFilter):
    """Raw coordinates with zero filtering and zero latency."""

    @property
    def filter_type(self) -> str:
        return GazeFilterType.NONE.value

    def update(
        self,
        raw_x: Optional[float],
        raw_y: Optional[float],
        timestamp: float,
        is_valid: bool = True,
    ) -> Tuple[Optional[float], Optional[float]]:
        if not is_valid or raw_x is None or raw_y is None or not (math.isfinite(raw_x) and math.isfinite(raw_y)):
            return None, None
        return float(raw_x), float(raw_y)

    def reset(self) -> None:
        pass


class EMAGazeFilter(BaseGazeFilter):
    """
    Saccade-Adaptive Exponential Moving Average (EMA) filter.

    Formula:
        S_t = alpha * X_t + (1 - alpha) * S_{t-1}

    Increases alpha during rapid saccadic movements to eliminate drag.
    """

    def __init__(
        self,
        alpha: float = 0.35,
        saccade_threshold: float = 0.08,
        saccade_alpha: float = 0.85,
        max_invalid_frames: int = 4,
    ) -> None:
        self.alpha = float(min(1.0, max(0.01, alpha)))
        self.saccade_threshold = float(max(0.001, saccade_threshold))
        self.saccade_alpha = float(min(1.0, max(self.alpha, saccade_alpha)))
        self.max_invalid_frames = max(0, int(max_invalid_frames))

        self._filtered_x: Optional[float] = None
        self._filtered_y: Optional[float] = None
        self._invalid_count: int = 0

    @property
    def filter_type(self) -> str:
        return GazeFilterType.EMA.value

    def update(
        self,
        raw_x: Optional[float],
        raw_y: Optional[float],
        timestamp: float,
        is_valid: bool = True,
    ) -> Tuple[Optional[float], Optional[float]]:
        if not is_valid or raw_x is None or raw_y is None or not (math.isfinite(raw_x) and math.isfinite(raw_y)):
            self._invalid_count += 1
            if self._invalid_count <= self.max_invalid_frames and self._filtered_x is not None:
                return self._filtered_x, self._filtered_y
            self.reset()
            return None, None

        self._invalid_count = 0
        rx, ry = float(raw_x), float(raw_y)

        if self._filtered_x is None or self._filtered_y is None:
            self._filtered_x, self._filtered_y = rx, ry
            return self._filtered_x, self._filtered_y

        dist = math.hypot(rx - self._filtered_x, ry - self._filtered_y)
        eff_alpha = self.saccade_alpha if dist > self.saccade_threshold else self.alpha

        self._filtered_x = eff_alpha * rx + (1.0 - eff_alpha) * self._filtered_x
        self._filtered_y = eff_alpha * ry + (1.0 - eff_alpha) * self._filtered_y
        return self._filtered_x, self._filtered_y

    def reset(self) -> None:
        self._filtered_x = None
        self._filtered_y = None
        self._invalid_count = 0


class OneEuroGazeFilter(BaseGazeFilter):
    """
    1-Euro Adaptive Low-Pass Filter (Casiez, Roussel, Vogel, CHI 2012).

    Dynamically adjusts cutoff frequency based on instantaneous speed:
    - At low speed (fixation): Low cutoff suppresses ocular tremor and sensor noise.
    - At high speed (saccade): High cutoff removes phase lag and visual drag.

    Parameters:
        min_cutoff: Minimum cutoff frequency (Hz) at zero speed. Lower = smoother fixation.
        beta: Speed coefficient. Higher = faster response during saccades.
        d_cutoff: Cutoff frequency (Hz) for velocity derivative smoothing.
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 12.0,
        d_cutoff: float = 1.0,
        max_invalid_frames: int = 4,
    ) -> None:
        self.min_cutoff = float(max(0.01, min_cutoff))
        self.beta = float(max(0.0, beta))
        self.d_cutoff = float(max(0.01, d_cutoff))
        self.max_invalid_frames = max(0, int(max_invalid_frames))

        self._x: Optional[float] = None
        self._y: Optional[float] = None
        self._dx: float = 0.0
        self._dy: float = 0.0
        self._last_time: Optional[float] = None
        self._invalid_count: int = 0

    @property
    def filter_type(self) -> str:
        return GazeFilterType.ONE_EURO.value

    @staticmethod
    def _compute_alpha(cutoff_hz: float, dt_seconds: float) -> float:
        """Compute smoothing factor alpha from cutoff frequency and delta time."""
        tau = 1.0 / (2.0 * math.pi * max(1e-4, cutoff_hz))
        return 1.0 / (1.0 + tau / max(1e-4, dt_seconds))

    def update(
        self,
        raw_x: Optional[float],
        raw_y: Optional[float],
        timestamp: float,
        is_valid: bool = True,
    ) -> Tuple[Optional[float], Optional[float]]:
        if not is_valid or raw_x is None or raw_y is None or not (math.isfinite(raw_x) and math.isfinite(raw_y)):
            self._invalid_count += 1
            if self._invalid_count <= self.max_invalid_frames and self._x is not None:
                return self._x, self._y
            self.reset()
            return None, None

        self._invalid_count = 0
        rx, ry = float(raw_x), float(raw_y)
        t = float(timestamp)

        if self._x is None or self._y is None or self._last_time is None:
            self._x, self._y = rx, ry
            self._dx, self._dy = 0.0, 0.0
            self._last_time = t
            return self._x, self._y

        # Bound dt between 1ms and 250ms to prevent division by zero or huge jumps
        dt = max(1e-3, min(0.25, t - self._last_time))
        self._last_time = t

        # 1. Filtered velocity derivative estimate
        alpha_d = self._compute_alpha(self.d_cutoff, dt)
        inst_vx = (rx - self._x) / dt
        inst_vy = (ry - self._y) / dt
        self._dx = alpha_d * inst_vx + (1.0 - alpha_d) * self._dx
        self._dy = alpha_d * inst_vy + (1.0 - alpha_d) * self._dy

        # 2. Velocity-adaptive cutoff frequency
        speed_x = abs(self._dx)
        speed_y = abs(self._dy)
        cutoff_x = self.min_cutoff + self.beta * speed_x
        cutoff_y = self.min_cutoff + self.beta * speed_y

        # 3. Position filtering with adaptive alpha
        alpha_x = self._compute_alpha(cutoff_x, dt)
        alpha_y = self._compute_alpha(cutoff_y, dt)

        self._x = alpha_x * rx + (1.0 - alpha_x) * self._x
        self._y = alpha_y * ry + (1.0 - alpha_y) * self._y
        return self._x, self._y

    def reset(self) -> None:
        self._x = None
        self._y = None
        self._dx = 0.0
        self._dy = 0.0
        self._last_time = None
        self._invalid_count = 0


def create_gaze_filter(
    filter_type: str = GazeFilterType.ONE_EURO.value,
    min_cutoff: float = 1.2,
    beta: float = 12.0,
    d_cutoff: float = 1.0,
    ema_alpha: float = 0.35,
    saccade_threshold: float = 0.08,
    saccade_alpha: float = 0.85,
) -> BaseGazeFilter:
    """Factory creating the appropriate BaseGazeFilter instance."""
    normalized_type = str(filter_type).strip().lower()
    if normalized_type in ("none", "raw", "passthrough"):
        return PassThroughGazeFilter()
    elif normalized_type == "ema":
        return EMAGazeFilter(
            alpha=ema_alpha,
            saccade_threshold=saccade_threshold,
            saccade_alpha=saccade_alpha,
        )
    elif normalized_type in ("one_euro", "oneeuro", "1euro"):
        return OneEuroGazeFilter(
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
        )
    else:
        # Default fallback to OneEuro
        return OneEuroGazeFilter(
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
        )
