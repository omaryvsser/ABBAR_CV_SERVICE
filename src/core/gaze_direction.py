"""
Relative Gaze Direction Classification Module (Enhanced with Hysteresis & Baseline Calibration).

Classifies eye gaze direction relative to neutral forward-facing gaze:
- CENTER
- LEFT
- RIGHT
- UP
- DOWN
(with optional compound directions: UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT)

Features:
- Dual-threshold Schmitt trigger hysteresis to eliminate boundary flicker.
- Baseline-relative deviation scoring (centering deadzones on personalized baseline).
- Temporal persistence window to filter out single-frame tracking glitches.
"""

from enum import Enum
from typing import Optional, Tuple


class GazeDirection(str, Enum):
    """Categorical relative gaze directions."""
    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    UP_LEFT = "UP_LEFT"
    UP_RIGHT = "UP_RIGHT"
    DOWN_LEFT = "DOWN_LEFT"
    DOWN_RIGHT = "DOWN_RIGHT"
    UNKNOWN = "UNKNOWN"


class GazeDirectionClassifier:
    """
    Classifies normalized iris ratios into categorical relative gaze directions.

    Applies dual-threshold hysteresis and optional temporal persistence
    around a personalized neutral resting baseline (b_x, b_y).
    """

    def __init__(
        self,
        horizontal_center_min: float = 0.42,
        horizontal_center_max: float = 0.58,
        vertical_center_min: float = 0.38,
        vertical_center_max: float = 0.62,
        h_activation_threshold: Optional[float] = None,
        h_deactivation_threshold: Optional[float] = None,
        v_activation_threshold: Optional[float] = None,
        v_deactivation_threshold: Optional[float] = None,
        v_up_activation_threshold: Optional[float] = None,
        v_up_deactivation_threshold: Optional[float] = None,
        v_down_activation_threshold: Optional[float] = None,
        v_down_deactivation_threshold: Optional[float] = None,
        persistence_frames: int = 1,
        allow_compound_directions: bool = False,
    ) -> None:
        """
        Initialize classifier with configurable thresholds and hysteresis parameters.

        Args:
            horizontal_center_min: Fallback lower bound for neutral horizontal gaze.
            horizontal_center_max: Fallback upper bound for neutral horizontal gaze.
            vertical_center_min: Fallback lower bound for neutral vertical gaze.
            vertical_center_max: Fallback upper bound for neutral vertical gaze.
            h_activation_threshold: Deviation from baseline required to enter LEFT or RIGHT.
            h_deactivation_threshold: Deviation below which active horizontal direction drops to CENTER.
            v_activation_threshold: Symmetrical fallback vertical deviation required to enter UP or DOWN.
            v_deactivation_threshold: Symmetrical fallback vertical deviation below which direction drops to CENTER.
            v_up_activation_threshold: Specific upward deviation to enter UP (looking up).
            v_up_deactivation_threshold: Upward deviation below which UP drops to CENTER.
            v_down_activation_threshold: Specific downward deviation to enter DOWN (looking down).
            v_down_deactivation_threshold: Downward deviation below which DOWN drops to CENTER.
            persistence_frames: Number of consecutive frames a direction must hold to be confirmed.
            allow_compound_directions: Whether to emit diagonal directions (e.g. UP_LEFT).
        """
        self.horizontal_center_min = horizontal_center_min
        self.horizontal_center_max = horizontal_center_max
        self.vertical_center_min = vertical_center_min
        self.vertical_center_max = vertical_center_max

        # Derive activation/deactivation deadzones from limits if not explicitly specified
        default_h_act = (horizontal_center_max - horizontal_center_min) / 2.0
        self.h_activation = h_activation_threshold if h_activation_threshold is not None else default_h_act
        self.h_deactivation = (
            h_deactivation_threshold if h_deactivation_threshold is not None else self.h_activation * 0.60
        )

        default_v_act = (vertical_center_max - vertical_center_min) / 2.0
        base_v_act = v_activation_threshold if v_activation_threshold is not None else default_v_act
        base_v_deact = v_deactivation_threshold if v_deactivation_threshold is not None else base_v_act * 0.60
        self.v_activation = base_v_act
        self.v_deactivation = base_v_deact

        # Asymmetric vertical thresholds: physiological elevation range is narrower than depression
        self.v_up_activation = (
            v_up_activation_threshold if v_up_activation_threshold is not None else base_v_act * 0.80
        )
        self.v_up_deactivation = (
            v_up_deactivation_threshold if v_up_deactivation_threshold is not None else self.v_up_activation * 0.55
        )
        self.v_down_activation = (
            v_down_activation_threshold if v_down_activation_threshold is not None else base_v_act
        )
        self.v_down_deactivation = (
            v_down_deactivation_threshold if v_down_deactivation_threshold is not None else self.v_down_activation * 0.55
        )

        self.persistence_frames = max(1, persistence_frames)
        self.allow_compound_directions = allow_compound_directions

        # State tracking for hysteresis & temporal persistence
        self._confirmed_direction: GazeDirection = GazeDirection.CENTER
        self._candidate_direction: GazeDirection = GazeDirection.CENTER
        self._candidate_count: int = 0
        self._active_h_state: Optional[str] = None  # "LEFT", "RIGHT", or None (CENTER)
        self._active_v_state: Optional[str] = None  # "UP", "DOWN", or None (CENTER)

    @property
    def confirmed_direction(self) -> GazeDirection:
        """Currently confirmed stable direction."""
        return self._confirmed_direction

    @property
    def active_vertical_state(self) -> Optional[str]:
        """Currently active vertical hysteresis state ('UP', 'DOWN', or None)."""
        return self._active_v_state

    @property
    def active_horizontal_state(self) -> Optional[str]:
        """Currently active horizontal hysteresis state ('LEFT', 'RIGHT', or None)."""
        return self._active_h_state

    @property
    def candidate_direction(self) -> GazeDirection:
        """Instantaneous candidate direction prior to temporal persistence confirmation."""
        return self._candidate_direction

    def reset(self) -> None:
        """Reset internal hysteresis and persistence state."""
        self._confirmed_direction = GazeDirection.CENTER
        self._candidate_direction = GazeDirection.CENTER
        self._candidate_count = 0
        self._active_h_state = None
        self._active_v_state = None

    def classify(
        self,
        horizontal_ratio: Optional[float],
        vertical_ratio: Optional[float],
        baseline_x: float = 0.50,
        baseline_y: float = 0.50,
        baseline_std_y: Optional[float] = None,
    ) -> GazeDirection:
        """
        Classify normalized ocular coordinates into a stable GazeDirection.

        Args:
            horizontal_ratio: Normalized horizontal iris ratio in anatomical gaze space.
            vertical_ratio: Normalized vertical iris ratio in anatomical gaze space.
            baseline_x: Neutral resting baseline horizontal ratio.
            baseline_y: Neutral resting baseline vertical ratio.
            baseline_std_y: Optional measured resting vertical noise standard deviation.

        Returns:
            GazeDirection enum value.
        """
        if horizontal_ratio is None or vertical_ratio is None:
            self._confirmed_direction = GazeDirection.UNKNOWN
            self._candidate_direction = GazeDirection.UNKNOWN
            self._candidate_count = 0
            self._active_h_state = None
            self._active_v_state = None
            return GazeDirection.UNKNOWN

        # Compute deviations from neutral baseline
        delta_x = horizontal_ratio - baseline_x
        delta_y = vertical_ratio - baseline_y

        # Dynamic vertical threshold adaptation: ensure deadzone >= 2.5 * baseline noise std
        if baseline_std_y is not None and baseline_std_y > 0.0:
            effective_up_act = max(self.v_up_activation, 2.5 * baseline_std_y)
            effective_down_act = max(self.v_down_activation, 2.5 * baseline_std_y)
        else:
            effective_up_act = self.v_up_activation
            effective_down_act = self.v_down_activation

        # 1. Horizontal Hysteresis Evaluation (preserved)
        if delta_x > self.h_activation:
            self._active_h_state = "RIGHT"
        elif delta_x < -self.h_activation:
            self._active_h_state = "LEFT"
        elif self._active_h_state == "RIGHT" and delta_x < self.h_deactivation:
            self._active_h_state = None
        elif self._active_h_state == "LEFT" and delta_x > -self.h_deactivation:
            self._active_h_state = None

        # 2. Vertical Hysteresis Evaluation (Asymmetric UP and DOWN)
        if delta_y < -effective_up_act:
            self._active_v_state = "UP"
        elif delta_y > effective_down_act:
            self._active_v_state = "DOWN"
        elif self._active_v_state == "UP" and delta_y > -self.v_up_deactivation:
            self._active_v_state = None
        elif self._active_v_state == "DOWN" and delta_y < self.v_down_deactivation:
            self._active_v_state = None

        # 3. Form Instantaneous Raw Candidate Direction
        raw_direction: GazeDirection
        h_dir = self._active_h_state
        v_dir = self._active_v_state

        if h_dir is None and v_dir is None:
            raw_direction = GazeDirection.CENTER
        elif self.allow_compound_directions and h_dir is not None and v_dir is not None:
            raw_direction = GazeDirection(f"{v_dir}_{h_dir}")
        elif h_dir is not None and v_dir is not None:
            # Dominant axis mode: prioritize axis with larger relative activation margin
            norm_h = abs(delta_x) / max(1e-4, self.h_activation)
            eff_v_act = self.v_up_activation if delta_y < 0 else self.v_down_activation
            norm_v = abs(delta_y) / max(1e-4, eff_v_act)
            raw_direction = GazeDirection(h_dir) if norm_h >= norm_v else GazeDirection(v_dir)
        elif h_dir is not None:
            raw_direction = GazeDirection(h_dir)
        else:
            assert v_dir is not None
            raw_direction = GazeDirection(v_dir)

        # 4. Temporal Persistence Window
        if self.persistence_frames <= 1:
            self._confirmed_direction = raw_direction
            return raw_direction

        if raw_direction == self._candidate_direction:
            self._candidate_count += 1
            if self._candidate_count >= self.persistence_frames:
                self._confirmed_direction = raw_direction
        else:
            self._candidate_direction = raw_direction
            self._candidate_count = 1

        return self._confirmed_direction
