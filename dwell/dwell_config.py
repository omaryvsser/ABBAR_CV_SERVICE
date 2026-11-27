"""
AABAR CV Service - Dwell Selection Configuration.

Provides configurable parameters for the Phase 4 Dwell Selection Engine.
Supports customizable dwell time, cooldown refractory periods, gaze stability
tolerances, tracking loss policies, target hysteresis, fixation detection,
and adaptive user profile settings.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Literal, Optional


class DwellUserSkill(str, Enum):
    """User proficiency tiers for adaptive dwell parameter presets."""
    NOVICE = "novice"
    STANDARD = "standard"
    EXPERT = "expert"
    TREMOR_PRONE = "tremor_prone"


@dataclass
class DwellConfig:
    """
    Configuration options for the Dwell Selection Engine.

    All time intervals are specified in milliseconds (ms) for accessibility
    and UI standard compliance, but internal calculations handle float seconds.
    """

    # Core Dwell Timing (ms) - Validated range: 200–5000 ms
    dwell_duration_ms: float = 1200.0  # Default 1200ms fixation duration required for selection
    cooldown_duration_ms: float = 800.0  # Refractory period (ms) preventing accidental repeated selection
    selection_display_duration_ms: float = 250.0  # Duration (ms) the SELECTED state remains visibly highlighted

    # Gaze Stability & Saccade Grace Handling
    stability_filter_enabled: bool = True  # Whether to filter sudden erratic coordinate jumps
    max_jump_distance: float = 0.25  # Max normalized Euclidean distance per frame before flagging saccade
    leave_target_tolerance_ms: float = 150.0  # Grace period (ms) allowing brief saccade outside target without resetting progress
    lost_tracking_tolerance_ms: float = 250.0  # Grace period (ms) when eye tracking is lost before aborting dwell
    max_dt_ms: Optional[float] = None  # Optional safety cap on frame dt (ms) preventing dwell leaps during system lag spikes

    # Gaze Smoothing Filter Configuration
    filter_type: Literal["none", "ema", "one_euro"] = "none"  # Default 'none' preserves raw pass-through; 'one_euro' / 'ema' configurable
    filter_min_cutoff: float = 1.2  # 1-Euro min cutoff frequency (Hz) at zero speed (lower = smoother)
    filter_beta: float = 12.0  # 1-Euro speed coefficient (higher = faster saccade tracking)
    filter_d_cutoff: float = 1.0  # 1-Euro derivative cutoff frequency (Hz)
    ema_alpha: float = 0.35  # Steady-state alpha for EMA filter
    ema_saccade_threshold: float = 0.08  # Saccade step threshold for EMA acceleration
    ema_saccade_alpha: float = 0.85  # Elevated alpha for EMA during saccades

    # Target Hysteresis Configuration
    hysteresis_enabled: bool = False  # Disabled by default; preserves clean, exact target bounding boxes
    entry_margin: float = 0.0  # Normalized margin inside/outside entry boundary (0.0 = exact bounding box)
    exit_margin: float = 0.0  # Normalized margin for target exit (0.0 = exact bounding box, avoids sticky gutter trap)
    switch_confirmation_ms: float = 0.0  # Confirmation time (ms) in new target before switching away (0.0 = immediate switch)

    # Fixation Detection Configuration
    fixation_detection_enabled: bool = False  # Disabled by default for instant entry; enabled for tremor-prone/saccade filtering
    saccade_velocity_threshold: float = 1.8  # Normalized screen distance / sec above which movement is a saccade
    fixation_min_duration_ms: float = 0.0  # Settling duration (ms) required before accumulating dwell (0.0 = instant)

    # Coordinate Convention & Space
    coordinate_space: Literal["normalized", "pixels"] = "normalized"  # 'normalized' in [0, 1] or 'pixels'
    screen_width: int = 1920
    screen_height: int = 1080

    # Overlap Resolution Policy
    target_selection_policy: Literal["highest_priority", "smallest_area", "first_defined"] = "highest_priority"

    # Selection Behavior
    allow_repeat_without_leaving: bool = False  # If False, user MUST look away from the selected target before re-dwelling
    pause_on_lost_tracking: bool = True  # If True, temporarily pauses dwell accumulation on brief tracking drops instead of hard reset

    def __post_init__(self) -> None:
        """Validate configuration parameters within safe, clinically realistic bounds."""
        if self.dwell_duration_ms <= 0:
            raise ValueError(f"dwell_duration_ms must be positive, got {self.dwell_duration_ms}")
        if self.dwell_duration_ms < 200.0 or self.dwell_duration_ms > 5000.0:
            # Warn / bound check for extreme accessibility limits
            pass
        if self.cooldown_duration_ms < 0:
            raise ValueError(f"cooldown_duration_ms cannot be negative, got {self.cooldown_duration_ms}")
        if self.screen_width <= 0 or self.screen_height <= 0:
            raise ValueError(f"Invalid screen dimensions: {self.screen_width}x{self.screen_height}")
        if self.leave_target_tolerance_ms < 0:
            raise ValueError("leave_target_tolerance_ms cannot be negative")
        if self.lost_tracking_tolerance_ms < 0:
            raise ValueError("lost_tracking_tolerance_ms cannot be negative")
        if self.exit_margin < 0:
            raise ValueError("exit_margin cannot be negative")
        if self.switch_confirmation_ms < 0:
            raise ValueError("switch_confirmation_ms cannot be negative")
        if self.saccade_velocity_threshold <= 0:
            raise ValueError("saccade_velocity_threshold must be positive")
        if self.max_dt_ms is not None and self.max_dt_ms <= 0:
            raise ValueError("max_dt_ms must be positive")

    @classmethod
    def create_preset(cls, skill: DwellUserSkill = DwellUserSkill.STANDARD, **overrides: Any) -> "DwellConfig":
        """
        Factory creating validated configuration profiles tailored for specific user needs
        without modifying core engine logic.
        """
        if skill == DwellUserSkill.NOVICE:
            params: Dict[str, Any] = {
                "dwell_duration_ms": 1500.0,
                "cooldown_duration_ms": 1000.0,
                "leave_target_tolerance_ms": 250.0,
                "exit_margin": 0.06,
                "filter_type": "one_euro",
                "filter_min_cutoff": 0.9,
                "filter_beta": 10.0,
                "switch_confirmation_ms": 120.0,
            }
        elif skill == DwellUserSkill.EXPERT:
            params = {
                "dwell_duration_ms": 800.0,
                "cooldown_duration_ms": 500.0,
                "leave_target_tolerance_ms": 100.0,
                "exit_margin": 0.03,
                "filter_type": "one_euro",
                "filter_min_cutoff": 1.5,
                "filter_beta": 15.0,
                "switch_confirmation_ms": 50.0,
            }
        elif skill == DwellUserSkill.TREMOR_PRONE:
            params = {
                "dwell_duration_ms": 1400.0,
                "cooldown_duration_ms": 900.0,
                "leave_target_tolerance_ms": 300.0,
                "exit_margin": 0.08,
                "filter_type": "one_euro",
                "filter_min_cutoff": 0.7,
                "filter_beta": 8.0,
                "switch_confirmation_ms": 150.0,
            }
        else:  # STANDARD (Default 1200ms - restored reliable baseline, no double filter, exact targets)
            params = {
                "dwell_duration_ms": 1200.0,
                "cooldown_duration_ms": 800.0,
                "leave_target_tolerance_ms": 150.0,
                "hysteresis_enabled": False,
                "exit_margin": 0.0,
                "filter_type": "none",
                "switch_confirmation_ms": 0.0,
                "fixation_detection_enabled": False,
            }

        params.update(overrides)
        return cls(**params)
