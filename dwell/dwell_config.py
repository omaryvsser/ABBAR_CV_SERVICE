"""
AABAR CV Service - Dwell Selection Configuration.

Provides configurable parameters for the Phase 4 Dwell Selection Engine.
Supports customizable dwell time, cooldown refractory periods, gaze stability
tolerances, tracking loss policies, and target overlap resolution rules.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class DwellConfig:
    """
    Configuration options for the Dwell Selection Engine.

    All time intervals are specified in milliseconds (ms) for accessibility
    and UI standard compliance, but internal calculations handle float seconds.
    """

    # Core Dwell Timing (ms)
    dwell_duration_ms: float = 1200.0  # Time (ms) required to fixate on a target to trigger selection
    cooldown_duration_ms: float = 800.0  # Refractory period (ms) after selection preventing repeat triggers
    selection_display_duration_ms: float = 250.0  # Duration (ms) the SELECTED state remains visibly highlighted

    # Gaze Stability & Saccade Grace Handling
    stability_filter_enabled: bool = True  # Whether to filter sudden erratic coordinate jumps
    max_jump_distance: float = 0.25  # Max normalized Euclidean distance per frame before flagging saccade
    leave_target_tolerance_ms: float = 150.0  # Grace period (ms) allowing brief saccade outside target without resetting progress
    lost_tracking_tolerance_ms: float = 250.0  # Grace period (ms) when eye tracking is lost before aborting dwell

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
        """Validate configuration parameters."""
        if self.dwell_duration_ms <= 0:
            raise ValueError(f"dwell_duration_ms must be positive, got {self.dwell_duration_ms}")
        if self.cooldown_duration_ms < 0:
            raise ValueError(f"cooldown_duration_ms cannot be negative, got {self.cooldown_duration_ms}")
        if self.screen_width <= 0 or self.screen_height <= 0:
            raise ValueError(f"Invalid screen dimensions: {self.screen_width}x{self.screen_height}")
        if self.leave_target_tolerance_ms < 0:
            raise ValueError("leave_target_tolerance_ms cannot be negative")
        if self.lost_tracking_tolerance_ms < 0:
            raise ValueError("lost_tracking_tolerance_ms cannot be negative")
