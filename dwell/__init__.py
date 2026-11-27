"""
AABAR CV Service - Dwell Selection Engine (Phase 4).

Modular, production-quality gaze-based dwell selection module for AAC interfaces.
Enhanced with target hysteresis, adaptive gaze filtering, and fixation detection.
"""

from .dwell_config import DwellConfig, DwellUserSkill
from .dwell_engine import DwellEngine
from .dwell_state import DwellSnapshot, DwellState, SelectionEvent
from .gaze_filter import (
    BaseGazeFilter,
    EMAGazeFilter,
    GazeFilterType,
    OneEuroGazeFilter,
    PassThroughGazeFilter,
    create_gaze_filter,
)
from .target_manager import DwellTarget, TargetManager

__all__ = [
    "BaseGazeFilter",
    "DwellConfig",
    "DwellEngine",
    "DwellSnapshot",
    "DwellState",
    "DwellTarget",
    "DwellUserSkill",
    "EMAGazeFilter",
    "GazeFilterType",
    "OneEuroGazeFilter",
    "PassThroughGazeFilter",
    "SelectionEvent",
    "TargetManager",
    "create_gaze_filter",
]
