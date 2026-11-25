"""
AABAR CV Service - Dwell Selection Engine (Phase 4).

Modular, production-quality gaze-based dwell selection module for AAC interfaces.
"""

from .dwell_config import DwellConfig
from .dwell_engine import DwellEngine
from .dwell_state import DwellSnapshot, DwellState, SelectionEvent
from .target_manager import DwellTarget, TargetManager

__all__ = [
    "DwellConfig",
    "DwellEngine",
    "DwellSnapshot",
    "DwellState",
    "DwellTarget",
    "SelectionEvent",
    "TargetManager",
]
