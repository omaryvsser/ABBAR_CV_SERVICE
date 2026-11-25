"""
AABAR CV Service - Dwell State Definitions & Selection Events.

Defines the explicit state machine enumeration, selection event data structures,
and real-time dwell status snapshots for the Phase 4 Dwell Selection Engine.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, Optional
import uuid


class DwellState(str, Enum):
    """
    State enumeration for the Dwell Selection State Machine.

    States:
        IDLE: No valid target is currently being fixated or gaze is outside all targets.
        DWELLING: Gaze is steadily fixating inside an enabled target; timer is accumulating.
        SELECTED: Dwell threshold reached; a single selection event is emitted.
        COOLDOWN: Refractory period preventing repeated accidental selection.
        LOST_TRACKING: Gaze input is missing, invalid (NaN/Inf), or face is lost.
    """
    IDLE = "IDLE"
    DWELLING = "DWELLING"
    SELECTED = "SELECTED"
    COOLDOWN = "COOLDOWN"
    LOST_TRACKING = "LOST_TRACKING"


@dataclass
class SelectionEvent:
    """
    Structured, transport-independent selection event emitted when dwell completes.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_id: str = ""
    target_label: str = ""
    timestamp: float = field(default_factory=time.time)
    gaze_x: Optional[float] = None
    gaze_y: Optional[float] = None
    dwell_duration_ms: float = 0.0
    selection_state: str = "CONFIRMED"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to clean JSON-serializable dictionary."""
        return {
            "event_id": self.event_id,
            "target_id": self.target_id,
            "target_label": self.target_label,
            "timestamp": round(self.timestamp, 4),
            "gaze_x": round(self.gaze_x, 4) if self.gaze_x is not None else None,
            "gaze_y": round(self.gaze_y, 4) if self.gaze_y is not None else None,
            "dwell_duration_ms": round(self.dwell_duration_ms, 1),
            "selection_state": self.selection_state,
            "metadata": dict(self.metadata),
        }


@dataclass
class DwellSnapshot:
    """
    Real-time status snapshot produced on every engine update.

    Decoupled from OpenCV rendering and accessible for visualizer, UI, and testing.
    """
    state: DwellState = DwellState.IDLE
    current_target_id: Optional[str] = None
    current_target_label: Optional[str] = None
    dwell_time_ms: float = 0.0
    dwell_progress: float = 0.0  # Strictly clamped to [0.0, 1.0]
    remaining_time_ms: float = 0.0
    is_tracking_valid: bool = False
    gaze_x: Optional[float] = None
    gaze_y: Optional[float] = None
    selection_event: Optional[SelectionEvent] = None
    cooldown_progress: float = 0.0  # Clamped to [0.0, 1.0] when in COOLDOWN

    @property
    def target_id(self) -> Optional[str]:
        return self.current_target_id

    @property
    def target_label(self) -> Optional[str]:
        return self.current_target_label

    @property
    def progress(self) -> float:
        return self.dwell_progress

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "state": self.state.value,
            "current_target_id": self.current_target_id,
            "current_target_label": self.current_target_label,
            "dwell_time_ms": round(self.dwell_time_ms, 1),
            "dwell_progress": round(self.dwell_progress, 4),
            "remaining_time_ms": round(self.remaining_time_ms, 1),
            "is_tracking_valid": self.is_tracking_valid,
            "gaze_x": round(self.gaze_x, 4) if self.gaze_x is not None else None,
            "gaze_y": round(self.gaze_y, 4) if self.gaze_y is not None else None,
            "has_selection": self.selection_event is not None,
            "selection_event": self.selection_event.to_dict() if self.selection_event else None,
            "cooldown_progress": round(self.cooldown_progress, 4),
        }
