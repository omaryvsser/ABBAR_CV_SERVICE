"""
Structured Output Schema for AABAR Gaze Tracking.

Defines typed, JSON-serializable representations of real-time gaze estimation,
ocular geometry, and head pose orientation.
"""

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, Optional

from .gaze_direction import GazeDirection


@dataclass(frozen=True)
class SingleEyeOutput:
    """Output metrics for a single eye."""
    iris_x_ratio: Optional[float]
    iris_y_ratio: Optional[float]
    openness: float
    valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iris_x_ratio": round(self.iris_x_ratio, 4) if self.iris_x_ratio is not None else None,
            "iris_y_ratio": round(self.iris_y_ratio, 4) if self.iris_y_ratio is not None else None,
            "openness": round(self.openness, 4),
            "valid": self.valid,
        }


@dataclass(frozen=True)
class EyesOutput:
    """Output metrics for both eyes."""
    left: SingleEyeOutput
    right: SingleEyeOutput

    def to_dict(self) -> Dict[str, Any]:
        return {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


@dataclass(frozen=True)
class GazeOutput:
    """Fused relative gaze estimation output."""
    raw_x: Optional[float]
    raw_y: Optional[float]
    smoothed_x: Optional[float]
    smoothed_y: Optional[float]
    direction: str
    confidence: float
    baseline_x: Optional[float] = None
    baseline_y: Optional[float] = None
    delta_x: Optional[float] = None
    delta_y: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "raw_x": round(self.raw_x, 4) if self.raw_x is not None else None,
            "raw_y": round(self.raw_y, 4) if self.raw_y is not None else None,
            "smoothed_x": round(self.smoothed_x, 4) if self.smoothed_x is not None else None,
            "smoothed_y": round(self.smoothed_y, 4) if self.smoothed_y is not None else None,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
        }
        if self.baseline_x is not None:
            d["baseline_x"] = round(self.baseline_x, 4)
        if self.baseline_y is not None:
            d["baseline_y"] = round(self.baseline_y, 4)
        if self.delta_x is not None:
            d["delta_x"] = round(self.delta_x, 4)
        if self.delta_y is not None:
            d["delta_y"] = round(self.delta_y, 4)
        return d


@dataclass(frozen=True)
class HeadPoseOutput:
    """Head pose orientation output."""
    yaw: float
    pitch: float
    roll: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
        }


@dataclass(frozen=True)
class ScreenGazeOutput:
    """Predicted continuous 2D screen gaze coordinates and quality metrics (Phase 3)."""
    gaze_x: Optional[float]
    gaze_y: Optional[float]
    pixel_x: Optional[int]
    pixel_y: Optional[int]
    is_calibrated: bool
    tracking_quality: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gaze_x": round(self.gaze_x, 4) if self.gaze_x is not None else None,
            "gaze_y": round(self.gaze_y, 4) if self.gaze_y is not None else None,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "is_calibrated": self.is_calibrated,
            "tracking_quality": self.tracking_quality,
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class GazeFrameResult:
    """
    Root output container for a single processed video frame.
    """
    timestamp: float
    face_detected: bool
    gaze: GazeOutput
    eyes: EyesOutput
    head_pose: Optional[HeadPoseOutput]
    tracking_state: str = "TRACKING"
    screen_gaze: Optional[ScreenGazeOutput] = None
    dwell: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "timestamp": round(self.timestamp, 4),
            "face_detected": self.face_detected,
            "tracking_state": self.tracking_state,
            "gaze": self.gaze.to_dict(),
            "eyes": self.eyes.to_dict(),
            "head_pose": self.head_pose.to_dict() if self.head_pose else None,
        }
        if self.screen_gaze is not None:
            d["screen_gaze"] = self.screen_gaze.to_dict()
        if self.dwell is not None:
            d["dwell"] = self.dwell
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)
