"""
Tracking Quality and State Definitions for AABAR CV Service.

Categorizes the health, reliability, and lifecycle of the gaze tracking pipeline.
"""

from enum import Enum


class TrackingState(str, Enum):
    """Lifecycle and reliability state of the gaze tracking pipeline."""
    TRACKING = "TRACKING"                         # High-fidelity tracking with valid baseline
    CALIBRATING = "CALIBRATING"                   # Currently sampling neutral center baseline
    LOW_CONFIDENCE = "LOW_CONFIDENCE"             # High head rotation, high ocular discrepancy, or partial occlusion
    NO_FACE = "NO_FACE"                           # FaceLandmarker detector found no face
    INVALID_EYES = "INVALID_EYES"                 # Eyes blinking, closed, or degenerate geometry
    RECALIBRATION_REQUIRED = "RECALIBRATION_REQUIRED" # Baseline is missing or uncalibrated
