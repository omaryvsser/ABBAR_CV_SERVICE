"""
Gaze Accuracy Evaluation & Validation System for AABAR CV Service (Phase 3 Enhanced).

Evaluates the empirical accuracy of a trained gaze-to-screen mapping model using
independent screen test targets that were not part of the training set.

Key Capabilities:
1. Strict train/test separation (zero data leakage; 5 interior targets vs 9 calibration targets).
2. Multi-trial repetition per target and optional target sequence randomization.
3. Online sample quality gating and explicit trial rejection tracking.
4. Robust trimmed-median feature aggregation.
5. Exact error calculations (signed, absolute, normalized Euclidean, pixel Euclidean).
6. Screen diagonal threshold compliance (<=2%, <=5%, <=10% screen diagonal).
7. 90th percentile error and per-target spatial breakdown with bias/pattern analysis.
8. Real-time live metrics feed (ValidationLiveMetrics) for instant HUD and canvas rendering.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .accuracy_metrics import compute_validation_metrics_from_trials
from .eye_features import BinocularEyeFeatures
from .gaze_baseline import GazeBaseline
from .gaze_mapper import BaseGazeMapper, extract_gaze_features
from .head_pose import HeadPose
from .screen_calibration import CalibrationTarget

logger = logging.getLogger(__name__)


# 5-Point Validation Grid (distinct intermediate targets in screen interior, strictly separate from 3x3 training grid)
DEFAULT_VALIDATION_TARGETS: List[CalibrationTarget] = [
    CalibrationTarget(0, "val_top_left",     0.30, 0.30),
    CalibrationTarget(1, "val_top_right",    0.70, 0.30),
    CalibrationTarget(2, "val_bottom_left",  0.30, 0.70),
    CalibrationTarget(3, "val_bottom_right", 0.70, 0.70),
    CalibrationTarget(4, "val_center",       0.50, 0.50),
]

# Full 9-Region Independent Validation Grid (covers Top/Middle/Bottom rows and Left/Center/Right columns)
FULL_9_REGION_VALIDATION_TARGETS: List[CalibrationTarget] = [
    CalibrationTarget(0, "val_top_left",      0.18, 0.18),
    CalibrationTarget(1, "val_top_center",    0.50, 0.18),
    CalibrationTarget(2, "val_top_right",     0.82, 0.18),
    CalibrationTarget(3, "val_middle_left",   0.18, 0.50),
    CalibrationTarget(4, "val_center",        0.50, 0.50),
    CalibrationTarget(5, "val_middle_right",  0.82, 0.50),
    CalibrationTarget(6, "val_bottom_left",   0.18, 0.82),
    CalibrationTarget(7, "val_bottom_center", 0.50, 0.82),
    CalibrationTarget(8, "val_bottom_right",  0.82, 0.82),
]


class ValidationPhase(str, Enum):
    """Validation target phase."""
    PENDING = "PENDING"
    SETTLING = "SETTLING"
    COLLECTING = "COLLECTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ValidationState(str, Enum):
    """Workflow state for validation."""
    IDLE = "IDLE"
    IN_PROGRESS = "IN_PROGRESS"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


@dataclass
class ValidationTrialRecord:
    """Individual trial result for a validation target."""
    experiment_id: str
    trial_id: int
    target_id: int
    target_name: str
    actual_x: float
    actual_y: float
    predicted_x: Optional[float]
    predicted_y: Optional[float]
    error_x: Optional[float]
    error_y: Optional[float]
    euclidean_error_norm: Optional[float]
    euclidean_error_pixels: Optional[float]
    screen_width: int
    screen_height: int
    tracking_quality: str
    accepted_sample_count: int
    timestamp: float = field(default_factory=time.time)
    # Enhanced Phase 3 attributes with backwards-compatible defaults
    calibration_profile_id: Optional[str] = None
    model_name: str = "unknown"
    model_version: str = "3.0"
    abs_error_x: Optional[float] = None
    abs_error_y: Optional[float] = None
    pixel_error_x: Optional[float] = None
    pixel_error_y: Optional[float] = None
    error_pct_diagonal: Optional[float] = None
    target_hit: Optional[bool] = None
    target_boundary: str = ""
    rejected_sample_count: int = 0
    trial_duration_sec: float = 0.0
    trial_status: str = "VALID"  # "VALID", "REJECTED", "CANCELLED"
    rejection_reason: Optional[str] = None
    session_id: str = ""
    raw_x: Optional[float] = None
    raw_y: Optional[float] = None
    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None

    def __post_init__(self) -> None:
        if self.error_x is not None:
            if self.abs_error_x is None:
                self.abs_error_x = abs(self.error_x)
            if self.pixel_error_x is None:
                self.pixel_error_x = self.error_x * float(self.screen_width)
        if self.error_y is not None:
            if self.abs_error_y is None:
                self.abs_error_y = abs(self.error_y)
            if self.pixel_error_y is None:
                self.pixel_error_y = self.error_y * float(self.screen_height)
        if self.euclidean_error_norm is None and self.error_x is not None and self.error_y is not None:
            self.euclidean_error_norm = float(math.hypot(self.error_x, self.error_y))
        if self.euclidean_error_pixels is None and self.pixel_error_x is not None and self.pixel_error_y is not None:
            self.euclidean_error_pixels = float(math.hypot(self.pixel_error_x, self.pixel_error_y))
        if self.euclidean_error_pixels is not None and self.error_pct_diagonal is None:
            diag = math.hypot(float(self.screen_width), float(self.screen_height))
            if diag > 0.0:
                self.error_pct_diagonal = (self.euclidean_error_pixels / diag) * 100.0

    def to_csv_dict(self) -> Dict[str, Any]:
        """Convert record to CSV column dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "session_id": self.session_id,
            "calibration_profile_id": self.calibration_profile_id or "",
            "model_name": self.model_name,
            "model_version": self.model_version,
            "trial_id": self.trial_id,
            "target_id": self.target_id,
            "target_name": self.target_name,
            "actual_x": round(self.actual_x, 4),
            "actual_y": round(self.actual_y, 4),
            "predicted_x": round(self.predicted_x, 4) if self.predicted_x is not None else None,
            "predicted_y": round(self.predicted_y, 4) if self.predicted_y is not None else None,
            "error_x": round(self.error_x, 4) if self.error_x is not None else None,
            "error_y": round(self.error_y, 4) if self.error_y is not None else None,
            "abs_error_x": round(self.abs_error_x, 4) if self.abs_error_x is not None else None,
            "abs_error_y": round(self.abs_error_y, 4) if self.abs_error_y is not None else None,
            "pixel_error_x": round(self.pixel_error_x, 2) if self.pixel_error_x is not None else None,
            "pixel_error_y": round(self.pixel_error_y, 2) if self.pixel_error_y is not None else None,
            "euclidean_error_normalized": round(self.euclidean_error_norm, 4) if self.euclidean_error_norm is not None else None,
            "euclidean_error_pixels": round(self.euclidean_error_pixels, 2) if self.euclidean_error_pixels is not None else None,
            "error_pct_diagonal": round(self.error_pct_diagonal, 2) if self.error_pct_diagonal is not None else None,
            "target_hit": int(self.target_hit) if self.target_hit is not None else None,
            "target_boundary": self.target_boundary,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "tracking_quality": self.tracking_quality,
            "accepted_sample_count": self.accepted_sample_count,
            "rejected_sample_count": self.rejected_sample_count,
            "trial_duration_sec": round(self.trial_duration_sec, 2),
            "trial_status": self.trial_status,
            "rejection_reason": self.rejection_reason or "",
            "raw_x": round(self.raw_x, 4) if self.raw_x is not None else None,
            "raw_y": round(self.raw_y, 4) if self.raw_y is not None else None,
            "head_yaw": round(self.head_yaw, 2) if self.head_yaw is not None else None,
            "head_pitch": round(self.head_pitch, 2) if self.head_pitch is not None else None,
            "timestamp": round(self.timestamp, 4),
        }


@dataclass
class ValidationLiveMetrics:
    """Real-time live experiment metrics for display during active validation."""
    total_trials: int
    completed_trials: int
    valid_trials: int
    rejected_trials: int
    current_trial_id: int
    current_target_name: str
    current_target_x: float
    current_target_y: float
    current_predicted_x: Optional[float]
    current_predicted_y: Optional[float]
    current_error_norm: Optional[float]
    current_error_px: Optional[float]
    running_mean_error_norm: float
    running_mean_error_px: float
    running_median_error_px: float
    running_mae_x_px: float
    running_mae_y_px: float
    running_rmse_x_px: float
    running_rmse_y_px: float
    running_p90_error_px: Optional[float]
    pct_within_2pct: float
    pct_within_5pct: float
    pct_within_10pct: float
    current_accepted_samples: int
    current_rejected_samples: int
    current_phase: str
    successful_selections: int = 0
    failed_selections: int = 0
    pct_target_selection: float = 0.0


@dataclass
class AccuracySummary:
    """Aggregated numerical accuracy and error evaluation report."""
    experiment_id: str
    model_name: str
    model_configuration: Dict[str, Any]
    total_trials: int
    successful_trials: int
    rejected_trials: int
    mae_x: float
    mae_y: float
    rmse_x: float
    rmse_y: float
    mean_euclidean_error: float
    median_euclidean_error: float
    maximum_error: float
    mean_error_pixels: float
    median_error_pixels: float
    max_error_pixels: float
    screen_resolution: str
    evaluation_timestamp: float = field(default_factory=time.time)
    target_accuracies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Enhanced Phase 3 attributes with backwards-compatible defaults
    mae_x_px: float = 0.0
    mae_y_px: float = 0.0
    rmse_x_px: float = 0.0
    rmse_y_px: float = 0.0
    p90_error_norm: Optional[float] = None
    p90_error_pixels: Optional[float] = None
    screen_width: int = 1920
    screen_height: int = 1080
    screen_diagonal_px: float = 2202.9
    threshold_2pct_frac: float = 0.02
    threshold_5pct_frac: float = 0.05
    threshold_10pct_frac: float = 0.10
    threshold_2pct_px: float = 44.06
    threshold_5pct_px: float = 110.15
    threshold_10pct_px: float = 220.29
    trials_within_2pct: int = 0
    pct_within_2pct: float = 0.0
    trials_within_5pct: int = 0
    pct_within_5pct: float = 0.0
    trials_within_10pct: int = 0
    pct_within_10pct: float = 0.0
    successful_selections: int = 0
    failed_selections: int = 0
    target_selection_accuracy_pct: float = 0.0
    target_boundary_description: str = ""
    mean_error_pct_diagonal: float = 0.0
    median_error_pct_diagonal: float = 0.0
    max_error_pct_diagonal: float = 0.0
    normalization_method: str = "None"
    training_sample_count: int = 0
    training_metrics: Optional[Dict[str, float]] = None
    calibration_profile_id: Optional[str] = None
    data_leakage_prevented: bool = True
    pattern_analysis: Dict[str, Any] = field(default_factory=dict)
    direction_accuracies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    region_accuracies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    region_accuracies_9: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rejected_trial_pct: float = 0.0
    validation_status: str = "COMPLETED"
    failure_reason: Optional[str] = None
    coordinate_convention: str = "interface_left0_right1_v1"

    def to_csv_dict(self) -> Dict[str, Any]:
        """Convert summary to CSV dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "calibration_profile_id": self.calibration_profile_id or "",
            "model_name": self.model_name,
            "model_configuration": str(self.model_configuration.get("model_type", "unknown")),
            "normalization_method": self.normalization_method,
            "training_sample_count": self.training_sample_count,
            "total_trials": self.total_trials,
            "successful_trials": self.successful_trials,
            "rejected_trials": self.rejected_trials,
            "rejected_trial_pct": round(self.rejected_trial_pct, 2),
            "validation_status": self.validation_status,
            "failure_reason": self.failure_reason or "",
            "coordinate_convention": self.coordinate_convention,
            "successful_selections": self.successful_selections,
            "failed_selections": self.failed_selections,
            "target_selection_accuracy_pct": round(self.target_selection_accuracy_pct, 2),
            "target_boundary": self.target_boundary_description,
            "MAE_X": round(self.mae_x, 4),
            "MAE_Y": round(self.mae_y, 4),
            "MAE_X_pixels": round(self.mae_x_px, 2),
            "MAE_Y_pixels": round(self.mae_y_px, 2),
            "RMSE_X": round(self.rmse_x, 4),
            "RMSE_Y": round(self.rmse_y, 4),
            "RMSE_X_pixels": round(self.rmse_x_px, 2),
            "RMSE_Y_pixels": round(self.rmse_y_px, 2),
            "mean_euclidean_error": round(self.mean_euclidean_error, 4),
            "median_euclidean_error": round(self.median_euclidean_error, 4),
            "p90_euclidean_error": round(self.p90_error_norm, 4) if self.p90_error_norm is not None else "N/A",
            "maximum_error": round(self.maximum_error, 4),
            "mean_error_pixels": round(self.mean_error_pixels, 2),
            "mean_error_pct_diagonal": round(self.mean_error_pct_diagonal, 2),
            "median_error_pixels": round(self.median_error_pixels, 2),
            "median_error_pct_diagonal": round(self.median_error_pct_diagonal, 2),
            "p90_error_pixels": round(self.p90_error_pixels, 2) if self.p90_error_pixels is not None else "N/A",
            "max_error_pixels": round(self.max_error_pixels, 2),
            "max_error_pct_diagonal": round(self.max_error_pct_diagonal, 2),
            "screen_diagonal_pixels": round(self.screen_diagonal_px, 1),
            "pct_within_2pct_diagonal": round(self.pct_within_2pct, 2),
            "pct_within_5pct_diagonal": round(self.pct_within_5pct, 2),
            "pct_within_10pct_diagonal": round(self.pct_within_10pct, 2),
            "threshold_strict_pct": round(self.threshold_2pct_frac * 100.0, 2),
            "threshold_moderate_pct": round(self.threshold_5pct_frac * 100.0, 2),
            "threshold_relaxed_pct": round(self.threshold_10pct_frac * 100.0, 2),
            "screen_resolution": self.screen_resolution,
            "evaluation_timestamp": round(self.evaluation_timestamp, 4),
        }

    def to_report_dict(self) -> Dict[str, Any]:
        """Convert summary to formatted report dictionary for JSON output."""
        return {
            "experiment_id": self.experiment_id,
            "calibration_profile_id": self.calibration_profile_id,
            "model_name": self.model_name,
            "normalization_method": self.normalization_method,
            "training_sample_count": self.training_sample_count,
            "training_metrics": self.training_metrics,
            "model_configuration": self.model_configuration,
            "total_trials": self.total_trials,
            "successful_trials": self.successful_trials,
            "rejected_trials": self.rejected_trials,
            "rejected_trial_pct": round(self.rejected_trial_pct, 2),
            "validation_status": self.validation_status,
            "failure_reason": self.failure_reason or "",
            "coordinate_convention": self.coordinate_convention,
            "data_leakage_prevented": self.data_leakage_prevented,
            "validation_completed": self.successful_trials > 0,
            "accuracy_rates": {
                "target_selection_accuracy_pct": round(self.target_selection_accuracy_pct, 2),
                "successful_selections": self.successful_selections,
                "failed_selections": self.failed_selections,
                "target_boundary": self.target_boundary_description,
                "accuracy_within_2pct": round(self.pct_within_2pct, 2),
                "accuracy_within_5pct": round(self.pct_within_5pct, 2),
                "accuracy_within_10pct": round(self.pct_within_10pct, 2),
                "threshold_strict_pct": round(self.threshold_2pct_frac * 100.0, 2),
                "threshold_moderate_pct": round(self.threshold_5pct_frac * 100.0, 2),
                "threshold_relaxed_pct": round(self.threshold_10pct_frac * 100.0, 2),
            },
            "metrics": {
                "mae_x": round(self.mae_x, 4),
                "mae_y": round(self.mae_y, 4),
                "mae_x_pixels": round(self.mae_x_px, 2),
                "mae_y_pixels": round(self.mae_y_px, 2),
                "rmse_x": round(self.rmse_x, 4),
                "rmse_y": round(self.rmse_y, 4),
                "rmse_x_pixels": round(self.rmse_x_px, 2),
                "rmse_y_pixels": round(self.rmse_y_px, 2),
                "mean_euclidean_normalized": round(self.mean_euclidean_error, 4),
                "median_euclidean_normalized": round(self.median_euclidean_error, 4),
                "p90_euclidean_normalized": round(self.p90_error_norm, 4) if self.p90_error_norm is not None else None,
                "max_euclidean_normalized": round(self.maximum_error, 4),
                "mean_euclidean_pixels": round(self.mean_error_pixels, 2),
                "mean_error_pct_diagonal": round(self.mean_error_pct_diagonal, 2),
                "median_euclidean_pixels": round(self.median_error_pixels, 2),
                "median_error_pct_diagonal": round(self.median_error_pct_diagonal, 2),
                "p90_euclidean_pixels": round(self.p90_error_pixels, 2) if self.p90_error_pixels is not None else None,
                "max_euclidean_pixels": round(self.max_error_pixels, 2),
                "max_error_pct_diagonal": round(self.max_error_pct_diagonal, 2),
            },
            "threshold_analysis": {
                "screen_diagonal_pixels": round(self.screen_diagonal_px, 1),
                "threshold_2pct_fraction": self.threshold_2pct_frac,
                "threshold_5pct_fraction": self.threshold_5pct_frac,
                "threshold_10pct_fraction": self.threshold_10pct_frac,
                "threshold_2pct_pixels": round(self.threshold_2pct_px, 1),
                "threshold_5pct_pixels": round(self.threshold_5pct_px, 1),
                "threshold_10pct_pixels": round(self.threshold_10pct_px, 1),
                "trials_within_2pct": self.trials_within_2pct,
                "pct_within_2pct": round(self.pct_within_2pct, 2),
                "trials_within_5pct": self.trials_within_5pct,
                "pct_within_5pct": round(self.pct_within_5pct, 2),
                "trials_within_10pct": self.trials_within_10pct,
                "pct_within_10pct": round(self.pct_within_10pct, 2),
            },
            "screen_resolution": self.screen_resolution,
            "evaluation_timestamp": self.evaluation_timestamp,
            "per_target_results": self.target_accuracies,
            "target_accuracies": self.target_accuracies,
            "direction_accuracies": self.direction_accuracies,
            "region_accuracies": self.region_accuracies,
            "region_accuracies_9": self.region_accuracies_9,
            "pattern_analysis": self.pattern_analysis,
        }


class ScreenValidator:
    """
    Coordinates the validation session by presenting independent targets,
    recording predictions, and computing statistical evaluation metrics.
    """

    def __init__(
        self,
        targets: Optional[List[CalibrationTarget]] = None,
        settling_duration: float = 0.8,
        collection_duration: float = 1.2,
        min_samples_per_target: int = 12,
        screen_width: int = 1920,
        screen_height: int = 1080,
        experiment_id: str = "exp_001",
        session_id: str = "sess_001",
        num_repeats: int = 1,
        randomize_order: bool = False,
        calibration_profile_id: Optional[str] = None,
        threshold_2pct: float = 0.02,
        threshold_5pct: float = 0.05,
        threshold_10pct: float = 0.10,
        display_width: int = 1280,
        display_height: int = 720,
        default_display_radius_px: float = 22.0,
        target_radius_px: Optional[float] = None,
        normalization_method: Optional[str] = None,
        training_sample_count: int = 0,
        training_metrics: Optional[Dict[str, float]] = None,
        max_collection_timeout: Optional[float] = None,
        fixation_max_std: float = 0.025,
        max_interframe_step: float = 0.050,
        max_head_motion_step: float = 4.0,
        allow_target_retry: bool = False,
        initial_settling_duration: Optional[float] = None,
    ) -> None:
        self.base_targets = list(targets or DEFAULT_VALIDATION_TARGETS)
        self.settling_duration = max(0.01, settling_duration)
        self.initial_settling_duration = (
            max(0.01, float(initial_settling_duration))
            if initial_settling_duration is not None
            else (max(1.2, self.settling_duration * 1.5) if self.settling_duration >= 0.5 else self.settling_duration)
        )
        self.collection_duration = max(0.01, collection_duration)
        self.min_samples_per_target = max(1, min_samples_per_target)
        self.max_collection_timeout = (
            max(self.collection_duration, float(max_collection_timeout))
            if max_collection_timeout is not None
            else self.collection_duration
        )
        self.fixation_max_std = max(0.002, float(fixation_max_std))
        self.max_interframe_step = max(0.005, float(max_interframe_step))
        self.max_head_motion_step = max(0.5, float(max_head_motion_step))
        self.allow_target_retry = bool(allow_target_retry)

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.experiment_id = experiment_id
        self.session_id = session_id
        self.num_repeats = max(1, num_repeats)
        self.randomize_order = randomize_order
        self.calibration_profile_id = calibration_profile_id

        # Configurable diagonal threshold fractions (default 2%, 5%, 10%)
        self.threshold_2pct = max(1e-4, float(threshold_2pct))
        self.threshold_5pct = max(1e-4, float(threshold_5pct))
        self.threshold_10pct = max(1e-4, float(threshold_10pct))

        # Actual rendered display geometry (used for exact hit-testing)
        self.display_width = int(max(1, display_width))
        self.display_height = int(max(1, display_height))
        self.default_display_radius_px = float(default_display_radius_px)
        self.target_radius_px = float(target_radius_px) if target_radius_px is not None else None

        # Provenance from calibration session/profile (never mixed into validation accuracy)
        self.normalization_method: Optional[str] = normalization_method
        self.training_sample_count: int = int(training_sample_count)
        self.training_metrics: Optional[Dict[str, float]] = training_metrics

        # Screen geometry
        self.screen_diagonal_px = float(np.hypot(self.screen_width, self.screen_height))

        # Build full trial sequence
        self._trial_sequence: List[CalibrationTarget] = []
        self._build_trial_sequence()
        self.targets = list(self._trial_sequence)

        self.mapper: Optional[BaseGazeMapper] = None
        self.state: ValidationState = ValidationState.IDLE
        self.current_target_index: int = 0
        self.target_phase: ValidationPhase = ValidationPhase.PENDING
        self.phase_start_time: float = 0.0
        self.fixation_status: str = "IDLE"
        self.fixation_feedback: str = ""
        self.failure_reason: Optional[str] = None
        self.retried_trials: Dict[int, int] = {}

        # Sample collection and fixation stability buffers
        self._active_target_samples: List[np.ndarray] = []
        self._active_raw_xy: List[Tuple[float, float]] = []
        self._active_rejected_samples_count: int = 0
        self._active_rejection_reasons: Dict[str, int] = {}
        self._recent_raw_xy: List[Tuple[float, float]] = []
        self._prev_pose_yp: Optional[Tuple[float, float]] = None
        self._post_instability_cooldown: int = 0
        self._trial_records: List[ValidationTrialRecord] = []
        self.summary: Optional[AccuracySummary] = None

        # Live prediction memory for real-time visualization
        self.last_live_pred_x: Optional[float] = None
        self.last_live_pred_y: Optional[float] = None

    def _build_trial_sequence(self) -> None:
        """Construct the ordered list of trials supporting repeats and randomization."""
        seq: List[CalibrationTarget] = []
        for rep in range(self.num_repeats):
            rep_targets = list(self.base_targets)
            if self.randomize_order:
                # Deterministic or randomized shuffle per repetition block
                random.shuffle(rep_targets)
            seq.extend(rep_targets)
        self._trial_sequence = seq
        self.targets = list(self._trial_sequence)

    @property
    def is_active(self) -> bool:
        """True if validation is actively running."""
        return self.state == ValidationState.IN_PROGRESS

    @property
    def current_target(self) -> Optional[CalibrationTarget]:
        """Return active validation target."""
        if not self.is_active or self.current_target_index >= len(self._trial_sequence):
            return None
        return self._trial_sequence[self.current_target_index]

    @property
    def progress(self) -> float:
        """Overall progress in [0.0, 1.0]."""
        if self.state == ValidationState.COMPLETED:
            return 1.0
        if not self.is_active or len(self._trial_sequence) == 0:
            return 0.0
        return float(np.clip(self.current_target_index / float(len(self._trial_sequence)), 0.0, 1.0))

    def start(self, mapper: BaseGazeMapper) -> None:
        """Begin validation experiment with given trained mapper."""
        if not mapper.is_fitted:
            raise ValueError("Cannot evaluate an unfitted BaseGazeMapper.")
        self.mapper = mapper
        self.state = ValidationState.IN_PROGRESS
        self.current_target_index = 0
        self.failure_reason = None
        self.fixation_status = "SETTLING"
        self.fixation_feedback = ""
        self.retried_trials.clear()
        self._trial_records.clear()
        self._active_target_samples.clear()
        self._active_raw_xy.clear()
        self._active_rejected_samples_count = 0
        self._active_rejection_reasons.clear()
        self._recent_raw_xy.clear()
        self._prev_pose_yp = None
        self._post_instability_cooldown = 0
        self.summary = None
        self._build_trial_sequence()
        self._enter_target(0)
        logger.info(
            f"Starting Screen Accuracy Validation: {len(self._trial_sequence)} total trials "
            f"({len(self.base_targets)} targets x {self.num_repeats} repeats, randomized={self.randomize_order})..."
        )

    def pause(self) -> None:
        """Pause validation session."""
        if self.state == ValidationState.IN_PROGRESS:
            self.state = ValidationState.PAUSED
            logger.info("Validation experiment PAUSED.")

    def resume(self) -> None:
        """Resume paused validation session."""
        if self.state == ValidationState.PAUSED:
            self.state = ValidationState.IN_PROGRESS
            self.phase_start_time = time.time()  # Reset phase timer
            logger.info("Validation experiment RESUMED.")

    def cancel(self) -> None:
        """Cancel validation while preserving already completed trials."""
        logger.info(
            f"Validation cancelled by user. Preserving {len(self._trial_records)} completed trials."
        )
        self.state = ValidationState.CANCELLED
        self.target_phase = ValidationPhase.PENDING

    def _enter_target(self, target_idx: int) -> None:
        self.current_target_index = target_idx
        self.target_phase = ValidationPhase.SETTLING
        self.phase_start_time = time.time()
        self._active_target_samples.clear()
        self._active_raw_xy.clear()
        self._active_rejected_samples_count = 0
        self._active_rejection_reasons.clear()
        self._recent_raw_xy.clear()
        self._prev_pose_yp = None
        self._post_instability_cooldown = 0
        target = self._trial_sequence[target_idx]
        self.fixation_status = "SETTLING"
        self.fixation_feedback = f"Look at Validation Target {target_idx + 1}/{len(self._trial_sequence)} ({target.name})"
        logger.info(
            f"Validation Trial [{target_idx + 1}/{len(self._trial_sequence)}] "
            f"'{target.name}' at ({target.x:.2f}, {target.y:.2f}) - SETTLING ({self.settling_duration:.1f}s)..."
        )

    def retry_current_target(self) -> bool:
        """Retry the current failed validation target without discarding completed trials."""
        if not self.is_active or self.current_target_index >= len(self._trial_sequence):
            return False
        self.retried_trials[self.current_target_index] = self.retried_trials.get(self.current_target_index, 0) + 1
        self.failure_reason = None
        logger.info(
            f"Retrying Validation Trial [{self.current_target_index + 1}/{len(self._trial_sequence)}]..."
        )
        self._enter_target(self.current_target_index)
        return True

    def skip_failed_target(self) -> Optional[AccuracySummary]:
        """Record the failed target as REJECTED and advance to next target (or finalize)."""
        if not self.is_active or self.target_phase != ValidationPhase.FAILED:
            return None
        target = self.current_target
        if target is None:
            return None
        rejection_msg = self.failure_reason or "INSUFFICIENT_STABLE_SAMPLES"
        self._evaluate_current_target(target, is_valid=False, rejection_reason=rejection_msg)
        next_idx = self.current_target_index + 1
        if next_idx < len(self._trial_sequence):
            self._enter_target(next_idx)
            return None
        return self._finalize_validation()

    def update(
        self,
        eye_features: BinocularEyeFeatures,
        head_pose: Optional[HeadPose] = None,
        baseline: Optional[GazeBaseline] = None,
    ) -> Optional[AccuracySummary]:
        """
        Process frame and evaluate target. Returns AccuracySummary upon completion.
        """
        if self.state == ValidationState.COMPLETED:
            return self.summary

        if not self.is_active or self.mapper is None or self.target_phase == ValidationPhase.FAILED:
            return None

        target = self.current_target
        if target is None:
            return None

        now = time.time()
        elapsed = now - self.phase_start_time

        # Update live gaze prediction for visualization
        features = extract_gaze_features(eye_features, head_pose, baseline)
        if features is not None and self.mapper is not None:
            try:
                live_pred = self.mapper.predict(features)
                self.last_live_pred_x = float(live_pred[0])
                self.last_live_pred_y = float(live_pred[1])
            except Exception:
                pass

        # 1. Settling Phase
        if self.target_phase == ValidationPhase.SETTLING:
            target_settle = (
                self.initial_settling_duration
                if self.current_target_index == 0
                else self.settling_duration
            )
            if elapsed >= target_settle:
                self.target_phase = ValidationPhase.COLLECTING
                self.phase_start_time = now
                self._recent_raw_xy.clear()
                self.fixation_status = "COLLECTING"
                self.fixation_feedback = f"Hold gaze steady on Target {self.current_target_index + 1}/{len(self._trial_sequence)}"
                logger.info(
                    f"Validation Trial [{self.current_target_index + 1}/{len(self._trial_sequence)}] "
                    f"'{target.name}' - COLLECTING samples ({self.collection_duration:.1f}s)..."
                )
            return None

        # 2. Collecting Phase: Sample quality & fixation stability evaluation
        is_sample_valid = False
        sample_rejection_cause: Optional[str] = None

        head_motion_bad = False
        if head_pose and head_pose.is_valid:
            cur_yp = (float(head_pose.yaw), float(head_pose.pitch))
            if self._prev_pose_yp is not None:
                if abs(cur_yp[0] - self._prev_pose_yp[0]) > self.max_head_motion_step or abs(cur_yp[1] - self._prev_pose_yp[1]) > self.max_head_motion_step:
                    head_motion_bad = True
            self._prev_pose_yp = cur_yp

        if (eye_features.right_eye and eye_features.right_eye.is_blink) or (
            eye_features.left_eye and eye_features.left_eye.is_blink
        ):
            sample_rejection_cause = "BLINK_DETECTED"
            self.fixation_status = "NO_EYE_TRACKING"
            self.fixation_feedback = "Blink detected — keep eyes open on target"
            self._recent_raw_xy.clear()
            self._post_instability_cooldown = 2
        elif not eye_features.is_valid:
            sample_rejection_cause = "INVALID_EYE_LANDMARKS"
            self.fixation_status = "NO_EYE_TRACKING"
            self.fixation_feedback = "Eyes not detected clearly — face camera"
            self._recent_raw_xy.clear()
            self._post_instability_cooldown = 2
        elif eye_features.confidence < 0.40:
            sample_rejection_cause = f"LOW_CONFIDENCE_{eye_features.confidence:.2f}"
            self.fixation_status = "LOW_CONFIDENCE"
            self.fixation_feedback = "Low eye tracking confidence — look directly at target"
            self._recent_raw_xy.clear()
            self._post_instability_cooldown = 2
        elif (eye_features.right_eye and eye_features.right_eye.openness < 0.25) or (
            eye_features.left_eye and eye_features.left_eye.openness < 0.25
        ):
            sample_rejection_cause = "PARTIAL_EYE_CLOSURE"
            self.fixation_status = "PARTIAL_CLOSURE"
            self.fixation_feedback = "Open eyes wider to complete validation"
            self._recent_raw_xy.clear()
            self._post_instability_cooldown = 2
        elif head_pose and head_pose.is_valid and (
            abs(head_pose.yaw) > 25.0 or abs(head_pose.pitch) > 20.0
        ):
            sample_rejection_cause = f"EXCESSIVE_HEAD_ROTATION_Y{head_pose.yaw:.1f}"
            self.fixation_status = "HEAD_MOVEMENT_HIGH"
            self.fixation_feedback = "Head rotated too far — face screen center"
            self._recent_raw_xy.clear()
            self._post_instability_cooldown = 2
        elif head_motion_bad:
            sample_rejection_cause = "HEAD_MOVEMENT_UNSTABLE"
            self.fixation_status = "HEAD_MOVEMENT_HIGH"
            self.fixation_feedback = "Hold head steady while fixating"
            self._recent_raw_xy.clear()
            self._post_instability_cooldown = 2
        elif features is None or eye_features.combined_horizontal_ratio is None or eye_features.combined_vertical_ratio is None:
            sample_rejection_cause = "FEATURE_EXTRACTION_FAILED"
            self.fixation_status = "NO_EYE_TRACKING"
            self._recent_raw_xy.clear()
            self._post_instability_cooldown = 2
        else:
            rx = float(eye_features.combined_horizontal_ratio)
            ry = float(eye_features.combined_vertical_ratio)
            if self._recent_raw_xy:
                px, py = self._recent_raw_xy[-1]
                if math.hypot(rx - px, ry - py) > self.max_interframe_step:
                    sample_rejection_cause = "FIXATION_SACCADE_JUMP"
                    self.fixation_status = "UNSTABLE_FIXATION"
                    self.fixation_feedback = "Eyes moving — hold gaze steady on target"
                    self._recent_raw_xy = [(rx, ry)]
                    self._post_instability_cooldown = 2
            if sample_rejection_cause is None:
                self._recent_raw_xy.append((rx, ry))
                if len(self._recent_raw_xy) > 6:
                    self._recent_raw_xy.pop(0)
                if self._post_instability_cooldown > 0:
                    self._post_instability_cooldown -= 1
                    sample_rejection_cause = "POST_JUMP_STABILIZING"
                    self.fixation_status = "UNSTABLE_FIXATION"
                    self.fixation_feedback = "Re-acquiring steady gaze on target..."
                elif len(self._recent_raw_xy) >= 4:
                    arr_w = np.array(self._recent_raw_xy, dtype=np.float64)
                    if float(np.std(arr_w[:, 0])) > self.fixation_max_std or float(np.std(arr_w[:, 1])) > self.fixation_max_std:
                        sample_rejection_cause = "FIXATION_UNSTABLE_DISPERSION"
                        self.fixation_status = "UNSTABLE_FIXATION"
                        self.fixation_feedback = "Hold gaze steady on target dot"
                if sample_rejection_cause is None and len(self._active_raw_xy) >= 4:
                    med_x = float(np.median([p[0] for p in self._active_raw_xy]))
                    med_y = float(np.median([p[1] for p in self._active_raw_xy]))
                    cluster_tol = max(0.06, 3.0 * self.fixation_max_std)
                    if math.hypot(rx - med_x, ry - med_y) > cluster_tol:
                        sample_rejection_cause = "FIXATION_OUTLIER_DRIFT"
                        self.fixation_status = "UNSTABLE_FIXATION"
                        self.fixation_feedback = "Gaze drifted from target — hold on dot"
                if sample_rejection_cause is None:
                    is_sample_valid = True
                    self.fixation_status = "FIXATION_STEADY"
                    self.fixation_feedback = (
                        f"Steady ({len(self._active_target_samples) + 1}/{self.min_samples_per_target})"
                    )

        if is_sample_valid and features is not None:
            self._active_target_samples.append(features)
            if eye_features.combined_horizontal_ratio is not None and eye_features.combined_vertical_ratio is not None:
                self._active_raw_xy.append(
                    (float(eye_features.combined_horizontal_ratio), float(eye_features.combined_vertical_ratio))
                )
        else:
            self._active_rejected_samples_count += 1
            if sample_rejection_cause:
                self._active_rejection_reasons[sample_rejection_cause] = (
                    self._active_rejection_reasons.get(sample_rejection_cause, 0) + 1
                )

        # 3. Check target completion (wait up to max_collection_timeout for stable samples)
        if elapsed >= self.collection_duration:
            if len(self._active_target_samples) >= self.min_samples_per_target:
                self._evaluate_current_target(target, is_valid=True)
                next_idx = self.current_target_index + 1
                if next_idx < len(self._trial_sequence):
                    self._enter_target(next_idx)
                else:
                    return self._finalize_validation()
            elif elapsed >= self.max_collection_timeout:
                dominant_cause = (
                    max(self._active_rejection_reasons, key=self._active_rejection_reasons.get)
                    if self._active_rejection_reasons
                    else "LOW_SAMPLE_RATE"
                )
                rejection_msg = (
                    f"INSUFFICIENT_SAMPLES (accepted {len(self._active_target_samples)}/{self.min_samples_per_target}, "
                    f"dominant cause: {dominant_cause})"
                )
                if self.allow_target_retry:
                    self.target_phase = ValidationPhase.FAILED
                    self.fixation_status = "TARGET_FAILED"
                    self.failure_reason = rejection_msg
                    self.fixation_feedback = (
                        f"Target {self.current_target_index + 1}/{len(self._trial_sequence)} failed "
                        f"({dominant_cause}). Press 'y' to retry or 'k' to skip."
                    )
                    logger.warning(
                        f"Validation Trial [{self.current_target_index + 1}/{len(self._trial_sequence)}] "
                        f"paused for retry: {rejection_msg}"
                    )
                    return None

                logger.warning(
                    f"Validation Trial [{self.current_target_index + 1}] REJECTED: {rejection_msg}"
                )
                self._evaluate_current_target(target, is_valid=False, rejection_reason=rejection_msg)
                next_idx = self.current_target_index + 1
                if next_idx < len(self._trial_sequence):
                    self._enter_target(next_idx)
                else:
                    return self._finalize_validation()

        return None

    def _evaluate_target_hit(
        self,
        target: CalibrationTarget,
        pred_x: float,
        pred_y: float,
    ) -> Tuple[bool, str]:
        """Check whether predicted gaze falls inside the target's actual boundary."""
        return target.contains_point(
            pred_x=pred_x,
            pred_y=pred_y,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            display_width=self.display_width,
            display_height=self.display_height,
            default_display_radius_px=self.default_display_radius_px,
            fallback_radius_px=self.target_radius_px,
        )

    def _evaluate_current_target(
        self,
        target: CalibrationTarget,
        is_valid: bool = True,
        rejection_reason: Optional[str] = None,
    ) -> None:
        """Predict coordinates from aggregated target samples and record trial metrics."""
        trial_num = len(self._trial_records) + 1
        trial_duration = time.time() - self.phase_start_time

        if is_valid and len(self._active_target_samples) > 0:
            samples_arr = np.array(self._active_target_samples)

            # 1. Spatial MAD outlier pruning if paired raw coordinates exist
            if len(samples_arr) >= 6 and len(self._active_raw_xy) == len(samples_arr):
                raw_arr = np.array(self._active_raw_xy, dtype=np.float64)
                med_rx = float(np.median(raw_arr[:, 0]))
                med_ry = float(np.median(raw_arr[:, 1]))
                mad_x = 1.4826 * float(np.median(np.abs(raw_arr[:, 0] - med_rx)))
                mad_y = 1.4826 * float(np.median(np.abs(raw_arr[:, 1] - med_ry)))
                tol_x = max(0.025, 3.0 * mad_x)
                tol_y = max(0.025, 3.0 * mad_y)
                spat_mask = (np.abs(raw_arr[:, 0] - med_rx) <= tol_x) & (np.abs(raw_arr[:, 1] - med_ry) <= tol_y)
                if int(np.sum(spat_mask)) >= max(4, min(self.min_samples_per_target, len(samples_arr) // 2)):
                    samples_arr = samples_arr[spat_mask]

            # 2. Robust aggregation: Trimmed Median (reject outer 10% outliers per dimension)
            if len(samples_arr) >= 5:
                p10 = np.percentile(samples_arr, 10, axis=0)
                p90 = np.percentile(samples_arr, 90, axis=0)
                inliers = np.all((samples_arr >= p10) & (samples_arr <= p90), axis=1)
                clean_samples = samples_arr[inliers] if np.sum(inliers) >= 3 else samples_arr
            else:
                clean_samples = samples_arr

            median_features = np.median(clean_samples, axis=0)
            raw_x_val = float(median_features[0])
            raw_y_val = float(median_features[1])
            yaw_val = float(median_features[8] * 30.0) if len(median_features) >= 10 else None
            pitch_val = float(median_features[9] * 20.0) if len(median_features) >= 10 else None

            assert self.mapper is not None
            pred = self.mapper.predict(median_features)
            pred_x, pred_y = float(pred[0]), float(pred[1])

            # Exact errors: error_x = predicted_x - actual_x
            err_x = pred_x - target.x
            err_y = pred_y - target.y
            abs_x = abs(err_x)
            abs_y = abs(err_y)
            e_norm = float(np.hypot(err_x, err_y))

            # Pixel conversions: pixel_error_x = error_x * screen_width
            err_px_x = err_x * float(self.screen_width)
            err_px_y = err_y * float(self.screen_height)
            e_px = float(np.hypot(err_px_x, err_px_y))
            e_pct = (e_px / self.screen_diagonal_px) * 100.0 if self.screen_diagonal_px > 0 else 0.0

            # Actual target/button hit test
            is_hit, boundary_spec = self._evaluate_target_hit(target, pred_x, pred_y)

            quality = "EXCELLENT" if len(self._active_target_samples) >= 20 else "GOOD"
            status = "VALID"
            reason = None
        else:
            raw_x_val, raw_y_val, yaw_val, pitch_val = None, None, None, None
            pred_x, pred_y = None, None
            err_x, err_y = None, None
            abs_x, abs_y = None, None
            err_px_x, err_px_y = None, None
            e_norm, e_px, e_pct = None, None, None
            is_hit = None
            _, boundary_spec = self._evaluate_target_hit(target, target.x, target.y)
            quality = "POOR"
            status = "REJECTED"
            reason = rejection_reason or "INSUFFICIENT_VALID_FRAMES"

        record = ValidationTrialRecord(
            experiment_id=self.experiment_id,
            session_id=self.session_id,
            calibration_profile_id=self.calibration_profile_id,
            model_name=self.mapper.model_type if self.mapper else "unknown",
            model_version="3.0",
            trial_id=trial_num,
            target_id=target.target_id,
            target_name=target.name,
            actual_x=target.x,
            actual_y=target.y,
            predicted_x=pred_x,
            predicted_y=pred_y,
            error_x=err_x,
            error_y=err_y,
            abs_error_x=abs_x,
            abs_error_y=abs_y,
            pixel_error_x=err_px_x,
            pixel_error_y=err_px_y,
            euclidean_error_norm=e_norm,
            euclidean_error_pixels=e_px,
            error_pct_diagonal=e_pct,
            target_hit=is_hit,
            target_boundary=boundary_spec,
            raw_x=raw_x_val,
            raw_y=raw_y_val,
            head_yaw=yaw_val,
            head_pitch=pitch_val,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            tracking_quality=quality,
            accepted_sample_count=len(self._active_target_samples),
            rejected_sample_count=self._active_rejected_samples_count,
            trial_duration_sec=trial_duration,
            trial_status=status,
            rejection_reason=reason,
            timestamp=time.time(),
        )
        self._trial_records.append(record)

        if status == "VALID" and pred_x is not None and pred_y is not None and e_norm is not None and e_px is not None:
            hit_tag = "HIT" if is_hit else "MISS"
            logger.info(
                f"Validation Trial [{trial_num}/{len(self._trial_sequence)}] '{target.name}' [{hit_tag}]: "
                f"Actual=({target.x:.2f}, {target.y:.2f}) | Pred=({pred_x:.2f}, {pred_y:.2f}) | "
                f"Err={e_px:.1f} px ({e_pct:.1f}% diag)"
            )
        else:
            logger.warning(
                f"Validation Trial [{trial_num}/{len(self._trial_sequence)}] '{target.name}' REJECTED: {reason}"
            )

    def get_live_metrics(self) -> ValidationLiveMetrics:
        """Calculate and return running live experiment metrics for HUD/dashboard rendering."""
        valid_records = [r for r in self._trial_records if r.trial_status == "VALID" and r.error_x is not None]
        rejected_records = [r for r in self._trial_records if r.trial_status == "REJECTED"]

        cur_target = self.current_target
        tx = cur_target.x if cur_target else 0.5
        ty = cur_target.y if cur_target else 0.5
        tname = cur_target.name if cur_target else "idle"

        # Calculate current real-time error against active target
        if self.last_live_pred_x is not None and self.last_live_pred_y is not None and cur_target is not None:
            live_err_x = self.last_live_pred_x - cur_target.x
            live_err_y = self.last_live_pred_y - cur_target.y
            cur_e_norm = float(np.hypot(live_err_x, live_err_y))
            cur_e_px = float(np.hypot(live_err_x * self.screen_width, live_err_y * self.screen_height))
        else:
            cur_e_norm = None
            cur_e_px = None

        if valid_records:
            e_norm = np.array([r.euclidean_error_norm for r in valid_records if r.euclidean_error_norm is not None])
            e_px = np.array([r.euclidean_error_pixels for r in valid_records if r.euclidean_error_pixels is not None])
            err_x = np.array([r.error_x for r in valid_records if r.error_x is not None])
            err_y = np.array([r.error_y for r in valid_records if r.error_y is not None])

            mean_norm = float(np.mean(e_norm))
            mean_px = float(np.mean(e_px))
            med_px = float(np.median(e_px))
            mae_x_px = float(np.mean(np.abs(err_x))) * float(self.screen_width)
            mae_y_px = float(np.mean(np.abs(err_y))) * float(self.screen_height)
            rmse_x_px = float(np.sqrt(np.mean(err_x ** 2))) * float(self.screen_width)
            rmse_y_px = float(np.sqrt(np.mean(err_y ** 2))) * float(self.screen_height)
            p90_px = float(np.percentile(e_px, 90)) if len(valid_records) >= 5 else None

            # Threshold compliance percentages
            diag_px = self.screen_diagonal_px
            pct_2 = float(np.sum(e_px <= self.threshold_2pct * diag_px + 1e-9) / len(valid_records) * 100.0)
            pct_5 = float(np.sum(e_px <= self.threshold_5pct * diag_px + 1e-9) / len(valid_records) * 100.0)
            pct_10 = float(np.sum(e_px <= self.threshold_10pct * diag_px + 1e-9) / len(valid_records) * 100.0)

            hits = sum(1 for r in valid_records if r.target_hit is True)
            fails = len(valid_records) - hits
            pct_sel = float(hits / len(valid_records) * 100.0)
        else:
            mean_norm, mean_px, med_px = 0.0, 0.0, 0.0
            mae_x_px, mae_y_px = 0.0, 0.0
            rmse_x_px, rmse_y_px = 0.0, 0.0
            p90_px = None
            pct_2, pct_5, pct_10 = 0.0, 0.0, 0.0
            hits, fails, pct_sel = 0, 0, 0.0

        return ValidationLiveMetrics(
            total_trials=len(self._trial_sequence),
            completed_trials=len(self._trial_records),
            valid_trials=len(valid_records),
            rejected_trials=len(rejected_records),
            current_trial_id=self.current_target_index + 1,
            current_target_name=tname,
            current_target_x=tx,
            current_target_y=ty,
            current_predicted_x=self.last_live_pred_x,
            current_predicted_y=self.last_live_pred_y,
            current_error_norm=cur_e_norm,
            current_error_px=cur_e_px,
            running_mean_error_norm=mean_norm,
            running_mean_error_px=mean_px,
            running_median_error_px=med_px,
            running_mae_x_px=mae_x_px,
            running_mae_y_px=mae_y_px,
            running_rmse_x_px=rmse_x_px,
            running_rmse_y_px=rmse_y_px,
            running_p90_error_px=p90_px,
            pct_within_2pct=pct_2,
            pct_within_5pct=pct_5,
            pct_within_10pct=pct_10,
            current_accepted_samples=len(self._active_target_samples),
            current_rejected_samples=self._active_rejected_samples_count,
            current_phase=self.target_phase.value,
            successful_selections=hits,
            failed_selections=fails,
            pct_target_selection=pct_sel,
        )

    def _summarize_trial_subset(
        self,
        valid_subset: List[ValidationTrialRecord],
        rejected_subset: List[ValidationTrialRecord],
        thresh_2_px: float,
        thresh_5_px: float,
        thresh_10_px: float,
    ) -> Dict[str, Any]:
        """Compute standardized accuracy and error summary metrics for a subset of trials."""
        n_val = len(valid_subset)
        n_rej = len(rejected_subset)
        if n_val == 0:
            return {
                "total_attempts": n_rej,
                "valid_trials": 0,
                "rejected_trials": n_rej,
                "successful_selections": 0,
                "failed_selections": 0,
                "target_selection_accuracy_pct": None,
                "pct_within_2pct": None,
                "pct_within_5pct": None,
                "pct_within_10pct": None,
                "mean_euclidean_pixels": None,
                "median_euclidean_pixels": None,
                "signed_bias_x_px": None,
                "signed_bias_y_px": None,
            }
        e_px = np.array([r.euclidean_error_pixels for r in valid_subset if r.euclidean_error_pixels is not None], dtype=np.float64)
        err_x = np.array([r.error_x for r in valid_subset if r.error_x is not None], dtype=np.float64)
        err_y = np.array([r.error_y for r in valid_subset if r.error_y is not None], dtype=np.float64)
        hits = sum(1 for r in valid_subset if r.target_hit is True)
        return {
            "total_attempts": n_val + n_rej,
            "valid_trials": n_val,
            "rejected_trials": n_rej,
            "successful_selections": hits,
            "failed_selections": n_val - hits,
            "target_selection_accuracy_pct": round(float(hits / n_val * 100.0), 2),
            "pct_within_2pct": round(float(np.sum(e_px <= thresh_2_px + 1e-9) / n_val * 100.0), 2),
            "pct_within_5pct": round(float(np.sum(e_px <= thresh_5_px + 1e-9) / n_val * 100.0), 2),
            "pct_within_10pct": round(float(np.sum(e_px <= thresh_10_px + 1e-9) / n_val * 100.0), 2),
            "mean_euclidean_pixels": round(float(np.mean(e_px)), 2),
            "median_euclidean_pixels": round(float(np.median(e_px)), 2),
            "signed_bias_x_px": round(float(np.mean(err_x)) * self.screen_width, 2),
            "signed_bias_y_px": round(float(np.mean(err_y)) * self.screen_height, 2),
        }

    def _finalize_validation(self) -> AccuracySummary:
        """Compute aggregated statistical metrics across all valid validation trials."""
        assert self.mapper is not None

        valid_raw = [r for r in self._trial_records if r.trial_status == "VALID" and r.error_x is not None]
        rejected_raw = [r for r in self._trial_records if r.trial_status == "REJECTED"]

        if not valid_raw:
            self.state = ValidationState.ERROR
            self.failure_reason = (
                f"Validation failed with 0 successful valid trials "
                f"({len(rejected_raw)} trials rejected due to poor tracking, unstable gaze, or blinks)."
            )
            raise RuntimeError(self.failure_reason)

        # Ensure custom display/target-radius hit boundaries are populated using ScreenValidator config
        target_lookup = {t.target_id: t for t in self.base_targets}
        for rec in valid_raw:
            if rec.target_hit is None and rec.predicted_x is not None and rec.predicted_y is not None:
                ref_target = target_lookup.get(
                    rec.target_id,
                    CalibrationTarget(rec.target_id, rec.target_name, rec.actual_x, rec.actual_y),
                )
                hit, b_spec = self._evaluate_target_hit(ref_target, rec.predicted_x, rec.predicted_y)
                rec.target_hit = hit
                if not rec.target_boundary:
                    rec.target_boundary = b_spec

        # Single-source-of-truth metric computation shared with tracker & reports
        calc = compute_validation_metrics_from_trials(
            self._trial_records,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            threshold_2pct=self.threshold_2pct,
            threshold_5pct=self.threshold_5pct,
            threshold_10pct=self.threshold_10pct,
            default_experiment_id=self.experiment_id,
            default_session_id=self.session_id,
        )

        target_accs = calc["target_accuracies"]
        direction_accs = calc["direction_accuracies"]
        region_accs = calc["region_accuracies"]
        region_accs_9 = calc["region_accuracies_9"]
        diag_px = float(calc["screen_diagonal_px"])
        thresh_2pct_px = float(calc["threshold_2pct_px"])
        thresh_5pct_px = float(calc["threshold_5pct_px"])
        thresh_10pct_px = float(calc["threshold_10pct_px"])

        mean_e_px = float(calc["mean_error_pixels"])
        median_e_px = float(calc["median_error_pixels"])
        max_e_px = float(calc["max_error_pixels"])
        mean_err_pct_diag = float(calc["mean_error_pct_diagonal"])
        pct_2 = float(calc["pct_within_2pct"])
        pct_5 = float(calc["pct_within_5pct"])
        pct_10 = float(calc["pct_within_10pct"])
        successful_selections = int(calc["successful_selections"])
        failed_selections = int(calc["failed_selections"])
        target_selection_acc_pct = float(calc["target_selection_accuracy_pct"])
        rejected_pct = float(calc["rejected_trial_pct"])
        boundary_desc = str(calc["target_boundary_description"])

        # Enrich pattern_analysis while preserving exact backwards-compatible keys
        err_x = np.array([r.error_x for r in valid_raw if r.error_x is not None], dtype=np.float64)
        err_y = np.array([r.error_y for r in valid_raw if r.error_y is not None], dtype=np.float64)
        valid_target_accs = {k: v for k, v in target_accs.items() if v.get("valid_trials", 0) > 0}
        valid_dir_accs = {k: v for k, v in direction_accs.items() if v.get("valid_trials", 0) > 0 and v.get("mean_euclidean_pixels") is not None}
        valid_reg_accs = {k: v for k, v in region_accs.items() if v.get("valid_trials", 0) > 0 and v.get("mean_euclidean_pixels") is not None}

        if valid_target_accs:
            best_t = min(valid_target_accs.items(), key=lambda item: item[1]["mean_euclidean_pixels"])
            worst_t = max(valid_target_accs.items(), key=lambda item: item[1]["mean_euclidean_pixels"])
            weakest_dir = (
                max(valid_dir_accs.items(), key=lambda item: item[1]["mean_euclidean_pixels"])
                if valid_dir_accs
                else None
            )
            best_dir = (
                min(valid_dir_accs.items(), key=lambda item: item[1]["mean_euclidean_pixels"])
                if valid_dir_accs
                else None
            )
            weakest_reg = (
                max(valid_reg_accs.items(), key=lambda item: item[1]["mean_euclidean_pixels"])
                if valid_reg_accs
                else None
            )
            left_errs = [v["mean_euclidean_pixels"] for k, v in valid_target_accs.items() if "left" in k]
            right_errs = [v["mean_euclidean_pixels"] for k, v in valid_target_accs.items() if "right" in k]
            center_errs = [v["mean_euclidean_pixels"] for k, v in valid_target_accs.items() if "center" in k]
            edge_err = region_accs.get("edge_regions", {}).get("mean_euclidean_pixels")
            ctr_err = region_accs.get("center_region", {}).get("mean_euclidean_pixels")
            edge_vs_ctr_ratio = (
                round(float(edge_err / max(1e-6, ctr_err)), 2)
                if (edge_err is not None and ctr_err is not None and ctr_err > 1e-6)
                else None
            )
            pattern_analysis: Dict[str, Any] = {
                "best_target": {
                    "name": best_t[0],
                    "mean_pixel_error": best_t[1]["mean_euclidean_pixels"],
                    "target_selection_accuracy_pct": best_t[1]["target_selection_accuracy_pct"],
                },
                "worst_target": {
                    "name": worst_t[0],
                    "mean_pixel_error": worst_t[1]["mean_euclidean_pixels"],
                    "target_selection_accuracy_pct": worst_t[1]["target_selection_accuracy_pct"],
                },
                "best_direction": {
                    "direction": best_dir[0],
                    "mean_pixel_error": best_dir[1]["mean_euclidean_pixels"],
                    "target_selection_accuracy_pct": best_dir[1]["target_selection_accuracy_pct"],
                } if best_dir else None,
                "weakest_direction": {
                    "direction": weakest_dir[0],
                    "mean_pixel_error": weakest_dir[1]["mean_euclidean_pixels"],
                    "target_selection_accuracy_pct": weakest_dir[1]["target_selection_accuracy_pct"],
                } if weakest_dir else None,
                "weakest_region": {
                    "region": weakest_reg[0],
                    "mean_pixel_error": weakest_reg[1]["mean_euclidean_pixels"],
                    "target_selection_accuracy_pct": weakest_reg[1]["target_selection_accuracy_pct"],
                } if weakest_reg else None,
                "overall_horizontal_bias_px": round(float(np.mean(err_x)) * self.screen_width, 2),
                "overall_vertical_bias_px": round(float(np.mean(err_y)) * self.screen_height, 2),
                "left_targets_mean_px": round(float(np.mean(left_errs)), 2) if left_errs else None,
                "right_targets_mean_px": round(float(np.mean(right_errs)), 2) if right_errs else None,
                "center_target_mean_px": round(float(np.mean(center_errs)), 2) if center_errs else None,
                "edge_vs_center_error_ratio": edge_vs_ctr_ratio,
                "sample_size_adequate": len(valid_raw) >= 9,
                "sample_size_note": (
                    "Full multi-target coverage (>= 9 trials)"
                    if len(valid_raw) >= 9
                    else f"Preliminary ({len(valid_raw)} trials; run full 9-point grid for broader statistical confidence)"
                ),
            }
        else:
            pattern_analysis = {"note": "No valid trials available for pattern analysis"}

        norm_method = self.normalization_method or getattr(self.mapper, "normalization_method", "None")
        train_count = self.training_sample_count or getattr(self.mapper, "training_sample_count", 0)

        val_status = "COMPLETED"
        fail_reason: Optional[str] = None
        if len(self._trial_sequence) > 0 and len(self._trial_records) < len(self._trial_sequence):
            val_status = "INCOMPLETE"
            fail_reason = f"Only {len(self._trial_records)} of {len(self._trial_sequence)} planned trials finished."

        summary = AccuracySummary(
            experiment_id=self.experiment_id,
            calibration_profile_id=self.calibration_profile_id,
            model_name=self.mapper.model_type,
            model_configuration=self.mapper.to_dict(),
            total_trials=len(self._trial_sequence),
            successful_trials=int(calc["successful_trials"]),
            rejected_trials=int(calc["rejected_trials"]),
            rejected_trial_pct=rejected_pct,
            mae_x=float(calc["mae_x"]),
            mae_y=float(calc["mae_y"]),
            mae_x_px=float(calc["mae_x_px"]),
            mae_y_px=float(calc["mae_y_px"]),
            rmse_x=float(calc["rmse_x"]),
            rmse_y=float(calc["rmse_y"]),
            rmse_x_px=float(calc["rmse_x_px"]),
            rmse_y_px=float(calc["rmse_y_px"]),
            mean_euclidean_error=float(calc["mean_euclidean_norm"]),
            median_euclidean_error=float(calc["median_euclidean_norm"]),
            maximum_error=float(calc["max_euclidean_norm"]),
            mean_error_pixels=mean_e_px,
            median_error_pixels=median_e_px,
            max_error_pixels=max_e_px,
            mean_error_pct_diagonal=mean_err_pct_diag,
            median_error_pct_diagonal=float(calc["median_error_pct_diagonal"]),
            max_error_pct_diagonal=float(calc["max_error_pct_diagonal"]),
            p90_error_norm=calc["p90_norm"],
            p90_error_pixels=calc["p90_error_pixels"],
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            screen_diagonal_px=diag_px,
            threshold_2pct_frac=self.threshold_2pct,
            threshold_5pct_frac=self.threshold_5pct,
            threshold_10pct_frac=self.threshold_10pct,
            threshold_2pct_px=thresh_2pct_px,
            threshold_5pct_px=thresh_5pct_px,
            threshold_10pct_px=thresh_10pct_px,
            trials_within_2pct=int(calc["trials_within_2pct"]),
            pct_within_2pct=pct_2,
            trials_within_5pct=int(calc["trials_within_5pct"]),
            pct_within_5pct=pct_5,
            trials_within_10pct=int(calc["trials_within_10pct"]),
            pct_within_10pct=pct_10,
            successful_selections=successful_selections,
            failed_selections=failed_selections,
            target_selection_accuracy_pct=target_selection_acc_pct,
            target_boundary_description=boundary_desc,
            normalization_method=norm_method,
            training_sample_count=train_count,
            training_metrics=self.training_metrics,
            screen_resolution=f"{self.screen_width}x{self.screen_height}",
            evaluation_timestamp=time.time(),
            target_accuracies=target_accs,
            direction_accuracies=direction_accs,
            region_accuracies=region_accs,
            region_accuracies_9=region_accs_9,
            pattern_analysis=pattern_analysis,
            data_leakage_prevented=True,
            validation_status=val_status,
            failure_reason=fail_reason,
        )

        self.summary = summary
        self.state = ValidationState.COMPLETED if val_status == "COMPLETED" else ValidationState.ERROR
        self.target_phase = ValidationPhase.COMPLETED
        self.fixation_status = val_status
        self.fixation_feedback = (
            f"Validation Completed! Target Hit Accuracy: {target_selection_acc_pct:.1f}% ({successful_selections}/{len(valid_raw)})"
        )
        logger.info(
            f"Validation Finalized ({val_status})! Valid Trials: {len(valid_raw)}/{len(self._trial_sequence)} | "
            f"Target Selection Accuracy: {target_selection_acc_pct:.1f}% ({successful_selections}/{len(valid_raw)}) | "
            f"Within 5% Diag: {pct_5:.1f}% | Mean Error: {mean_e_px:.1f}px ({mean_err_pct_diag:.1f}% diag)"
        )
        return summary

    def get_trial_records(self) -> List[ValidationTrialRecord]:
        """Return list of all recorded validation trials."""
        return self._trial_records
