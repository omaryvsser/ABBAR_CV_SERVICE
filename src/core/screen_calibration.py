"""
9-Point Screen Gaze Calibration Workflow for AABAR CV Service (Phase 3).

Coordinates the systematic acquisition of empirical eye features corresponding
to known 2D screen targets. Handles settling delays, sample filtering, outlier
rejection, target progression, sample aggregation, and mapping model training.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from .calibration_profile import COORDINATE_CONVENTION_VERSION, CalibrationProfile
from .eye_features import BinocularEyeFeatures
from .gaze_baseline import GazeBaseline
from .gaze_mapper import (
    BaseGazeMapper,
    compare_and_select_mapper,
    create_mapper,
    evaluate_mapper_predictions,
    extract_gaze_features,
    FEATURE_NAMES,
)
from .head_pose import HeadPose

logger = logging.getLogger(__name__)


class TargetPhase(str, Enum):
    """Execution phase for the active calibration target."""
    PENDING = "PENDING"
    SETTLING = "SETTLING"        # Waiting for user's eyes to locate and fixate on target
    COLLECTING = "COLLECTING"    # Sampling valid eye features during steady fixation
    COMPLETED = "COMPLETED"      # Sufficient valid samples gathered for target
    FAILED = "FAILED"            # Insufficient samples after timeout


class CalibrationState(str, Enum):
    """High-level state of the calibration process."""
    IDLE = "IDLE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class CalibrationTarget:
    """
    A geometric calibration/validation point on the display screen.

    Supports either explicit rectangular button dimensions (width_px, height_px),
    explicit circular radius (radius_px), or the default rendered OpenCV target
    circle (radius 22 display px mapped to target screen dimensions).
    """
    target_id: int
    name: str
    x: float  # Normalized horizontal position in [0.0, 1.0]
    y: float  # Normalized vertical position in [0.0, 1.0]
    radius_px: Optional[float] = None
    width_px: Optional[float] = None
    height_px: Optional[float] = None

    def to_pixel(self, screen_width: int, screen_height: int) -> Tuple[int, int]:
        """Convert normalized coordinates to integer pixel coordinates."""
        px = int(np.clip(round(self.x * screen_width), 0, screen_width - 1))
        py = int(np.clip(round(self.y * screen_height), 0, screen_height - 1))
        return px, py

    def contains_point(
        self,
        pred_x: float,
        pred_y: float,
        screen_width: int,
        screen_height: int,
        display_width: int = 1280,
        display_height: int = 720,
        default_display_radius_px: float = 22.0,
        fallback_radius_px: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Check whether predicted gaze (pred_x, pred_y) lies within this target's actual boundary.

        Uses explicit button dimensions if provided, otherwise checks the exact rendered
        circular target boundary rendered on the UI canvas.
        """
        err_x = pred_x - self.x
        err_y = pred_y - self.y

        if self.width_px is not None and self.height_px is not None:
            px_err_x = err_x * float(screen_width)
            px_err_y = err_y * float(screen_height)
            inside = (
                abs(px_err_x) <= (self.width_px / 2.0) + 1e-9
                and abs(px_err_y) <= (self.height_px / 2.0) + 1e-9
            )
            spec = f"Button Box ({self.width_px:.0f}x{self.height_px:.0f} px)"
            return inside, spec

        if self.radius_px is not None:
            dist_px = math.hypot(err_x * float(screen_width), err_y * float(screen_height))
            spec = f"Target Circle (radius {self.radius_px:.1f} px)"
            return dist_px <= self.radius_px + 1e-9, spec

        if fallback_radius_px is not None:
            dist_px = math.hypot(err_x * float(screen_width), err_y * float(screen_height))
            spec = f"Configured Target Circle (radius {fallback_radius_px:.1f} px)"
            return dist_px <= fallback_radius_px + 1e-9, spec

        # Exact rendered OpenCV target ring boundary on display window (default radius 22 px)
        dx_disp = err_x * float(max(1, display_width))
        dy_disp = err_y * float(max(1, display_height))
        screen_equiv = default_display_radius_px * (float(screen_width) / float(max(1, display_width)))
        spec = (
            f"Rendered Target Circle (radius {default_display_radius_px:.0f} display px / "
            f"{screen_equiv:.1f} px @ {screen_width}x{screen_height})"
        )
        return math.hypot(dx_disp, dy_disp) <= default_display_radius_px + 1e-9, spec


# Standard 9-Point Calibration Grid (3x3 layout with safe screen margins)
DEFAULT_9_TARGETS: List[CalibrationTarget] = [
    CalibrationTarget(0, "top_left",      0.10, 0.10),
    CalibrationTarget(1, "top_center",    0.50, 0.10),
    CalibrationTarget(2, "top_right",     0.90, 0.10),
    CalibrationTarget(3, "middle_left",   0.10, 0.50),
    CalibrationTarget(4, "center",        0.50, 0.50),
    CalibrationTarget(5, "middle_right",  0.90, 0.50),
    CalibrationTarget(6, "bottom_left",   0.10, 0.90),
    CalibrationTarget(7, "bottom_center", 0.50, 0.90),
    CalibrationTarget(8, "bottom_right",  0.90, 0.90),
]


@dataclass
class CalibrationSampleRecord:
    """Detailed observation record captured during calibration."""
    experiment_id: str
    session_id: str
    target_id: int
    target_x: float
    target_y: float
    timestamp: float
    features: Optional[np.ndarray]
    raw_x: Optional[float]
    raw_y: Optional[float]
    r_iris_x: Optional[float]
    r_iris_y: Optional[float]
    l_iris_x: Optional[float]
    l_iris_y: Optional[float]
    r_openness: Optional[float]
    l_openness: Optional[float]
    yaw: Optional[float]
    pitch: Optional[float]
    roll: Optional[float]
    tracking_quality: str
    confidence: float
    accepted: bool
    rejection_reason: Optional[str] = None

    def to_csv_dict(self) -> Dict[str, Any]:
        """Flatten record into CSV-compatible column dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "session_id": self.session_id,
            "target_id": self.target_id,
            "target_x": round(self.target_x, 4),
            "target_y": round(self.target_y, 4),
            "timestamp": round(self.timestamp, 4),
            "raw_x": round(self.raw_x, 4) if self.raw_x is not None else "",
            "raw_y": round(self.raw_y, 4) if self.raw_y is not None else "",
            "r_iris_x": round(self.r_iris_x, 4) if self.r_iris_x is not None else "",
            "r_iris_y": round(self.r_iris_y, 4) if self.r_iris_y is not None else "",
            "l_iris_x": round(self.l_iris_x, 4) if self.l_iris_x is not None else "",
            "l_iris_y": round(self.l_iris_y, 4) if self.l_iris_y is not None else "",
            "r_openness": round(self.r_openness, 4) if self.r_openness is not None else "",
            "l_openness": round(self.l_openness, 4) if self.l_openness is not None else "",
            "yaw": round(self.yaw, 2) if self.yaw is not None else "",
            "pitch": round(self.pitch, 2) if self.pitch is not None else "",
            "roll": round(self.roll, 2) if self.roll is not None else "",
            "tracking_quality": self.tracking_quality,
            "confidence": round(self.confidence, 4),
            "sample_accepted": int(self.accepted),
            "rejection_reason": self.rejection_reason or "",
        }


class ScreenCalibrator:
    """
    Coordinates interactive 9-point screen calibration.

    Sequence:
    For each target in targets:
      1. SETTLING (e.g. 0.8s): User moves eyes to target. Samples ignored.
      2. COLLECTING (e.g. 1.2s): Valid eye feature samples accumulated.
      3. COMPLETED: If >= min_samples collected, advance to next target.
    Finally:
      Aggregate samples per target, fit mapping model, generate profile.
    """

    def __init__(
        self,
        targets: Optional[List[CalibrationTarget]] = None,
        settling_duration: float = 0.8,
        collection_duration: float = 1.2,
        min_samples_per_target: int = 15,
        max_collection_timeout: float = 5.0,
        screen_width: int = 1920,
        screen_height: int = 1080,
        camera_index: int = 0,
        user_id: str = "user_default",
        experiment_id: str = "exp_001",
        session_id: str = "sess_001",
        model_type: str = "auto",
        ridge_alpha: float = 0.01,
        normalize_features: bool = True,
        fixation_max_std: float = 0.022,
        max_interframe_step: float = 0.045,
        max_head_motion_step: float = 4.0,
        allow_target_retry: bool = True,
    ) -> None:
        self.targets = list(targets or DEFAULT_9_TARGETS)
        self.settling_duration = max(0.01, settling_duration)
        self.collection_duration = max(0.01, collection_duration)
        self.min_samples_per_target = max(1, min_samples_per_target)
        self.max_collection_timeout = max(self.collection_duration + 0.5, max_collection_timeout)

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.camera_index = camera_index
        self.user_id = user_id
        self.experiment_id = experiment_id
        self.session_id = session_id
        self.model_type = model_type
        self.ridge_alpha = ridge_alpha
        self.normalize_features: bool = bool(normalize_features)
        self.fixation_max_std = max(0.002, float(fixation_max_std))
        self.max_interframe_step = max(0.005, float(max_interframe_step))
        self.max_head_motion_step = max(0.5, float(max_head_motion_step))
        self.allow_target_retry = bool(allow_target_retry)

        # Runtime state
        self.state: CalibrationState = CalibrationState.IDLE
        self.current_target_index: int = 0
        self.target_phase: TargetPhase = TargetPhase.PENDING
        self.phase_start_time: float = 0.0
        self.fixation_status: str = "IDLE"
        self.fixation_feedback: str = ""
        self.retried_targets: Dict[int, int] = {}

        # Fixation stability buffers
        self._recent_raw_xy: List[Tuple[float, float]] = []
        self._prev_pose_yp: Optional[Tuple[float, float]] = None

        # Data store
        self.all_samples: List[CalibrationSampleRecord] = []
        self._target_samples: Dict[int, List[CalibrationSampleRecord]] = {
            t.target_id: [] for t in self.targets
        }
        self._target_all_samples: Dict[int, List[CalibrationSampleRecord]] = {
            t.target_id: [] for t in self.targets
        }

        # Results
        self.trained_mapper: Optional[BaseGazeMapper] = None
        self.profile: Optional[CalibrationProfile] = None
        self.comparison_report: Dict[str, Any] = {}
        self.calibration_metadata: Dict[str, Any] = {}
        self.error_message: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """True if calibration is actively in progress."""
        return self.state == CalibrationState.IN_PROGRESS

    @property
    def is_completed(self) -> bool:
        """True if calibration finished successfully."""
        return self.state == CalibrationState.COMPLETED

    @property
    def current_target(self) -> Optional[CalibrationTarget]:
        """Return the active calibration target point."""
        if not self.is_active or self.current_target_index >= len(self.targets):
            return None
        return self.targets[self.current_target_index]

    @property
    def progress(self) -> float:
        """Overall calibration progress in [0.0, 1.0]."""
        if self.state == CalibrationState.COMPLETED:
            return 1.0
        if not self.is_active or len(self.targets) == 0:
            return 0.0
        base_progress = self.current_target_index / float(len(self.targets))
        target_prog = self.target_progress / float(len(self.targets))
        return float(np.clip(base_progress + target_prog, 0.0, 1.0))

    @property
    def target_progress(self) -> float:
        """Progress on current target in [0.0, 1.0]."""
        if not self.is_active:
            return 0.0
        now = time.time()
        elapsed = now - self.phase_start_time
        if self.target_phase == TargetPhase.SETTLING:
            return min(0.4, (elapsed / self.settling_duration) * 0.4)
        elif self.target_phase == TargetPhase.COLLECTING:
            collect_prog = min(0.6, (elapsed / self.collection_duration) * 0.6)
            return 0.4 + collect_prog
        return 1.0

    def start(self) -> None:
        """Start or restart the 9-point calibration process."""
        logger.info(f"Starting 9-Point Screen Calibration with {len(self.targets)} targets...")
        self.state = CalibrationState.IN_PROGRESS
        self.current_target_index = 0
        self.all_samples.clear()
        self._target_samples = {t.target_id: [] for t in self.targets}
        self._target_all_samples = {t.target_id: [] for t in self.targets}
        self.retried_targets.clear()
        self.trained_mapper = None
        self.profile = None
        self.calibration_metadata = {}
        self.error_message = None
        self._enter_target(0)

    def cancel(self) -> None:
        """Cancel the active calibration session."""
        logger.info("Screen calibration cancelled by user.")
        self.state = CalibrationState.CANCELLED
        self.target_phase = TargetPhase.PENDING
        self.fixation_status = "CANCELLED"
        self.fixation_feedback = "Calibration cancelled."

    def _enter_target(self, target_idx: int) -> None:
        """Transition to a new calibration target."""
        self.current_target_index = target_idx
        self.target_phase = TargetPhase.SETTLING
        self.phase_start_time = time.time()
        self._recent_raw_xy.clear()
        self._prev_pose_yp = None
        self.error_message = None
        target = self.targets[target_idx]
        self.fixation_status = "SETTLING"
        self.fixation_feedback = f"Look at Target {target_idx + 1}/{len(self.targets)} ({target.name})"
        logger.info(
            f"Calibration Target [{target_idx + 1}/{len(self.targets)}] "
            f"'{target.name}' at ({target.x:.2f}, {target.y:.2f}) - SETTLING ({self.settling_duration:.1f}s)..."
        )

    def retry_current_target(self) -> bool:
        """Clear samples for the active/failed target and re-collect fresh fixations."""
        if not self.is_active or self.current_target_index >= len(self.targets):
            return False
        target = self.targets[self.current_target_index]
        tid = target.target_id
        self._target_samples[tid] = []
        self._target_all_samples[tid] = []
        self.all_samples = [s for s in self.all_samples if s.target_id != tid]
        self.retried_targets[tid] = self.retried_targets.get(tid, 0) + 1
        logger.info(f"Retrying Calibration Target [{self.current_target_index + 1}/{len(self.targets)}] '{target.name}'...")
        self._enter_target(self.current_target_index)
        return True

    def update(
        self,
        eye_features: BinocularEyeFeatures,
        head_pose: Optional[HeadPose] = None,
        baseline: Optional[GazeBaseline] = None,
    ) -> Optional[BaseGazeMapper]:
        """
        Feed current video frame features into the calibration state machine.

        Returns:
            Fitted BaseGazeMapper if calibration just completed on this frame, else None.
        """
        if self.state == CalibrationState.COMPLETED:
            return self.trained_mapper
        if not self.is_active or self.target_phase == TargetPhase.FAILED:
            return None

        target = self.current_target
        if target is None:
            return None

        now = time.time()
        elapsed = now - self.phase_start_time

        # 1. Check phase transitions
        if self.target_phase == TargetPhase.SETTLING:
            if elapsed >= self.settling_duration:
                self.target_phase = TargetPhase.COLLECTING
                self.phase_start_time = now
                self._recent_raw_xy.clear()
                self.fixation_status = "COLLECTING"
                self.fixation_feedback = f"Hold gaze steady on Target {self.current_target_index + 1}/{len(self.targets)}"
                logger.info(f"Target '{target.name}' - COLLECTING samples ({self.collection_duration:.1f}s)...")
            return None

        # 2. In COLLECTING phase: evaluate sample quality & fixation stability
        features = extract_gaze_features(eye_features, head_pose, baseline)

        accepted = False
        rejection_reason: Optional[str] = None
        tracking_quality = "POOR"

        head_motion_bad = False
        if head_pose and head_pose.is_valid:
            cur_yp = (float(head_pose.yaw), float(head_pose.pitch))
            if self._prev_pose_yp is not None:
                dy = abs(cur_yp[0] - self._prev_pose_yp[0])
                dp = abs(cur_yp[1] - self._prev_pose_yp[1])
                if dy > self.max_head_motion_step or dp > self.max_head_motion_step:
                    head_motion_bad = True
            self._prev_pose_yp = cur_yp

        if (eye_features.right_eye and eye_features.right_eye.is_blink) or (
            eye_features.left_eye and eye_features.left_eye.is_blink
        ):
            rejection_reason = "BLINK_DETECTED"
            self.fixation_status = "NO_EYE_TRACKING"
            self.fixation_feedback = "Blink detected — keep eyes open on target"
            self._recent_raw_xy.clear()
        elif not eye_features.is_valid:
            rejection_reason = "INVALID_EYE_LANDMARKS"
            self.fixation_status = "NO_EYE_TRACKING"
            self.fixation_feedback = "Eyes not detected clearly — face camera"
            self._recent_raw_xy.clear()
        elif eye_features.confidence < 0.40:
            rejection_reason = f"LOW_CONFIDENCE_{eye_features.confidence:.2f}"
            self.fixation_status = "LOW_CONFIDENCE"
            self.fixation_feedback = "Low eye tracking confidence — look straight at target"
            self._recent_raw_xy.clear()
        elif (eye_features.right_eye and eye_features.right_eye.openness < 0.25) or (
            eye_features.left_eye and eye_features.left_eye.openness < 0.25
        ):
            rejection_reason = "PARTIAL_EYE_CLOSURE"
            self.fixation_status = "PARTIAL_CLOSURE"
            self.fixation_feedback = "Open eyes wider to improve fixation quality"
            self._recent_raw_xy.clear()
        elif head_pose and head_pose.is_valid and (
            abs(head_pose.yaw) > 25.0 or abs(head_pose.pitch) > 20.0
        ):
            rejection_reason = f"EXCESSIVE_HEAD_ROTATION_Y{head_pose.yaw:.1f}_P{head_pose.pitch:.1f}"
            self.fixation_status = "HEAD_MOVEMENT_HIGH"
            self.fixation_feedback = "Turn head back toward center — keep head still"
            self._recent_raw_xy.clear()
        elif head_motion_bad:
            rejection_reason = "HEAD_MOVEMENT_UNSTABLE"
            self.fixation_status = "HEAD_MOVEMENT_HIGH"
            self.fixation_feedback = "Keep head still during target fixation"
            self._recent_raw_xy.clear()
        elif features is None or eye_features.combined_horizontal_ratio is None or eye_features.combined_vertical_ratio is None:
            rejection_reason = "FEATURE_EXTRACTION_FAILED"
            self.fixation_status = "NO_EYE_TRACKING"
            self._recent_raw_xy.clear()
        else:
            rx = float(eye_features.combined_horizontal_ratio)
            ry = float(eye_features.combined_vertical_ratio)
            if self._recent_raw_xy:
                prev_x, prev_y = self._recent_raw_xy[-1]
                if math.hypot(rx - prev_x, ry - prev_y) > self.max_interframe_step:
                    rejection_reason = "FIXATION_SACCADE_JUMP"
                    self.fixation_status = "UNSTABLE_FIXATION"
                    self.fixation_feedback = "Hold eyes steady on target dot"
                    self._recent_raw_xy = [(rx, ry)]
            if rejection_reason is None:
                self._recent_raw_xy.append((rx, ry))
                if len(self._recent_raw_xy) > 6:
                    self._recent_raw_xy.pop(0)
                if len(self._recent_raw_xy) >= 4:
                    arr_w = np.array(self._recent_raw_xy, dtype=np.float64)
                    if float(np.std(arr_w[:, 0])) > self.fixation_max_std or float(np.std(arr_w[:, 1])) > self.fixation_max_std:
                        rejection_reason = "FIXATION_UNSTABLE_DISPERSION"
                        self.fixation_status = "UNSTABLE_FIXATION"
                        self.fixation_feedback = "Hold gaze steady on target — reduce eye wander"
                if rejection_reason is None:
                    accepted = True
                    tracking_quality = "EXCELLENT" if eye_features.confidence >= 0.70 else "GOOD"
                    self.fixation_status = "FIXATION_STEADY"
                    self.fixation_feedback = (
                        f"Fixation steady ({len(self._target_samples[target.target_id]) + 1}/{self.min_samples_per_target})"
                    )

        # Record observation
        record = CalibrationSampleRecord(
            experiment_id=self.experiment_id,
            session_id=self.session_id,
            target_id=target.target_id,
            target_x=target.x,
            target_y=target.y,
            timestamp=now,
            features=features,
            raw_x=eye_features.combined_horizontal_ratio,
            raw_y=eye_features.combined_vertical_ratio,
            r_iris_x=eye_features.right_eye.horizontal_ratio if eye_features.right_eye else None,
            r_iris_y=eye_features.right_eye.vertical_ratio if eye_features.right_eye else None,
            l_iris_x=eye_features.left_eye.horizontal_ratio if eye_features.left_eye else None,
            l_iris_y=eye_features.left_eye.vertical_ratio if eye_features.left_eye else None,
            r_openness=eye_features.right_eye.openness if eye_features.right_eye else None,
            l_openness=eye_features.left_eye.openness if eye_features.left_eye else None,
            yaw=head_pose.yaw if (head_pose and head_pose.is_valid) else None,
            pitch=head_pose.pitch if (head_pose and head_pose.is_valid) else None,
            roll=head_pose.roll if (head_pose and head_pose.is_valid) else None,
            tracking_quality=tracking_quality,
            confidence=eye_features.confidence,
            accepted=accepted,
            rejection_reason=rejection_reason,
        )

        self.all_samples.append(record)
        self._target_all_samples[target.target_id].append(record)
        if accepted:
            self._target_samples[target.target_id].append(record)

        # 3. Check if target collection time and minimum samples reached
        valid_count = len(self._target_samples[target.target_id])
        elapsed_collect = time.time() - self.phase_start_time

        if elapsed_collect >= self.collection_duration:
            if valid_count >= self.min_samples_per_target:
                logger.info(
                    f"Target [{self.current_target_index + 1}/{len(self.targets)}] '{target.name}' "
                    f"completed with {valid_count} accepted fixation samples."
                )
                next_idx = self.current_target_index + 1
                if next_idx < len(self.targets):
                    self._enter_target(next_idx)
                else:
                    return self._finalize_calibration()
            elif elapsed_collect >= self.max_collection_timeout:
                target_records = self._target_all_samples.get(target.target_id, [])
                rej_counts: Dict[str, int] = {}
                for r in target_records:
                    if not r.accepted and r.rejection_reason:
                        rej_counts[r.rejection_reason] = rej_counts.get(r.rejection_reason, 0) + 1
                top_cause = max(rej_counts, key=rej_counts.get) if rej_counts else "LOW_FRAME_RATE"
                msg = (
                    f"Target {self.current_target_index + 1}/{len(self.targets)} '{target.name}' failed: "
                    f"only {valid_count}/{self.min_samples_per_target} stable samples collected "
                    f"(cause: {top_cause})."
                )
                logger.warning(msg)
                self.error_message = msg
                if self.allow_target_retry:
                    self.target_phase = TargetPhase.FAILED
                    self.fixation_status = "TARGET_FAILED"
                    self.fixation_feedback = f"{msg} Press 'y' to retry."
                else:
                    self.state = CalibrationState.ERROR
                    self.fixation_status = "ERROR"
                return None

        return None

    def _finalize_calibration(self) -> Optional[BaseGazeMapper]:
        """Aggregate samples across all targets, record rich metadata, and train mapping models."""
        logger.info("Aggregating calibration samples across all targets...")
        X_rows: List[np.ndarray] = []
        y_rows: List[Tuple[float, float]] = []
        per_target_stats: List[Dict[str, Any]] = []

        for target in self.targets:
            accepted_records = self._target_samples.get(target.target_id, [])
            all_target_records = self._target_all_samples.get(target.target_id, accepted_records)
            if len(accepted_records) < self.min_samples_per_target:
                self.state = CalibrationState.ERROR
                self.error_message = (
                    f"Incomplete calibration: Target {target.name} has only "
                    f"{len(accepted_records)} accepted samples."
                )
                logger.error(self.error_message)
                return None

            feat_matrix = np.array([r.features for r in accepted_records if r.features is not None])

            # Robust aggregation: Trimmed Median (reject outer 10% extreme frames across features)
            if len(feat_matrix) >= 6:
                p10 = np.percentile(feat_matrix, 10, axis=0)
                p90 = np.percentile(feat_matrix, 90, axis=0)
                inlier_mask = np.all((feat_matrix >= p10) & (feat_matrix <= p90), axis=1)
                clean_matrix = feat_matrix[inlier_mask] if np.sum(inlier_mask) >= 5 else feat_matrix
            else:
                clean_matrix = feat_matrix

            aggregated_feature = np.median(clean_matrix, axis=0)
            X_rows.append(aggregated_feature)
            y_rows.append((target.x, target.y))

            rej_counts: Dict[str, int] = {}
            for rec in all_target_records:
                if not rec.accepted and rec.rejection_reason:
                    rej_counts[rec.rejection_reason] = rej_counts.get(rec.rejection_reason, 0) + 1

            raw_xs = [r.raw_x for r in accepted_records if r.raw_x is not None]
            raw_ys = [r.raw_y for r in accepted_records if r.raw_y is not None]
            yaws = [r.yaw for r in accepted_records if r.yaw is not None]
            pitches = [r.pitch for r in accepted_records if r.pitch is not None]
            per_target_stats.append({
                "target_id": target.target_id,
                "name": target.name,
                "x": target.x,
                "y": target.y,
                "collected_samples": len(all_target_records),
                "accepted_samples": len(accepted_records),
                "rejected_samples": max(0, len(all_target_records) - len(accepted_records)),
                "retried_count": self.retried_targets.get(target.target_id, 0),
                "rejection_breakdown": rej_counts,
                "feature_stats": {
                    "raw_x_median": round(float(np.median(raw_xs)), 4) if raw_xs else None,
                    "raw_x_std": round(float(np.std(raw_xs)), 4) if raw_xs else None,
                    "raw_y_median": round(float(np.median(raw_ys)), 4) if raw_ys else None,
                    "raw_y_std": round(float(np.std(raw_ys)), 4) if raw_ys else None,
                    "yaw_median": round(float(np.median(yaws)), 2) if yaws else None,
                    "pitch_median": round(float(np.median(pitches)), 2) if pitches else None,
                },
            })

        X_train = np.array(X_rows, dtype=np.float64)
        y_train = np.array(y_rows, dtype=np.float64)

        m_type_req = self.model_type.strip().lower()
        if m_type_req in ("linear", "polynomial", "compact_linear", "compact_polynomial"):
            mapper = create_mapper(
                m_type_req, ridge_alpha=self.ridge_alpha, use_scaler=self.normalize_features
            ).fit(X_train, y_train)
            train_metrics = evaluate_mapper_predictions(
                y_train, mapper.predict(X_train), self.screen_width, self.screen_height
            )
            comp_report = {
                "selected_model": mapper.model_type,
                "selected_ridge_alpha": mapper.ridge_alpha,
                "normalization_method": mapper.normalization_method,
                "training_metrics": train_metrics,
            }
        else:
            mapper, comp_report = compare_and_select_mapper(
                X_train=X_train,
                y_train=y_train,
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                ridge_alpha=self.ridge_alpha,
                use_scaler=self.normalize_features,
            )
            train_metrics = comp_report.get("training_metrics", {})

        self.trained_mapper = mapper
        self.comparison_report = comp_report

        accepted_count = len(self.get_accepted_samples())
        total_collected = len(self.all_samples)
        rejected_count = max(0, total_collected - accepted_count)

        calib_ts = time.time()
        compact_comp = {
            k: v for k, v in comp_report.items() if k != "candidate_benchmarks"
        }
        self.calibration_metadata = {
            "calibration_timestamp": calib_ts,
            "total_targets": len(self.targets),
            "collected_sample_count": total_collected,
            "accepted_sample_count": accepted_count,
            "rejected_sample_count": rejected_count,
            "model_selection_summary": compact_comp,
            "per_target_stats": per_target_stats,
        }

        profile_id = f"{self.user_id}_{int(calib_ts)}"
        self.profile = CalibrationProfile(
            profile_id=profile_id,
            user_id=self.user_id,
            created_at=calib_ts,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            camera_index=self.camera_index,
            model_type=mapper.model_type,
            model_data=mapper.to_dict(),
            feature_names=list(FEATURE_NAMES),
            normalization_method=mapper.normalization_method,
            normalization_params=mapper.scaler.to_dict() if mapper.scaler is not None else None,
            training_sample_count=int(len(X_train)),
            calibration_frame_count=accepted_count,
            coordinate_convention=COORDINATE_CONVENTION_VERSION,
            training_metrics=train_metrics,
            validation_metrics=None,
            calibration_metadata=self.calibration_metadata,
            model_version="3.1",
        )

        self.state = CalibrationState.COMPLETED
        self.target_phase = TargetPhase.COMPLETED
        self.fixation_status = "COMPLETED"
        self.fixation_feedback = "9-point calibration completed."
        logger.info(
            f"Screen Calibration Successfully Finalized! Model: [{mapper.model_type.upper()}] | "
            f"Scaler: [{mapper.normalization_method}] | Profile: [{profile_id}]"
        )
        return self.trained_mapper

    def get_all_samples(self) -> List[CalibrationSampleRecord]:
        """Return all recorded frame samples."""
        return self.all_samples

    def get_accepted_samples(self) -> List[CalibrationSampleRecord]:
        """Return only accepted valid frame samples."""
        return [s for s in self.all_samples if s.accepted]
