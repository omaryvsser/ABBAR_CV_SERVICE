"""
Computer Vision Processing Pipeline for AABAR (Phase 2 Enhanced).

Orchestrates:
1. Frame acquisition from CameraManager (or synthetic mock camera).
2. Facial and iris landmark detection via MediaPipe FaceLandmarkerDetector.
3. Eye and iris feature extraction with normalized openness and ocular reliability.
4. Saccade-adaptive Exponential Moving Average (EMA) gaze smoothing.
5. Relative gaze direction classification (CENTER, LEFT, RIGHT, UP, DOWN).
6. 3D Head Pose estimation (Yaw, Pitch, Roll) via solvePnP.
7. Structured, typed output schema serialization (GazeFrameResult).
8. Real-time performance monitoring via FPSCounter.
9. Rich visual overlays via CVVisualizer.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import Optional, Tuple
import cv2
import numpy as np

# Set MPLCONFIGDIR to writable directory in temp to suppress font cache warnings
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_aabar"

from .camera.camera_manager import CameraManager
from .config import AppConfig, load_config
from .core.calibration_profile import (
    CalibrationProfile,
    IncompatibleCalibrationProfileError,
    ProfileManager,
)
from .core.data_logger import DiagnosticDataLogger
from .core.experiment_logger import ExperimentLogger
from .core.eye_features import BinocularEyeFeatures, compensate_eye_features, extract_eye_features
from .core.face_landmarker import FaceLandmarkerDetectionResult, FaceLandmarkerDetector
from .core.gaze_baseline import GazeBaseline, GazeBaselineCalibrator
from .core.gaze_direction import GazeDirection, GazeDirectionClassifier
from .core.gaze_mapper import BaseGazeMapper, extract_gaze_features
from .core.gaze_output import (
    EyesOutput,
    GazeFrameResult,
    GazeOutput,
    HeadPoseOutput,
    ScreenGazeOutput,
    SingleEyeOutput,
)
from .core.gaze_smoother import GazeSmoother, ScreenGazeStabilizer
from .core.head_pose import HeadPose, HeadPoseEstimator
from .core.screen_calibration import CalibrationState, CalibrationTarget, ScreenCalibrator, TargetPhase
from .core.screen_validator import (
    AccuracySummary,
    DEFAULT_VALIDATION_TARGETS,
    FULL_9_REGION_VALIDATION_TARGETS,
    ScreenValidator,
    ValidationPhase,
)
from .core.tracking_state import TrackingState
from .utils.fps_counter import FPSCounter, FPSMetrics
from .visualization.visualizer import CVVisualizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineOutput:
    """
    Standard output produced for every processed video frame.

    Attributes:
        raw_frame: Original BGR frame from webcam.
        detection_result: MediaPipe detection output.
        eye_features: Normalized geometric eye and iris ratios.
        fps_metrics: Current frame rate, processing time, and detection latency.
        gaze_direction: Relative gaze direction classification.
        smoothed_x: Smoothed horizontal gaze ratio.
        smoothed_y: Smoothed vertical gaze ratio.
        head_pose: Estimated 3D head pose (or None if disabled/untracked).
        gaze_frame_result: Structured JSON-serializable output schema.
        annotated_frame: Frame with visual overlays (None if visualization disabled).
        tracking_state: Current tracking state enum.
        baseline: Active neutral gaze baseline statistics.
    """
    raw_frame: np.ndarray
    detection_result: FaceLandmarkerDetectionResult
    eye_features: BinocularEyeFeatures
    fps_metrics: FPSMetrics
    gaze_direction: GazeDirection
    smoothed_x: Optional[float]
    smoothed_y: Optional[float]
    head_pose: Optional[HeadPose]
    gaze_frame_result: GazeFrameResult
    annotated_frame: Optional[np.ndarray]
    tracking_state: TrackingState = TrackingState.TRACKING
    baseline: Optional[GazeBaseline] = None
    screen_gaze: Optional[ScreenGazeOutput] = None
    screen_calibrator: Optional[ScreenCalibrator] = None
    screen_validator: Optional[ScreenValidator] = None


class CVPipeline:
    """
    Central pipeline coordinating camera capture, detection, feature extraction,
    smoothing, head pose estimation, baseline calibration, and rendering.
    """

    def __init__(self, config: Optional[AppConfig] = None, use_mock_camera: bool = False) -> None:
        """Initialize pipeline components using provided configuration."""
        self.config = config or load_config()
        self.use_mock_camera = use_mock_camera

        self.camera = (
            CameraManager(
                camera_index=self.config.camera.camera_index,
                frame_width=self.config.camera.frame_width,
                frame_height=self.config.camera.frame_height,
                target_fps=self.config.camera.target_fps,
                buffer_size=self.config.camera.buffer_size,
                use_threading=self.config.camera.use_threading,
            )
            if not use_mock_camera
            else None
        )

        self.detector = FaceLandmarkerDetector(
            model_path=self.config.landmarker.model_path,
            num_faces=self.config.landmarker.num_faces,
            min_face_detection_confidence=self.config.landmarker.min_face_detection_confidence,
            min_face_presence_confidence=self.config.landmarker.min_face_presence_confidence,
            min_tracking_confidence=self.config.landmarker.min_tracking_confidence,
            running_mode=self.config.landmarker.running_mode,
        )

        self.head_pose_estimator = (
            HeadPoseEstimator() if self.config.gaze.enable_head_pose else None
        )

        self.calibrator = GazeBaselineCalibrator(
            calibration_duration_seconds=self.config.gaze.calibration_duration_seconds,
            target_fps=self.config.camera.target_fps,
            max_head_yaw_during_calib=self.config.gaze.max_head_yaw_limit * 0.6,
            max_head_pitch_during_calib=self.config.gaze.max_head_pitch_limit * 0.6,
        )

        self.data_logger = DiagnosticDataLogger()

        self.gaze_classifier = GazeDirectionClassifier(
            horizontal_center_min=self.config.gaze.horizontal_center_min,
            horizontal_center_max=self.config.gaze.horizontal_center_max,
            vertical_center_min=self.config.gaze.vertical_center_min,
            vertical_center_max=self.config.gaze.vertical_center_max,
            h_activation_threshold=self.config.gaze.h_activation_threshold,
            h_deactivation_threshold=self.config.gaze.h_deactivation_threshold,
            v_activation_threshold=self.config.gaze.v_activation_threshold,
            v_deactivation_threshold=self.config.gaze.v_deactivation_threshold,
            v_up_activation_threshold=self.config.gaze.v_up_activation_threshold,
            v_up_deactivation_threshold=self.config.gaze.v_up_deactivation_threshold,
            v_down_activation_threshold=self.config.gaze.v_down_activation_threshold,
            v_down_deactivation_threshold=self.config.gaze.v_down_deactivation_threshold,
            persistence_frames=self.config.gaze.direction_persistence_frames,
            allow_compound_directions=self.config.gaze.allow_compound_directions,
        )

        self.gaze_smoother = GazeSmoother(
            alpha=self.config.gaze.smoothing_alpha,
            saccade_threshold=self.config.gaze.saccade_threshold,
            saccade_alpha=self.config.gaze.saccade_alpha,
            saccade_threshold_y=self.config.gaze.saccade_threshold_y,
            max_invalid_frames=self.config.gaze.max_invalid_frames,
        )
        self.screen_gaze_stabilizer: Optional[ScreenGazeStabilizer] = (
            ScreenGazeStabilizer(
                min_cutoff=self.config.screen.screen_filter_min_cutoff,
                beta=self.config.screen.screen_filter_beta,
                max_invalid_frames=self.config.gaze.max_invalid_frames,
            )
            if self.config.screen.stabilize_screen_gaze
            else None
        )

        self.fps_counter = FPSCounter(window_size=30)
        self.visualizer = (
            CVVisualizer(config=self.config.visualization)
            if self.config.visualization.enabled
            else None
        )

        self._frame_interval = (
            1.0 / self.config.camera.target_fps if self.config.camera.target_fps > 0 else 0.0
        )
        self._is_running = False
        self._mock_frame_count = 0
        self._frame_index = 0

        # Phase 3 Screen Calibration, Mapping, and Validation components
        self.profile_manager = ProfileManager(profiles_dir=Path(self.config.screen.profiles_dir))
        self.experiment_logger = ExperimentLogger(base_dir=Path(self.config.screen.experiments_dir))
        self.screen_calibrator = ScreenCalibrator(
            settling_duration=self.config.screen.settling_duration,
            collection_duration=self.config.screen.collection_duration,
            min_samples_per_target=self.config.screen.min_samples_per_target,
            max_collection_timeout=self.config.screen.calib_max_collection_timeout,
            screen_width=self.config.screen.screen_width,
            screen_height=self.config.screen.screen_height,
            camera_index=self.config.camera.camera_index,
            model_type=self.config.screen.model_type,
            ridge_alpha=self.config.screen.ridge_alpha,
            normalize_features=self.config.screen.normalize_features,
            fixation_max_std=self.config.screen.calib_fixation_max_std,
            allow_target_retry=self.config.visualization.enabled,
        )
        base_val_targets = (
            FULL_9_REGION_VALIDATION_TARGETS
            if self.config.screen.validation_grid_mode == "full_9"
            else DEFAULT_VALIDATION_TARGETS
        )
        btn_w = self.config.screen.validation_target_width_px
        btn_h = self.config.screen.validation_target_height_px
        if btn_w is not None and btn_h is not None:
            val_targets = [
                CalibrationTarget(
                    target_id=t.target_id,
                    name=t.name,
                    x=t.x,
                    y=t.y,
                    radius_px=t.radius_px,
                    width_px=float(btn_w),
                    height_px=float(btn_h),
                )
                for t in base_val_targets
            ]
        else:
            val_targets = base_val_targets
        self.screen_validator = ScreenValidator(
            targets=val_targets,
            settling_duration=self.config.screen.settling_duration,
            collection_duration=self.config.screen.collection_duration,
            max_collection_timeout=self.config.screen.validation_max_timeout,
            screen_width=self.config.screen.screen_width,
            screen_height=self.config.screen.screen_height,
            num_repeats=self.config.screen.validation_repeats,
            randomize_order=self.config.screen.randomize_validation,
            threshold_2pct=self.config.screen.threshold_2pct,
            threshold_5pct=self.config.screen.threshold_5pct,
            threshold_10pct=self.config.screen.threshold_10pct,
            display_width=self.config.camera.frame_width,
            display_height=self.config.camera.frame_height,
            target_radius_px=self.config.screen.validation_target_radius_px,
            allow_target_retry=self.config.visualization.enabled,
        )
        self.active_mapper: Optional[BaseGazeMapper] = None
        self.active_profile: Optional[CalibrationProfile] = None
        self._active_profile_file: Optional[Path] = None
        self._queue_validation_after_calibration: bool = False
        self._validation_linked_to_active_calib: bool = False
        self.last_experiment_dir: Optional[Path] = None
        self.last_tracker_path: Optional[Path] = None

        # Preload saved calibration profile if configured
        if self.config.screen.active_profile_path:
            p_path = Path(self.config.screen.active_profile_path)
            if p_path.exists():
                try:
                    self.load_calibration_profile(p_path)
                except Exception as e:
                    logger.warning(f"Failed to load active profile from {p_path}: {e}")

        if self.config.gaze.auto_calibrate_on_start:
            self.calibrator.start_calibration()

    def recalibrate(self) -> None:
        """Trigger interactive recalibration of neutral resting baseline."""
        logger.info("Initiating baseline recalibration...")
        self.calibrator.start_calibration()
        self.gaze_classifier.reset()
        self.gaze_smoother.reset()
        if self.screen_gaze_stabilizer is not None:
            self.screen_gaze_stabilizer.reset()

    def recalibrate_vertical(self) -> None:
        """Trigger interactive recalibration of vertical neutral resting baseline."""
        logger.info("Initiating vertical-only baseline recalibration...")
        self.calibrator.recalibrate_vertical()
        self.gaze_classifier.reset()
        self.gaze_smoother.reset()
        if self.screen_gaze_stabilizer is not None:
            self.screen_gaze_stabilizer.reset()

    def has_unvalidated_calibration_session(self) -> bool:
        """True if calibration completed in this session without finished validation metrics yet."""
        return (
            self.screen_calibrator.is_completed
            and self.active_mapper is not None
            and self.screen_validator.summary is None
        )

    def retry_failed_target(self) -> bool:
        """Retry an active failed calibration or validation target."""
        if self.screen_validator.is_active and self.screen_validator.target_phase == ValidationPhase.FAILED:
            return self.screen_validator.retry_current_target()
        if self.screen_calibrator.is_active and self.screen_calibrator.target_phase == TargetPhase.FAILED:
            return self.screen_calibrator.retry_current_target()
        return False

    def skip_failed_target(self) -> bool:
        """Skip a failed validation target (marking it REJECTED) so validation can continue."""
        if self.screen_validator.is_active and self.screen_validator.target_phase == ValidationPhase.FAILED:
            val_summary = self.screen_validator.skip_failed_target()
            if val_summary is not None:
                self._persist_completed_validation(val_summary)
            return True
        return False

    def start_screen_calibration(self) -> None:
        """Initiate 9-point screen gaze calibration in a fresh experiment session."""
        logger.info("Starting 9-point screen calibration...")
        self.experiment_logger = ExperimentLogger(base_dir=Path(self.config.screen.experiments_dir))
        self.screen_calibrator.experiment_id = self.experiment_logger.experiment_id
        self._validation_linked_to_active_calib = False
        if self.screen_gaze_stabilizer is not None:
            self.screen_gaze_stabilizer.reset()
        self.screen_calibrator.start()

    def cancel_screen_calibration(self) -> None:
        """Cancel in-progress 9-point screen calibration."""
        logger.info("Cancelling 9-point screen calibration...")
        self._queue_validation_after_calibration = False
        self.screen_calibrator.cancel()

    def queue_screen_validation(self) -> None:
        """Queue automatic transition to independent validation once calibration completes."""
        self._queue_validation_after_calibration = True

    def start_screen_validation(self, reuse_active_experiment: bool = False) -> None:
        """Initiate screen gaze accuracy validation."""
        if self.active_mapper is None or not self.active_mapper.is_fitted:
            raise RuntimeError("Cannot start screen validation: No calibrated gaze mapper model available.")
        logger.info("Starting screen gaze accuracy validation...")
        if not reuse_active_experiment:
            self.experiment_logger = ExperimentLogger(base_dir=Path(self.config.screen.experiments_dir))
            self._validation_linked_to_active_calib = False
        else:
            self._validation_linked_to_active_calib = True

        self.screen_validator.experiment_id = self.experiment_logger.experiment_id
        self.screen_validator.session_id = getattr(self.screen_calibrator, "session_id", "sess_001")
        self.screen_validator.normalization_method = getattr(self.active_mapper, "normalization_method", "None")
        self.screen_validator.training_sample_count = int(
            self.active_profile.training_sample_count
            if self.active_profile is not None and self.active_profile.training_sample_count > 0
            else getattr(self.active_mapper, "training_sample_count", 0)
        )
        self.screen_validator.training_metrics = (
            dict(self.active_profile.training_metrics)
            if self.active_profile is not None and self.active_profile.training_metrics
            else None
        )
        if self.active_profile is not None:
            self.screen_validator.calibration_profile_id = self.active_profile.profile_id
        if self.screen_gaze_stabilizer is not None:
            self.screen_gaze_stabilizer.reset()
        self.screen_validator.start(self.active_mapper)

    def _persist_completed_validation(self, val_summary: Optional[AccuracySummary], failed_zero_valid: bool = False) -> None:
        """Update active profile and write complete validation reports & artifacts to disk."""
        if val_summary is not None and self.active_profile is not None:
            self.active_profile.validation_metrics = val_summary.to_report_dict()
            if self._active_profile_file is not None:
                try:
                    self.profile_manager.save_profile(self.active_profile, filepath=self._active_profile_file)
                except Exception as e:
                    logger.warning(f"Could not update saved profile with validation metrics: {e}")

        calib_samples = (
            self.screen_calibrator.get_all_samples()
            if self._validation_linked_to_active_calib or self.screen_calibrator.is_completed
            else []
        )
        try:
            created = self.experiment_logger.log_experiment(
                calibration_samples=calib_samples,
                validation_results=self.screen_validator.get_trial_records(),
                accuracy_summary=val_summary,
                profile=self.active_profile,
                metadata_extra={
                    "screen_width": self.config.screen.screen_width,
                    "screen_height": self.config.screen.screen_height,
                    "normalization_method": getattr(self.active_mapper, "normalization_method", "None") if self.active_mapper else "None",
                    "validation_grid_mode": self.config.screen.validation_grid_mode,
                    "calibration_metadata": self.screen_calibrator.calibration_metadata if self.screen_calibrator.is_completed else None,
                    "status": "COMPLETED" if val_summary is not None and not failed_zero_valid else "NO_VALID_TRIALS",
                    "failure_reason": self.screen_validator.failure_reason,
                },
                export_excel=self.config.screen.export_excel,
                generate_plots=(val_summary is not None),
            )
            self.last_experiment_dir = self.experiment_logger.experiment_dir
            if created.get("central_tracker_xlsx"):
                self.last_tracker_path = created["central_tracker_xlsx"]
        except Exception as e:
            logger.warning(f"Failed to log validation experiment: {e}")

    def cancel_screen_validation(self) -> None:
        """Cancel in-progress screen gaze validation while preserving completed trials."""
        logger.info("Cancelling screen gaze validation...")
        self.screen_validator.cancel()
        completed_trials = self.screen_validator.get_trial_records()
        calib_samples = (
            self.screen_calibrator.get_all_samples()
            if self._validation_linked_to_active_calib or self.screen_calibrator.is_completed
            else []
        )
        if completed_trials or calib_samples:
            try:
                self.experiment_logger.log_experiment(
                    calibration_samples=calib_samples,
                    validation_results=completed_trials,
                    accuracy_summary=None,
                    profile=self.active_profile,
                    metadata_extra={
                        "screen_width": self.config.screen.screen_width,
                        "screen_height": self.config.screen.screen_height,
                        "validation_grid_mode": self.config.screen.validation_grid_mode,
                        "status": "INTERRUPTED_BY_USER",
                        "completed_trials_count": len(completed_trials),
                    },
                    export_excel=self.config.screen.export_excel,
                    generate_plots=False,
                )
                self.last_experiment_dir = self.experiment_logger.experiment_dir
                logger.info(f"Preserved {len(completed_trials)} completed trials to {self.experiment_logger.experiment_dir}.")
            except Exception as e:
                logger.warning(f"Failed to preserve interrupted validation trials: {e}")

    def load_calibration_profile(self, filepath: Path) -> CalibrationProfile:
        """Load calibration profile from disk and set active mapper if convention-compatible."""
        profile = self.profile_manager.load_profile(filepath)
        compat, msg = self.profile_manager.validate_compatibility(
            profile, self.config.screen.screen_width, self.config.screen.screen_height
        )
        if not compat:
            logger.warning(f"Profile compatibility notice: {msg}")
        try:
            mapper = self.profile_manager.extract_mapper(profile)
        except IncompatibleCalibrationProfileError as e:
            self.active_mapper = None
            self.active_profile = None
            self._active_profile_file = None
            logger.error(f"Refused to activate incompatible calibration profile: {e}")
            raise
        self.active_mapper = mapper
        self.active_profile = profile
        self._active_profile_file = Path(filepath)
        if self.screen_gaze_stabilizer is not None:
            self.screen_gaze_stabilizer.reset()
        logger.info(
            f"Loaded and activated profile [{profile.profile_id}] ({profile.model_type}, "
            f"convention={profile.coordinate_convention})"
        )
        return profile

    def save_calibration_profile(self, filepath: Optional[Path] = None) -> Optional[Path]:
        """Save active calibration profile to disk."""
        if self.active_profile is None:
            logger.warning("No active calibration profile available to save.")
            return None
        return self.profile_manager.save_profile(self.active_profile, filepath)

    def toggle_data_logging(self) -> bool:
        """Toggle CSV diagnostic recording on or off."""
        return self.data_logger.toggle_logging()

    def start_data_logging(self, filename: Optional[str] = None) -> Path:
        """Start recording diagnostic telemetry to CSV."""
        return self.data_logger.start_logging(filename)

    def stop_data_logging(self) -> Optional[Path]:
        """Stop recording diagnostic telemetry to CSV."""
        return self.data_logger.stop_logging()

    def start(self) -> None:
        """Start underlying camera and prepare pipeline."""
        logger.info("Starting CV Pipeline...")
        if self.camera is not None:
            self.camera.start()
        self._is_running = True

    def process_next_frame(self) -> Optional[PipelineOutput]:
        """
        Grab and process the next frame from the camera.

        Returns:
            PipelineOutput with landmarks, eye features, head pose, smoothed gaze, and metrics.
        """
        if not self._is_running:
            return None

        t_start = self.fps_counter.start_frame()

        if self.use_mock_camera:
            self._mock_frame_count += 1
            w, h = self.config.camera.frame_width, self.config.camera.frame_height
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Draw synthetic face canvas
            cv2.ellipse(frame, (w // 2, h // 2), (180, 240), 0, 0, 360, (60, 60, 60), -1)
            ret = True
        else:
            if not self.camera or not self.camera.is_opened:
                return None
            ret, frame = self.camera.read()

        if not ret or frame is None:
            return None

        self._frame_index += 1
        now = time.time()
        timestamp_ms = int(now * 1000)
        h, w = frame.shape[:2]

        # 1. MediaPipe Face & Iris Landmark Detection
        detection_result = self.detector.process_frame(frame, timestamp_ms=timestamp_ms)

        # 2. Eye & Iris Feature Extraction (canthal vector normalization)
        eye_features = extract_eye_features(
            landmarks=detection_result.all_landmarks if detection_result.has_face else None,
            ear_min=self.config.gaze.ear_min,
            ear_max=self.config.gaze.ear_max,
            blink_threshold=self.config.gaze.blink_threshold,
        )

        # 3. Head Pose Estimation (computed early for calibration & compensation)
        head_pose: Optional[HeadPose] = None
        if self.head_pose_estimator and detection_result.has_face and detection_result.all_landmarks:
            head_pose = self.head_pose_estimator.estimate(
                landmarks=detection_result.all_landmarks,
                image_width=w,
                image_height=h,
            )

        # 4. Neutral Baseline Calibration Update
        calib_finished = self.calibrator.update(eye_features, head_pose)
        if calib_finished:
            logger.info(
                f"Baseline calibration completed! Neutral: ({self.calibrator.baseline.neutral_x:.3f}, "
                f"{self.calibrator.baseline.neutral_y:.3f}) | Noise std: ({self.calibrator.baseline.noise_std_x:.3f}, "
                f"{self.calibrator.baseline.noise_std_y:.3f})"
            )
            self.gaze_classifier.reset()
            self.gaze_smoother.reset()

        baseline = self.calibrator.baseline

        # 5. Tracking State Evaluation
        if not detection_result.has_face:
            tracking_state = TrackingState.NO_FACE
        elif self.calibrator.is_calibrating:
            tracking_state = TrackingState.CALIBRATING
        elif not eye_features.is_valid or eye_features.confidence <= 0.0:
            tracking_state = TrackingState.INVALID_EYES
        elif (
            (head_pose and (abs(head_pose.yaw) > self.config.gaze.max_head_yaw_limit or abs(head_pose.pitch) > self.config.gaze.max_head_pitch_limit))
            or eye_features.confidence < 0.40
        ):
            tracking_state = TrackingState.LOW_CONFIDENCE
        else:
            tracking_state = TrackingState.TRACKING

        # 6. Head Pose Compensation (corrected VOR counter-rotation) & single-eye alignment
        raw_x = eye_features.combined_horizontal_ratio
        raw_y = eye_features.combined_vertical_ratio
        comp_eye = compensate_eye_features(
            eye_features=eye_features,
            head_pose=head_pose,
            baseline=baseline,
            enable_head_compensation=self.config.gaze.enable_head_pose_compensation,
            yaw_factor=self.config.gaze.head_yaw_compensation_factor,
            pitch_factor=self.config.gaze.head_pitch_compensation_factor,
        )
        comp_x = comp_eye.combined_horizontal_ratio if comp_eye is not None else raw_x
        comp_y = comp_eye.combined_vertical_ratio if comp_eye is not None else raw_y

        # 7. Gaze Smoothing (EMA on stabilized coordinates)
        is_gaze_valid = eye_features.is_valid and (comp_x is not None) and (comp_y is not None)
        smoothed_x, smoothed_y = self.gaze_smoother.update(comp_x, comp_y, is_valid=is_gaze_valid)

        # 8. Baseline Deviation & Direction Classification with Hysteresis
        eval_x = smoothed_x if smoothed_x is not None else comp_x
        eval_y = smoothed_y if smoothed_y is not None else comp_y

        gaze_direction = self.gaze_classifier.classify(
            eval_x,
            eval_y,
            baseline_x=baseline.neutral_x,
            baseline_y=baseline.neutral_y,
            baseline_std_y=baseline.noise_std_y,
        )

        delta_x = (eval_x - baseline.neutral_x) if eval_x is not None else None
        delta_y = (eval_y - baseline.neutral_y) if eval_y is not None else None

        # 8b. Phase 3: Screen Calibration / Validation Workflow Update (using VOR-stabilized eyes)
        phase3_eye = comp_eye if comp_eye is not None else eye_features
        if self.screen_calibrator.is_active:
            fitted_mapper = self.screen_calibrator.update(phase3_eye, head_pose, baseline)
            if fitted_mapper is not None:
                self.active_mapper = fitted_mapper
                self.active_profile = self.screen_calibrator.profile
                if self.screen_gaze_stabilizer is not None:
                    self.screen_gaze_stabilizer.reset()
                if self.active_profile is not None:
                    try:
                        self._active_profile_file = self.profile_manager.save_profile(self.active_profile)
                    except Exception as e:
                        logger.warning(f"Failed to auto-save calibration profile: {e}")
                try:
                    created_c = self.experiment_logger.log_experiment(
                        calibration_samples=self.screen_calibrator.get_all_samples(),
                        profile=self.active_profile,
                        metadata_extra={
                            "screen_width": self.config.screen.screen_width,
                            "screen_height": self.config.screen.screen_height,
                            "model_type": self.active_mapper.model_type,
                            "normalization_method": getattr(self.active_mapper, "normalization_method", "None"),
                            "training_metrics": self.active_profile.training_metrics if self.active_profile else None,
                            "calibration_metadata": self.screen_calibrator.calibration_metadata,
                            "status": "IN_PROGRESS" if self._queue_validation_after_calibration else "NOT_STARTED",
                        },
                        export_excel=self.config.screen.export_excel,
                    )
                    self.last_experiment_dir = self.experiment_logger.experiment_dir
                    if created_c.get("central_tracker_xlsx"):
                        self.last_tracker_path = created_c["central_tracker_xlsx"]
                except Exception as e:
                    logger.warning(f"Failed to log calibration experiment: {e}")

                if self._queue_validation_after_calibration:
                    self._queue_validation_after_calibration = False
                    self.start_screen_validation(reuse_active_experiment=True)

        if self.screen_validator.is_active:
            val_summary: Optional[AccuracySummary] = None
            val_failed_zero_valid = False
            try:
                val_summary = self.screen_validator.update(phase3_eye, head_pose, baseline)
            except RuntimeError as e:
                logger.warning(f"Validation finished with 0 valid trials: {e}")
                self.screen_validator.state = self.screen_validator.state.ERROR
                self.screen_validator.failure_reason = str(e)
                val_failed_zero_valid = True

            if val_summary is not None or val_failed_zero_valid:
                self._persist_completed_validation(val_summary, failed_zero_valid=val_failed_zero_valid)

        # 8c. Phase 3: Screen Gaze Coordinate Prediction + Cursor Stabilization
        screen_gaze_out: Optional[ScreenGazeOutput] = None
        if self.active_mapper is not None and self.active_mapper.is_fitted:
            feat_vec = extract_gaze_features(phase3_eye, head_pose, baseline)
            if feat_vec is not None and tracking_state in (TrackingState.TRACKING, TrackingState.LOW_CONFIDENCE):
                pred_coords = self.active_mapper.predict(feat_vec)
                gx = float(pred_coords[0])
                gy = float(pred_coords[1])
                if self.screen_gaze_stabilizer is not None:
                    sx, sy = self.screen_gaze_stabilizer.update(
                        gx,
                        gy,
                        timestamp=now,
                        is_valid=True,
                        confidence=eye_features.confidence,
                    )
                    if sx is not None and sy is not None:
                        gx, gy = sx, sy
                px = int(np.clip(round(gx * self.config.screen.screen_width), 0, self.config.screen.screen_width - 1))
                py = int(np.clip(round(gy * self.config.screen.screen_height), 0, self.config.screen.screen_height - 1))
                t_quality = "EXCELLENT" if eye_features.confidence >= 0.70 else "GOOD"
                screen_gaze_out = ScreenGazeOutput(
                    gaze_x=gx,
                    gaze_y=gy,
                    pixel_x=px,
                    pixel_y=py,
                    is_calibrated=True,
                    tracking_quality=t_quality,
                    confidence=eye_features.confidence,
                )
            else:
                if self.screen_gaze_stabilizer is not None:
                    self.screen_gaze_stabilizer.update(None, None, timestamp=now, is_valid=False, confidence=0.0)
                screen_gaze_out = ScreenGazeOutput(
                    gaze_x=None,
                    gaze_y=None,
                    pixel_x=None,
                    pixel_y=None,
                    is_calibrated=True,
                    tracking_quality="POOR",
                    confidence=0.0,
                )
        else:
            screen_gaze_out = ScreenGazeOutput(
                gaze_x=None,
                gaze_y=None,
                pixel_x=None,
                pixel_y=None,
                is_calibrated=False,
                tracking_quality="UNCALIBRATED",
                confidence=0.0,
            )

        # 9. Structured Output Representation
        left_eye_out = (
            SingleEyeOutput(
                iris_x_ratio=eye_features.left_eye.horizontal_ratio,
                iris_y_ratio=eye_features.left_eye.vertical_ratio,
                openness=eye_features.left_eye.openness,
                valid=eye_features.left_eye.is_valid,
            )
            if eye_features.left_eye
            else SingleEyeOutput(None, None, 0.0, False)
        )
        right_eye_out = (
            SingleEyeOutput(
                iris_x_ratio=eye_features.right_eye.horizontal_ratio,
                iris_y_ratio=eye_features.right_eye.vertical_ratio,
                openness=eye_features.right_eye.openness,
                valid=eye_features.right_eye.is_valid,
            )
            if eye_features.right_eye
            else SingleEyeOutput(None, None, 0.0, False)
        )

        gaze_out = GazeOutput(
            raw_x=raw_x,
            raw_y=raw_y,
            smoothed_x=smoothed_x,
            smoothed_y=smoothed_y,
            direction=gaze_direction.value,
            confidence=eye_features.confidence,
            baseline_x=baseline.neutral_x if baseline.is_calibrated else None,
            baseline_y=baseline.neutral_y if baseline.is_calibrated else None,
            delta_x=delta_x,
            delta_y=delta_y,
        )

        head_pose_out = (
            HeadPoseOutput(
                yaw=head_pose.yaw,
                pitch=head_pose.pitch,
                roll=head_pose.roll,
            )
            if head_pose and head_pose.is_valid
            else None
        )

        gaze_frame_result = GazeFrameResult(
            timestamp=now,
            face_detected=detection_result.has_face,
            gaze=gaze_out,
            eyes=EyesOutput(left=left_eye_out, right=right_eye_out),
            head_pose=head_pose_out,
            tracking_state=tracking_state.value,
            screen_gaze=screen_gaze_out,
        )

        # 10. Diagnostic Data Logging (if enabled)
        if self.data_logger.is_logging:
            self.data_logger.log_frame(
                frame_idx=self._frame_index,
                tracking_state=tracking_state.value,
                face_detected=detection_result.has_face,
                r_iris_x=eye_features.right_eye.horizontal_ratio if eye_features.right_eye else None,
                r_iris_y=eye_features.right_eye.vertical_ratio if eye_features.right_eye else None,
                l_iris_x=eye_features.left_eye.horizontal_ratio if eye_features.left_eye else None,
                l_iris_y=eye_features.left_eye.vertical_ratio if eye_features.left_eye else None,
                raw_x=raw_x,
                raw_y=raw_y,
                smoothed_x=smoothed_x,
                smoothed_y=smoothed_y,
                baseline_x=baseline.neutral_x,
                baseline_y=baseline.neutral_y,
                delta_x=delta_x,
                delta_y=delta_y,
                direction=gaze_direction.value,
                confidence=eye_features.confidence,
                yaw=head_pose.yaw if head_pose and head_pose.is_valid else None,
                pitch=head_pose.pitch if head_pose and head_pose.is_valid else None,
                roll=head_pose.roll if head_pose and head_pose.is_valid else None,
                r_openness=eye_features.right_eye.openness if eye_features.right_eye else None,
                l_openness=eye_features.left_eye.openness if eye_features.left_eye else None,
                v_state=self.gaze_classifier.active_vertical_state,
            )

        # 11. Update Performance Metrics
        fps_metrics = self.fps_counter.update(
            frame_start_time=t_start,
            detection_latency_ms=detection_result.inference_time_ms,
        )

        # 12. Visualization Overlay (if enabled)
        annotated_frame: Optional[np.ndarray] = None
        if self.visualizer is not None:
            annotated_frame = self.visualizer.render(
                frame=frame,
                detection_result=detection_result,
                eye_features=eye_features,
                fps_metrics=fps_metrics,
                gaze_direction=gaze_direction,
                smoothed_x=smoothed_x,
                smoothed_y=smoothed_y,
                head_pose=head_pose,
                baseline=baseline,
                tracking_state=tracking_state,
                calibrator_progress=self.calibrator.progress if self.calibrator.is_calibrating else None,
                is_recording=self.data_logger.is_logging,
                screen_gaze=screen_gaze_out,
                screen_calibrator=self.screen_calibrator,
                screen_validator=self.screen_validator,
            )

        return PipelineOutput(
            raw_frame=frame,
            detection_result=detection_result,
            eye_features=eye_features,
            fps_metrics=fps_metrics,
            gaze_direction=gaze_direction,
            smoothed_x=smoothed_x,
            smoothed_y=smoothed_y,
            head_pose=head_pose,
            gaze_frame_result=gaze_frame_result,
            annotated_frame=annotated_frame,
            tracking_state=tracking_state,
            baseline=baseline,
            screen_gaze=screen_gaze_out,
            screen_calibrator=self.screen_calibrator,
            screen_validator=self.screen_validator,
        )

    def close(self) -> None:
        """Clean up all camera, logger, and detector resources."""
        logger.info("Closing CV Pipeline...")
        self._is_running = False
        if self.screen_validator is not None and self.screen_validator.is_active:
            try:
                self.cancel_screen_validation()
            except Exception as e:
                logger.warning(f"Error preserving interrupted validation session on close: {e}")

        if self.data_logger is not None and self.data_logger.is_logging:
            try:
                self.data_logger.stop_logging()
            except Exception as e:
                logger.warning(f"Error stopping data logger: {e}")

        if self.camera is not None:
            try:
                self.camera.release()
            except Exception as e:
                logger.warning(f"Error releasing camera: {e}")

        try:
            self.detector.close()
        except Exception as e:
            logger.warning(f"Error closing detector: {e}")

        if self.visualizer is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def __enter__(self) -> "CVPipeline":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
