"""
Personalized Neutral Gaze Baseline Calibration Module.

Provides robust empirical sampling of the user's neutral resting gaze,
computing personalized center offsets (b_x, b_y) and baseline noise variance.
Eliminates reliance on hardcoded 0.50 center assumptions.
"""

from dataclasses import dataclass
import logging
import time
from typing import List, Optional, Tuple
import numpy as np

from .eye_features import BinocularEyeFeatures
from .head_pose import HeadPose

logger = logging.getLogger(__name__)


@dataclass
class GazeBaseline:
    """
    Stores empirical resting gaze coordinates and noise statistics.

    Attributes:
        neutral_x: Calibrated horizontal center ratio in anatomical gaze space.
        neutral_y: Calibrated vertical center ratio in anatomical gaze space.
        neutral_yaw: Calibrated neutral head yaw in degrees.
        neutral_pitch: Calibrated neutral head pitch in degrees.
        noise_std_x: Observed standard deviation of horizontal gaze at rest.
        noise_std_y: Observed standard deviation of vertical gaze at rest.
        sample_count: Number of valid frames used to compute this baseline.
        is_calibrated: True if calibration completed successfully.
        calibrated_at: Monotonic timestamp when calibration completed.
        neutral_r_x: Calibrated resting horizontal ratio for right eye.
        neutral_r_y: Calibrated resting vertical ratio for right eye.
        neutral_l_x: Calibrated resting horizontal ratio for left eye.
        neutral_l_y: Calibrated resting vertical ratio for left eye.
    """
    neutral_x: float = 0.50
    neutral_y: float = 0.50
    neutral_yaw: float = 0.0
    neutral_pitch: float = 0.0
    noise_std_x: float = 0.015
    noise_std_y: float = 0.015
    sample_count: int = 0
    is_calibrated: bool = False
    calibrated_at: float = 0.0
    neutral_r_x: Optional[float] = None
    neutral_r_y: Optional[float] = None
    neutral_l_x: Optional[float] = None
    neutral_l_y: Optional[float] = None

    def to_dict(self) -> dict:
        """Return dictionary representation for logging and telemetry."""
        d = {
            "neutral_x": round(self.neutral_x, 4),
            "neutral_y": round(self.neutral_y, 4),
            "neutral_yaw": round(self.neutral_yaw, 2),
            "neutral_pitch": round(self.neutral_pitch, 2),
            "noise_std_x": round(self.noise_std_x, 4),
            "noise_std_y": round(self.noise_std_y, 4),
            "sample_count": self.sample_count,
            "is_calibrated": self.is_calibrated,
        }
        if self.neutral_r_x is not None and self.neutral_l_x is not None:
            d["neutral_r_x"] = round(self.neutral_r_x, 4)
            d["neutral_l_x"] = round(self.neutral_l_x, 4)
        return d


class GazeBaselineCalibrator:
    """
    Collects valid gaze observations while the user looks at the center,
    computes robust trimmed medians, and establishes a personalized neutral baseline.
    """

    def __init__(
        self,
        calibration_duration_seconds: float = 1.5,
        target_fps: int = 30,
        min_required_samples: int = 25,
        max_head_yaw_during_calib: float = 15.0,
        max_head_pitch_during_calib: float = 15.0,
    ) -> None:
        """
        Initialize the calibrator.

        Args:
            calibration_duration_seconds: Duration in seconds to sample neutral gaze.
            target_fps: Expected video frame rate.
            min_required_samples: Minimum valid samples required to compute baseline.
            max_head_yaw_during_calib: Maximum allowed head yaw to ensure frontal posture.
            max_head_pitch_during_calib: Maximum allowed head pitch.
        """
        self.target_samples = max(min_required_samples, int(calibration_duration_seconds * target_fps))
        self.min_required_samples = min_required_samples
        self.max_head_yaw = max_head_yaw_during_calib
        self.max_head_pitch = max_head_pitch_during_calib

        self.baseline = GazeBaseline()
        self._samples_x: List[float] = []
        self._samples_y: List[float] = []
        self._samples_r_x: List[float] = []
        self._samples_r_y: List[float] = []
        self._samples_l_x: List[float] = []
        self._samples_l_y: List[float] = []
        self._samples_yaw: List[float] = []
        self._samples_pitch: List[float] = []
        self._is_calibrating: bool = False
        self._vertical_only: bool = False
        self._start_time: float = 0.0

    @property
    def is_calibrating(self) -> bool:
        """True if the calibrator is actively sampling frames."""
        return self._is_calibrating

    @property
    def is_calibrated(self) -> bool:
        """True if a valid baseline has been established."""
        return self.baseline.is_calibrated

    @property
    def progress(self) -> float:
        """Calibration progress in [0.0, 1.0]."""
        if not self._is_calibrating or self.target_samples <= 0:
            return 1.0 if self.baseline.is_calibrated else 0.0
        return min(1.0, len(self._samples_x) / float(self.target_samples))

    def start_calibration(self) -> None:
        """Initiate or restart neutral baseline sampling."""
        logger.info("Starting neutral gaze baseline calibration...")
        self._is_calibrating = True
        self._vertical_only = False
        self._samples_x.clear()
        self._samples_y.clear()
        self._samples_r_x.clear()
        self._samples_r_y.clear()
        self._samples_l_x.clear()
        self._samples_l_y.clear()
        self._samples_yaw.clear()
        self._samples_pitch.clear()
        self._start_time = time.time()

    def recalibrate_vertical(self) -> None:
        """Initiate vertical-only neutral baseline sampling, keeping horizontal baseline."""
        logger.info("Starting vertical-only gaze baseline recalibration...")
        self._is_calibrating = True
        self._vertical_only = True
        self._samples_x.clear()
        self._samples_y.clear()
        self._samples_r_x.clear()
        self._samples_r_y.clear()
        self._samples_l_x.clear()
        self._samples_l_y.clear()
        self._samples_yaw.clear()
        self._samples_pitch.clear()
        self._start_time = time.time()

    def cancel_calibration(self) -> None:
        """Cancel an in-progress calibration without altering existing baseline."""
        self._is_calibrating = False
        self._samples_x.clear()
        self._samples_y.clear()
        self._samples_r_x.clear()
        self._samples_r_y.clear()
        self._samples_l_x.clear()
        self._samples_l_y.clear()
        self._samples_yaw.clear()
        self._samples_pitch.clear()

    def update(
        self,
        eye_features: BinocularEyeFeatures,
        head_pose: Optional[HeadPose] = None,
    ) -> bool:
        """
        Process a frame observation during calibration.

        Args:
            eye_features: Extracted eye features for current frame.
            head_pose: Estimated 3D head pose (if enabled).

        Returns:
            True if calibration just successfully finished on this frame.
        """
        if not self._is_calibrating:
            return False

        # Verify eye features are valid and confident
        if not eye_features.is_valid or eye_features.confidence < 0.40:
            return False

        raw_x = eye_features.combined_horizontal_ratio
        raw_y = eye_features.combined_vertical_ratio
        if raw_x is None or raw_y is None:
            return False

        # Filter out observations with excessive head rotation
        yaw = head_pose.yaw if (head_pose and head_pose.is_valid) else 0.0
        pitch = head_pose.pitch if (head_pose and head_pose.is_valid) else 0.0
        if abs(yaw) > self.max_head_yaw or abs(pitch) > self.max_head_pitch:
            return False

        self._samples_x.append(raw_x)
        self._samples_y.append(raw_y)
        if eye_features.right_eye and eye_features.right_eye.is_valid:
            self._samples_r_x.append(eye_features.right_eye.horizontal_ratio)
            self._samples_r_y.append(eye_features.right_eye.vertical_ratio)
        if eye_features.left_eye and eye_features.left_eye.is_valid:
            self._samples_l_x.append(eye_features.left_eye.horizontal_ratio)
            self._samples_l_y.append(eye_features.left_eye.vertical_ratio)
        self._samples_yaw.append(yaw)
        self._samples_pitch.append(pitch)

        # Check if sampling goal reached
        if len(self._samples_x) >= self.target_samples:
            return self._finalize_baseline()

        return False

    def _finalize_baseline(self) -> bool:
        """Compute robust median and variance metrics from collected samples."""
        if len(self._samples_x) < self.min_required_samples:
            logger.warning(
                f"Calibration failed: only {len(self._samples_x)} valid samples "
                f"(minimum required: {self.min_required_samples})."
            )
            self._is_calibrating = False
            return False

        arr_x = np.array(self._samples_x, dtype=np.float64)
        arr_y = np.array(self._samples_y, dtype=np.float64)
        arr_yaw = np.array(self._samples_yaw, dtype=np.float64)
        arr_pitch = np.array(self._samples_pitch, dtype=np.float64)

        # Outlier rejection: Trim top and bottom 10%
        p10_x, p90_x = np.percentile(arr_x, [10, 90])
        p10_y, p90_y = np.percentile(arr_y, [10, 90])
        mask = (arr_x >= p10_x) & (arr_x <= p90_x) & (arr_y >= p10_y) & (arr_y <= p90_y)

        clean_x = arr_x[mask] if np.sum(mask) >= 10 else arr_x
        clean_y = arr_y[mask] if np.sum(mask) >= 10 else arr_y

        neutral_x = float(np.median(clean_x))
        neutral_y = float(np.median(clean_y))
        std_x = float(np.std(clean_x))
        std_y = float(np.std(clean_y))

        neutral_r_x = float(np.median(self._samples_r_x)) if self._samples_r_x else None
        neutral_r_y = float(np.median(self._samples_r_y)) if self._samples_r_y else None
        neutral_l_x = float(np.median(self._samples_l_x)) if self._samples_l_x else None
        neutral_l_y = float(np.median(self._samples_l_y)) if self._samples_l_y else None

        neutral_yaw = float(np.median(arr_yaw))
        neutral_pitch = float(np.median(arr_pitch))

        if self._vertical_only and self.baseline.is_calibrated:
            neutral_x = self.baseline.neutral_x
            std_x = self.baseline.noise_std_x
            neutral_yaw = self.baseline.neutral_yaw
            neutral_r_x = self.baseline.neutral_r_x
            neutral_l_x = self.baseline.neutral_l_x

        self.baseline = GazeBaseline(
            neutral_x=neutral_x,
            neutral_y=neutral_y,
            neutral_yaw=neutral_yaw,
            neutral_pitch=neutral_pitch,
            noise_std_x=max(1e-4, std_x),
            noise_std_y=max(1e-4, std_y),
            sample_count=len(self._samples_x),
            is_calibrated=True,
            calibrated_at=time.time(),
            neutral_r_x=neutral_r_x,
            neutral_r_y=neutral_r_y,
            neutral_l_x=neutral_l_x,
            neutral_l_y=neutral_l_y,
        )

        self._is_calibrating = False
        self._vertical_only = False
        logger.info(
            f"Baseline Calibration Complete! Neutral=({neutral_x:.3f}, {neutral_y:.3f}) | "
            f"Noise=({std_x:.4f}, {std_y:.4f}) | Pose=({neutral_yaw:.1f}deg, {neutral_pitch:.1f}deg)"
        )
        return True

    def reset(self) -> None:
        """Reset calibration back to uncalibrated default state."""
        self.baseline = GazeBaseline()
        self._is_calibrating = False
        self._samples_x.clear()
        self._samples_y.clear()
        self._samples_r_x.clear()
        self._samples_r_y.clear()
        self._samples_l_x.clear()
        self._samples_l_y.clear()
        self._samples_yaw.clear()
        self._samples_pitch.clear()
