"""
Configuration Module for AABAR CV Service.

Handles typed, validated settings loaded from environment variables and .env file.
"""

import os
from pathlib import Path
from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)


class CameraConfig(BaseModel):
    """Configuration for camera device and capture properties."""

    camera_index: int = Field(
        default_factory=lambda: int(os.getenv("AABAR_CAMERA_INDEX", "0")),
        description="Index of video device (0 = default webcam)",
    )
    frame_width: int = Field(
        default_factory=lambda: int(os.getenv("AABAR_FRAME_WIDTH", "1280")),
        description="Target capture width in pixels",
    )
    frame_height: int = Field(
        default_factory=lambda: int(os.getenv("AABAR_FRAME_HEIGHT", "720")),
        description="Target capture height in pixels",
    )
    target_fps: int = Field(
        default_factory=lambda: int(os.getenv("AABAR_TARGET_FPS", "30")),
        description="Desired capture and processing frames per second",
    )
    buffer_size: int = Field(
        default_factory=lambda: int(os.getenv("AABAR_BUFFER_SIZE", "1")),
        description="OpenCV buffer size (1 minimizes latency)",
    )
    use_threading: bool = Field(
        default=True,
        description="Run frame capture on a background thread to prevent buffer buildup",
    )


class LandmarkerConfig(BaseModel):
    """Configuration for MediaPipe Face Landmarker model."""

    model_path: str = Field(
        default_factory=lambda: os.getenv(
            "AABAR_MODEL_PATH",
            str(Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"),
        ),
        description="Path to MediaPipe face_landmarker.task model bundle",
    )
    num_faces: int = Field(
        default_factory=lambda: int(os.getenv("AABAR_NUM_FACES", "1")),
        description="Maximum number of faces to detect (AABAR tracks a single user)",
    )
    min_face_detection_confidence: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_MIN_FACE_DETECTION_CONFIDENCE", "0.5")),
        description="Minimum confidence score for face detection",
    )
    min_face_presence_confidence: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_MIN_FACE_PRESENCE_CONFIDENCE", "0.5")),
        description="Minimum confidence score for face presence in landmark detection",
    )
    min_tracking_confidence: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_MIN_TRACKING_CONFIDENCE", "0.5")),
        description="Minimum confidence score for landmark tracking across video frames",
    )
    running_mode: Literal["VIDEO", "LIVE_STREAM", "IMAGE"] = Field(
        default_factory=lambda: os.getenv("AABAR_RUNNING_MODE", "VIDEO"),  # type: ignore[assignment]
        description="MediaPipe running mode (VIDEO enables deterministic temporal tracking)",
    )


class VisualizationConfig(BaseModel):
    """Configuration for OpenCV real-time visualizer."""

    enabled: bool = Field(
        default_factory=lambda: os.getenv("AABAR_VIZ_ENABLED", "true").lower() in ("true", "1", "yes"),
        description="Whether the GUI display window is enabled",
    )
    window_name: str = Field(
        default_factory=lambda: os.getenv("AABAR_WINDOW_NAME", "AABAR - CV Service | Face & Iris Tracking"),
        description="Title of display window",
    )
    mirror_display: bool = Field(
        default_factory=lambda: os.getenv("AABAR_VIZ_MIRROR_DISPLAY", "true").lower() in ("true", "1", "yes"),
        description="Horizontally mirror camera feed in debug window for intuitive selfie mirror display",
    )
    draw_face_mesh: bool = Field(
        default_factory=lambda: os.getenv("AABAR_DRAW_FACE_MESH", "false").lower() in ("true", "1", "yes"),
        description="Render complete facial mesh connections",
    )
    draw_eye_contours: bool = Field(
        default_factory=lambda: os.getenv("AABAR_DRAW_EYE_CONTOURS", "true").lower() in ("true", "1", "yes"),
        description="Render left and right eyelid boundaries",
    )
    draw_bounding_boxes: bool = Field(
        default_factory=lambda: os.getenv("AABAR_DRAW_BOUNDING_BOXES", "true").lower() in ("true", "1", "yes"),
        description="Render bounding boxes around eye sockets",
    )
    draw_iris: bool = Field(
        default_factory=lambda: os.getenv("AABAR_DRAW_IRIS", "true").lower() in ("true", "1", "yes"),
        description="Render iris landmarks and outline",
    )
    draw_iris_center: bool = Field(
        default_factory=lambda: os.getenv("AABAR_DRAW_IRIS_CENTER", "true").lower() in ("true", "1", "yes"),
        description="Render pinpoint iris center dot",
    )
    draw_ratios: bool = Field(
        default_factory=lambda: os.getenv("AABAR_DRAW_RATIOS", "true").lower() in ("true", "1", "yes"),
        description="Display text HUD with normalized horizontal/vertical ratios",
    )
    draw_fps: bool = Field(
        default_factory=lambda: os.getenv("AABAR_DRAW_FPS", "true").lower() in ("true", "1", "yes"),
        description="Display real-time FPS and latency counter",
    )
    draw_gaze_direction: bool = Field(
        default_factory=lambda: os.getenv("AABAR_DRAW_GAZE_DIRECTION", "true").lower() in ("true", "1", "yes"),
        description="Display relative gaze direction badge and indicator",
    )
    draw_gaze_crosshair: bool = Field(
        default_factory=lambda: os.getenv("AABAR_DRAW_GAZE_CROSSHAIR", "true").lower() in ("true", "1", "yes"),
        description="Render raw vs smoothed 2D gaze indicator crosshairs",
    )
    draw_head_pose: bool = Field(
        default_factory=lambda: os.getenv("AABAR_DRAW_HEAD_POSE", "true").lower() in ("true", "1", "yes"),
        description="Display 3D head orientation angles and axes",
    )
    draw_eye_openness: bool = Field(
        default_factory=lambda: os.getenv("AABAR_DRAW_EYE_OPENNESS", "true").lower() in ("true", "1", "yes"),
        description="Display eye openness meters for both eyes",
    )


class GazeConfig(BaseModel):
    """Configuration for relative gaze estimation, classification, and smoothing."""

    horizontal_center_min: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_GAZE_H_CENTER_MIN", "0.42")),
        description="Lower bound for horizontal center deadzone",
    )
    horizontal_center_max: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_GAZE_H_CENTER_MAX", "0.58")),
        description="Upper bound for horizontal center deadzone",
    )
    vertical_center_min: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_GAZE_V_CENTER_MIN", "0.38")),
        description="Lower bound for vertical center deadzone (looking up)",
    )
    vertical_center_max: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_GAZE_V_CENTER_MAX", "0.62")),
        description="Upper bound for vertical center deadzone (looking down)",
    )
    ear_min: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_EAR_MIN", "0.16")),
        description="EAR threshold corresponding to complete eye closure (openness = 0.0)",
    )
    ear_max: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_EAR_MAX", "0.32")),
        description="EAR threshold corresponding to full normal eye opening (openness = 1.0)",
    )
    blink_threshold: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_BLINK_THRESHOLD", "0.18")),
        description="EAR threshold below which eye is marked as closed/blinking",
    )
    smoothing_alpha: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_SMOOTHING_ALPHA", "0.25")),
        description="Exponential Moving Average (EMA) smoothing weight",
    )
    saccade_threshold: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_SACCADE_THRESHOLD", "0.12")),
        description="Gaze distance jump triggering adaptive saccade smoothing acceleration",
    )
    saccade_alpha: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_SACCADE_ALPHA", "0.85")),
        description="Responsive smoothing weight during fast saccades",
    )
    max_invalid_frames: int = Field(
        default_factory=lambda: int(os.getenv("AABAR_MAX_INVALID_FRAMES", "5")),
        description="Consecutive blink frames to hold previous gaze before resetting",
    )
    enable_head_pose: bool = Field(
        default_factory=lambda: os.getenv("AABAR_ENABLE_HEAD_POSE", "true").lower() in ("true", "1", "yes"),
        description="Enable solvePnP 3D head pose estimation",
    )
    enable_head_pose_compensation: bool = Field(
        default_factory=lambda: os.getenv("AABAR_ENABLE_POSE_COMPENSATION", "true").lower() in ("true", "1", "yes"),
        description="Apply gentle head yaw and pitch compensation to prevent false directions during minor head turns",
    )
    head_yaw_compensation_factor: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_POSE_COMPENSATION_FACTOR", "0.0065")),
        description="Linear VOR compensation weight subtracting spurious head-yaw eye counter-rotation",
    )
    head_pitch_compensation_factor: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_PITCH_COMPENSATION_FACTOR", "0.0020")),
        description="Linear compensation weight for head pitch deviation",
    )
    max_head_yaw_limit: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_MAX_HEAD_YAW", "25.0")),
        description="Maximum head yaw in degrees before flagging tracking as LOW_CONFIDENCE",
    )
    max_head_pitch_limit: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_MAX_HEAD_PITCH", "20.0")),
        description="Maximum head pitch in degrees before flagging tracking as LOW_CONFIDENCE",
    )
    auto_calibrate_on_start: bool = Field(
        default_factory=lambda: os.getenv("AABAR_AUTO_CALIBRATE", "true").lower() in ("true", "1", "yes"),
        description="Automatically sample neutral center gaze on pipeline startup",
    )
    calibration_duration_seconds: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_CALIBRATION_DURATION", "1.5")),
        description="Duration in seconds to sample neutral center gaze",
    )
    h_activation_threshold: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_GAZE_H_ACT_THRESH", "0.06")),
        description="Horizontal deviation from neutral required to enter LEFT or RIGHT",
    )
    h_deactivation_threshold: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_GAZE_H_DEACT_THRESH", "0.035")),
        description="Horizontal deviation below which active direction drops back to CENTER",
    )
    v_activation_threshold: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_GAZE_V_ACT_THRESH", "0.07")),
        description="Symmetrical fallback vertical deviation required to enter UP or DOWN",
    )
    v_deactivation_threshold: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_GAZE_V_DEACT_THRESH", "0.04")),
        description="Symmetrical fallback vertical deviation below which active direction drops back to CENTER",
    )
    v_up_activation_threshold: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_GAZE_V_UP_ACT_THRESH", "0.045")),
        description="Vertical deviation from neutral required to enter UP",
    )
    v_up_deactivation_threshold: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_GAZE_V_UP_DEACT_THRESH", "0.025")),
        description="Vertical deviation below which active UP drops back to CENTER",
    )
    v_down_activation_threshold: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_GAZE_V_DOWN_ACT_THRESH", "0.060")),
        description="Vertical deviation from neutral required to enter DOWN",
    )
    v_down_deactivation_threshold: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_GAZE_V_DOWN_DEACT_THRESH", "0.035")),
        description="Vertical deviation below which active DOWN drops back to CENTER",
    )
    saccade_threshold_y: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_SACCADE_THRESHOLD_Y", "0.050")),
        description="Vertical gaze delta triggering saccade acceleration",
    )
    direction_persistence_frames: int = Field(
        default_factory=lambda: int(os.getenv("AABAR_DIRECTION_PERSISTENCE", "3")),
        description="Consecutive frames required to confirm a new gaze direction",
    )
    allow_compound_directions: bool = Field(
        default_factory=lambda: os.getenv("AABAR_ALLOW_COMPOUND_DIRECTIONS", "false").lower() in ("true", "1", "yes"),
        description="Allow diagonal compound directions (e.g. UP_LEFT)",
    )


class ScreenConfig(BaseModel):
    """Configuration for Phase 3 screen calibration, mapping, and validation."""

    screen_width: int = Field(
        default_factory=lambda: int(os.getenv("AABAR_SCREEN_WIDTH", "1920")),
        description="Target display width in pixels",
    )
    screen_height: int = Field(
        default_factory=lambda: int(os.getenv("AABAR_SCREEN_HEIGHT", "1080")),
        description="Target display height in pixels",
    )
    settling_duration: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_CALIB_SETTLE_DURATION", "0.8")),
        description="Seconds to allow user eyes to locate and settle on calibration target",
    )
    collection_duration: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_CALIB_COLLECT_DURATION", "1.2")),
        description="Seconds to sample valid eye features during steady target fixation",
    )
    min_samples_per_target: int = Field(
        default_factory=lambda: int(os.getenv("AABAR_CALIB_MIN_SAMPLES", "15")),
        description="Minimum valid eye feature frames required per calibration target",
    )
    model_type: str = Field(
        default_factory=lambda: os.getenv("AABAR_MAPPER_MODEL_TYPE", "auto"),
        description="Regression model algorithm: 'auto', 'polynomial', or 'linear'",
    )
    ridge_alpha: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_MAPPER_RIDGE_ALPHA", "0.01")),
        description="L2 Ridge regularization penalty parameter",
    )
    normalize_features: bool = Field(
        default_factory=lambda: os.getenv("AABAR_NORMALIZE_FEATURES", "true").lower() in ("true", "1", "yes"),
        description="Standardize regression input features with StandardScaler (fitted on training data only)",
    )
    validation_target_radius_px: Optional[float] = Field(
        default=None,
        description="Optional fixed target radius in screen pixels; None derives exact boundary from rendered target",
    )
    profiles_dir: str = Field(
        default_factory=lambda: os.getenv("AABAR_PROFILES_DIR", "data/calibration"),
        description="Directory for saving and loading calibration profile JSONs",
    )
    experiments_dir: str = Field(
        default_factory=lambda: os.getenv("AABAR_EXPERIMENTS_DIR", "data/experiments"),
        description="Directory for storing calibration and validation CSV/Excel experiment logs",
    )
    export_excel: bool = Field(
        default_factory=lambda: os.getenv("AABAR_EXPORT_EXCEL", "true").lower() in ("true", "1", "yes"),
        description="Export multi-sheet Excel workbook if openpyxl is installed",
    )
    active_profile_path: Optional[str] = Field(
        default_factory=lambda: os.getenv("AABAR_ACTIVE_PROFILE_PATH"),
        description="Optional path to existing profile JSON to preload on startup",
    )
    stabilize_screen_gaze: bool = Field(
        default=True,
        description="Apply adaptive One-Euro cursor stabilization to predicted screen gaze coordinates",
    )
    screen_filter_min_cutoff: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_SCREEN_MIN_CUTOFF", "1.6")),
        description="Minimum cutoff frequency (Hz) for steady fixation smoothing",
    )
    screen_filter_beta: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_SCREEN_BETA", "18.0")),
        description="Speed coefficient increasing cutoff during fast saccades to minimize lag",
    )
    calib_max_collection_timeout: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_CALIB_MAX_TIMEOUT", "5.0")),
        description="Maximum seconds to wait for stable fixation frames on a single calibration point",
    )
    calib_fixation_max_std: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_CALIB_FIXATION_MAX_STD", "0.022")),
        description="Maximum rolling horizontal/vertical eye dispersion allowed for a steady fixation sample",
    )
    validation_max_timeout: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_VAL_MAX_TIMEOUT", "4.5")),
        description="Maximum seconds to gather minimum valid samples per validation target before failing/retrying",
    )
    validation_grid_mode: Literal["full_9", "interior_5"] = Field(
        default_factory=lambda: os.getenv("AABAR_VAL_GRID_MODE", "full_9"),  # type: ignore[assignment]
        description="Independent validation target layout ('full_9' covers all 9 screen regions; 'interior_5' uses 5 interior targets)",
    )
    validation_repeats: int = Field(
        default_factory=lambda: int(os.getenv("AABAR_VAL_REPEATS", "1")),
        description="Number of trial repetitions per validation target (default: 1)",
    )
    randomize_validation: bool = Field(
        default_factory=lambda: os.getenv("AABAR_VAL_RANDOMIZE", "false").lower() in ("true", "1", "yes"),
        description="Randomize order of validation targets to eliminate sequential bias",
    )
    threshold_2pct: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_VAL_THRESH_2PCT", "0.02")),
        description="Strict accuracy threshold as fraction of screen diagonal (default: 0.02)",
    )
    threshold_5pct: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_VAL_THRESH_5PCT", "0.05")),
        description="Moderate accuracy threshold as fraction of screen diagonal (default: 0.05)",
    )
    threshold_10pct: float = Field(
        default_factory=lambda: float(os.getenv("AABAR_VAL_THRESH_10PCT", "0.10")),
        description="Relaxed accuracy threshold as fraction of screen diagonal (default: 0.10)",
    )


class AppConfig(BaseModel):
    """Master application configuration."""

    camera: CameraConfig = Field(default_factory=CameraConfig)
    landmarker: LandmarkerConfig = Field(default_factory=LandmarkerConfig)
    gaze: GazeConfig = Field(default_factory=GazeConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    screen: ScreenConfig = Field(default_factory=ScreenConfig)


def load_config() -> AppConfig:
    """Load and return the validated application configuration."""
    return AppConfig()
