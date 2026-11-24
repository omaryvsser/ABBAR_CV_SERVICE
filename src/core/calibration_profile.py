"""
Calibration Profile Management for AABAR CV Service (Phase 3).

Stores, validates, loads, and manages user calibration profiles containing
trained regression model parameters, screen resolution context, feature
configurations, and calibration metadata.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from .gaze_mapper import BaseGazeMapper, FEATURE_NAMES, load_mapper_auto

logger = logging.getLogger(__name__)


COORDINATE_CONVENTION_VERSION = "interface_left0_right1_v1"
LEGACY_MIRRORED_CONVENTION = "legacy_mirrored_target_overlay"


class IncompatibleCalibrationProfileError(ValueError):
    """Raised when a calibration profile uses an incompatible coordinate convention."""
    pass


@dataclass
class CalibrationProfile:
    """
    Persistent container for a complete calibration profile.

    Attributes:
        profile_id: Unique identifier for this calibration profile.
        user_id: User/subject identifier.
        created_at: Epoch timestamp of creation.
        screen_width: Display width in pixels at calibration time.
        screen_height: Display height in pixels at calibration time.
        camera_index: Index of video capture device used during calibration.
        model_type: Regression model algorithm ("linear" or "polynomial").
        model_data: Serialized model weights and parameters dict.
        feature_names: List of feature names used by this profile.
        normalization_method: Feature scaling method (e.g. "StandardScaler" or "None").
        normalization_params: Serialized scaler statistics fitted on training data.
        training_sample_count: Number of calibration target samples used to train the model.
        calibration_frame_count: Number of accepted camera frames accumulated during calibration.
        coordinate_convention: Screen/gaze horizontal coordinate convention identifier.
        training_metrics: Error metrics evaluated on calibration training set (not validation accuracy).
        validation_metrics: Evaluation metrics on independent validation targets.
        model_version: Calibration schema version (e.g. "3.1").
    """
    profile_id: str
    user_id: str = "default_user"
    created_at: float = field(default_factory=time.time)
    screen_width: int = 1920
    screen_height: int = 1080
    camera_index: int = 0
    model_type: str = "polynomial"
    model_data: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    normalization_method: str = "StandardScaler"
    normalization_params: Optional[Dict[str, Any]] = None
    training_sample_count: int = 0
    calibration_frame_count: int = 0
    coordinate_convention: str = COORDINATE_CONVENTION_VERSION
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Optional[Dict[str, float]] = None
    calibration_metadata: Optional[Dict[str, Any]] = None
    model_version: str = "3.1"

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationProfile":
        """Construct profile from dictionary with explicit coordinate convention checking."""
        model_data = dict(data.get("model_data", {}))
        scaler_in_model = model_data.get("scaler")
        norm_params = data.get("normalization_params")
        if norm_params is None and isinstance(scaler_in_model, dict):
            norm_params = scaler_in_model

        if "normalization_method" in data:
            norm_method = str(data["normalization_method"])
        elif norm_params is not None:
            norm_method = str(norm_params.get("method", "StandardScaler"))
        elif model_data.get("is_fitted") and scaler_in_model is None:
            # Legacy pre-normalization profile: preserve unscaled inference
            norm_method = "None"
        else:
            norm_method = "StandardScaler"

        # Any pre-fix JSON profile lacking `coordinate_convention` was calibrated while
        # the target overlay inverted X in selfie-mirror mode. Flag it explicitly!
        coord_convention = str(data.get("coordinate_convention", LEGACY_MIRRORED_CONVENTION))

        train_count = int(
            data.get("training_sample_count", model_data.get("training_sample_count", 0))
        )

        return cls(
            profile_id=data.get("profile_id", "default_profile"),
            user_id=data.get("user_id", "default_user"),
            created_at=float(data.get("created_at", time.time())),
            screen_width=int(data.get("screen_width", 1920)),
            screen_height=int(data.get("screen_height", 1080)),
            camera_index=int(data.get("camera_index", 0)),
            model_type=data.get("model_type", "polynomial"),
            model_data=model_data,
            feature_names=list(data.get("feature_names", FEATURE_NAMES)),
            normalization_method=norm_method,
            normalization_params=norm_params,
            training_sample_count=train_count,
            calibration_frame_count=int(data.get("calibration_frame_count", 0)),
            coordinate_convention=coord_convention,
            training_metrics=dict(data.get("training_metrics", {})),
            validation_metrics=data.get("validation_metrics"),
            calibration_metadata=data.get("calibration_metadata"),
            model_version=str(data.get("model_version", "3.0")),
        )


class ProfileManager:
    """Manages profile saving, loading, inspection, and compatibility verification."""

    def __init__(self, profiles_dir: Optional[Path] = None) -> None:
        self.profiles_dir = Path(profiles_dir) if profiles_dir else Path("data/calibration")
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def save_profile(
        self,
        profile: CalibrationProfile,
        filepath: Optional[Path] = None,
    ) -> Path:
        """
        Save a calibration profile to JSON.

        Args:
            profile: CalibrationProfile instance.
            filepath: Optional destination path; defaults to `data/calibration/profile_<profile_id>.json`.

        Returns:
            Absolute Path to written file.
        """
        if filepath is None:
            filename = f"profile_{profile.profile_id}.json"
            filepath = self.profiles_dir / filename
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2)

        logger.info(f"Calibration profile saved: {filepath.resolve()}")
        return filepath.resolve()

    def load_profile(self, filepath: Path) -> CalibrationProfile:
        """Load and parse calibration profile from JSON."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Calibration profile does not exist: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        profile = CalibrationProfile.from_dict(data)
        logger.info(
            f"Loaded profile [{profile.profile_id}] (model: {profile.model_type}, "
            f"resolution: {profile.screen_width}x{profile.screen_height})"
        )
        return profile

    def extract_mapper(
        self,
        profile: CalibrationProfile,
        allow_incompatible_convention: bool = False,
    ) -> BaseGazeMapper:
        """
        Instantiate fitted BaseGazeMapper and restore its trained StandardScaler
        from profile's serialized model_data / normalization_params.

        Raises:
            IncompatibleCalibrationProfileError: if the profile was created with a
                legacy/mismatched horizontal coordinate convention.
        """
        from .gaze_mapper import StandardScaler, create_mapper

        if (
            not allow_incompatible_convention
            and profile.coordinate_convention != COORDINATE_CONVENTION_VERSION
        ):
            raise IncompatibleCalibrationProfileError(
                f"Calibration profile '{profile.profile_id}' uses incompatible coordinate convention "
                f"'{profile.coordinate_convention}' (trained with mirrored target overlay where screen X "
                f"was inverted relative to gaze). Expected '{COORDINATE_CONVENTION_VERSION}' "
                f"(gaze_x=0 Left, gaze_x=1 Right). Please run a fresh 9-point calibration."
            )

        mapper = create_mapper(profile.model_type).from_dict(profile.model_data)

        # Ensure external normalization_params stored at profile level are synced if missing in model_data
        if mapper.scaler is None and profile.normalization_params is not None:
            mapper.scaler = StandardScaler.from_dict(profile.normalization_params)
            mapper.use_scaler = True
            mapper.normalization_method = profile.normalization_method

        if mapper.is_fitted and profile.normalization_method == "StandardScaler":
            if mapper.scaler is None or not mapper.scaler.is_fitted:
                raise RuntimeError(
                    f"Calibration profile [{profile.profile_id}] specifies StandardScaler normalization "
                    f"but lacks fitted scaler parameters."
                )

        return mapper

    def validate_compatibility(
        self,
        profile: CalibrationProfile,
        current_screen_w: int,
        current_screen_h: int,
        current_camera_index: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Verify whether an existing profile matches coordinate convention and hardware settings.

        Returns:
            (is_compatible, explanation_message)
        """
        if profile.coordinate_convention != COORDINATE_CONVENTION_VERSION:
            msg = (
                f"Incompatible coordinate convention: Profile '{profile.profile_id}' uses "
                f"'{profile.coordinate_convention}' (legacy mirrored target overlay), whereas current "
                f"pipeline requires '{COORDINATE_CONVENTION_VERSION}' (gaze_x=0 Left, gaze_x=1 Right). "
                f"Recalibration required."
            )
            return False, msg

        if profile.screen_width != current_screen_w or profile.screen_height != current_screen_h:
            msg = (
                f"Resolution mismatch: Profile was calibrated for "
                f"{profile.screen_width}x{profile.screen_height}, but current display is "
                f"{current_screen_w}x{current_screen_h}. Recalibration recommended."
            )
            return False, msg

        if current_camera_index is not None and profile.camera_index != current_camera_index:
            msg = (
                f"Camera device mismatch: Profile calibrated with camera {profile.camera_index}, "
                f"current device is {current_camera_index}."
            )
            return True, msg  # Non-fatal warning

        return True, "Profile is fully compatible with current screen resolution."

    def list_profiles(self) -> List[Path]:
        """Return list of all profile JSON files in profiles directory."""
        if not self.profiles_dir.exists():
            return []
        return sorted(self.profiles_dir.glob("profile_*.json"))
