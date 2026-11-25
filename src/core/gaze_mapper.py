"""
Gaze-to-Screen Coordinate Mapping Regression Models for AABAR (Phase 3).

Provides:
1. Feature extraction from continuous ocular and head pose measurements.
2. Linear and Polynomial (degree 2) Ridge regression mapping models.
3. Multi-output screen coordinate prediction clamped to [0.0, 1.0].
4. Pure JSON-serializable model persistence without binary pickle dependencies.
5. Model evaluation and automated model comparison / selection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np

from .eye_features import BinocularEyeFeatures
from .gaze_baseline import GazeBaseline
from .head_pose import HeadPose

logger = logging.getLogger(__name__)


FEATURE_NAMES: List[str] = [
    "raw_x",
    "raw_y",
    "r_iris_x",
    "r_iris_y",
    "l_iris_x",
    "l_iris_y",
    "r_openness",
    "l_openness",
    "head_yaw",
    "head_pitch",
]


def _sanitize_feature_array(X: Any, expected_dim: Optional[int] = None) -> Tuple[np.ndarray, bool]:
    """
    Convert input array-like into finite-checked float64 2D matrix (N, D).

    Replaces Python None and non-finite (+/-Inf) entries with NaN so downstream
    imputation can handle them safely without crashing.

    Returns:
        (X_2d, is_1d_input)
    """
    try:
        arr = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        # Handle object arrays containing None values
        obj_arr = np.asarray(X, dtype=object)
        single = (obj_arr.ndim == 1)
        if single:
            obj_arr = obj_arr.reshape(1, -1)
        arr = np.full(obj_arr.shape, np.nan, dtype=np.float64)
        for i in range(obj_arr.shape[0]):
            for j in range(obj_arr.shape[1]):
                v = obj_arr[i, j]
                if v is not None:
                    try:
                        arr[i, j] = float(v)
                    except (TypeError, ValueError):
                        arr[i, j] = np.nan
        is_1d = single
    else:
        is_1d = (arr.ndim == 1)
        if is_1d:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D feature array, got {arr.ndim}D with shape {arr.shape}")

    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: expected {expected_dim} features, got {arr.shape[1]}"
        )

    # Normalize +/-inf to NaN for uniform missing/invalid value imputation
    arr = arr.copy()
    arr[np.isinf(arr)] = np.nan
    return arr, is_1d


class StandardScaler:
    """
    Pure-NumPy, JSON-serializable feature standardizer (z-score normalization).

    Computes z = (x - mean_) / scale_ using population mean and standard deviation
    fitted strictly on training data.

    Edge-case protections:
    1. Constant features (std < eps) receive scale_ = 1.0 so they transform to 0.0
       without division-by-zero or NaN generation.
    2. Missing/NaN/Inf entries at inference time are imputed to the column training
       mean_ (which maps cleanly to 0.0 standardized units).
    3. Optional z_clip bounds extreme out-of-distribution spikes before non-linear
       polynomial expansion.
    """

    method: str = "StandardScaler"

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        eps: float = 1e-8,
        z_clip: Optional[float] = 8.0,
    ) -> None:
        self.feature_names: List[str] = list(feature_names) if feature_names else list(FEATURE_NAMES)
        self.eps: float = float(eps)
        self.z_clip: Optional[float] = float(z_clip) if z_clip is not None else None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.n_samples_seen_: int = 0
        self.is_fitted: bool = False

    def fit(self, X: Any) -> "StandardScaler":
        """Fit per-feature mean and standard deviation using training features ONLY."""
        X_arr, _ = _sanitize_feature_array(X)
        N, D = X_arr.shape
        if N < 1 or D < 1:
            raise ValueError(f"Cannot fit StandardScaler on empty matrix with shape {X_arr.shape}")

        if len(self.feature_names) != D:
            self.feature_names = [f"f_{i}" for i in range(D)]

        # Compute column-wise finite mean with fallback 0.0 for completely invalid columns
        means = np.zeros(D, dtype=np.float64)
        mins = np.zeros(D, dtype=np.float64)
        maxs = np.zeros(D, dtype=np.float64)
        X_clean = X_arr.copy()

        for j in range(D):
            col = X_arr[:, j]
            finite_mask = np.isfinite(col)
            if np.any(finite_mask):
                col_valid = col[finite_mask]
                means[j] = float(np.mean(col_valid))
                mins[j] = float(np.min(col_valid))
                maxs[j] = float(np.max(col_valid))
                X_clean[~finite_mask, j] = means[j]
            else:
                logger.warning(f"StandardScaler: feature column {j} ('{self.feature_names[j]}') has no finite training values; using neutral mean=0.0, scale=1.0.")
                X_clean[:, j] = 0.0

        var = np.mean((X_clean - means) ** 2, axis=0)
        std = np.sqrt(var)
        # Guard zero-variance / constant features cleanly
        scale = np.where((~np.isfinite(std)) | (std < self.eps), 1.0, std)

        self.mean_ = means
        self.var_ = var
        self.scale_ = scale
        self.min_ = mins
        self.max_ = maxs
        self.n_samples_seen_ = N
        self.is_fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Transform features using previously fitted training mean_ and scale_."""
        if not self.is_fitted or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted on training data before calling transform().")

        expected_dim = len(self.mean_)
        X_arr, is_1d = _sanitize_feature_array(X, expected_dim=expected_dim)

        # Impute any NaN/Inf/None entries with the fitted training mean -> 0.0 normalized value
        nan_mask = ~np.isfinite(X_arr)
        if np.any(nan_mask):
            n_bad = int(np.sum(nan_mask))
            logger.debug(f"StandardScaler sanitized {n_bad} non-finite feature value(s) to training mean.")
            col_indices = np.where(nan_mask)[1]
            X_arr[nan_mask] = self.mean_[col_indices]

        Z = (X_arr - self.mean_) / self.scale_

        if self.z_clip is not None:
            Z = np.clip(Z, -self.z_clip, self.z_clip)

        return Z[0] if is_1d else Z

    def fit_transform(self, X: Any) -> np.ndarray:
        """Fit scaler on X and return normalized X."""
        return self.fit(X).transform(X)

    def get_feature_ranges(self) -> List[Dict[str, Any]]:
        """Return inspected training statistics and value ranges for every feature."""
        if not self.is_fitted or self.mean_ is None or self.scale_ is None:
            return []
        stats: List[Dict[str, Any]] = []
        for idx, name in enumerate(self.feature_names):
            std_val = float(np.sqrt(self.var_[idx])) if self.var_ is not None else float(self.scale_[idx])
            stats.append({
                "name": name,
                "mean": round(float(self.mean_[idx]), 6),
                "std": round(std_val, 6),
                "scale": round(float(self.scale_[idx]), 6),
                "min": round(float(self.min_[idx]), 6) if self.min_ is not None else None,
                "max": round(float(self.max_[idx]), 6) if self.max_ is not None else None,
                "is_constant": bool(std_val < self.eps),
            })
        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scaler parameters to pure JSON-compatible dictionary."""
        return {
            "method": self.method,
            "is_fitted": self.is_fitted,
            "eps": self.eps,
            "z_clip": self.z_clip,
            "n_samples_seen": self.n_samples_seen_,
            "feature_names": list(self.feature_names),
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
            "var": self.var_.tolist() if self.var_ is not None else None,
            "min": self.min_.tolist() if self.min_ is not None else None,
            "max": self.max_.tolist() if self.max_ is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardScaler":
        """Reconstruct fitted StandardScaler from serialized JSON dictionary."""
        scaler = cls(
            feature_names=data.get("feature_names"),
            eps=float(data.get("eps", 1e-8)),
            z_clip=data.get("z_clip", 8.0),
        )
        scaler.is_fitted = bool(data.get("is_fitted", False))
        scaler.n_samples_seen_ = int(data.get("n_samples_seen", 0))
        if data.get("mean") is not None:
            scaler.mean_ = np.array(data["mean"], dtype=np.float64)
        if data.get("scale") is not None:
            scaler.scale_ = np.array(data["scale"], dtype=np.float64)
        if data.get("var") is not None:
            scaler.var_ = np.array(data["var"], dtype=np.float64)
        if data.get("min") is not None:
            scaler.min_ = np.array(data["min"], dtype=np.float64)
        if data.get("max") is not None:
            scaler.max_ = np.array(data["max"], dtype=np.float64)
        return scaler


def extract_gaze_features(
    eye_features: Optional[BinocularEyeFeatures],
    head_pose: Optional[HeadPose] = None,
    baseline: Optional[GazeBaseline] = None,
) -> Optional[np.ndarray]:
    """
    Extract a normalized continuous 1D numerical feature vector for regression mapping.

    Features:
    0: raw_x (combined horizontal ratio in anatomical space)
    1: raw_y (combined vertical ratio in anatomical space)
    2: r_iris_x (right eye horizontal ratio)
    3: r_iris_y (right eye vertical ratio)
    4: l_iris_x (left eye horizontal ratio)
    5: l_iris_y (left eye vertical ratio)
    6: r_openness (right eye openness [0, 1])
    7: l_openness (left eye openness [0, 1])
    8: head_yaw (normalized head yaw: yaw / 30.0)
    9: head_pitch (normalized head pitch: pitch / 30.0)

    Returns:
        np.ndarray of shape (10,) with float64 values, or None if eye tracking is invalid.
    """
    if eye_features is None or not eye_features.is_valid:
        return None

    raw_x = eye_features.combined_horizontal_ratio
    raw_y = eye_features.combined_vertical_ratio
    if raw_x is None or raw_y is None:
        return None

    r_iris_x = eye_features.right_eye.horizontal_ratio if eye_features.right_eye else raw_x
    r_iris_y = eye_features.right_eye.vertical_ratio if eye_features.right_eye else raw_y
    l_iris_x = eye_features.left_eye.horizontal_ratio if eye_features.left_eye else raw_x
    l_iris_y = eye_features.left_eye.vertical_ratio if eye_features.left_eye else raw_y

    r_open = eye_features.right_eye.openness if eye_features.right_eye else 1.0
    l_open = eye_features.left_eye.openness if eye_features.left_eye else 1.0

    # Head pose normalized by typical working range (+-30 deg)
    yaw_norm = (head_pose.yaw / 30.0) if (head_pose and head_pose.is_valid) else 0.0
    pitch_norm = (head_pose.pitch / 30.0) if (head_pose and head_pose.is_valid) else 0.0

    vec = np.array(
        [
            float(raw_x),
            float(raw_y),
            float(r_iris_x),
            float(r_iris_y),
            float(l_iris_x),
            float(l_iris_y),
            float(r_open),
            float(l_open),
            float(np.clip(yaw_norm, -1.0, 1.0)),
            float(np.clip(pitch_norm, -1.0, 1.0)),
        ],
        dtype=np.float64,
    )
    return vec


class BaseGazeMapper(ABC):
    """Abstract base class for gaze-to-screen regression mappers."""

    def __init__(
        self,
        ridge_alpha: float = 0.01,
        use_scaler: bool = True,
        z_clip: Optional[float] = 8.0,
    ) -> None:
        self.ridge_alpha = float(ridge_alpha)
        self.use_scaler: bool = bool(use_scaler)
        self.z_clip: Optional[float] = z_clip
        self.normalization_method: str = "StandardScaler" if self.use_scaler else "None"
        self.scaler: Optional[StandardScaler] = (
            StandardScaler(z_clip=z_clip) if self.use_scaler else None
        )
        self.weights: Optional[np.ndarray] = None  # Shape: (K, 2) where col 0 is X, col 1 is Y
        self.is_fitted: bool = False
        self.feature_names: List[str] = list(FEATURE_NAMES)
        self.training_sample_count: int = 0

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Unique identifier of the regression model type."""
        pass

    @abstractmethod
    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        """Transform normalized feature matrix X (N, D) into model basis matrix Phi (N, K)."""
        pass

    def normalize_features(self, X: Any) -> Tuple[np.ndarray, bool]:
        """
        Apply this model's exact fitted normalization transform to input features.

        Never fits on new data; ensures training, validation, and real-time inference
        receive identical feature representation.
        """
        if self.use_scaler and self.scaler is not None:
            if not self.scaler.is_fitted:
                raise RuntimeError("Gaze mapper's StandardScaler is not fitted yet.")
            X_norm = self.scaler.transform(X)
            is_1d = (np.asarray(X, dtype=object).ndim == 1)
            if is_1d:
                X_norm = X_norm.reshape(1, -1)
            return X_norm, is_1d

        # Unscaled legacy mode: sanitize NaN/Inf safely to 0.0
        X_arr, is_1d = _sanitize_feature_array(X, expected_dim=len(self.feature_names) if self.is_fitted and self.feature_names else None)
        if not np.all(np.isfinite(X_arr)):
            X_arr = np.where(np.isfinite(X_arr), X_arr, 0.0)
        return X_arr, is_1d

    def fit(self, X: Any, y: Any) -> "BaseGazeMapper":
        """
        Fit feature scaler (if enabled) and regression mapping model strictly on training data.

        Args:
            X: Input training features of shape (N, D).
            y: Target screen coordinates of shape (N, 2) normalized to [0.0, 1.0].

        Returns:
            Self (fitted mapper).
        """
        X_raw, _ = _sanitize_feature_array(X)
        y_arr = np.asarray(y, dtype=np.float64)

        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(1, -1)

        N, D = X_raw.shape
        if N < 2:
            raise ValueError(f"At least 2 samples required to fit regression mapper, got {N}")
        if y_arr.shape[0] != N or y_arr.shape[1] != 2:
            raise ValueError(f"Target matrix y must have shape ({N}, 2), got {y_arr.shape}")
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("Target coordinate matrix y contains non-finite values.")

        if len(self.feature_names) != D:
            self.feature_names = [f"f_{i}" for i in range(D)]

        # Fit feature scaler strictly on training features ONLY (never on validation/test data)
        if self.use_scaler:
            self.scaler = StandardScaler(feature_names=self.feature_names, z_clip=self.z_clip)
            X_model = self.scaler.fit_transform(X_raw)
            self.normalization_method = self.scaler.method
        else:
            self.scaler = None
            self.normalization_method = "None"
            X_model = np.where(np.isfinite(X_raw), X_raw, 0.0)

        Phi = self._transform_features(X_model)  # Shape (N, K)
        K = Phi.shape[1]

        # Normal equation with Ridge regularization: (Phi^T Phi + lambda * I) W = Phi^T y
        Phi_T_Phi = Phi.T @ Phi  # (K, K)
        reg_matrix = self.ridge_alpha * np.eye(K, dtype=np.float64)
        # Do not regularize the bias/intercept term (index 0)
        reg_matrix[0, 0] = 0.0

        A = Phi_T_Phi + reg_matrix
        b = Phi.T @ y_arr  # (K, 2)

        try:
            self.weights = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to least squares pseudo-inverse if singular
            self.weights, _, _, _ = np.linalg.lstsq(A, b, rcond=1e-5)

        self.is_fitted = True
        self.training_sample_count = N
        return self

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict continuous normalized screen coordinates in [0.0, 1.0].

        Applies the saved training scaler before mapping features.

        Args:
            X: Input feature matrix of shape (N, D) or 1D array of shape (D,).

        Returns:
            Array of shape (N, 2) or (2,) clamped to [0.0, 1.0].
        """
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Cannot predict with unfitted gaze mapper.")

        X_model, single_sample = self.normalize_features(X)
        Phi = self._transform_features(X_model)  # Shape (N, K)
        raw_pred = Phi @ self.weights             # Shape (N, 2)

        # Strictly enforce screen boundary invariants [0.0, 1.0]
        clamped_pred = np.clip(raw_pred, 0.0, 1.0)

        if single_sample:
            return clamped_pred[0]
        return clamped_pred

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model weights, scaler parameters, and hyperparameters to JSON-safe dict."""
        return {
            "model_type": self.model_type,
            "ridge_alpha": self.ridge_alpha,
            "use_scaler": self.use_scaler,
            "normalization_method": self.normalization_method,
            "scaler": self.scaler.to_dict() if self.scaler is not None else None,
            "is_fitted": self.is_fitted,
            "training_sample_count": self.training_sample_count,
            "feature_names": self.feature_names,
            "weights": self.weights.tolist() if self.weights is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseGazeMapper":
        """Reconstruct model and scaler instance from serialized dictionary."""
        scaler_data = data.get("scaler")
        has_scaler = scaler_data is not None and isinstance(scaler_data, dict)
        # Backwards-compatible detection: legacy models without 'scaler' run unscaled
        use_scaler = bool(data["use_scaler"]) if "use_scaler" in data else has_scaler

        mapper = cls(
            ridge_alpha=data.get("ridge_alpha", 0.01),
            use_scaler=use_scaler,
        )
        mapper.is_fitted = bool(data.get("is_fitted", False))
        mapper.training_sample_count = int(data.get("training_sample_count", 0))
        mapper.feature_names = list(data.get("feature_names", FEATURE_NAMES))

        if use_scaler and has_scaler:
            mapper.scaler = StandardScaler.from_dict(scaler_data)
            mapper.normalization_method = mapper.scaler.method
        elif not use_scaler:
            mapper.scaler = None
            mapper.normalization_method = str(data.get("normalization_method", "None"))

        weights_data = data.get("weights")
        if weights_data is not None:
            mapper.weights = np.array(weights_data, dtype=np.float64)
        return mapper

    def save(self, filepath: Path) -> None:
        """Save model configuration and weights to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved {self.model_type} model to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "BaseGazeMapper":
        """Load model from JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file does not exist: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class LinearGazeMapper(BaseGazeMapper):
    """
    Linear Ridge Regression mapper.

    Basis: [1, x_1, x_2, ..., x_D]
    Number of parameters: (D + 1) * 2
    """

    @property
    def model_type(self) -> str:
        return "linear"

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        # Prepend bias term 1.0
        ones = np.ones((N, 1), dtype=np.float64)
        return np.hstack([ones, X])


class CompactLinearGazeMapper(BaseGazeMapper):
    """
    Parsimonious Linear Ridge Regression mapper over primary binocular ocular coordinates
    [1, raw_x, raw_y] (avoiding overfitting on collinear 10-D calibration features when N=9).
    """

    @property
    def model_type(self) -> str:
        return "compact_linear"

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        if D >= 2:
            return np.column_stack([np.ones(N, dtype=np.float64), X[:, 0], X[:, 1]])
        return np.hstack([np.ones((N, 1), dtype=np.float64), X])


class CompactPolynomialGazeMapper(BaseGazeMapper):
    """
    Low-variance Degree-2 Spatial Polynomial Ridge mapper over primary binocular ocular
    coordinates: [1, x, y, x^2, y^2, x*y] (plus first extra features only if D < 10).

    Retains quadratic curvature sensitivity near screen edges without expanding all 10
    features into 36 collinear terms when calibrated on 9 fixation points.
    """

    @property
    def model_type(self) -> str:
        return "compact_polynomial"

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        if D < 2:
            return np.hstack([np.ones((N, 1), dtype=np.float64), X])
        x = X[:, 0]
        y = X[:, 1]
        cols = [
            np.ones(N, dtype=np.float64),
            x,
            y,
            x * x,
            y * y,
            x * y,
        ]
        # For synthetic multi-feature benchmarks (or if binocular channels need residual pose),
        # keep purely spatial 6-param basis on 10-D eye vectors unless extra channels differ.
        if D >= 10:
            # Also include head_yaw / head_pitch linear compensation gently regularized
            return np.column_stack(cols)
        if D > 2:
            for k in range(2, min(4, D)):
                cols.append(X[:, k])
        return np.column_stack(cols)


class PolynomialGazeMapper(BaseGazeMapper):
    """
    Degree-2 Polynomial Ridge Regression mapper.

    Basis includes:
    1. Bias: 1
    2. Linear terms: x_i
    3. Pure quadratic terms: x_i^2
    4. Cross-interaction terms between ocular features: x_i * x_j (i < j)

    Captures natural non-linear curvature in ocular-to-screen perspective geometry.
    """

    @property
    def model_type(self) -> str:
        return "polynomial"

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        terms = [np.ones((N, 1), dtype=np.float64), X]

        # 1. Quadratic terms x_i^2
        X_sq = X * X
        terms.append(X_sq)

        # 2. Key ocular interaction terms:
        cross_terms = []
        for i in range(min(6, D)):
            for j in range(i + 1, min(6, D)):
                cross_terms.append((X[:, i] * X[:, j]).reshape(-1, 1))

        if cross_terms:
            terms.append(np.hstack(cross_terms))

        return np.hstack(terms)


class DecoupledBaseGazeMapper(BaseGazeMapper):
    """
    Abstract base class for axis-decoupled gaze-to-screen regression mappers.

    In human visual physiology, horizontal eye movement (adduction / abduction)
    is driven by the medial/lateral recti and does not alter vertical target gaze.
    Coupled models fit all 10 features onto Y, creating severe cross-talk (e.g. looking
    right shifts vertical prediction by hundreds of pixels).

    Decoupled mappers fit:
      Screen X = f(horizontal_features)
      Screen Y = g(vertical_features)
    completely eliminating cross-axis contamination and keeping parameter count
    strictly within the sample size (2-4 parameters per axis for N=9 targets).
    """

    @abstractmethod
    def _transform_x(self, X: np.ndarray) -> np.ndarray:
        """Transform normalized features X (N, D) into horizontal basis Phi_x (N, K_x)."""
        pass

    @abstractmethod
    def _transform_y(self, X: np.ndarray) -> np.ndarray:
        """Transform normalized features X (N, D) into vertical basis Phi_y (N, K_y)."""
        pass

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        """Combined basis representation for compatibility and inspection."""
        phi_x = self._transform_x(X)
        phi_y = self._transform_y(X)
        return np.hstack([phi_x, phi_y])

    def fit(self, X: Any, y: Any) -> "DecoupledBaseGazeMapper":
        X_raw, _ = _sanitize_feature_array(X)
        y_arr = np.asarray(y, dtype=np.float64)

        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(1, -1)

        N, D = X_raw.shape
        if N < 2:
            raise ValueError(f"At least 2 samples required to fit regression mapper, got {N}")
        if y_arr.shape[0] != N or y_arr.shape[1] != 2:
            raise ValueError(f"Target matrix y must have shape ({N}, 2), got {y_arr.shape}")
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("Target coordinate matrix y contains non-finite values.")

        if len(self.feature_names) != D:
            self.feature_names = [f"f_{i}" for i in range(D)]

        if self.use_scaler:
            self.scaler = StandardScaler(feature_names=self.feature_names, z_clip=self.z_clip)
            X_model = self.scaler.fit_transform(X_raw)
            self.normalization_method = self.scaler.method
        else:
            self.scaler = None
            self.normalization_method = "None"
            X_model = np.where(np.isfinite(X_raw), X_raw, 0.0)

        Phi_x = self._transform_x(X_model)  # (N, K_x)
        Phi_y = self._transform_y(X_model)  # (N, K_y)
        K_x = Phi_x.shape[1]
        K_y = Phi_y.shape[1]

        # Fit X axis independently
        reg_x = self.ridge_alpha * np.eye(K_x, dtype=np.float64)
        reg_x[0, 0] = 0.0
        # Prevent head-pose overfitting and double-compensation on small sample sizes (N=9):
        # When auxiliary head pose is present in the basis (last column for D >= 10),
        # apply strong differential L2 regularization so head pose cannot overpower ocular ratios.
        if K_x >= 3 and X_raw.shape[1] >= 10:
            reg_x[-1, -1] = max(self.ridge_alpha * 25.0, 2.5)
            if K_x >= 4:
                # Quadratic term (x^2) regularization to prevent edge flare
                reg_x[2, 2] = max(self.ridge_alpha * 5.0, 0.5)

        Ax = Phi_x.T @ Phi_x + reg_x
        bx = Phi_x.T @ y_arr[:, 0]
        try:
            wx = np.linalg.solve(Ax, bx)
        except np.linalg.LinAlgError:
            wx, _, _, _ = np.linalg.lstsq(Ax, bx, rcond=1e-5)

        # Fit Y axis independently
        reg_y = self.ridge_alpha * np.eye(K_y, dtype=np.float64)
        reg_y[0, 0] = 0.0
        if K_y >= 3 and X_raw.shape[1] >= 10:
            reg_y[-1, -1] = max(self.ridge_alpha * 25.0, 2.5)
            if K_y >= 4:
                # Quadratic term (y^2) regularization to prevent edge flare
                reg_y[2, 2] = max(self.ridge_alpha * 5.0, 0.5)

        Ay = Phi_y.T @ Phi_y + reg_y
        by = Phi_y.T @ y_arr[:, 1]
        try:
            wy = np.linalg.solve(Ay, by)
        except np.linalg.LinAlgError:
            wy, _, _, _ = np.linalg.lstsq(Ay, by, rcond=1e-5)

        if K_x == K_y:
            self.weights = np.column_stack([wx, wy])
        else:
            K_max = max(K_x, K_y)
            w_mat = np.zeros((K_max, 2), dtype=np.float64)
            w_mat[:K_x, 0] = wx
            w_mat[:K_y, 1] = wy
            self.weights = w_mat

        self.is_fitted = True
        self.training_sample_count = N
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Cannot predict with unfitted gaze mapper.")

        X_model, single_sample = self.normalize_features(X)
        Phi_x = self._transform_x(X_model)
        Phi_y = self._transform_y(X_model)

        K_x = Phi_x.shape[1]
        K_y = Phi_y.shape[1]
        pred_x = Phi_x @ self.weights[:K_x, 0]
        pred_y = Phi_y @ self.weights[:K_y, 1]

        raw_pred = np.column_stack([pred_x, pred_y])
        clamped_pred = np.clip(raw_pred, 0.0, 1.0)
        if single_sample:
            return clamped_pred[0]
        return clamped_pred


class DecoupledLinearGazeMapper(DecoupledBaseGazeMapper):
    """
    Decoupled Linear Ridge Regression mapper.

    X basis: [1, raw_x, (head_yaw if D >= 10)]
    Y basis: [1, raw_y, (head_pitch if D >= 10)]
    """

    @property
    def model_type(self) -> str:
        return "decoupled_linear"

    def _transform_x(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        if D >= 10:
            return np.column_stack([np.ones(N, dtype=np.float64), X[:, 0], X[:, 8]])
        return np.column_stack([np.ones(N, dtype=np.float64), X[:, 0]])

    def _transform_y(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        if D >= 10:
            return np.column_stack([np.ones(N, dtype=np.float64), X[:, 1], X[:, 9]])
        return np.column_stack([np.ones(N, dtype=np.float64), X[:, 1]])


class DecoupledPolynomialGazeMapper(DecoupledBaseGazeMapper):
    """
    Decoupled Degree-2 Polynomial Ridge Regression mapper.

    X basis: [1, raw_x, raw_x^2, (head_yaw if D >= 10)]
    Y basis: [1, raw_y, raw_y^2, (head_pitch if D >= 10)]
    """

    @property
    def model_type(self) -> str:
        return "decoupled_polynomial"

    def _transform_x(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        x = X[:, 0]
        if D >= 10:
            return np.column_stack([np.ones(N, dtype=np.float64), x, x * x, X[:, 8]])
        return np.column_stack([np.ones(N, dtype=np.float64), x, x * x])

    def _transform_y(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        y = X[:, 1]
        if D >= 10:
            return np.column_stack([np.ones(N, dtype=np.float64), y, y * y, X[:, 9]])
        return np.column_stack([np.ones(N, dtype=np.float64), y, y * y])


class PureDecoupledLinearGazeMapper(DecoupledBaseGazeMapper):
    """
    Pure Decoupled Linear Ridge Regression mapper.

    Operates strictly on ocular horizontal and vertical ratios without
    auxiliary head pose features, preventing posture shifts from swaying
    gaze predictions. Upstream physiological VOR compensation in Step 6
    is fully preserved.

    X basis: [1, raw_x]
    Y basis: [1, raw_y]
    """

    @property
    def model_type(self) -> str:
        return "pure_decoupled_linear"

    def _transform_x(self, X: np.ndarray) -> np.ndarray:
        N, _ = X.shape
        return np.column_stack([np.ones(N, dtype=np.float64), X[:, 0]])

    def _transform_y(self, X: np.ndarray) -> np.ndarray:
        N, _ = X.shape
        return np.column_stack([np.ones(N, dtype=np.float64), X[:, 1]])


def create_mapper(
    model_type: str = "polynomial",
    ridge_alpha: float = 0.01,
    use_scaler: bool = True,
) -> BaseGazeMapper:
    """Factory creating mapper instance by type name."""
    type_clean = model_type.strip().lower()
    if type_clean in ("pure_decoupled_linear", "pure_decoupled_lin", "pure_linear"):
        return PureDecoupledLinearGazeMapper(ridge_alpha=ridge_alpha, use_scaler=use_scaler)
    elif type_clean in ("decoupled_linear", "decoupled_lin"):
        return DecoupledLinearGazeMapper(ridge_alpha=ridge_alpha, use_scaler=use_scaler)
    elif type_clean in ("decoupled_polynomial", "decoupled_poly"):
        return DecoupledPolynomialGazeMapper(ridge_alpha=ridge_alpha, use_scaler=use_scaler)
    elif type_clean in ("polynomial", "poly", "deg2"):
        return PolynomialGazeMapper(ridge_alpha=ridge_alpha, use_scaler=use_scaler)
    elif type_clean in ("compact_polynomial", "compact_poly", "spatial_poly"):
        return CompactPolynomialGazeMapper(ridge_alpha=ridge_alpha, use_scaler=use_scaler)
    elif type_clean in ("compact_linear", "compact_lin"):
        return CompactLinearGazeMapper(ridge_alpha=ridge_alpha, use_scaler=use_scaler)
    elif type_clean in ("linear", "lin", "ols"):
        return LinearGazeMapper(ridge_alpha=ridge_alpha, use_scaler=use_scaler)
    else:
        raise ValueError(
            f"Unknown mapper model type: '{model_type}'. "
            f"Choose 'pure_decoupled_linear', 'decoupled_linear', 'decoupled_polynomial', 'linear', 'compact_linear', 'compact_polynomial', or 'polynomial'."
        )


def load_mapper_auto(filepath: Path) -> BaseGazeMapper:
    """Load model from JSON, automatically dispatching by serialized model_type."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file does not exist: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    m_type = data.get("model_type", "polynomial").lower()
    if m_type in ("pure_decoupled_linear", "pure_linear"):
        return PureDecoupledLinearGazeMapper.from_dict(data)
    if m_type == "decoupled_linear":
        return DecoupledLinearGazeMapper.from_dict(data)
    if m_type == "decoupled_polynomial":
        return DecoupledPolynomialGazeMapper.from_dict(data)
    if m_type == "linear":
        return LinearGazeMapper.from_dict(data)
    if m_type == "compact_linear":
        return CompactLinearGazeMapper.from_dict(data)
    if m_type == "compact_polynomial":
        return CompactPolynomialGazeMapper.from_dict(data)
    return PolynomialGazeMapper.from_dict(data)


def evaluate_mapper_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> Dict[str, float]:
    """
    Calculate comprehensive numerical error metrics on predicted coordinates.

    Args:
        y_true: Ground truth screen coordinates of shape (N, 2) in [0.0, 1.0].
        y_pred: Model predictions of shape (N, 2) in [0.0, 1.0].
        screen_width: Screen width in pixels.
        screen_height: Screen height in pixels.

    Returns:
        Dictionary of error metrics.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    err_x = y_pred[:, 0] - y_true[:, 0]
    err_y = y_pred[:, 1] - y_true[:, 1]

    mae_x = float(np.mean(np.abs(err_x)))
    mae_y = float(np.mean(np.abs(err_y)))
    rmse_x = float(np.sqrt(np.mean(err_x ** 2)))
    rmse_y = float(np.sqrt(np.mean(err_y ** 2)))

    # Normalized Euclidean error (in screen ratio space [0, 1])
    euclidean_norm = np.sqrt(err_x ** 2 + err_y ** 2)
    mean_euclidean_norm = float(np.mean(euclidean_norm))
    median_euclidean_norm = float(np.median(euclidean_norm))
    max_euclidean_norm = float(np.max(euclidean_norm))

    # Pixel Euclidean error
    err_px_x = err_x * float(screen_width)
    err_px_y = err_y * float(screen_height)
    euclidean_px = np.sqrt(err_px_x ** 2 + err_px_y ** 2)
    mean_euclidean_px = float(np.mean(euclidean_px))
    median_euclidean_px = float(np.median(euclidean_px))
    max_euclidean_px = float(np.max(euclidean_px))

    return {
        "mae_x": mae_x,
        "mae_y": mae_y,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "mean_euclidean_norm": mean_euclidean_norm,
        "median_euclidean_norm": median_euclidean_norm,
        "max_euclidean_norm": max_euclidean_norm,
        "mean_euclidean_pixels": mean_euclidean_px,
        "median_euclidean_pixels": median_euclidean_px,
        "max_euclidean_pixels": max_euclidean_px,
    }


def _run_loocv_metrics(
    cls: Type[BaseGazeMapper],
    X: np.ndarray,
    y: np.ndarray,
    screen_width: int,
    screen_height: int,
    ridge_alpha: float,
    use_scaler: bool,
) -> Dict[str, float]:
    """Run strict Leave-One-Out Cross-Validation fitting scaler inside every fold."""
    N = len(X)
    preds = np.zeros_like(y, dtype=np.float64)
    for i in range(N):
        mask = np.arange(N) != i
        m = cls(ridge_alpha=ridge_alpha, use_scaler=use_scaler).fit(X[mask], y[mask])
        preds[i] = m.predict(X[i])
    return evaluate_mapper_predictions(y, preds, screen_width, screen_height)


def compare_and_select_mapper(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    screen_width: int = 1920,
    screen_height: int = 1080,
    ridge_alpha: float = 0.01,
    use_scaler: bool = True,
) -> Tuple[BaseGazeMapper, Dict[str, Any]]:
    """
    Benchmark Linear, CompactLinear, CompactPolynomial, and Polynomial Ridge regressors
    across regularization alphas using training-only normalization and out-of-sample CV,
    selecting the model with the lowest reliable generalization error (never training error).

    If explicit X_val/y_val are provided, evaluates candidates on X_val (using each
    model's training-fitted scaler). If omitted and N >= 4, uses Leave-One-Out CV
    where both scaler and regressor are refitted inside each fold.
    """
    X_tr = np.asarray(X_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64)
    n_samples = len(X_tr)

    has_val = X_val is not None and y_val is not None and len(X_val) > 0
    val_y_arr = np.asarray(y_val, dtype=np.float64) if has_val else None

    # Candidate alpha grids tailored per model complexity to prevent overfitting small 9-pt sets
    base_alpha = max(1e-4, float(ridge_alpha))
    if n_samples <= 12 and not has_val:
        cand_specs: List[Tuple[str, Type[BaseGazeMapper], List[float], float]] = [
            ("pure_decoupled_linear", PureDecoupledLinearGazeMapper, sorted(set([0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0])), 0.0),
            ("decoupled_linear", DecoupledLinearGazeMapper, sorted(set([0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0])), 0.002),
            ("decoupled_polynomial", DecoupledPolynomialGazeMapper, sorted(set([0.02, 0.05, 0.1, 0.25, 0.5, 1.0])), 0.015),
            ("compact_linear", CompactLinearGazeMapper, sorted(set([0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])), 0.005),
            ("compact_polynomial", CompactPolynomialGazeMapper, sorted(set([0.1, 0.25, 0.5, 1.0, 2.5, 5.0])), 0.020),
            ("linear", LinearGazeMapper, sorted(set([0.1, 0.25, 0.5, 1.0, 2.5, 5.0])), 0.005),
            ("polynomial", PolynomialGazeMapper, sorted(set([0.5, 1.0, 2.5, 5.0])), 0.025),
        ]
    else:
        cand_specs = [
            ("pure_decoupled_linear", PureDecoupledLinearGazeMapper, sorted(set([base_alpha, 0.02, 0.1, 0.5, 1.0])), 0.0),
            ("decoupled_linear", DecoupledLinearGazeMapper, sorted(set([base_alpha, 0.02, 0.1, 0.5, 1.0])), 0.002),
            ("decoupled_polynomial", DecoupledPolynomialGazeMapper, sorted(set([base_alpha, 0.05, 0.25, 1.0, 2.0])), 0.015),
            ("compact_linear", CompactLinearGazeMapper, sorted(set([base_alpha, 0.02, 0.1, 0.5, 1.0])), 0.005),
            ("compact_polynomial", CompactPolynomialGazeMapper, sorted(set([base_alpha, 0.05, 0.25, 1.0, 2.5])), 0.020),
            ("linear", LinearGazeMapper, sorted(set([base_alpha, 0.05, 0.25, 1.0, 2.5])), 0.005),
            ("polynomial", PolynomialGazeMapper, sorted(set([base_alpha, 0.1, 0.5, 1.5, 5.0])), 0.025),
        ]

    benchmarks: List[Dict[str, Any]] = []
    best_by_family: Dict[str, Tuple[Dict[str, float], Dict[str, float], BaseGazeMapper, float]] = {}

    for fam_name, cls, alphas, complexity_tiebreak in cand_specs:
        for a in alphas:
            fitted = cls(ridge_alpha=a, use_scaler=use_scaler).fit(X_tr, y_tr)
            train_met = evaluate_mapper_predictions(y_tr, fitted.predict(X_tr), screen_width, screen_height)

            if has_val and val_y_arr is not None:
                eval_met = evaluate_mapper_predictions(val_y_arr, fitted.predict(X_val), screen_width, screen_height)
                selection_basis = "validation"
                eval_count = len(val_y_arr)
            elif n_samples >= 4:
                eval_met = _run_loocv_metrics(cls, X_tr, y_tr, screen_width, screen_height, a, use_scaler)
                selection_basis = "loocv"
                eval_count = n_samples
            else:
                eval_met = train_met
                selection_basis = "training"
                eval_count = n_samples

            score = float(eval_met["mean_euclidean_norm"]) + complexity_tiebreak
            entry = {
                "model_type": fam_name,
                "ridge_alpha": a,
                "selection_score": score,
                "eval_metrics": eval_met,
                "train_metrics": train_met,
                "mapper": fitted,
            }
            benchmarks.append(entry)

            prev_best = best_by_family.get(fam_name)
            if prev_best is None or eval_met["mean_euclidean_norm"] < prev_best[0]["mean_euclidean_norm"]:
                best_by_family[fam_name] = (eval_met, train_met, fitted, a)

    best_entry = min(benchmarks, key=lambda b: b["selection_score"])
    selected: BaseGazeMapper = best_entry["mapper"]
    selected_type: str = best_entry["model_type"]
    selected_alpha: float = float(best_entry["ridge_alpha"])
    selected_training_metrics: Dict[str, float] = best_entry["train_metrics"]
    selected_eval_metrics: Dict[str, float] = best_entry["eval_metrics"]

    lin_metrics, lin_train_metrics, _, _ = best_by_family.get("linear", (selected_eval_metrics, selected_training_metrics, None, None))
    poly_metrics, poly_train_metrics, _, _ = best_by_family.get("polynomial", (selected_eval_metrics, selected_training_metrics, None, None))
    comp_lin_metrics, _, _, _ = best_by_family.get("compact_linear", (lin_metrics, None, None, None))
    comp_poly_metrics, _, _, _ = best_by_family.get("compact_polynomial", (poly_metrics, None, None, None))
    pure_dec_lin_metrics, _, _, _ = best_by_family.get("pure_decoupled_linear", (lin_metrics, None, None, None))
    dec_lin_metrics, _, _, _ = best_by_family.get("decoupled_linear", (lin_metrics, None, None, None))
    dec_poly_metrics, _, _, _ = best_by_family.get("decoupled_polynomial", (poly_metrics, None, None, None))

    comparison_report = {
        "selected_model": selected_type,
        "selected_ridge_alpha": selected_alpha,
        "normalization_method": selected.normalization_method,
        "selection_basis": selection_basis,
        "validation_samples": eval_count,
        "linear_metrics": lin_metrics,
        "polynomial_metrics": poly_metrics,
        "compact_linear_metrics": comp_lin_metrics,
        "compact_polynomial_metrics": comp_poly_metrics,
        "pure_decoupled_linear_metrics": pure_dec_lin_metrics,
        "decoupled_linear_metrics": dec_lin_metrics,
        "decoupled_polynomial_metrics": dec_poly_metrics,
        "linear_training_metrics": lin_train_metrics,
        "polynomial_training_metrics": poly_train_metrics,
        "training_metrics": selected_training_metrics,
        "selected_eval_metrics": selected_eval_metrics,
        "candidate_benchmarks": [
            {
                "model_type": b["model_type"],
                "ridge_alpha": b["ridge_alpha"],
                "mean_euclidean_pixels": round(b["eval_metrics"]["mean_euclidean_pixels"], 2),
                "train_mean_pixels": round(b["train_metrics"]["mean_euclidean_pixels"], 2),
            }
            for b in benchmarks
        ],
        "performance_gain_pct": float(
            (lin_metrics["mean_euclidean_norm"] - selected_eval_metrics["mean_euclidean_norm"])
            / max(1e-6, lin_metrics["mean_euclidean_norm"]) * 100.0
        ),
    }

    logger.info(
        f"Mapper Comparison Complete ({selection_basis}, scaler={selected.normalization_method}). "
        f"Selected: [{selected_type.upper()} alpha={selected_alpha}] ({selected_eval_metrics['mean_euclidean_pixels']:.1f}px CV) | "
        f"PureDecLin: {pure_dec_lin_metrics['mean_euclidean_pixels']:.1f}px | "
        f"DecoupledLin: {dec_lin_metrics['mean_euclidean_pixels']:.1f}px | "
        f"DecoupledPoly: {dec_poly_metrics['mean_euclidean_pixels']:.1f}px | "
        f"CompactLin: {comp_lin_metrics['mean_euclidean_pixels']:.1f}px | "
        f"CompactPoly: {comp_poly_metrics['mean_euclidean_pixels']:.1f}px | "
        f"Linear: {lin_metrics['mean_euclidean_pixels']:.1f}px | "
        f"Poly: {poly_metrics['mean_euclidean_pixels']:.1f}px | "
        f"Selected Train Fit: {selected_training_metrics['mean_euclidean_pixels']:.1f}px"
    )
    return selected, comparison_report
