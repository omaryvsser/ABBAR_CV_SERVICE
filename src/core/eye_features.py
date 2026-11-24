"""
Eye Feature Extraction Module (Phase 2 Enhanced).

Calculates:
- Normalized iris positions relative to eye boundaries (resolution-independent)
- Continuous normalized eye openness [0.0 = closed, 1.0 = fully open]
- Categorical eye openness states (OPEN, PARTIALLY_CLOSED, CLOSED, INVALID)
- Single-eye reliability scores based on openness and contour stability
- Unified binocular feature fusion with dynamic single-eye occlusion fallback
- Meaningful composite confidence score

DISCLAIMER:
The normalized horizontal and vertical iris ratios computed in this module represent
relative eye geometry within ocular canthi and eyelid margins.
They do NOT constitute calibrated screen gaze coordinates; screen gaze mapping
will be developed in Phase 3 with explicit calibration.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np

from .landmark_indices import (
    LEFT_EYE_BOTTOM_MARGIN,
    LEFT_EYE_CONTOUR,
    LEFT_EYE_EAR_HORIZONTAL,
    LEFT_EYE_EAR_VERTICAL_1,
    LEFT_EYE_EAR_VERTICAL_2,
    LEFT_EYE_INNER_CANTHUS,
    LEFT_EYE_OUTER_CANTHUS,
    LEFT_EYE_TOP_MARGIN,
    LEFT_IRIS_ALL,
    LEFT_IRIS_CENTER,
    RIGHT_EYE_BOTTOM_MARGIN,
    RIGHT_EYE_CONTOUR,
    RIGHT_EYE_EAR_HORIZONTAL,
    RIGHT_EYE_EAR_VERTICAL_1,
    RIGHT_EYE_EAR_VERTICAL_2,
    RIGHT_EYE_INNER_CANTHUS,
    RIGHT_EYE_OUTER_CANTHUS,
    RIGHT_EYE_TOP_MARGIN,
    RIGHT_IRIS_ALL,
    RIGHT_IRIS_CENTER,
    TOTAL_LANDMARKS,
)


class EyeOpennessState(str, Enum):
    """Categorical openness state for an eye."""
    OPEN = "OPEN"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    CLOSED = "CLOSED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class Point2D:
    """Normalized 2D/3D coordinate (coordinates in [0.0, 1.0])."""
    x: float
    y: float
    z: float = 0.0

    def to_pixel(self, image_width: int, image_height: int) -> Tuple[int, int]:
        """Convert normalized coordinates to integer pixel coordinates."""
        px = int(np.clip(round(self.x * image_width), 0, image_width - 1))
        py = int(np.clip(round(self.y * image_height), 0, image_height - 1))
        return px, py


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box in normalized coordinates."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    def to_pixel(self, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        """Return (x_min, y_min, x_max, y_max) in integer pixel units."""
        x1 = int(np.clip(round(self.x_min * image_width), 0, image_width - 1))
        y1 = int(np.clip(round(self.y_min * image_height), 0, image_height - 1))
        x2 = int(np.clip(round(self.x_max * image_width), 0, image_width - 1))
        y2 = int(np.clip(round(self.y_max * image_height), 0, image_height - 1))
        return x1, y1, x2, y2


@dataclass(frozen=True)
class SingleEyeFeatures:
    """
    Extracted geometric features for a single eye.

    Attributes:
        side: "left" or "right" (subject perspective).
        iris_center: Normalized coordinates of iris center.
        bounding_box: Normalized bounding box of eye contour.
        horizontal_ratio: Iris position in anatomical gaze space [0.0 = subject looking LEFT, 1.0 = subject looking RIGHT].
        vertical_ratio: Iris position vertically inside eye bounding box [0.0 = top eyelid / UP, 1.0 = bottom eyelid / DOWN].
        eye_aspect_ratio: Raw Eye Aspect Ratio (EAR).
        openness: Normalized continuous openness score [0.0 = closed, 1.0 = fully open].
        openness_state: EyeOpennessState classification.
        is_blink: True if eye is closed or below blink threshold.
        reliability: Quality/confidence metric for this eye [0.0 = unusable, 1.0 = ideal].
        is_valid: True if landmarks were present, valid, and non-degenerate.
    """
    side: str
    iris_center: Point2D
    bounding_box: BoundingBox
    horizontal_ratio: float
    vertical_ratio: float
    eye_aspect_ratio: float
    openness: float
    openness_state: EyeOpennessState
    is_blink: bool
    reliability: float
    is_valid: bool


@dataclass(frozen=True)
class BinocularEyeFeatures:
    """
    Unified binocular eye features combining left and right ocular data.

    Attributes:
        left_eye: Features for subject's left eye (if detected).
        right_eye: Features for subject's right eye (if detected).
        combined_horizontal_ratio: Reliability-weighted horizontal iris ratio.
        combined_vertical_ratio: Reliability-weighted vertical iris ratio.
        inter_iris_distance: Normalized Euclidean distance between left and right iris centers.
        confidence: Composite reliability score in [0.0, 1.0] representing tracking fidelity.
        is_valid: True if at least one eye has valid features and is open.
    """
    left_eye: Optional[SingleEyeFeatures]
    right_eye: Optional[SingleEyeFeatures]
    combined_horizontal_ratio: Optional[float]
    combined_vertical_ratio: Optional[float]
    inter_iris_distance: Optional[float]
    confidence: float
    is_valid: bool

    @property
    def average_horizontal_ratio(self) -> Optional[float]:
        """Backward-compatible alias for combined_horizontal_ratio."""
        return self.combined_horizontal_ratio

    @property
    def average_vertical_ratio(self) -> Optional[float]:
        """Backward-compatible alias for combined_vertical_ratio."""
        return self.combined_vertical_ratio


def _euclidean_distance(p1: Point2D, p2: Point2D) -> float:
    """Compute 2D Euclidean distance between two normalized points."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def _compute_ear(
    p_vert1_top: Point2D,
    p_vert1_bot: Point2D,
    p_vert2_top: Point2D,
    p_vert2_bot: Point2D,
    p_horiz_inner: Point2D,
    p_horiz_outer: Point2D,
) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) using Soukupová & Čech formula:
    EAR = (||p_v1_top - p_v1_bot|| + ||p_v2_top - p_v2_bot||) / (2 * ||p_h_in - p_h_out||)
    """
    d_v1 = _euclidean_distance(p_vert1_top, p_vert1_bot)
    d_v2 = _euclidean_distance(p_vert2_top, p_vert2_bot)
    d_h = _euclidean_distance(p_horiz_inner, p_horiz_outer)

    if d_h < 1e-6:
        return 0.0
    return (d_v1 + d_v2) / (2.0 * d_h)


def calculate_eye_openness(
    ear: float,
    ear_min: float = 0.16,
    ear_max: float = 0.32,
) -> Tuple[float, EyeOpennessState]:
    """
    Map raw Eye Aspect Ratio (EAR) to a normalized openness score and state.

    Args:
        ear: Raw Eye Aspect Ratio.
        ear_min: EAR threshold corresponding to complete eye closure (0.0).
        ear_max: EAR threshold corresponding to normal fully open eye (1.0).

    Returns:
        Tuple of (normalized_openness in [0.0, 1.0], EyeOpennessState).
    """
    if ear <= 0.0 or math.isnan(ear):
        return 0.0, EyeOpennessState.INVALID

    span = max(1e-4, ear_max - ear_min)
    openness = float(np.clip((ear - ear_min) / span, 0.0, 1.0))

    if openness >= 0.65:
        state = EyeOpennessState.OPEN
    elif openness >= 0.25:
        state = EyeOpennessState.PARTIALLY_CLOSED
    else:
        state = EyeOpennessState.CLOSED

    return openness, state


def _extract_single_eye(
    landmarks: Sequence[Any],
    side: str,
    contour_indices: Sequence[int],
    iris_center_idx: int,
    iris_all_indices: Sequence[int],
    ear_v1_pair: Tuple[int, int],
    ear_v2_pair: Tuple[int, int],
    ear_h_pair: Tuple[int, int],
    top_margin_idx: int,
    bottom_margin_idx: int,
    ear_min: float = 0.16,
    ear_max: float = 0.32,
    blink_threshold: float = 0.18,
) -> SingleEyeFeatures:
    """
    Extract geometric features for one eye using eye-local anatomical coordinate normalization.
    
    Eye-local normalization projects the iris center onto the true anatomical canthal vector
    (inner canthus to outer canthus), achieving complete invariance to head roll, face scale,
    and camera distance.
    """
    # Safety checks
    max_idx = max(
        max(contour_indices),
        max(iris_all_indices),
        iris_center_idx,
        ear_v1_pair[0], ear_v1_pair[1],
        ear_v2_pair[0], ear_v2_pair[1],
        ear_h_pair[0], ear_h_pair[1],
        top_margin_idx, bottom_margin_idx,
    )
    if len(landmarks) <= max_idx:
        return SingleEyeFeatures(
            side=side,
            iris_center=Point2D(0.0, 0.0),
            bounding_box=BoundingBox(0.0, 0.0, 0.0, 0.0),
            horizontal_ratio=0.5,
            vertical_ratio=0.5,
            eye_aspect_ratio=0.0,
            openness=0.0,
            openness_state=EyeOpennessState.INVALID,
            is_blink=True,
            reliability=0.0,
            is_valid=False,
        )

    # Convert landmarks to Point2D with finiteness guard
    def get_pt(idx: int) -> Point2D:
        lm = landmarks[idx]
        return Point2D(x=float(lm.x), y=float(lm.y), z=float(getattr(lm, "z", 0.0)))

    def _is_pt_valid(pt: Point2D) -> bool:
        return (
            math.isfinite(pt.x)
            and math.isfinite(pt.y)
            and math.isfinite(pt.z)
            and (-0.25 <= pt.x <= 1.25)
            and (-0.25 <= pt.y <= 1.25)
        )

    center_pt = get_pt(iris_center_idx)
    all_iris_pts = [get_pt(i) for i in iris_all_indices]
    rim_pts = [get_pt(i) for i in iris_all_indices if i != iris_center_idx] or all_iris_pts

    # Reject non-finite or wildly out-of-frame landmarks early
    core_pts = [center_pt, get_pt(ear_h_pair[0]), get_pt(ear_h_pair[1]), get_pt(top_margin_idx), get_pt(bottom_margin_idx)]
    if not all(_is_pt_valid(p) for p in core_pts) or not all(_is_pt_valid(p) for p in all_iris_pts):
        return SingleEyeFeatures(
            side=side,
            iris_center=Point2D(0.0, 0.0),
            bounding_box=BoundingBox(0.0, 0.0, 0.0, 0.0),
            horizontal_ratio=0.5,
            vertical_ratio=0.5,
            eye_aspect_ratio=0.0,
            openness=0.0,
            openness_state=EyeOpennessState.INVALID,
            is_blink=True,
            reliability=0.0,
            is_valid=False,
        )

    # Compute iris center: blend explicit iris center with cardinal rim centroid,
    # falling back to rim centroid if center landmark is displaced disproportionately.
    avg_rim_x = sum(p.x for p in rim_pts) / len(rim_pts)
    avg_rim_y = sum(p.y for p in rim_pts) / len(rim_pts)
    avg_rim_z = sum(p.z for p in rim_pts) / len(rim_pts)
    rim_centroid = Point2D(x=avg_rim_x, y=avg_rim_y, z=avg_rim_z)
    h_span_prelim = max(1e-6, _euclidean_distance(get_pt(ear_h_pair[0]), get_pt(ear_h_pair[1])))
    if len(rim_pts) >= 4 and _euclidean_distance(center_pt, rim_centroid) > 0.35 * h_span_prelim:
        iris_center = rim_centroid
    else:
        avg_iris_x = sum(p.x for p in all_iris_pts) / len(all_iris_pts)
        avg_iris_y = sum(p.y for p in all_iris_pts) / len(all_iris_pts)
        avg_iris_z = sum(p.z for p in all_iris_pts) / len(all_iris_pts)
        iris_center = Point2D(
            x=(center_pt.x + avg_iris_x) / 2.0,
            y=(center_pt.y + avg_iris_y) / 2.0,
            z=(center_pt.z + avg_iris_z) / 2.0,
        )

    # Compute bounding box of eye contour (for visualization and boundary checks)
    contour_pts = [get_pt(i) for i in contour_indices]
    xs = [p.x for p in contour_pts]
    ys = [p.y for p in contour_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    # Key anatomical landmarks
    p_lateral = get_pt(ear_h_pair[0] if side == "right" else ear_h_pair[1])
    p_medial = get_pt(ear_h_pair[1] if side == "right" else ear_h_pair[0])
    p_top = get_pt(top_margin_idx)
    p_bottom = get_pt(bottom_margin_idx)

    # Eye-local horizontal canthal axis (from anatomical LEFT to anatomical RIGHT):
    # For Right Eye: anatomical LEFT is medial (nose, 133), anatomical RIGHT is lateral (ear, 33)
    # For Left Eye:  anatomical LEFT is lateral (ear, 263), anatomical RIGHT is medial (nose, 362)
    if side == "right":
        p_origin_h = p_medial    # 0.0 = looking left
        p_target_h = p_lateral   # 1.0 = looking right
    else:
        p_origin_h = p_lateral   # 0.0 = looking left
        p_target_h = p_medial    # 1.0 = looking right

    u_x = p_target_h.x - p_origin_h.x
    u_y = p_target_h.y - p_origin_h.y
    canthal_len_sq = u_x * u_x + u_y * u_y
    canthal_len = math.sqrt(canthal_len_sq) if canthal_len_sq > 1e-8 else 0.0

    # Rigid canthal midpoint (unmoving anatomical reference center for the eye socket)
    canthal_mid_x = (p_medial.x + p_lateral.x) / 2.0
    canthal_mid_y = (p_medial.y + p_lateral.y) / 2.0

    # Unified horizontal canthal vector pointing from Image-Left to Image-Right:
    # Right Eye: lateral (33, image-left) -> medial (133, image-right)
    # Left Eye:  medial (362, image-left) -> lateral (263, image-right)
    if side == "right":
        c_left_x, c_left_y = p_lateral.x, p_lateral.y
        c_right_x, c_right_y = p_medial.x, p_medial.y
    else:
        c_left_x, c_left_y = p_medial.x, p_medial.y
        c_right_x, c_right_y = p_lateral.x, p_lateral.y

    h_disp_x = c_right_x - c_left_x
    h_disp_y = c_right_y - c_left_y
    disp_len = math.hypot(h_disp_x, h_disp_y)

    # Eye-local downward perpendicular unit vector w_down:
    # Rotating u_display = (h_disp_x, h_disp_y) by -90 deg produces (-u_y, u_x), pointing down.
    if disp_len > 1e-4:
        w_down_x = -h_disp_y / disp_len
        w_down_y = h_disp_x / disp_len
    else:
        w_down_x = 0.0
        w_down_y = 1.0

    is_valid_geometry = (canthal_len_sq > 1e-8)

    # 1. Horizontal ratio (anatomical: 0.0 = looking LEFT, 1.0 = looking RIGHT)
    if is_valid_geometry:
        d_hx = iris_center.x - p_origin_h.x
        d_hy = iris_center.y - p_origin_h.y
        h_ratio = float(np.clip((d_hx * u_x + d_hy * u_y) / canthal_len_sq, 0.0, 1.0))
    else:
        h_ratio = 0.50

    # 2. Vertical ratio (anatomical: 0.0 = looking UP, 1.0 = looking DOWN)
    # Measured relative to the rigid canthal midpoint along w_down, normalized by canthal length.
    # Completely decoupled from moving eyelid borders and invulnerable to aperture collapse!
    # Scale factor k_v_scale = 0.35 maps physiological vertical excursion symmetrically to [0.0, 1.0].
    k_v_scale = 0.35
    if is_valid_geometry and canthal_len > 1e-4:
        d_vx = iris_center.x - canthal_mid_x
        d_vy = iris_center.y - canthal_mid_y
        e_iris = (d_vx * w_down_x + d_vy * w_down_y) / canthal_len
        v_ratio = float(np.clip(0.50 + (e_iris / k_v_scale), 0.0, 1.0))
    else:
        v_ratio = 0.50

    # Compute EAR and Openness
    ear = _compute_ear(
        get_pt(ear_v1_pair[0]),
        get_pt(ear_v1_pair[1]),
        get_pt(ear_v2_pair[0]),
        get_pt(ear_v2_pair[1]),
        get_pt(ear_h_pair[0]),
        get_pt(ear_h_pair[1]),
    )
    openness, openness_state = calculate_eye_openness(ear, ear_min=ear_min, ear_max=ear_max)
    is_blink = (ear < blink_threshold) or (openness_state == EyeOpennessState.CLOSED)

    # Palpebral opening height along w_down
    palpebral_opening = abs((p_bottom.x - p_top.x) * w_down_x + (p_bottom.y - p_top.y) * w_down_y)
    norm_palpebral_height = palpebral_opening / canthal_len if canthal_len > 1e-4 else 0.0

    # Reliability calculation:
    # When eye is closed, blinking, or aperture is degenerate, vertical feature is marked invalid
    # rather than generating a false UP or DOWN result.
    if not is_valid_geometry or openness_state == EyeOpennessState.INVALID or norm_palpebral_height < 0.08:
        reliability = 0.0
        is_valid = False
        v_ratio = 0.50
    elif is_blink or openness < 0.20:
        reliability = 0.05
        is_valid = False
        v_ratio = 0.50
    elif openness < 0.40:
        # Partial eye closure reduces tracking reliability
        reliability = float(np.clip(openness * 0.7, 0.15, 0.50))
        is_valid = True
    else:
        # Fully open eye
        reliability = float(np.clip(openness, 0.40, 1.0))
        is_valid = True

    return SingleEyeFeatures(
        side=side,
        iris_center=iris_center,
        bounding_box=bbox,
        horizontal_ratio=h_ratio,
        vertical_ratio=v_ratio,
        eye_aspect_ratio=ear,
        openness=openness,
        openness_state=openness_state,
        is_blink=is_blink,
        reliability=reliability,
        is_valid=is_valid,
    )


def extract_eye_features(
    landmarks: Optional[Sequence[Any]],
    ear_min: float = 0.16,
    ear_max: float = 0.32,
    blink_threshold: float = 0.18,
) -> BinocularEyeFeatures:
    """
    Extract comprehensive normalized eye and iris features from MediaPipe face landmarks.

    Features:
    - Independent left and right eye feature extraction.
    - Continuous normalized openness score.
    - Reliability-weighted binocular fusion with graceful fallback if one eye is occluded/closed.
    - Meaningful composite confidence score.

    Args:
        landmarks: List or sequence of NormalizedLandmark objects (expected 478 points).
        ear_min: EAR corresponding to closed eye.
        ear_max: EAR corresponding to fully open eye.
        blink_threshold: EAR threshold below which eye is marked blinking.

    Returns:
        BinocularEyeFeatures containing individual eye metrics and fused gaze features.
    """
    if landmarks is None or len(landmarks) < TOTAL_LANDMARKS:
        return BinocularEyeFeatures(
            left_eye=None,
            right_eye=None,
            combined_horizontal_ratio=None,
            combined_vertical_ratio=None,
            inter_iris_distance=None,
            confidence=0.0,
            is_valid=False,
        )

    right_features = _extract_single_eye(
        landmarks=landmarks,
        side="right",
        contour_indices=RIGHT_EYE_CONTOUR,
        iris_center_idx=RIGHT_IRIS_CENTER,
        iris_all_indices=RIGHT_IRIS_ALL,
        ear_v1_pair=RIGHT_EYE_EAR_VERTICAL_1,
        ear_v2_pair=RIGHT_EYE_EAR_VERTICAL_2,
        ear_h_pair=RIGHT_EYE_EAR_HORIZONTAL,
        top_margin_idx=RIGHT_EYE_TOP_MARGIN,
        bottom_margin_idx=RIGHT_EYE_BOTTOM_MARGIN,
        ear_min=ear_min,
        ear_max=ear_max,
        blink_threshold=blink_threshold,
    )

    left_features = _extract_single_eye(
        landmarks=landmarks,
        side="left",
        contour_indices=LEFT_EYE_CONTOUR,
        iris_center_idx=LEFT_IRIS_CENTER,
        iris_all_indices=LEFT_IRIS_ALL,
        ear_v1_pair=LEFT_EYE_EAR_VERTICAL_1,
        ear_v2_pair=LEFT_EYE_EAR_VERTICAL_2,
        ear_h_pair=LEFT_EYE_EAR_HORIZONTAL,
        top_margin_idx=LEFT_EYE_TOP_MARGIN,
        bottom_margin_idx=LEFT_EYE_BOTTOM_MARGIN,
        ear_min=ear_min,
        ear_max=ear_max,
        blink_threshold=blink_threshold,
    )

    r_valid = right_features.is_valid and not right_features.is_blink
    l_valid = left_features.is_valid and not left_features.is_blink

    r_weight = right_features.reliability if r_valid else 0.0
    l_weight = left_features.reliability if l_valid else 0.0
    total_weight = r_weight + l_weight

    # Binocular fusion based on ocular reliability
    if total_weight > 1e-4:
        combined_h = float(
            (right_features.horizontal_ratio * r_weight + left_features.horizontal_ratio * l_weight)
            / total_weight
        )
        combined_v = float(
            (right_features.vertical_ratio * r_weight + left_features.vertical_ratio * l_weight)
            / total_weight
        )
        is_valid = True
        # If both eyes are open, confidence is higher; if single eye, confidence is scaled down
        if r_valid and l_valid:
            # Measure bilateral agreement (discrepancy penalty)
            h_diff = abs(right_features.horizontal_ratio - left_features.horizontal_ratio)
            v_diff = abs(right_features.vertical_ratio - left_features.vertical_ratio)
            # Penalize discrepancy between eyes (if eyes disagree by >0.20, penalize steeply)
            agreement_penalty = min(0.45, (h_diff + v_diff) * 0.6)
            confidence = float(np.clip((r_weight + l_weight) / 2.0 - agreement_penalty, 0.15, 1.0))
        else:
            # Single eye fallback: max confidence capped at 0.65 to reflect occlusion
            confidence = float(np.clip(total_weight * 0.65, 0.2, 0.65))
    else:
        combined_h = None
        combined_v = None
        is_valid = False
        confidence = 0.0

    # Inter-iris distance
    inter_iris_dist: Optional[float] = None
    if right_features.is_valid and left_features.is_valid:
        inter_iris_dist = _euclidean_distance(right_features.iris_center, left_features.iris_center)

    return BinocularEyeFeatures(
        left_eye=left_features,
        right_eye=right_features,
        combined_horizontal_ratio=combined_h,
        combined_vertical_ratio=combined_v,
        inter_iris_distance=inter_iris_dist,
        confidence=confidence,
        is_valid=is_valid,
    )


def compensate_eye_features(
    eye_features: Optional[BinocularEyeFeatures],
    head_pose: Optional[Any] = None,
    baseline: Optional[Any] = None,
    enable_head_compensation: bool = True,
    yaw_factor: float = 0.0065,
    pitch_factor: float = 0.0020,
) -> Optional[BinocularEyeFeatures]:
    """
    Produce head-pose-compensated and single-eye-baseline-aligned binocular ratios.

    1. When only one eye is open/valid and the baseline recorded per-eye resting
       offsets, shifts the surviving eye into the calibrated binocular frame to
       prevent horizontal/vertical jumps from inter-eye canthal asymmetry.
    2. Applies corrected Vestibulo-Ocular Reflex (VOR) yaw/pitch compensation:
       - Turning head LEFT (+yaw) counter-rotates eyes RIGHT (+raw_x); subtracting
         yaw_factor * delta_yaw removes the head-motion artifact.
       - Tilting head UP (+pitch) counter-rotates eyes DOWN (+raw_y); subtracting
         pitch_factor * delta_pitch removes the vertical head-motion artifact.
    """
    if eye_features is None or not eye_features.is_valid:
        return eye_features
    if eye_features.combined_horizontal_ratio is None or eye_features.combined_vertical_ratio is None:
        return eye_features

    h_val = float(eye_features.combined_horizontal_ratio)
    v_val = float(eye_features.combined_vertical_ratio)

    # 1. Single-eye occlusion offset alignment using personalized resting eye offsets
    r_valid = bool(eye_features.right_eye and eye_features.right_eye.is_valid and not eye_features.right_eye.is_blink)
    l_valid = bool(eye_features.left_eye and eye_features.left_eye.is_valid and not eye_features.left_eye.is_blink)
    if baseline is not None and getattr(baseline, "is_calibrated", False) and (r_valid != l_valid):
        n_x = getattr(baseline, "neutral_x", None)
        n_y = getattr(baseline, "neutral_y", None)
        if r_valid and getattr(baseline, "neutral_r_x", None) is not None and n_x is not None:
            h_val += float(n_x - baseline.neutral_r_x)
            if getattr(baseline, "neutral_r_y", None) is not None and n_y is not None:
                v_val += float(n_y - baseline.neutral_r_y)
        elif l_valid and getattr(baseline, "neutral_l_x", None) is not None and n_x is not None:
            h_val += float(n_x - baseline.neutral_l_x)
            if getattr(baseline, "neutral_l_y", None) is not None and n_y is not None:
                v_val += float(n_y - baseline.neutral_l_y)

    # 2. VOR head pose compensation
    if enable_head_compensation and head_pose is not None and getattr(head_pose, "is_valid", False):
        ref_yaw = float(baseline.neutral_yaw) if (baseline is not None and getattr(baseline, "is_calibrated", False)) else 0.0
        ref_pitch = float(baseline.neutral_pitch) if (baseline is not None and getattr(baseline, "is_calibrated", False)) else 0.0
        yaw_diff = float(head_pose.yaw) - ref_yaw
        pitch_diff = float(head_pose.pitch) - ref_pitch
        h_val -= float(yaw_factor) * yaw_diff
        v_val -= float(pitch_factor) * pitch_diff

    comp_h = float(np.clip(h_val, 0.0, 1.0))
    comp_v = float(np.clip(v_val, 0.0, 1.0))

    return BinocularEyeFeatures(
        left_eye=eye_features.left_eye,
        right_eye=eye_features.right_eye,
        combined_horizontal_ratio=comp_h,
        combined_vertical_ratio=comp_v,
        inter_iris_distance=eye_features.inter_iris_distance,
        confidence=eye_features.confidence,
        is_valid=eye_features.is_valid,
    )
