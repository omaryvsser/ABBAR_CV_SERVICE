"""
Centralized MediaPipe Face & Iris Landmark Topology Indices.

This module is the single source of truth for all MediaPipe facial and iris landmark
indices used throughout the AABAR Computer Vision service.

Topology Reference:
- Total Landmarks: 478
  - 468 Face Mesh landmarks (indices 0..467)
  - 10 Iris landmarks (indices 468..477)
- Anatomical conventions:
  - Subject Right Eye (viewer left): indices 33..246, iris 468..472
  - Subject Left Eye (viewer right): indices 263..466, iris 473..477
"""

from typing import Final, FrozenSet, Tuple

# Total landmark counts
TOTAL_LANDMARKS: Final[int] = 478
NUM_FACE_MESH_LANDMARKS: Final[int] = 468
NUM_IRIS_LANDMARKS: Final[int] = 10

# ---------------------------------------------------------------------------
# Subject's Right Eye (viewer's left side in standard un-mirrored webcam frame)
# ---------------------------------------------------------------------------
RIGHT_EYE_OUTER_CANTHUS: Final[int] = 33      # Lateral corner
RIGHT_EYE_INNER_CANTHUS: Final[int] = 133     # Medial corner
RIGHT_EYE_TOP_MARGIN: Final[int] = 159        # Superior eyelid extreme
RIGHT_EYE_BOTTOM_MARGIN: Final[int] = 145     # Inferior eyelid extreme

# Complete closed right eye contour loop
RIGHT_EYE_CONTOUR: Final[Tuple[int, ...]] = (
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
)

# Landmarks used for Right Eye Aspect Ratio (EAR) computation
# EAR = (||p160 - p144|| + ||p158 - p153||) / (2 * ||p33 - p133||)
RIGHT_EYE_EAR_VERTICAL_1: Final[Tuple[int, int]] = (160, 144)
RIGHT_EYE_EAR_VERTICAL_2: Final[Tuple[int, int]] = (158, 153)
RIGHT_EYE_EAR_HORIZONTAL: Final[Tuple[int, int]] = (33, 133)

# Right Iris landmarks
RIGHT_IRIS_CENTER: Final[int] = 468
RIGHT_IRIS_CONTOUR: Final[Tuple[int, ...]] = (469, 470, 471, 472)
RIGHT_IRIS_ALL: Final[Tuple[int, ...]] = (468, 469, 470, 471, 472)


# ---------------------------------------------------------------------------
# Subject's Left Eye (viewer's right side in standard un-mirrored webcam frame)
# ---------------------------------------------------------------------------
LEFT_EYE_INNER_CANTHUS: Final[int] = 362     # Medial corner
LEFT_EYE_OUTER_CANTHUS: Final[int] = 263     # Lateral corner
LEFT_EYE_TOP_MARGIN: Final[int] = 386        # Superior eyelid extreme
LEFT_EYE_BOTTOM_MARGIN: Final[int] = 374     # Inferior eyelid extreme

# Complete closed left eye contour loop
LEFT_EYE_CONTOUR: Final[Tuple[int, ...]] = (
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
)

# Landmarks used for Left Eye Aspect Ratio (EAR) computation
# EAR = (||p385 - p380|| + ||p387 - p373||) / (2 * ||p362 - p263||)
LEFT_EYE_EAR_VERTICAL_1: Final[Tuple[int, int]] = (385, 380)
LEFT_EYE_EAR_VERTICAL_2: Final[Tuple[int, int]] = (387, 373)
LEFT_EYE_EAR_HORIZONTAL: Final[Tuple[int, int]] = (362, 263)

# Left Iris landmarks
LEFT_IRIS_CENTER: Final[int] = 473
LEFT_IRIS_CONTOUR: Final[Tuple[int, ...]] = (474, 475, 476, 477)
LEFT_IRIS_ALL: Final[Tuple[int, ...]] = (473, 474, 475, 476, 477)


# ---------------------------------------------------------------------------
# Facial Silhouette / Oval (for context visualization)
# ---------------------------------------------------------------------------
FACE_OVAL_CONTOUR: Final[Tuple[int, ...]] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
)

# Eyebrow contours (useful for expression & context)
RIGHT_EYEBROW: Final[Tuple[int, ...]] = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)
LEFT_EYEBROW: Final[Tuple[int, ...]] = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)


# ---------------------------------------------------------------------------
# Canonical 3D Facial Landmarks for Head Pose Estimation (solvePnP)
# ---------------------------------------------------------------------------
NOSE_TIP: Final[int] = 1
CHIN: Final[int] = 152
LEFT_EYE_OUTER_CORNER: Final[int] = 263
RIGHT_EYE_OUTER_CORNER: Final[int] = 33
LEFT_MOUTH_CORNER: Final[int] = 291
RIGHT_MOUTH_CORNER: Final[int] = 61

HEAD_POSE_LANDMARKS: Final[Tuple[int, ...]] = (
    NOSE_TIP,
    CHIN,
    LEFT_EYE_OUTER_CORNER,
    RIGHT_EYE_OUTER_CORNER,
    LEFT_MOUTH_CORNER,
    RIGHT_MOUTH_CORNER,
)


# Aggregated sets for fast membership checks and validation
ALL_EYE_LANDMARKS: Final[FrozenSet[int]] = frozenset(
    RIGHT_EYE_CONTOUR + LEFT_EYE_CONTOUR + RIGHT_IRIS_ALL + LEFT_IRIS_ALL
)

ALL_IRIS_LANDMARKS: Final[FrozenSet[int]] = frozenset(
    RIGHT_IRIS_ALL + LEFT_IRIS_ALL
)


def validate_topology() -> bool:
    """
    Validate internal consistency of landmark index definitions.

    Ensures:
    1. All indices are integers between 0 and TOTAL_LANDMARKS - 1.
    2. No intersection between right and left iris sets.
    3. No intersection between right and left eye contours.
    4. Expected total iris landmarks match NUM_IRIS_LANDMARKS.

    Returns:
        True if all checks pass.

    Raises:
        ValueError if an inconsistency is detected.
    """
    for name, seq in [
        ("RIGHT_EYE_CONTOUR", RIGHT_EYE_CONTOUR),
        ("LEFT_EYE_CONTOUR", LEFT_EYE_CONTOUR),
        ("RIGHT_IRIS_ALL", RIGHT_IRIS_ALL),
        ("LEFT_IRIS_ALL", LEFT_IRIS_ALL),
        ("FACE_OVAL_CONTOUR", FACE_OVAL_CONTOUR),
    ]:
        for idx in seq:
            if not (0 <= idx < TOTAL_LANDMARKS):
                raise ValueError(f"Index {idx} in {name} is out of range [0, {TOTAL_LANDMARKS - 1}].")

    right_iris_set = set(RIGHT_IRIS_ALL)
    left_iris_set = set(LEFT_IRIS_ALL)
    if right_iris_set.intersection(left_iris_set):
        raise ValueError("Right and Left iris index sets intersect!")

    if len(ALL_IRIS_LANDMARKS) != NUM_IRIS_LANDMARKS:
        raise ValueError(
            f"Expected {NUM_IRIS_LANDMARKS} unique iris landmarks, got {len(ALL_IRIS_LANDMARKS)}."
        )

    right_eye_set = set(RIGHT_EYE_CONTOUR)
    left_eye_set = set(LEFT_EYE_CONTOUR)
    if right_eye_set.intersection(left_eye_set):
        raise ValueError("Right and Left eye contour sets intersect!")

    return True


# Run self-validation on module load
validate_topology()
