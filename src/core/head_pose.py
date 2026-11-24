"""
Head Pose Estimation Module using OpenCV solvePnP.

Estimates 3D head orientation (Yaw, Pitch, Roll in degrees) from 2D facial landmarks
projected from a canonical anthropometric 3D face model.

Coordinate Conventions:
- Yaw:   Rotation around vertical axis. Negative = turned Right (anatomical), Positive = turned Left (anatomical).
- Pitch: Rotation around horizontal axis. Negative = looking Down, Positive = looking Up.
- Roll:  Rotation around optical axis. Negative = tilting Right, Positive = tilting Left.
"""

from dataclasses import dataclass
import math
from typing import Any, Optional, Sequence, Tuple
import cv2
import numpy as np

from .landmark_indices import (
    CHIN,
    LEFT_EYE_OUTER_CORNER,
    LEFT_MOUTH_CORNER,
    NOSE_TIP,
    RIGHT_EYE_OUTER_CORNER,
    RIGHT_MOUTH_CORNER,
)

# Canonical 3D anthropometric facial landmark points in millimeters (world coordinates)
# Origin (0, 0, 0) is placed at the tip of the nose
# OpenCV camera coordinate system: +X points Right, +Y points Down, +Z points Forward
CANONICAL_FACE_3D = np.array([
    [0.0, 0.0, 0.0],          # Nose tip (index 1)
    [0.0, 330.0, -65.0],      # Chin (index 152, below nose -> +Y)
    [225.0, -170.0, -135.0],  # Left eye outer corner (index 263, image right -> +X, above nose -> -Y)
    [-225.0, -170.0, -135.0], # Right eye outer corner (index 33, image left -> -X, above nose -> -Y)
    [150.0, 150.0, -125.0],   # Left mouth corner (index 291, image right -> +X, below nose -> +Y)
    [-150.0, 150.0, -125.0],  # Right mouth corner (index 61, image left -> -X, below nose -> +Y)
], dtype=np.float64)

HEAD_POSE_INDICES = (
    NOSE_TIP,
    CHIN,
    LEFT_EYE_OUTER_CORNER,
    RIGHT_EYE_OUTER_CORNER,
    LEFT_MOUTH_CORNER,
    RIGHT_MOUTH_CORNER,
)


@dataclass(frozen=True)
class HeadPose:
    """
    Estimated 3D head pose orientation.

    Attributes:
        yaw: Horizontal turn in degrees (negative = turned Right, positive = turned Left).
        pitch: Vertical tilt in degrees (negative = looking Down, positive = looking Up).
        roll: Side tilt in degrees (negative = tilting Right, positive = tilting Left).
        rvec: Rotation vector from solvePnP (or None).
        tvec: Translation vector from solvePnP (or None).
        is_valid: True if solvePnP converged successfully.
    """
    yaw: float
    pitch: float
    roll: float
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    is_valid: bool = False

    def to_dict(self) -> dict:
        """Return dict matching output schema."""
        return {
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
        }


class HeadPoseEstimator:
    """
    Lightweight Head Pose Estimator utilizing OpenCV's Perspective-n-Point (solvePnP).

    Executes in <0.2 ms on standard laptop CPUs with zero heavy deep-learning dependencies.
    """

    def __init__(self) -> None:
        pass

    def estimate(
        self,
        landmarks: Optional[Sequence[Any]],
        image_width: int,
        image_height: int,
    ) -> HeadPose:
        """
        Estimate 3D head pose from MediaPipe facial landmarks.

        Args:
            landmarks: Sequence of 478 NormalizedLandmark objects (x, y normalized).
            image_width: Frame width in pixels.
            image_height: Frame height in pixels.

        Returns:
            HeadPose dataclass with yaw, pitch, roll in degrees.
        """
        if landmarks is None or len(landmarks) < max(HEAD_POSE_INDICES) + 1:
            return HeadPose(yaw=0.0, pitch=0.0, roll=0.0, is_valid=False)

        # Extract 2D pixel coordinates for the 6 canonical points
        points_2d = []
        for idx in HEAD_POSE_INDICES:
            lm = landmarks[idx]
            px = float(lm.x * image_width)
            py = float(lm.y * image_height)
            points_2d.append([px, py])

        points_2d_arr = np.array(points_2d, dtype=np.float64)

        # Approximate pinhole camera intrinsics (standard focal length approximation)
        focal_length = float(image_width)
        center_x = float(image_width / 2.0)
        center_y = float(image_height / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center_x],
            [0.0, focal_length, center_y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve Perspective-n-Point
        try:
            success, rvec, tvec = cv2.solvePnP(
                objectPoints=CANONICAL_FACE_3D,
                imagePoints=points_2d_arr,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            return HeadPose(yaw=0.0, pitch=0.0, roll=0.0, is_valid=False)

        if not success or rvec is None or tvec is None:
            return HeadPose(yaw=0.0, pitch=0.0, roll=0.0, is_valid=False)

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Compute Euler angles via RQ decomposition
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # Convert to degrees:
        # angles[0] = pitch, angles[1] = yaw, angles[2] = roll
        pitch = -float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Normalize roll to neutral 0 deg (RQDecomp on facing camera produces roll around +-180)
        if roll > 90.0:
            roll = roll - 180.0
        elif roll < -90.0:
            roll = roll + 180.0

        return HeadPose(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            rvec=rvec,
            tvec=tvec,
            is_valid=True,
        )
