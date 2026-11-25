"""
OpenCV Real-Time Visualizer Module (Phase 2 Enhanced & Accuracy Improved).

Provides visual rendering for:
- Webcam RGB feed with multi-row status HUD
- Facial silhouette and optional full face mesh
- Eye contours and eye bounding boxes
- Iris contours and pinpoint center dots
- Real-time FPS, frame latency, and inference latency
- Prominent Relative Gaze Direction badge (CENTER, LEFT, RIGHT, UP, DOWN)
- Raw vs. Smoothed 2D gaze coordinates with a personalized baseline 2D gaze tracking radar
- Eye openness percentage meters (Left and Right)
- 3D Head Pose orientation indicator (Yaw, Pitch, Roll) and projected 3D axis
- Interactive Calibration countdown overlay
- Tracking state indicator (TRACKING, CALIBRATING, LOW_CONFIDENCE, etc.)
- Diagnostic data recording indicator [REC]
"""

import time
from typing import Any, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from ..config import VisualizationConfig
from ..core.eye_features import BinocularEyeFeatures, SingleEyeFeatures
from ..core.face_landmarker import FaceLandmarkerDetectionResult
from ..core.gaze_baseline import GazeBaseline
from ..core.gaze_direction import GazeDirection
from ..core.gaze_output import ScreenGazeOutput
from ..core.head_pose import CANONICAL_FACE_3D, HeadPose
from ..core.landmark_indices import (
    FACE_OVAL_CONTOUR,
    LEFT_EYE_CONTOUR,
    LEFT_IRIS_ALL,
    LEFT_IRIS_CENTER,
    NOSE_TIP,
    RIGHT_EYE_CONTOUR,
    RIGHT_IRIS_ALL,
    RIGHT_IRIS_CENTER,
)
from ..core.tracking_state import TrackingState
from ..utils.fps_counter import FPSMetrics


class CVVisualizer:
    """
    Decoupled OpenCV visualization overlay for real-time gaze & face tracking.
    """

    COLOR_EYE_CONTOUR: Tuple[int, int, int] = (255, 200, 0)       # Cyan/Sky Blue
    COLOR_IRIS_CONTOUR: Tuple[int, int, int] = (0, 255, 128)      # Spring Green
    COLOR_IRIS_CENTER: Tuple[int, int, int] = (0, 230, 255)       # Yellow
    COLOR_BBOX: Tuple[int, int, int] = (190, 190, 190)             # Silver
    COLOR_FACE_OVAL: Tuple[int, int, int] = (140, 120, 90)        # Muted Steel Blue
    COLOR_HUD_BG: Tuple[int, int, int] = (20, 20, 20)             # Dark Charcoal
    COLOR_STATUS_OK: Tuple[int, int, int] = (60, 220, 60)         # Green
    COLOR_STATUS_WARN: Tuple[int, int, int] = (50, 50, 230)       # Red
    COLOR_GAZE_CENTER: Tuple[int, int, int] = (0, 220, 220)       # Cyan/Yellow
    COLOR_GAZE_ACTIVE: Tuple[int, int, int] = (0, 165, 255)       # Bright Orange
    COLOR_GAZE_BLINK: Tuple[int, int, int] = (120, 120, 240)      # Salmon/Red

    def __init__(self, config: Optional[VisualizationConfig] = None) -> None:
        """Initialize visualizer with configuration."""
        self.config = config or VisualizationConfig()

    def render(
        self,
        frame: np.ndarray,
        detection_result: FaceLandmarkerDetectionResult,
        eye_features: BinocularEyeFeatures,
        fps_metrics: FPSMetrics,
        gaze_direction: Optional[GazeDirection] = None,
        smoothed_x: Optional[float] = None,
        smoothed_y: Optional[float] = None,
        head_pose: Optional[HeadPose] = None,
        baseline: Optional[GazeBaseline] = None,
        tracking_state: Optional[TrackingState] = None,
        calibrator_progress: Optional[float] = None,
        is_recording: bool = False,
        screen_gaze: Optional[ScreenGazeOutput] = None,
        screen_calibrator: Optional[Any] = None,
        screen_validator: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Draw tracking elements, overlays, gaze badge, radar, and performance HUD.

        Args:
            frame: Raw BGR webcam frame (original camera coordinates).
            detection_result: Landmark detection results from MediaPipe.
            eye_features: Extracted normalized eye features, openness, and ratios.
            fps_metrics: FPS and latency metrics.
            gaze_direction: Estimated relative gaze direction enum.
            smoothed_x: Smoothed horizontal gaze ratio.
            smoothed_y: Smoothed vertical gaze ratio.
            head_pose: Estimated 3D head pose.
            baseline: Calibrated neutral gaze baseline.
            tracking_state: Current tracking state enum.
            calibrator_progress: Progress of baseline calibration in [0.0, 1.0].
            is_recording: Whether diagnostic data is actively being recorded.

        Returns:
            Annotated BGR display frame (optionally mirrored for selfie view).
        """
        mirrored = self.config.mirror_display
        canvas = cv2.flip(frame, 1) if mirrored else frame.copy()
        h, w = canvas.shape[:2]

        if detection_result.has_face and detection_result.all_landmarks:
            landmarks = detection_result.all_landmarks

            # 1. Face Oval
            if len(landmarks) >= max(FACE_OVAL_CONTOUR):
                self._draw_contour(canvas, landmarks, FACE_OVAL_CONTOUR, self.COLOR_FACE_OVAL, thickness=1, w=w, h=h, mirrored=mirrored)

            # 2. Eye Contours
            if self.config.draw_eye_contours:
                self._draw_contour(canvas, landmarks, RIGHT_EYE_CONTOUR, self.COLOR_EYE_CONTOUR, thickness=1, w=w, h=h, mirrored=mirrored)
                self._draw_contour(canvas, landmarks, LEFT_EYE_CONTOUR, self.COLOR_EYE_CONTOUR, thickness=1, w=w, h=h, mirrored=mirrored)

            # 3. Eye Bounding Boxes
            if self.config.draw_bounding_boxes:
                if eye_features.right_eye and eye_features.right_eye.is_valid:
                    self._draw_bbox(canvas, eye_features.right_eye.bounding_box, w, h, label="R-EYE", mirrored=mirrored)
                if eye_features.left_eye and eye_features.left_eye.is_valid:
                    self._draw_bbox(canvas, eye_features.left_eye.bounding_box, w, h, label="L-EYE", mirrored=mirrored)

            # 4. Iris Outlines and Centers
            if self.config.draw_iris:
                self._draw_points(canvas, landmarks, RIGHT_IRIS_ALL, self.COLOR_IRIS_CONTOUR, radius=2, w=w, h=h, mirrored=mirrored)
                self._draw_points(canvas, landmarks, LEFT_IRIS_ALL, self.COLOR_IRIS_CONTOUR, radius=2, w=w, h=h, mirrored=mirrored)

            if self.config.draw_iris_center:
                if eye_features.right_eye and eye_features.right_eye.is_valid:
                    px, py = self._transform_pt(eye_features.right_eye.iris_center.x, eye_features.right_eye.iris_center.y, w, h, mirrored)
                    cv2.circle(canvas, (px, py), 4, self.COLOR_IRIS_CENTER, -1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)

                if eye_features.left_eye and eye_features.left_eye.is_valid:
                    px, py = self._transform_pt(eye_features.left_eye.iris_center.x, eye_features.left_eye.iris_center.y, w, h, mirrored)
                    cv2.circle(canvas, (px, py), 4, self.COLOR_IRIS_CENTER, -1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)

            # 5. Head Pose 3D Axes Projection
            if self.config.draw_head_pose and head_pose and head_pose.is_valid and head_pose.rvec is not None:
                self._draw_head_pose_axes(canvas, head_pose, landmarks[NOSE_TIP], w, h, mirrored=mirrored)

        # 6. Center Calibration Banner (if actively calibrating)
        if calibrator_progress is not None and calibrator_progress < 1.0:
            self._draw_calibration_overlay(canvas, calibrator_progress, w, h)

        # 6b. Phase 3: 9-Point Screen Calibration / Validation Target Overlay
        is_calib_active = bool(screen_calibrator is not None and screen_calibrator.is_active)
        is_val_active = bool(screen_validator is not None and screen_validator.is_active)
        is_screen_task_active = is_calib_active or is_val_active

        if is_calib_active:
            c_idx = getattr(screen_calibrator, "current_target_index", 0) + 1
            c_tot = max(1, len(getattr(screen_calibrator, "targets", [])))
            self._draw_screen_target_overlay(
                canvas,
                target=screen_calibrator.current_target,
                phase=screen_calibrator.target_phase.value,
                progress=screen_calibrator.target_progress,
                w=w,
                h=h,
                mirrored=mirrored,
                is_validation=False,
                target_index=c_idx,
                target_total=c_tot,
                feedback_text=getattr(screen_calibrator, "fixation_feedback", ""),
                fixation_status=getattr(screen_calibrator, "fixation_status", ""),
            )
        elif is_val_active:
            v_idx = getattr(screen_validator, "current_target_index", 0) + 1
            v_tot = max(1, len(getattr(screen_validator, "targets", [])))
            self._draw_screen_target_overlay(
                canvas,
                target=screen_validator.current_target,
                phase=screen_validator.target_phase.value,
                progress=screen_validator.progress,
                w=w,
                h=h,
                mirrored=mirrored,
                is_validation=True,
                screen_gaze=screen_gaze,
                screen_validator=screen_validator,
                target_index=v_idx,
                target_total=v_tot,
                feedback_text=getattr(screen_validator, "fixation_feedback", ""),
                fixation_status=getattr(screen_validator, "fixation_status", ""),
            )
        elif (
            (screen_validator is not None and getattr(screen_validator, "summary", None) is not None)
            or (screen_calibrator is not None and getattr(screen_calibrator, "is_completed", False))
        ):
            self._draw_post_session_results_panel(canvas, screen_calibrator, screen_validator, w, h)

        # 7. Top Information HUD (compact single-strip during active calibration/validation)
        self._draw_hud(
            canvas,
            detection_result,
            eye_features,
            fps_metrics,
            gaze_direction,
            smoothed_x,
            smoothed_y,
            head_pose,
            baseline,
            tracking_state,
            is_recording,
            screen_gaze,
            w,
            h,
            screen_calibrator=screen_calibrator,
            screen_validator=screen_validator,
        )

        # Keep bottom-corner widgets hidden during active calibration/validation so all 9 targets remain clear
        if not is_screen_task_active:
            # 8. Bottom-Right 2D Gaze Tracking Radar
            if self.config.draw_gaze_crosshair:
                raw_x = eye_features.combined_horizontal_ratio
                raw_y = eye_features.combined_vertical_ratio
                self._draw_gaze_radar(canvas, raw_x, raw_y, smoothed_x, smoothed_y, baseline, w, h)

            # 9. Bottom-Left Simulated Screen Gaze Widget (Phase 3)
            self._draw_screen_gaze_widget(canvas, screen_gaze, screen_calibrator, screen_validator, w, h)

        return canvas

    @staticmethod
    def compute_visible_content_rect(
        w: int,
        h: int,
        margin_x: Optional[int] = None,
        margin_top: Optional[int] = None,
        margin_bottom: Optional[int] = None,
    ) -> Tuple[int, int, int, int]:
        """
        Return (left, top, right, bottom) visible interface content rectangle clear of
        the compact calibration/validation top HUD (42 px) and bottom bar (24 px).
        """
        mx = max(24, int(round(w * 0.04))) if margin_x is None else max(0, int(margin_x))
        mt = max(56, int(round(h * 0.08))) if margin_top is None else max(0, int(margin_top))
        mb = max(36, int(round(h * 0.05))) if margin_bottom is None else max(0, int(margin_bottom))
        left = min(mx, max(0, w - 1))
        top = min(mt, max(0, h - 1))
        right = max(left, w - mx - 1)
        bottom = max(top, h - mb - 1)
        return left, top, right, bottom

    @staticmethod
    def interface_to_window_point(
        norm_x: float,
        norm_y: float,
        w: int,
        h: int,
        margin_x: Optional[int] = None,
        margin_top: Optional[int] = None,
        margin_bottom: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Proportionally map normalized patient-interface coordinates (norm_x, norm_y)
        in [0.0, 1.0] into the explicit visible content rectangle WITHOUT horizontal
        camera mirroring or hard clipping collapse between nearby rows.

        Convention:
        - norm_x = 0.0 -> Left boundary of visible content area
        - norm_x = 1.0 -> Right boundary of visible content area
        - norm_y = 0.0 -> Top boundary of visible content area
        - norm_y = 1.0 -> Bottom boundary of visible content area
        """
        left, top, right, bottom = CVVisualizer.compute_visible_content_rect(
            w, h, margin_x=margin_x, margin_top=margin_top, margin_bottom=margin_bottom
        )
        nx = float(np.clip(norm_x, 0.0, 1.0))
        ny = float(np.clip(norm_y, 0.0, 1.0))
        px = int(round(left + nx * float(right - left)))
        py = int(round(top + ny * float(bottom - top)))
        return px, py

    @staticmethod
    def compute_simulated_screen_rect(w: int, h: int) -> Tuple[int, int, int, int]:
        """Return (x0, y0, sw, sh) geometry for the 16:9 Simulated Screen widget."""
        sw = 170
        sh = 96
        x0 = 15
        y0 = h - sh - 32
        return x0, y0, sw, sh

    @classmethod
    def compute_simulated_screen_cursor(
        cls,
        gaze_x: float,
        gaze_y: float,
        w: int,
        h: int,
    ) -> Tuple[int, int]:
        """
        Map predicted gaze (gaze_x, gaze_y) to pixel coordinates inside the
        Simulated Screen widget (0.0 = widget left, 1.0 = widget right).
        """
        x0, y0, sw, sh = cls.compute_simulated_screen_rect(w, h)
        gx = int(np.clip(round(x0 + gaze_x * sw), x0 + 2, x0 + sw - 2))
        gy = int(np.clip(round(y0 + gaze_y * sh), y0 + 2, y0 + sh - 2))
        return gx, gy

    def _transform_pt(self, x: float, y: float, w: int, h: int, mirrored: bool) -> Tuple[int, int]:
        """
        Convert normalized camera-space (x, y) into display pixel coordinates, applying
        horizontal flip ONLY for camera pixels / facial landmarks when selfie mirror is on.
        """
        if mirrored:
            px = int(np.clip(round((1.0 - x) * w), 0, w - 1))
        else:
            px = int(np.clip(round(x * w), 0, w - 1))
        py = int(np.clip(round(y * h), 0, h - 1))
        return px, py

    def _draw_contour(
        self,
        img: np.ndarray,
        landmarks: Sequence[Any],
        indices: Sequence[int],
        color: Tuple[int, int, int],
        thickness: int,
        w: int,
        h: int,
        mirrored: bool = False,
    ) -> None:
        """Connect landmarks sequentially to draw a closed polygonal contour."""
        pts = []
        for idx in indices:
            lm = landmarks[idx]
            px, py = self._transform_pt(lm.x, lm.y, w, h, mirrored)
            pts.append([px, py])
        if len(pts) > 2:
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts_arr], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    def _draw_points(
        self,
        img: np.ndarray,
        landmarks: Sequence[Any],
        indices: Sequence[int],
        color: Tuple[int, int, int],
        radius: int,
        w: int,
        h: int,
        mirrored: bool = False,
    ) -> None:
        """Draw small solid circles at specified landmark locations."""
        for idx in indices:
            lm = landmarks[idx]
            px, py = self._transform_pt(lm.x, lm.y, w, h, mirrored)
            cv2.circle(img, (px, py), radius, color, -1, lineType=cv2.LINE_AA)

    def _draw_bbox(
        self,
        img: np.ndarray,
        bbox: Any,
        w: int,
        h: int,
        label: str,
        mirrored: bool = False,
    ) -> None:
        """Draw rectangular bounding box around an eye."""
        if mirrored:
            x1 = int(np.clip(round((1.0 - bbox.x_max) * w), 0, w - 1))
            x2 = int(np.clip(round((1.0 - bbox.x_min) * w), 0, w - 1))
        else:
            x1 = int(np.clip(round(bbox.x_min * w), 0, w - 1))
            x2 = int(np.clip(round(bbox.x_max * w), 0, w - 1))
        y1 = int(np.clip(round(bbox.y_min * h), 0, h - 1))
        y2 = int(np.clip(round(bbox.y_max * h), 0, h - 1))
        cv2.rectangle(img, (x1, y1), (x2, y2), self.COLOR_BBOX, 1, lineType=cv2.LINE_AA)

    def _draw_head_pose_axes(
        self,
        img: np.ndarray,
        pose: HeadPose,
        nose_landmark: Any,
        w: int,
        h: int,
        mirrored: bool = False,
    ) -> None:
        """Project 3D coordinate axes onto the nose tip."""
        if pose.rvec is None or pose.tvec is None:
            return

        focal = float(w)
        cam_mat = np.array([
            [focal, 0.0, float(w / 2.0)],
            [0.0, focal, float(h / 2.0)],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)

        # 3D axis points (length 80mm)
        axis_3d = np.array([
            [80.0, 0.0, 0.0],   # X axis (Right, Red)
            [0.0, 80.0, 0.0],   # Y axis (Down, Green)
            [0.0, 0.0, -80.0],  # Z axis (Forward, Blue)
        ], dtype=np.float64)

        try:
            proj_pts, _ = cv2.projectPoints(axis_3d, pose.rvec, pose.tvec, cam_mat, dist)
            p0 = self._transform_pt(nose_landmark.x, nose_landmark.y, w, h, mirrored)

            if mirrored:
                p_x = (w - 1 - int(proj_pts[0][0][0]), int(proj_pts[0][0][1]))
                p_y = (w - 1 - int(proj_pts[1][0][0]), int(proj_pts[1][0][1]))
                p_z = (w - 1 - int(proj_pts[2][0][0]), int(proj_pts[2][0][1]))
            else:
                p_x = (int(proj_pts[0][0][0]), int(proj_pts[0][0][1]))
                p_y = (int(proj_pts[1][0][0]), int(proj_pts[1][0][1]))
                p_z = (int(proj_pts[2][0][0]), int(proj_pts[2][0][1]))

            cv2.line(img, p0, p_x, (0, 0, 255), 2, lineType=cv2.LINE_AA)   # Red: X
            cv2.line(img, p0, p_y, (0, 255, 0), 2, lineType=cv2.LINE_AA)   # Green: Y
            cv2.line(img, p0, p_z, (255, 0, 0), 2, lineType=cv2.LINE_AA)   # Blue: Z
        except Exception:
            pass

    def _draw_calibration_overlay(self, img: np.ndarray, progress: float, w: int, h: int) -> None:
        """Render prominent calibration progress bar and instruction in center of screen."""
        box_w, box_h = 490, 115
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        sub = img[y1:y1 + box_h, x1:x1 + box_w]
        overlay = sub.copy()
        cv2.rectangle(sub, (0, 0), (box_w, box_h), (25, 25, 25), -1)
        cv2.addWeighted(sub, 0.82, overlay, 0.18, 0, sub)
        cv2.rectangle(img, (x1, y1), (x1 + box_w, y1 + box_h), (0, 220, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "CALIBRATING NEUTRAL BASELINE", (x1 + 35, y1 + 32), font, 0.65, (0, 240, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "Look comfortably at the camera center...", (x1 + 45, y1 + 58), font, 0.48, (210, 210, 210), 1, cv2.LINE_AA)

        # Progress bar
        bar_x1 = x1 + 40
        bar_y1 = y1 + 75
        bar_w = box_w - 80
        bar_h = 16
        cv2.rectangle(img, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * np.clip(progress, 0.0, 1.0))
        cv2.rectangle(img, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1 + bar_h), (0, 220, 255), -1)
        cv2.rectangle(img, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (120, 120, 120), 1)

    def _draw_hud(
        self,
        img: np.ndarray,
        detection: FaceLandmarkerDetectionResult,
        eyes: BinocularEyeFeatures,
        fps_metrics: FPSMetrics,
        gaze_direction: Optional[GazeDirection],
        smoothed_x: Optional[float],
        smoothed_y: Optional[float],
        head_pose: Optional[HeadPose],
        baseline: Optional[GazeBaseline],
        tracking_state: Optional[TrackingState],
        is_recording: bool,
        screen_gaze: Optional[ScreenGazeOutput],
        w: int,
        h: int,
        screen_calibrator: Optional[Any] = None,
        screen_validator: Optional[Any] = None,
    ) -> None:
        """Render high-contrast top overlay banner with real-time stats."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        is_calib_active = bool(screen_calibrator is not None and screen_calibrator.is_active)
        is_val_active = bool(screen_validator is not None and screen_validator.is_active)

        state_str = tracking_state.value if tracking_state else ("TRACKING" if detection.has_face else "NO_FACE")
        if state_str == "TRACKING":
            status_color = self.COLOR_STATUS_OK
        elif state_str == "CALIBRATING":
            status_color = (0, 220, 255)
        elif state_str in ("LOW_CONFIDENCE", "RECALIBRATION_REQUIRED"):
            status_color = (0, 165, 255)
        else:
            status_color = self.COLOR_STATUS_WARN

        # Compact HUD strip during active calibration/validation so all 9 targets remain 100% clear
        if is_calib_active or is_val_active:
            compact_h = min(42, h)
            overlay = img[:compact_h, :w].copy()
            cv2.rectangle(img, (0, 0), (w, compact_h), self.COLOR_HUD_BG, -1)
            cv2.addWeighted(img[:compact_h, :w], 0.80, overlay, 0.20, 0, img[:compact_h, :w])
            cv2.line(img, (0, compact_h), (w, compact_h), (50, 70, 90), 1)

            cv2.circle(img, (18, 21), 5, status_color, -1, lineType=cv2.LINE_AA)
            if is_calib_active:
                c_idx = getattr(screen_calibrator, "current_target_index", 0) + 1
                c_tot = max(1, len(getattr(screen_calibrator, "targets", [])))
                c_phase = screen_calibrator.target_phase.value
                task_str = f"CALIB [{c_idx}/{c_tot}] {c_phase}"
                task_color = (0, 220, 255)
            else:
                v_idx = getattr(screen_validator, "current_target_index", 0) + 1
                v_tot = max(1, len(getattr(screen_validator, "targets", [])))
                v_phase = screen_validator.target_phase.value
                task_str = f"VALID [{v_idx}/{v_tot}] {v_phase}"
                task_color = (0, 165, 255)

            cv2.putText(img, task_str, (32, 26), font, 0.48, task_color, 2, cv2.LINE_AA)

            blink_flag = bool(
                (eyes.left_eye and eyes.left_eye.is_blink) or (eyes.right_eye and eyes.right_eye.is_blink)
            )
            pose_bits = []
            if blink_flag:
                pose_bits.append("BLINK")
            if not detection.has_face:
                pose_bits.append("NO_FACE")
            if self.config.draw_head_pose and head_pose and head_pose.is_valid:
                pose_bits.append(f"Head Y:{head_pose.yaw:+.1f} P:{head_pose.pitch:+.1f}")
            mid_x = max(250, w // 3)
            if pose_bits:
                mid_color = (0, 255, 255) if (blink_flag or not detection.has_face) else (180, 220, 240)
                cv2.putText(img, " | ".join(pose_bits), (mid_x, 26), font, 0.42, mid_color, 1, cv2.LINE_AA)

            cam_tag = "MIRROR" if self.config.mirror_display else "RAW"
            fps_tag = f" | {fps_metrics.fps:.0f}FPS" if self.config.draw_fps else ""
            rec_tag = " | REC" if is_recording else ""
            right_txt = f"{cam_tag}{fps_tag}{rec_tag}"
            cv2.putText(img, right_txt, (max(mid_x + 170, w - 210), 26), font, 0.40, (200, 220, 220), 1, cv2.LINE_AA)

            # Bottom compact status/controls strip clear of content rect
            bar_top = max(compact_h + 1, h - 24)
            sub_b = img[bar_top:h, :w].copy()
            cv2.rectangle(img, (0, bar_top), (w, h), self.COLOR_HUD_BG, -1)
            cv2.addWeighted(img[bar_top:h, :w], 0.80, sub_b, 0.20, 0, img[bar_top:h, :w])

            if is_val_active and hasattr(screen_validator, "get_live_metrics"):
                m = screen_validator.get_live_metrics()
                bottom_str = (
                    f"Acc:{m.current_accepted_samples} Rej:{m.current_rejected_samples} | "
                    f"Valid:{m.valid_trials}/{m.total_trials} | Mean:{m.running_mean_error_px:.0f}px | "
                    f"Hit:{m.pct_target_selection:.0f}% ({m.successful_selections}/{m.valid_trials}) | "
                    "Keys: 'y'=Retry 'k'=Skip 'q'=Exit"
                )
            else:
                cur_t = getattr(screen_calibrator, "current_target", None) if is_calib_active else None
                cur_cnt = len(screen_calibrator._target_samples.get(cur_t.target_id, [])) if (cur_t is not None and hasattr(screen_calibrator, "_target_samples")) else 0
                min_cnt = getattr(screen_calibrator, "min_samples_per_target", 0) if is_calib_active else 0
                bottom_str = (
                    f"Samples:{cur_cnt}/{min_cnt} | Look steadily at the active target center | "
                    "Keys: 'y'=Retry 'c'=Re-center 'q'=Exit"
                )
            cv2.putText(img, bottom_str, (16, h - 7), font, 0.38, (185, 210, 225), 1, cv2.LINE_AA)
            return

        banner_h = 118
        overlay = img[:banner_h, :w].copy()
        cv2.rectangle(img, (0, 0), (w, banner_h), self.COLOR_HUD_BG, -1)
        cv2.addWeighted(img[:banner_h, :w], 0.78, overlay, 0.22, 0, img[:banner_h, :w])

        # Row 1: Detection Status & Tracking State pill
        cv2.circle(img, (20, 22), 6, status_color, -1, lineType=cv2.LINE_AA)
        status_text = f"STATE: [{state_str}]"
        cv2.putText(img, status_text, (35, 27), font, 0.52, status_color, 2, cv2.LINE_AA)

        # Explicit visual debug indicator for camera display mode (independent of draw_fps)
        cam_mode_label = "CAM: MIRRORED" if self.config.mirror_display else "CAM: NON-MIRRORED"
        cam_mode_color = (0, 230, 200) if self.config.mirror_display else (0, 165, 255)
        cv2.putText(img, cam_mode_label, (560, 27), font, 0.45, cam_mode_color, 1, cv2.LINE_AA)

        if self.config.draw_fps:
            fps_str = f"FPS: {fps_metrics.fps:.1f} | Latency: {detection.inference_time_ms:.1f}ms"
            cv2.putText(img, fps_str, (240, 27), font, 0.46, (220, 220, 220), 1, cv2.LINE_AA)

        if is_recording:
            cv2.circle(img, (w - 380, 22), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(img, "REC", (w - 370, 27), font, 0.50, (0, 0, 255), 2, cv2.LINE_AA)

        if self.config.draw_head_pose and head_pose and head_pose.is_valid:
            pose_str = f"Head: Y:{head_pose.yaw:+.1f}deg  P:{head_pose.pitch:+.1f}deg  R:{head_pose.roll:+.1f}deg"
            cv2.putText(img, pose_str, (w - 325, 27), font, 0.46, (180, 220, 240), 1, cv2.LINE_AA)

        # Row 2: Relative Gaze Direction Badge
        dir_val = gaze_direction.value if gaze_direction else "UNKNOWN"
        badge_text = f"GAZE: [ {dir_val} ]"
        if dir_val == "CENTER":
            badge_color = self.COLOR_GAZE_CENTER
        elif "LEFT" in dir_val or "RIGHT" in dir_val or "UP" in dir_val or "DOWN" in dir_val:
            badge_color = self.COLOR_GAZE_ACTIVE
        else:
            badge_color = (160, 160, 160)

        cv2.putText(img, badge_text, (20, 56), font, 0.65, badge_color, 2, cv2.LINE_AA)

        # Eye Openness & Workflow Status on Row 2
        if self.config.draw_eye_openness and eyes.left_eye and eyes.right_eye:
            l_op = int(eyes.left_eye.openness * 100)
            r_op = int(eyes.right_eye.openness * 100)
            blink_flag = " [BLINK]" if (eyes.left_eye.is_blink or eyes.right_eye.is_blink) else ""
            open_str = f"Open: L:{l_op}% ({eyes.left_eye.openness_state.value}) R:{r_op}% ({eyes.right_eye.openness_state.value}){blink_flag}"
            cv2.putText(img, open_str, (240, 56), font, 0.44, (0, 255, 255) if blink_flag else (200, 240, 200), 1, cv2.LINE_AA)

        if screen_calibrator is not None and screen_calibrator.is_active:
            c_idx = getattr(screen_calibrator, "current_target_index", 0) + 1
            c_tot = max(1, len(getattr(screen_calibrator, "targets", [])))
            c_stat = f"CALIB: Target {c_idx}/{c_tot} [{screen_calibrator.target_phase.value}]"
            cv2.putText(img, c_stat, (w - 340, 56), font, 0.45, (0, 215, 255), 1, cv2.LINE_AA)
        elif screen_validator is not None and screen_validator.is_active:
            v_idx = getattr(screen_validator, "current_target_index", 0) + 1
            v_tot = max(1, len(getattr(screen_validator, "targets", [])))
            v_stat = f"VALID: Target {v_idx}/{v_tot} [{screen_validator.target_phase.value}]"
            cv2.putText(img, v_stat, (w - 340, 56), font, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
        elif screen_validator is not None and getattr(screen_validator, "summary", None) is not None:
            vs = screen_validator.summary
            v_stat = f"VALIDATED: Hit {vs.target_selection_accuracy_pct:.0f}% | Mean {vs.mean_error_pixels:.0f}px"
            cv2.putText(img, v_stat, (w - 340, 56), font, 0.43, (0, 255, 128), 1, cv2.LINE_AA)
        elif screen_calibrator is not None and getattr(screen_calibrator, "is_completed", False):
            cv2.putText(img, "CALIB: DONE ('t' = Validate)", (w - 340, 56), font, 0.43, (0, 255, 200), 1, cv2.LINE_AA)

        # Row 3: Raw vs. Smoothed Gaze Coordinates & Baseline Deviations
        raw_x_str = f"{eyes.combined_horizontal_ratio:.2f}" if eyes.combined_horizontal_ratio is not None else "--"
        raw_y_str = f"{eyes.combined_vertical_ratio:.2f}" if eyes.combined_vertical_ratio is not None else "--"
        sm_x_str = f"{smoothed_x:.2f}" if smoothed_x is not None else "--"
        sm_y_str = f"{smoothed_y:.2f}" if smoothed_y is not None else "--"

        base_str = f"Base:({baseline.neutral_x:.2f},{baseline.neutral_y:.2f})" if (baseline and baseline.is_calibrated) else "Base:(Uncalibrated)"
        delta_x = (smoothed_x - baseline.neutral_x) if (smoothed_x is not None and baseline and baseline.is_calibrated) else None
        delta_y = (smoothed_y - baseline.neutral_y) if (smoothed_y is not None and baseline and baseline.is_calibrated) else None
        d_str = f"d:({delta_x:+.2f},{delta_y:+.2f})" if (delta_x is not None and delta_y is not None) else "d:(----,----)"

        l_vy_str = f"{eyes.left_eye.vertical_ratio:.2f}" if (eyes.left_eye and eyes.left_eye.vertical_ratio is not None) else "--"
        r_vy_str = f"{eyes.right_eye.vertical_ratio:.2f}" if (eyes.right_eye and eyes.right_eye.vertical_ratio is not None) else "--"

        coord_str = (
            f"Ocular: H:{sm_x_str}({raw_x_str}) V:{sm_y_str}({raw_y_str}) [L-V:{l_vy_str},R-V:{r_vy_str}] | "
            f"{base_str} {d_str} | Conf:{int(eyes.confidence * 100)}%"
        )
        cv2.putText(img, coord_str, (20, 84), font, 0.42, (210, 210, 210), 1, cv2.LINE_AA)

        # Row 4: Phase 3 Screen Gaze Coordinates & Simulated Screen Cursor Debug Readout
        if screen_gaze is not None and screen_gaze.is_calibrated and screen_gaze.gaze_x is not None and screen_gaze.gaze_y is not None:
            sim_gx, sim_gy = self.compute_simulated_screen_cursor(screen_gaze.gaze_x, screen_gaze.gaze_y, w, h)
            x0, y0, sw, sh = self.compute_simulated_screen_rect(w, h)
            screen_str = (
                f"SCREEN GAZE: ({screen_gaze.gaze_x:.3f}, {screen_gaze.gaze_y:.3f}) | "
                f"Px: ({screen_gaze.pixel_x:4d}, {screen_gaze.pixel_y:4d}) | "
                f"SimCursor: ({sim_gx - x0:3d}/{sw}, {sim_gy - y0:2d}/{sh}) | [{screen_gaze.tracking_quality}]"
            )
            cv2.putText(img, screen_str, (20, 108), font, 0.44, (0, 255, 128), 1, cv2.LINE_AA)
        elif screen_gaze is not None and screen_gaze.is_calibrated:
            cv2.putText(img, "SCREEN GAZE: [TRACKING PAUSED - BLINK / OCCLUSION]", (20, 108), font, 0.46, (0, 165, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "SCREEN GAZE: [UNCALIBRATED] (Press 's' for 9-Point Calibration + Validation)", (20, 108), font, 0.46, (140, 180, 220), 1, cv2.LINE_AA)

        # Bottom controls footer
        footer_text = "Controls: [ 's'=Calib+Val | 't'=Validate | 'y'=Retry | 'k'=Skip | 'c'=Re-center | 'p'=Save | 'q'=Exit ]"
        cv2.putText(img, footer_text, (20, h - 14), font, 0.39, (150, 150, 150), 1, cv2.LINE_AA)

    def _draw_screen_target_overlay(
        self,
        img: np.ndarray,
        target: Optional[Any],
        phase: str,
        progress: float,
        w: int,
        h: int,
        mirrored: bool = False,
        is_validation: bool = False,
        screen_gaze: Optional[ScreenGazeOutput] = None,
        screen_validator: Optional[Any] = None,
        target_index: int = 1,
        target_total: int = 9,
        feedback_text: str = "",
        fixation_status: str = "",
    ) -> None:
        """
        Render prominent on-screen bullseye, predicted gaze marker, error vector,
        progress count (Target i/N), and fixation stability feedback for the active target.
        """
        if target is None:
            return

        _ = mirrored  # Display mirroring applies ONLY to camera frames & facial landmarks.
        left, top, right, bottom = self.compute_visible_content_rect(w, h)
        tx, ty = self.interface_to_window_point(target.x, target.y, w, h)

        # Subtle visible interface boundary frame so user can see active workspace bounds cleanly
        cv2.rectangle(img, (left, top), (right, bottom), (55, 65, 75), 1)

        if phase == "FAILED":
            color_outer = (0, 0, 255)
            color_inner = (0, 80, 255)
        else:
            color_outer = (0, 140, 255) if is_validation else (0, 215, 255)
            color_inner = (0, 255, 255) if is_validation else (0, 255, 128)

        # Render exact actual target boundary matching ScreenValidator geometry inside visible content area
        screen_w = getattr(screen_validator, "screen_width", 1920) if screen_validator is not None else 1920
        screen_h = getattr(screen_validator, "screen_height", 1080) if screen_validator is not None else 1080
        content_w = max(1, right - left)
        content_h = max(1, bottom - top)
        scale_x = float(content_w) / float(max(1, screen_w))
        scale_y = float(content_h) / float(max(1, screen_h))

        t_width_px = getattr(target, "width_px", None)
        t_height_px = getattr(target, "height_px", None)
        t_radius_px = getattr(target, "radius_px", None)
        cfg_radius_px = getattr(screen_validator, "target_radius_px", None) if screen_validator is not None else None

        if t_width_px is not None and t_height_px is not None:
            half_w = int(max(4, round((t_width_px * scale_x) / 2.0)))
            half_h = int(max(4, round((t_height_px * scale_y) / 2.0)))
            cv2.rectangle(img, (tx - half_w, ty - half_h), (tx + half_w, ty + half_h), color_outer, 2, lineType=cv2.LINE_AA)
            radius_pulse = max(half_w, half_h)
        elif t_radius_px is not None or (is_validation and cfg_radius_px is not None):
            eff_r = t_radius_px if t_radius_px is not None else cfg_radius_px
            bound_r = int(max(4, round(float(eff_r) * min(scale_x, scale_y))))
            cv2.circle(img, (tx, ty), bound_r, color_outer, 2, lineType=cv2.LINE_AA)
            radius_pulse = bound_r
        else:
            # Default rendered OpenCV circle (static boundary radius = 22 display px)
            cv2.circle(img, (tx, ty), 22, (0, 200, 255) if is_validation else color_outer, 1, lineType=cv2.LINE_AA)
            radius_pulse = int(22 + 6 * np.sin(time.time() * 6.0))
            cv2.circle(img, (tx, ty), radius_pulse, color_outer, 2, lineType=cv2.LINE_AA)

        # Progress sector / center bullseye
        cv2.circle(img, (tx, ty), 14, (40, 40, 40), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (tx, ty), 14, color_outer, 2, lineType=cv2.LINE_AA)
        cv2.circle(img, (tx, ty), 6, color_inner, -1, lineType=cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # If in validation mode, render predicted gaze reticle and error vector inside same content area
        if is_validation and screen_gaze is not None and screen_gaze.gaze_x is not None and screen_gaze.gaze_y is not None:
            gx, gy = self.interface_to_window_point(screen_gaze.gaze_x, screen_gaze.gaze_y, w, h)

            # Predicted gaze marker (orange diamond + cross)
            cv2.drawMarker(img, (gx, gy), (0, 165, 255), markerType=cv2.MARKER_DIAMOND, markerSize=16, thickness=2, line_type=cv2.LINE_AA)
            cv2.circle(img, (gx, gy), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

            # Error vector connecting actual target to predicted gaze
            cv2.arrowedLine(img, (tx, ty), (gx, gy), (0, 80, 255), 2, tipLength=0.20, line_type=cv2.LINE_AA)

            # Error distance calculation in screen coordinates
            dist_norm = float(np.hypot(screen_gaze.gaze_x - target.x, screen_gaze.gaze_y - target.y))
            dist_px = int(np.hypot(
                (screen_gaze.gaze_x - target.x) * float(screen_w),
                (screen_gaze.gaze_y - target.y) * float(screen_h),
            ))
            mx = (tx + gx) // 2
            my = (ty + gy) // 2
            cv2.putText(img, f"Err: {dist_px}px ({dist_norm:.3f})", (mx + 8, my - 6), font, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        # Target progress & phase banner with HUD-safe smart placement above/below target
        title_prefix = "VALIDATION" if is_validation else "CALIBRATION"
        label = f"{title_prefix} Target {target_index}/{target_total}: {target.name.upper()} [{phase}]"
        label_x = int(np.clip(tx - 135, 16, max(16, w - 340)))
        prefer_below = (ty - radius_pulse - 26) < top
        near_bottom = (ty + radius_pulse + 42) > bottom
        if prefer_below:
            label_y = ty + radius_pulse + 20
            fb_y = ty + radius_pulse + 38
        elif near_bottom:
            fb_y = ty - radius_pulse - 10
            label_y = ty - radius_pulse - 28
        else:
            label_y = ty - radius_pulse - 10
            fb_y = ty + radius_pulse + 22
        cv2.putText(img, label, (label_x, label_y), font, 0.44, color_outer, 1, cv2.LINE_AA)

        if feedback_text:
            fb_color = (
                (0, 255, 128)
                if fixation_status == "FIXATION_STEADY"
                else ((0, 80, 255) if phase == "FAILED" else (0, 215, 255))
            )
            cv2.putText(img, feedback_text, (label_x, fb_y), font, 0.39, fb_color, 1, cv2.LINE_AA)

    def _draw_validation_dashboard(
        self,
        img: np.ndarray,
        screen_validator: Any,
        w: int,
        h: int,
    ) -> None:
        """Render dedicated live metrics dashboard panel during active validation experiment."""
        if not hasattr(screen_validator, "get_live_metrics"):
            return

        metrics = screen_validator.get_live_metrics()

        pw = min(375, max(100, w - 30))
        ph = 152
        x0 = 15
        y0 = 135  # Positioned cleanly below HUD Row 4
        if y0 + ph >= h or x0 + pw >= w:
            return

        # Semi-transparent background
        sub = img[y0:y0 + ph, x0:x0 + pw]
        overlay = sub.copy()
        cv2.rectangle(sub, (0, 0), (pw, ph), (18, 22, 28), -1)
        cv2.addWeighted(sub, 0.84, overlay, 0.16, 0, sub)
        cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (0, 165, 255), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Header
        cv2.putText(img, "LIVE ACCURACY & ERROR DASHBOARD", (x0 + 8, y0 + 16), font, 0.38, (0, 215, 255), 1, cv2.LINE_AA)
        trial_str = f"Trial [{metrics.current_trial_id}/{metrics.total_trials}]: {metrics.current_target_name}"
        cv2.putText(img, trial_str, (x0 + 8, y0 + 34), font, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

        # Current trial error & sample counts
        cur_err_str = f"Err: {metrics.current_error_px:.1f}px ({metrics.current_error_norm:.3f})" if metrics.current_error_px is not None else "Err: --"
        samp_str = f"Samples: {metrics.current_accepted_samples} [Acc] | {metrics.current_rejected_samples} [Rej]"
        cv2.putText(img, f"{cur_err_str} | {samp_str}", (x0 + 8, y0 + 52), font, 0.31, (0, 230, 255), 1, cv2.LINE_AA)

        # Running error metrics
        run_mean_str = f"Mean Err: {metrics.running_mean_error_px:.1f}px ({metrics.running_mean_error_norm:.3f}) | Med: {metrics.running_median_error_px:.1f}px"
        cv2.putText(img, run_mean_str, (x0 + 8, y0 + 72), font, 0.32, (220, 220, 220), 1, cv2.LINE_AA)

        mae_str = f"MAE: X:{metrics.running_mae_x_px:.1f} Y:{metrics.running_mae_y_px:.1f}px | RMSE: ({metrics.running_rmse_x_px:.1f}, {metrics.running_rmse_y_px:.1f})px"
        cv2.putText(img, mae_str, (x0 + 8, y0 + 91), font, 0.30, (180, 180, 180), 1, cv2.LINE_AA)

        # Accuracy / Success rates (Thresholds + Target Hits)
        t2_lbl = int(round(getattr(screen_validator, "threshold_2pct", 0.02) * 100))
        t5_lbl = int(round(getattr(screen_validator, "threshold_5pct", 0.05) * 100))
        t10_lbl = int(round(getattr(screen_validator, "threshold_10pct", 0.10) * 100))
        thresh_str = f"Acc <={t2_lbl}%:{metrics.pct_within_2pct:.0f}% | <={t5_lbl}%:{metrics.pct_within_5pct:.0f}% | <={t10_lbl}%:{metrics.pct_within_10pct:.0f}%"
        cv2.putText(img, thresh_str, (x0 + 8, y0 + 111), font, 0.31, (0, 255, 128), 1, cv2.LINE_AA)

        hit_pct = getattr(metrics, "pct_target_selection", 0.0)
        hits = getattr(metrics, "successful_selections", 0)
        hit_str = f"Target Hit Acc: {hit_pct:.0f}% ({hits}/{metrics.valid_trials}) | Valid:{metrics.valid_trials} Rej:{metrics.rejected_trials}"
        cv2.putText(img, hit_str, (x0 + 8, y0 + 130), font, 0.31, (0, 255, 200), 1, cv2.LINE_AA)

        cv2.putText(img, f"Phase: {metrics.current_phase}", (x0 + 8, y0 + 146), font, 0.28, (160, 160, 160), 1, cv2.LINE_AA)

    def _draw_post_session_results_panel(
        self,
        img: np.ndarray,
        screen_calibrator: Optional[Any],
        screen_validator: Optional[Any],
        w: int,
        h: int,
    ) -> None:
        """
        Render a prominent accuracy & validation status card on the CV window
        whenever calibration or validation has concluded (one single source of truth).
        """
        box_w = min(380, max(240, w - 30))
        box_h = 128
        x0 = w - box_w - 15
        y0 = 126
        if x0 < 0 or y0 + box_h > h - 20:
            return

        sub = img[y0:y0 + box_h, x0:x0 + box_w]
        overlay = sub.copy()
        cv2.rectangle(sub, (0, 0), (box_w, box_h), (18, 22, 30), -1)
        cv2.addWeighted(sub, 0.80, overlay, 0.20, 0, sub)

        font = cv2.FONT_HERSHEY_SIMPLEX
        summary = getattr(screen_validator, "summary", None) if screen_validator is not None else None

        if summary is not None and getattr(summary, "successful_trials", 0) > 0:
            cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 210, 120), 1)
            val_st = getattr(summary, "validation_status", "COMPLETED")
            cv2.putText(img, f"RESULTS SUMMARY | Status: [{val_st}]", (x0 + 10, y0 + 20), font, 0.42, (0, 255, 160), 1, cv2.LINE_AA)
            cv2.putText(
                img,
                f"Target Hit Accuracy: {summary.target_selection_accuracy_pct:.1f}% ({summary.successful_selections}/{summary.successful_trials})",
                (x0 + 10, y0 + 42),
                font,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"Within 2%: {summary.pct_within_2pct:.1f}% | 5%: {summary.pct_within_5pct:.1f}% | 10%: {summary.pct_within_10pct:.1f}%",
                (x0 + 10, y0 + 64),
                font,
                0.38,
                (140, 240, 190),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"Mean Error: {summary.mean_error_pixels:.1f} px ({summary.mean_error_pct_diagonal:.1f}% diag)",
                (x0 + 10, y0 + 86),
                font,
                0.40,
                (0, 230, 255),
                1,
                cv2.LINE_AA,
            )
            patt = getattr(summary, "pattern_analysis", None) or {}
            w_dir = (patt.get("weakest_direction") or {}).get("direction") if isinstance(patt, dict) else None
            extra_s = f"Weakest Dir: {w_dir}" if w_dir else f"Model: {summary.model_name.upper()}"
            cv2.putText(img, extra_s, (x0 + 10, y0 + 108), font, 0.36, (190, 210, 230), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 165, 255), 1)
            fail_r = getattr(screen_validator, "failure_reason", None) if screen_validator is not None else None
            st_lbl = "FAILED" if fail_r else "NOT_STARTED"
            cv2.putText(img, f"RESULTS SUMMARY | Status: [{st_lbl}]", (x0 + 10, y0 + 22), font, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "Target Hit: N/A | 2%/5%/10%: N/A | Mean: N/A", (x0 + 10, y0 + 48), font, 0.37, (220, 220, 220), 1, cv2.LINE_AA)
            msg = fail_r[:48] if fail_r else "Validation not completed yet."
            cv2.putText(img, msg, (x0 + 10, y0 + 74), font, 0.36, (140, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "Next: Press 't' to measure real accuracy", (x0 + 10, y0 + 100), font, 0.38, (0, 255, 200), 1, cv2.LINE_AA)

    def _draw_screen_gaze_widget(
        self,
        img: np.ndarray,
        screen_gaze: Optional[ScreenGazeOutput],
        screen_calibrator: Optional[Any],
        screen_validator: Optional[Any],
        w: int,
        h: int,
    ) -> None:
        """
        Render a 16:9 simulated display monitor widget depicting calibrated screen gaze (Phase 3).

        Coordinate convention inside the Simulated Screen widget:
        - gaze_x = 0.0 -> Left edge ('L')
        - gaze_x = 1.0 -> Right edge ('R')
        - gaze_y = 0.0 -> Top edge
        - gaze_y = 1.0 -> Bottom edge
        Never flipped by mirror_display.
        """
        x0, y0, sw, sh = self.compute_simulated_screen_rect(w, h)

        # Semi-transparent background
        sub = img[y0:y0 + sh, x0:x0 + sw]
        overlay = sub.copy()
        cv2.rectangle(sub, (0, 0), (sw, sh), (20, 20, 25), -1)
        cv2.addWeighted(sub, 0.75, overlay, 0.25, 0, sub)
        cv2.rectangle(img, (x0, y0), (x0 + sw, y0 + sh), (140, 140, 160), 1)

        # Subtle center axes and L/R horizontal orientation markers
        cv2.line(img, (x0 + sw // 2, y0 + 16), (x0 + sw // 2, y0 + sh - 15), (55, 55, 70), 1)
        cv2.line(img, (x0 + 12, y0 + sh // 2), (x0 + sw - 12, y0 + sh // 2), (55, 55, 70), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Simulated Screen", (x0 + 4, y0 + 12), font, 0.32, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(img, "L", (x0 + 3, y0 + sh // 2 + 3), font, 0.28, (130, 130, 150), 1, cv2.LINE_AA)
        cv2.putText(img, "R", (x0 + sw - 11, y0 + sh // 2 + 3), font, 0.28, (130, 130, 150), 1, cv2.LINE_AA)

        # 1. If 9-point screen calibration is active, draw calibration targets
        if screen_calibrator is not None and screen_calibrator.is_active:
            for t in screen_calibrator.targets:
                tx, ty = self.compute_simulated_screen_cursor(t.x, t.y, w, h)
                is_current = (screen_calibrator.current_target and t.target_id == screen_calibrator.current_target.target_id)
                if is_current:
                    cv2.circle(img, (tx, ty), 6, (0, 215, 255), 2, lineType=cv2.LINE_AA)
                    cv2.circle(img, (tx, ty), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                    lbl = f"{t.name[:6]}"
                    cv2.putText(img, lbl, (x0 + 4, y0 + sh - 5), font, 0.30, (0, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(img, (tx, ty), 2, (100, 100, 100), -1, lineType=cv2.LINE_AA)
            return

        # 2. If validation is active, draw validation targets and live gaze tracking
        if screen_validator is not None and screen_validator.is_active:
            val_targets = screen_validator.base_targets if hasattr(screen_validator, "base_targets") else screen_validator.targets
            for vt in val_targets:
                vtx, vty = self.compute_simulated_screen_cursor(vt.x, vt.y, w, h)
                cv2.circle(img, (vtx, vty), 3, (90, 90, 100), 1, lineType=cv2.LINE_AA)

            # Draw completed trials with color-coded error dots
            if hasattr(screen_validator, "get_trial_records"):
                for tr in screen_validator.get_trial_records():
                    if tr.trial_status == "VALID" and tr.actual_x is not None and tr.actual_y is not None:
                        ptx, pty = self.compute_simulated_screen_cursor(tr.actual_x, tr.actual_y, w, h)
                        if tr.euclidean_error_norm is not None:
                            if tr.euclidean_error_norm <= 0.05:
                                dot_c = (0, 255, 128)  # Green (<= 5% diagonal)
                            elif tr.euclidean_error_norm <= 0.10:
                                dot_c = (0, 200, 255)  # Orange (<= 10% diagonal)
                            else:
                                dot_c = (0, 60, 255)   # Red (> 10% diagonal)
                            cv2.circle(img, (ptx, pty), 3, dot_c, -1, lineType=cv2.LINE_AA)

            # Draw active target
            if screen_validator.current_target:
                cur_t = screen_validator.current_target
                atx, aty = self.compute_simulated_screen_cursor(cur_t.x, cur_t.y, w, h)
                cv2.circle(img, (atx, aty), 6, (0, 140, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(img, (atx, aty), 2, (0, 255, 255), -1, lineType=cv2.LINE_AA)

                # Draw predicted gaze cursor and error vector inside simulated screen
                if screen_gaze is not None and screen_gaze.gaze_x is not None and screen_gaze.gaze_y is not None:
                    gx, gy = self.compute_simulated_screen_cursor(screen_gaze.gaze_x, screen_gaze.gaze_y, w, h)
                    cv2.line(img, (atx, aty), (gx, gy), (0, 160, 255), 1, cv2.LINE_AA)
                    cv2.circle(img, (gx, gy), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)

                lbl = f"VAL: {cur_t.name[:6]}"
                cv2.putText(img, lbl, (x0 + 4, y0 + sh - 5), font, 0.30, (0, 200, 255), 1, cv2.LINE_AA)
            return

        # 3. Draw screen gaze cursor if calibrated and predicted
        if screen_gaze is not None and screen_gaze.is_calibrated and screen_gaze.gaze_x is not None and screen_gaze.gaze_y is not None:
            gx, gy = self.compute_simulated_screen_cursor(screen_gaze.gaze_x, screen_gaze.gaze_y, w, h)

            cv2.circle(img, (gx, gy), 6, (0, 255, 128), 1, lineType=cv2.LINE_AA)
            cv2.circle(img, (gx, gy), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.line(img, (gx - 5, gy), (gx + 5, gy), (0, 255, 128), 1)
            cv2.line(img, (gx, gy - 5), (gx, gy + 5), (0, 255, 128), 1)

            coord_txt = f"({screen_gaze.gaze_x:.2f},{screen_gaze.gaze_y:.2f})"
            px_txt = f"c:({gx - x0},{gy - y0})"
            cv2.putText(img, coord_txt, (x0 + 4, y0 + sh - 5), font, 0.28, (0, 255, 128), 1, cv2.LINE_AA)
            cv2.putText(img, px_txt, (x0 + sw - 62, y0 + sh - 5), font, 0.28, (180, 240, 200), 1, cv2.LINE_AA)
        elif screen_gaze is not None and not screen_gaze.is_calibrated:
            cv2.putText(img, "Uncalibrated ('s')", (x0 + 6, y0 + sh // 2 + 4), font, 0.32, (120, 120, 120), 1, cv2.LINE_AA)

    def _draw_gaze_radar(
        self,
        img: np.ndarray,
        raw_x: Optional[float],
        raw_y: Optional[float],
        smoothed_x: Optional[float],
        smoothed_y: Optional[float],
        baseline: Optional[GazeBaseline],
        w: int,
        h: int,
    ) -> None:
        """Render a 2D normalized gaze radar widget and dedicated vertical gaze meter."""
        radar_size = 110
        margin = 15
        x0 = w - radar_size - margin
        y0 = h - radar_size - margin

        # Draw semi-transparent background for 2D radar
        sub = img[y0:y0 + radar_size, x0:x0 + radar_size]
        overlay = sub.copy()
        cv2.rectangle(sub, (0, 0), (radar_size, radar_size), (30, 30, 30), -1)
        cv2.addWeighted(sub, 0.75, overlay, 0.25, 0, sub)
        cv2.rectangle(img, (x0, y0), (x0 + radar_size, y0 + radar_size), (120, 120, 120), 1)

        # Center crosshair: calibrated baseline or default (0.50, 0.50)
        bx = baseline.neutral_x if (baseline and baseline.is_calibrated) else 0.50
        by = baseline.neutral_y if (baseline and baseline.is_calibrated) else 0.50

        cx = int(x0 + bx * radar_size)
        cy = int(y0 + by * radar_size)
        cv2.line(img, (cx, y0 + 6), (cx, y0 + radar_size - 6), (70, 70, 70), 1)
        cv2.line(img, (x0 + 6, cy), (x0 + radar_size - 6, cy), (70, 70, 70), 1)

        # Neutral deadzone box (+- 0.06 horizontally, +- 0.06 vertically around baseline)
        h_dead = 0.06
        v_dead = 0.06
        bx1 = int(np.clip(x0 + (bx - h_dead) * radar_size, x0 + 2, x0 + radar_size - 2))
        bx2 = int(np.clip(x0 + (bx + h_dead) * radar_size, x0 + 2, x0 + radar_size - 2))
        by1 = int(np.clip(y0 + (by - v_dead) * radar_size, y0 + 2, y0 + radar_size - 2))
        by2 = int(np.clip(y0 + (by + v_dead) * radar_size, y0 + 2, y0 + radar_size - 2))
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (75, 75, 95), 1)

        # Plot Raw Gaze (small red dot)
        if raw_x is not None and raw_y is not None:
            rx = int(np.clip(x0 + raw_x * radar_size, x0 + 2, x0 + radar_size - 2))
            ry = int(np.clip(y0 + raw_y * radar_size, y0 + 2, y0 + radar_size - 2))
            cv2.circle(img, (rx, ry), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        # Plot Smoothed Gaze (green circle with yellow center dot)
        if smoothed_x is not None and smoothed_y is not None:
            sx = int(np.clip(x0 + smoothed_x * radar_size, x0 + 2, x0 + radar_size - 2))
            sy = int(np.clip(y0 + smoothed_y * radar_size, y0 + 2, y0 + radar_size - 2))
            cv2.circle(img, (sx, sy), 5, (0, 255, 128), 1, lineType=cv2.LINE_AA)
            cv2.circle(img, (sx, sy), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

        # 2D Radar Title
        cv2.putText(img, "Gaze Radar", (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # Dedicated Vertical Gaze Indicator Meter (Adjacent to 2D Radar)
        # ----------------------------------------------------------------------
        vm_w = 26
        vm_x = x0 - vm_w - 8
        vm_y = y0

        # Semi-transparent background
        v_sub = img[vm_y:vm_y + radar_size, vm_x:vm_x + vm_w]
        v_overlay = v_sub.copy()
        cv2.rectangle(v_sub, (0, 0), (vm_w, radar_size), (25, 25, 25), -1)
        cv2.addWeighted(v_sub, 0.75, v_overlay, 0.25, 0, v_sub)
        cv2.rectangle(img, (vm_x, vm_y), (vm_x + vm_w, vm_y + radar_size), (120, 120, 120), 1)

        # Neutral baseline tick
        v_cy = int(np.clip(vm_y + by * radar_size, vm_y + 4, vm_y + radar_size - 4))
        cv2.line(img, (vm_x + 2, v_cy), (vm_x + vm_w - 2, v_cy), (0, 220, 220), 1)

        # UP threshold tick (-0.045)
        up_y = int(np.clip(vm_y + (by - 0.045) * radar_size, vm_y + 4, vm_y + radar_size - 4))
        cv2.line(img, (vm_x + 3, up_y), (vm_x + vm_w - 3, up_y), (0, 165, 255), 1)

        # DOWN threshold tick (+0.060)
        dn_y = int(np.clip(vm_y + (by + 0.060) * radar_size, vm_y + 4, vm_y + radar_size - 4))
        cv2.line(img, (vm_x + 3, dn_y), (vm_x + vm_w - 3, dn_y), (0, 165, 255), 1)

        # Real-time vertical cursor
        val_y = smoothed_y if smoothed_y is not None else raw_y
        if val_y is not None:
            cur_y = int(np.clip(vm_y + val_y * radar_size, vm_y + 4, vm_y + radar_size - 4))
            d_y_val = val_y - by
            if d_y_val < -0.045:
                cur_color = self.COLOR_GAZE_ACTIVE  # Looking UP (Orange)
            elif d_y_val > 0.060:
                cur_color = self.COLOR_GAZE_ACTIVE  # Looking DOWN (Orange)
            else:
                cur_color = self.COLOR_GAZE_CENTER  # CENTER (Cyan)

            # Draw cursor pill
            cv2.rectangle(img, (vm_x + 3, cur_y - 3), (vm_x + vm_w - 3, cur_y + 3), cur_color, -1)
            cv2.rectangle(img, (vm_x + 3, cur_y - 3), (vm_x + vm_w - 3, cur_y + 3), (255, 255, 255), 1)

        # Meter labels
        font_sm = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "UP", (vm_x + 4, vm_y + 11), font_sm, 0.28, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "DN", (vm_x + 4, vm_y + radar_size - 4), font_sm, 0.28, (0, 200, 255), 1, cv2.LINE_AA)
