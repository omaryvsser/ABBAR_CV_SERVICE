"""
Canonical Accuracy Metrics & Spatial Breakdown Calculator (Single Source of Truth).

Ensures identical calculations across:
- Real-time ScreenValidator finalization
- Experiment JSON, Markdown, HTML, and plain-text reports
- Central AABAR_Experiments_Tracker.xlsx workbook
- Historical raw-trial recovery & CLI/GUI result displays

Strictly separates:
- Target Selection Accuracy: Hits / Valid Trials * 100 (actual target bounds)
- Threshold Accuracies: % of Valid Trials within 2%, 5%, 10% of screen diagonal
- Direction Classification Match Accuracy (evaluated separately from Target Selection)
- Rejected trials (never counted as valid or successful)
"""

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np


CANONICAL_DIRECTIONS: Tuple[str, ...] = ("LEFT", "RIGHT", "CENTER", "UP", "DOWN")

CANONICAL_SCREEN_REGIONS_9: Tuple[str, ...] = (
    "Top Left",
    "Top Center",
    "Top Right",
    "Middle Left",
    "Center",
    "Middle Right",
    "Bottom Left",
    "Bottom Center",
    "Bottom Right",
)

_REGION_CODE_TO_LABEL: Dict[str, str] = {
    "top_left": "Top Left",
    "top_center": "Top Center",
    "top_right": "Top Right",
    "middle_left": "Middle Left",
    "center": "Center",
    "middle_right": "Middle Right",
    "bottom_left": "Bottom Left",
    "bottom_center": "Bottom Center",
    "bottom_right": "Bottom Right",
}


def _to_float_or_none(val: Any) -> Optional[float]:
    if val is None or val == "" or str(val).strip().upper() in ("N/A", "NONE", "NAN", "NULL"):
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _to_int_or_default(val: Any, default: int = 0) -> int:
    f = _to_float_or_none(val)
    return int(round(f)) if f is not None else default


def _to_bool_or_none(val: Any) -> Optional[bool]:
    if val is None or val == "" or str(val).strip().upper() in ("N/A", "NONE", "NULL"):
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "hit"):
        return True
    if s in ("0", "false", "no", "miss"):
        return False
    return None


def classify_screen_point_direction(
    x: Optional[float],
    y: Optional[float],
    center_min: float = 0.40,
    center_max: float = 0.60,
) -> str:
    """
    Classify normalized screen coordinates (x, y) in [0.0, 1.0] into categorical
    gaze directions following the project's documented interface convention:
    - x=0 Left -> x=1 Right
    - y=0 Top  -> y=1 Bottom
    """
    if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)):
        return "UNKNOWN"

    h = "LEFT" if x < center_min else ("RIGHT" if x > center_max else None)
    v = "UP" if y < center_min else ("DOWN" if y > center_max else None)

    if v and h:
        return f"{v}_{h}"
    if h:
        return h
    if v:
        return v
    return "CENTER"


def matches_direction_bucket(direction_label: str, bucket: str) -> bool:
    """Return True if a point's categorical direction belongs to the requested direction bucket."""
    if not direction_label or direction_label == "UNKNOWN":
        return False
    if bucket == "CENTER":
        return direction_label == "CENTER"
    return bucket in direction_label


def classify_screen_region_9(x: Optional[float], y: Optional[float]) -> Optional[str]:
    """Map normalized screen coordinates (x, y) into one of the 9 screen calibration regions."""
    if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)):
        return None

    row = "Top" if y <= 0.35 else ("Bottom" if y >= 0.65 else "Middle")
    col = "Left" if x <= 0.35 else ("Right" if x >= 0.65 else "Center")
    if row == "Middle" and col == "Center":
        return "Center"
    return f"{row} {col}"


def normalize_region_label(name_or_coords: Tuple[Optional[str], Optional[float], Optional[float]]) -> Optional[str]:
    """Resolve canonical 9-region display label from coordinates or target name."""
    t_name, ax, ay = name_or_coords
    from_coords = classify_screen_region_9(ax, ay)
    if from_coords is not None:
        return from_coords
    if t_name:
        cleaned = t_name.lower().removeprefix("val_").strip()
        if cleaned in _REGION_CODE_TO_LABEL:
            return _REGION_CODE_TO_LABEL[cleaned]
    return None


def evaluate_boundary_from_spec(
    actual_x: float,
    actual_y: float,
    pred_x: float,
    pred_y: float,
    screen_width: int,
    screen_height: int,
    target_boundary: str = "",
    display_width: int = 1280,
    display_height: int = 720,
    default_display_radius_px: float = 22.0,
) -> Tuple[bool, str]:
    """
    Evaluate whether (pred_x, pred_y) lies within the rendered target boundary.
    Matches CalibrationTarget.contains_point() rules exactly.
    """
    err_x = pred_x - actual_x
    err_y = pred_y - actual_y
    spec = target_boundary or ""

    rect_m = re.search(r"Button Box\s*\(\s*([\d.]+)\s*x\s*([\d.]+)\s*px\s*\)", spec, re.IGNORECASE)
    if rect_m:
        bw = float(rect_m.group(1))
        bh = float(rect_m.group(2))
        inside = (
            abs(err_x * float(screen_width)) <= (bw / 2.0) + 1e-9
            and abs(err_y * float(screen_height)) <= (bh / 2.0) + 1e-9
        )
        return inside, spec

    disp_m = re.search(r"radius\s+([\d.]+)\s+display\s*px", spec, re.IGNORECASE)
    if disp_m:
        r_disp = float(disp_m.group(1))
        dx_disp = err_x * float(max(1, display_width))
        dy_disp = err_y * float(max(1, display_height))
        return math.hypot(dx_disp, dy_disp) <= r_disp + 1e-9, spec

    circ_m = re.search(r"radius\s+([\d.]+)\s*px", spec, re.IGNORECASE)
    if circ_m:
        r_px = float(circ_m.group(1))
        dist_px = math.hypot(err_x * float(screen_width), err_y * float(screen_height))
        return dist_px <= r_px + 1e-9, spec

    dx_disp = err_x * float(max(1, display_width))
    dy_disp = err_y * float(max(1, display_height))
    screen_equiv = default_display_radius_px * (float(screen_width) / float(max(1, display_width)))
    fallback_spec = (
        f"Rendered Target Circle (radius {default_display_radius_px:.0f} display px / "
        f"{screen_equiv:.1f} px @ {screen_width}x{screen_height})"
    )
    return math.hypot(dx_disp, dy_disp) <= default_display_radius_px + 1e-9, fallback_spec


def normalize_trial_record(
    raw_trial: Any,
    default_experiment_id: str = "",
    default_session_id: str = "",
    default_screen_width: int = 1920,
    default_screen_height: int = 1080,
) -> Dict[str, Any]:
    """
    Normalize either a ValidationTrialRecord dataclass or a CSV dict into one canonical schema,
    reconstructing missing derived error/hit/direction fields only from true underlying coordinates.
    """
    if hasattr(raw_trial, "to_csv_dict") and not isinstance(raw_trial, Mapping):
        d = {
            "experiment_id": getattr(raw_trial, "experiment_id", default_experiment_id),
            "session_id": getattr(raw_trial, "session_id", default_session_id),
            "calibration_profile_id": getattr(raw_trial, "calibration_profile_id", None),
            "trial_id": getattr(raw_trial, "trial_id", 0),
            "timestamp": getattr(raw_trial, "timestamp", None),
            "target_id": getattr(raw_trial, "target_id", 0),
            "target_name": getattr(raw_trial, "target_name", ""),
            "actual_x": getattr(raw_trial, "actual_x", None),
            "actual_y": getattr(raw_trial, "actual_y", None),
            "predicted_x": getattr(raw_trial, "predicted_x", None),
            "predicted_y": getattr(raw_trial, "predicted_y", None),
            "error_x": getattr(raw_trial, "error_x", None),
            "error_y": getattr(raw_trial, "error_y", None),
            "pixel_error_x": getattr(raw_trial, "pixel_error_x", None),
            "pixel_error_y": getattr(raw_trial, "pixel_error_y", None),
            "euclidean_error_norm": getattr(raw_trial, "euclidean_error_norm", None),
            "euclidean_error_pixels": getattr(raw_trial, "euclidean_error_pixels", None),
            "error_pct_diagonal": getattr(raw_trial, "error_pct_diagonal", None),
            "target_hit": getattr(raw_trial, "target_hit", None),
            "target_boundary": getattr(raw_trial, "target_boundary", ""),
            "screen_width": getattr(raw_trial, "screen_width", default_screen_width),
            "screen_height": getattr(raw_trial, "screen_height", default_screen_height),
            "accepted_sample_count": getattr(raw_trial, "accepted_sample_count", 0),
            "rejected_sample_count": getattr(raw_trial, "rejected_sample_count", 0),
            "trial_duration_sec": getattr(raw_trial, "trial_duration_sec", 0.0),
            "trial_status": getattr(raw_trial, "trial_status", "VALID"),
            "rejection_reason": getattr(raw_trial, "rejection_reason", ""),
        }
    else:
        r = dict(raw_trial)
        d = {
            "experiment_id": str(r.get("experiment_id") or default_experiment_id),
            "session_id": str(r.get("session_id") or default_session_id),
            "calibration_profile_id": r.get("calibration_profile_id") or None,
            "trial_id": _to_int_or_default(r.get("trial_id"), 0),
            "timestamp": _to_float_or_none(r.get("timestamp")),
            "target_id": _to_int_or_default(r.get("target_id"), 0),
            "target_name": str(r.get("target_name") or ""),
            "actual_x": _to_float_or_none(r.get("actual_x", r.get("target_x"))),
            "actual_y": _to_float_or_none(r.get("actual_y", r.get("target_y"))),
            "predicted_x": _to_float_or_none(r.get("predicted_x")),
            "predicted_y": _to_float_or_none(r.get("predicted_y")),
            "error_x": _to_float_or_none(r.get("error_x")),
            "error_y": _to_float_or_none(r.get("error_y")),
            "pixel_error_x": _to_float_or_none(r.get("pixel_error_x")),
            "pixel_error_y": _to_float_or_none(r.get("pixel_error_y")),
            "euclidean_error_norm": _to_float_or_none(
                r.get("euclidean_error_norm", r.get("euclidean_error_normalized", r.get("euclidean_error")))
            ),
            "euclidean_error_pixels": _to_float_or_none(
                r.get("euclidean_error_pixels", r.get("euclidean_error_px"))
            ),
            "error_pct_diagonal": _to_float_or_none(r.get("error_pct_diagonal")),
            "target_hit": _to_bool_or_none(r.get("target_hit")),
            "target_boundary": str(r.get("target_boundary") or ""),
            "screen_width": _to_int_or_default(r.get("screen_width"), default_screen_width),
            "screen_height": _to_int_or_default(r.get("screen_height"), default_screen_height),
            "accepted_sample_count": _to_int_or_default(r.get("accepted_sample_count", r.get("sample_count")), 0),
            "rejected_sample_count": _to_int_or_default(r.get("rejected_sample_count"), 0),
            "trial_duration_sec": _to_float_or_none(r.get("trial_duration_sec", r.get("duration_sec"))) or 0.0,
            "trial_status": str(r.get("trial_status", r.get("status")) or "VALID").upper(),
            "rejection_reason": str(r.get("rejection_reason") or ""),
            "raw_x": _to_float_or_none(r.get("raw_x")),
            "raw_y": _to_float_or_none(r.get("raw_y")),
            "head_yaw": _to_float_or_none(r.get("head_yaw")),
            "head_pitch": _to_float_or_none(r.get("head_pitch")),
        }

    sw = max(1, int(d["screen_width"] or default_screen_width))
    sh = max(1, int(d["screen_height"] or default_screen_height))
    diag_px = math.hypot(float(sw), float(sh))

    ax, ay = d["actual_x"], d["actual_y"]
    px, py = d["predicted_x"], d["predicted_y"]

    d["actual_direction"] = classify_screen_point_direction(ax, ay)
    d["predicted_direction"] = classify_screen_point_direction(px, py)
    d["region_9"] = normalize_region_label((d["target_name"], ax, ay))

    is_valid_status = d["trial_status"] == "VALID"
    if is_valid_status and ax is not None and ay is not None and px is not None and py is not None:
        if d["error_x"] is None:
            d["error_x"] = px - ax
        if d["error_y"] is None:
            d["error_y"] = py - ay
        if d["pixel_error_x"] is None:
            d["pixel_error_x"] = d["error_x"] * float(sw)
        if d["pixel_error_y"] is None:
            d["pixel_error_y"] = d["error_y"] * float(sh)
        if d["euclidean_error_norm"] is None:
            d["euclidean_error_norm"] = math.hypot(d["error_x"], d["error_y"])
        if d["euclidean_error_pixels"] is None:
            d["euclidean_error_pixels"] = math.hypot(d["pixel_error_x"], d["pixel_error_y"])
        if d["error_pct_diagonal"] is None and diag_px > 0.0:
            d["error_pct_diagonal"] = (d["euclidean_error_pixels"] / diag_px) * 100.0
        if d["target_hit"] is None:
            hit, spec = evaluate_boundary_from_spec(
                ax, ay, px, py, sw, sh, target_boundary=d["target_boundary"]
            )
            d["target_hit"] = hit
            if not d["target_boundary"]:
                d["target_boundary"] = spec
    else:
        if not is_valid_status:
            d["target_hit"] = None

    return d


def summarize_trial_subset(
    valid_subset: Sequence[Mapping[str, Any]],
    rejected_subset: Sequence[Mapping[str, Any]],
    screen_width: int,
    screen_height: int,
    thresh_2_px: float,
    thresh_5_px: float,
    thresh_10_px: float,
    direction_bucket: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute standardized metrics for a subset of validation trials.
    Direction classification match accuracy is kept strictly separate from Target Selection Accuracy.
    """
    n_val = len(valid_subset)
    n_rej = len(rejected_subset)
    if n_val == 0:
        out: Dict[str, Any] = {
            "total_attempts": n_rej,
            "valid_trials": 0,
            "rejected_trials": n_rej,
            "successful_selections": 0,
            "failed_selections": 0,
            "target_selection_accuracy_pct": None,
            "pct_within_2pct": None,
            "pct_within_5pct": None,
            "pct_within_10pct": None,
            "mean_euclidean_pixels": None,
            "median_euclidean_pixels": None,
            "signed_bias_x_px": None,
            "signed_bias_y_px": None,
        }
        if direction_bucket is not None:
            out["direction_match_accuracy_pct"] = None
        return out

    e_px = np.array([float(r["euclidean_error_pixels"]) for r in valid_subset], dtype=np.float64)
    err_x = np.array([float(r["error_x"]) for r in valid_subset], dtype=np.float64)
    err_y = np.array([float(r["error_y"]) for r in valid_subset], dtype=np.float64)
    hits = sum(1 for r in valid_subset if r.get("target_hit") is True)

    out = {
        "total_attempts": n_val + n_rej,
        "valid_trials": n_val,
        "rejected_trials": n_rej,
        "successful_selections": hits,
        "failed_selections": n_val - hits,
        "target_selection_accuracy_pct": round(float(hits / n_val * 100.0), 2),
        "pct_within_2pct": round(float(np.sum(e_px <= thresh_2_px + 1e-9) / n_val * 100.0), 2),
        "pct_within_5pct": round(float(np.sum(e_px <= thresh_5_px + 1e-9) / n_val * 100.0), 2),
        "pct_within_10pct": round(float(np.sum(e_px <= thresh_10_px + 1e-9) / n_val * 100.0), 2),
        "mean_euclidean_pixels": round(float(np.mean(e_px)), 2),
        "median_euclidean_pixels": round(float(np.median(e_px)), 2),
        "signed_bias_x_px": round(float(np.mean(err_x)) * float(screen_width), 2),
        "signed_bias_y_px": round(float(np.mean(err_y)) * float(screen_height), 2),
    }
    if direction_bucket is not None:
        dir_matches = sum(
            1
            for r in valid_subset
            if matches_direction_bucket(str(r.get("predicted_direction") or ""), direction_bucket)
        )
        out["direction_match_accuracy_pct"] = round(float(dir_matches / n_val * 100.0), 2)
    return out


def compute_validation_metrics_from_trials(
    raw_trials: Sequence[Any],
    screen_width: int = 1920,
    screen_height: int = 1080,
    threshold_2pct: float = 0.02,
    threshold_5pct: float = 0.05,
    threshold_10pct: float = 0.10,
    default_experiment_id: str = "",
    default_session_id: str = "",
) -> Dict[str, Any]:
    """
    Compute the canonical single-source-of-truth validation metrics dictionary
    directly from validation trial records.
    """
    normalized_trials = [
        normalize_trial_record(
            t,
            default_experiment_id=default_experiment_id,
            default_session_id=default_session_id,
            default_screen_width=screen_width,
            default_screen_height=screen_height,
        )
        for t in raw_trials
    ]

    valid_records = [
        r for r in normalized_trials
        if r["trial_status"] == "VALID"
        and r["error_x"] is not None
        and r["error_y"] is not None
        and r["euclidean_error_pixels"] is not None
    ]
    rejected_records = [r for r in normalized_trials if r["trial_status"] == "REJECTED"]

    diag_px = float(math.hypot(float(max(1, screen_width)), float(max(1, screen_height))))
    thresh_2_px = float(threshold_2pct) * diag_px
    thresh_5_px = float(threshold_5pct) * diag_px
    thresh_10_px = float(threshold_10pct) * diag_px

    total_trials = len(normalized_trials)
    n_valid = len(valid_records)
    n_rej = len(rejected_records)
    attempted = n_valid + n_rej
    rejected_pct = float(n_rej / attempted * 100.0) if attempted > 0 else 0.0

    if n_valid == 0:
        return {
            "has_valid_metrics": False,
            "normalized_trials": normalized_trials,
            "total_trials": total_trials,
            "successful_trials": 0,
            "rejected_trials": n_rej,
            "rejected_trial_pct": round(rejected_pct, 2),
            "successful_selections": 0,
            "failed_selections": 0,
            "target_selection_accuracy_pct": None,
            "pct_within_2pct": None,
            "pct_within_5pct": None,
            "pct_within_10pct": None,
            "trials_within_2pct": 0,
            "trials_within_5pct": 0,
            "trials_within_10pct": 0,
            "mean_error_pixels": None,
            "median_error_pixels": None,
            "max_error_pixels": None,
            "p90_error_pixels": None,
            "direction_accuracies": {},
            "region_accuracies": {},
            "region_accuracies_9": {},
            "target_accuracies": {},
            "pattern_analysis": {},
        }

    err_x = np.array([float(r["error_x"]) for r in valid_records], dtype=np.float64)
    err_y = np.array([float(r["error_y"]) for r in valid_records], dtype=np.float64)
    e_norm = np.array([float(r["euclidean_error_norm"]) for r in valid_records], dtype=np.float64)
    e_px = np.array([float(r["euclidean_error_pixels"]) for r in valid_records], dtype=np.float64)

    mae_x = float(np.mean(np.abs(err_x)))
    mae_y = float(np.mean(np.abs(err_y)))
    rmse_x = float(np.sqrt(np.mean(err_x ** 2)))
    rmse_y = float(np.sqrt(np.mean(err_y ** 2)))
    mean_e_norm = float(np.mean(e_norm))
    median_e_norm = float(np.median(e_norm))
    max_e_norm = float(np.max(e_norm))
    p90_norm = float(np.percentile(e_norm, 90)) if n_valid >= 5 else None

    mae_x_px = mae_x * float(screen_width)
    mae_y_px = mae_y * float(screen_height)
    rmse_x_px = rmse_x * float(screen_width)
    rmse_y_px = rmse_y * float(screen_height)
    mean_e_px = float(np.mean(e_px))
    median_e_px = float(np.median(e_px))
    max_e_px = float(np.max(e_px))
    p90_px = float(np.percentile(e_px, 90)) if n_valid >= 5 else None

    within_2 = int(np.sum(e_px <= thresh_2_px + 1e-9))
    within_5 = int(np.sum(e_px <= thresh_5_px + 1e-9))
    within_10 = int(np.sum(e_px <= thresh_10_px + 1e-9))

    successful_selections = sum(1 for r in valid_records if r.get("target_hit") is True)
    failed_selections = n_valid - successful_selections
    target_selection_acc_pct = float(successful_selections / n_valid * 100.0)

    boundary_desc = next((str(r["target_boundary"]) for r in valid_records if r.get("target_boundary")), "")

    # Per-target breakdown
    target_accs: Dict[str, Dict[str, Any]] = {}
    unique_names = sorted(list(set(str(r["target_name"]) for r in normalized_trials)))
    for tname in unique_names:
        t_val = [r for r in valid_records if str(r["target_name"]) == tname]
        t_rej = [r for r in rejected_records if str(r["target_name"]) == tname]
        if t_val:
            sub = summarize_trial_subset(t_val, t_rej, screen_width, screen_height, thresh_2_px, thresh_5_px, thresh_10_px)
            t_e_norm = np.array([float(r["euclidean_error_norm"]) for r in t_val], dtype=np.float64)
            t_e_px = np.array([float(r["euclidean_error_pixels"]) for r in t_val], dtype=np.float64)
            t_ex = np.array([float(r["error_x"]) for r in t_val], dtype=np.float64)
            t_ey = np.array([float(r["error_y"]) for r in t_val], dtype=np.float64)
            sub.update({
                "actual_x": t_val[0]["actual_x"],
                "actual_y": t_val[0]["actual_y"],
                "mean_error_pct_diagonal": round((float(np.mean(t_e_px)) / diag_px) * 100.0, 2) if diag_px > 0 else 0.0,
                "mean_euclidean_norm": round(float(np.mean(t_e_norm)), 4),
                "std_euclidean_pixels": round(float(np.std(t_e_px)), 2) if len(t_e_px) > 1 else 0.0,
                "mae_x_pixels": round(float(np.mean(np.abs(t_ex))) * float(screen_width), 2),
                "mae_y_pixels": round(float(np.mean(np.abs(t_ey))) * float(screen_height), 2),
                "target_boundary": t_val[0].get("target_boundary", ""),
            })
            target_accs[tname] = sub
        elif t_rej:
            target_accs[tname] = {
                "total_attempts": len(t_rej),
                "valid_trials": 0,
                "rejected_trials": len(t_rej),
                "successful_selections": 0,
                "failed_selections": 0,
                "target_selection_accuracy_pct": None,
                "actual_x": t_rej[0]["actual_x"],
                "actual_y": t_rej[0]["actual_y"],
                "rejection_status": "ALL_TRIALS_REJECTED",
            }

    # Direction accuracy breakdown (LEFT, RIGHT, CENTER, UP, DOWN)
    direction_accs: Dict[str, Dict[str, Any]] = {}
    for dname in CANONICAL_DIRECTIONS:
        d_val = [r for r in valid_records if matches_direction_bucket(str(r.get("actual_direction") or ""), dname)]
        d_rej = [r for r in rejected_records if matches_direction_bucket(str(r.get("actual_direction") or ""), dname)]
        if d_val or d_rej:
            direction_accs[dname] = summarize_trial_subset(
                d_val, d_rej, screen_width, screen_height, thresh_2_px, thresh_5_px, thresh_10_px, direction_bucket=dname
            )

    # Aggregate regional groups (Rows / Columns / Edges)
    region_preds = {
        "top_row": lambda r: (r["actual_y"] or 0.5) <= 0.35,
        "middle_row": lambda r: 0.35 < (r["actual_y"] or 0.5) < 0.65,
        "bottom_row": lambda r: (r["actual_y"] or 0.5) >= 0.65,
        "left_column": lambda r: (r["actual_x"] or 0.5) <= 0.35,
        "center_column": lambda r: 0.35 < (r["actual_x"] or 0.5) < 0.65,
        "right_column": lambda r: (r["actual_x"] or 0.5) >= 0.65,
        "edge_regions": lambda r: (
            (r["actual_x"] or 0.5) <= 0.35
            or (r["actual_x"] or 0.5) >= 0.65
            or (r["actual_y"] or 0.5) <= 0.35
            or (r["actual_y"] or 0.5) >= 0.65
        ),
        "center_region": lambda r: 0.35 < (r["actual_x"] or 0.5) < 0.65 and 0.35 < (r["actual_y"] or 0.5) < 0.65,
    }
    region_accs: Dict[str, Dict[str, Any]] = {}
    for reg_name, pred_fn in region_preds.items():
        g_val = [r for r in valid_records if r["actual_x"] is not None and r["actual_y"] is not None and pred_fn(r)]
        g_rej = [r for r in rejected_records if r["actual_x"] is not None and r["actual_y"] is not None and pred_fn(r)]
        if g_val or g_rej:
            region_accs[reg_name] = summarize_trial_subset(
                g_val, g_rej, screen_width, screen_height, thresh_2_px, thresh_5_px, thresh_10_px
            )

    # Exact 9 Screen Regions (Top Left .. Bottom Right)
    region_accs_9: Dict[str, Dict[str, Any]] = {}
    for reg9 in CANONICAL_SCREEN_REGIONS_9:
        r_val = [r for r in valid_records if r.get("region_9") == reg9]
        r_rej = [r for r in rejected_records if r.get("region_9") == reg9]
        if r_val or r_rej:
            region_accs_9[reg9] = summarize_trial_subset(
                r_val, r_rej, screen_width, screen_height, thresh_2_px, thresh_5_px, thresh_10_px
            )

    # Pattern analysis (best/weakest direction, region, target)
    valid_t = {k: v for k, v in target_accs.items() if v.get("valid_trials", 0) > 0}
    valid_d = {k: v for k, v in direction_accs.items() if v.get("valid_trials", 0) > 0}
    valid_r = {k: v for k, v in region_accs.items() if v.get("valid_trials", 0) > 0}
    valid_r9 = {k: v for k, v in region_accs_9.items() if v.get("valid_trials", 0) > 0}

    pattern: Dict[str, Any] = {}
    if valid_t:
        best_t = min(valid_t.items(), key=lambda kv: (kv[1]["mean_euclidean_pixels"], -(kv[1]["target_selection_accuracy_pct"] or 0.0)))
        worst_t = max(valid_t.items(), key=lambda kv: (kv[1]["mean_euclidean_pixels"], -(kv[1]["target_selection_accuracy_pct"] or 0.0)))
        pattern["best_target"] = {"target": best_t[0], "mean_error_px": best_t[1]["mean_euclidean_pixels"], "selection_accuracy_pct": best_t[1]["target_selection_accuracy_pct"]}
        pattern["worst_target"] = {"target": worst_t[0], "mean_error_px": worst_t[1]["mean_euclidean_pixels"], "selection_accuracy_pct": worst_t[1]["target_selection_accuracy_pct"]}
        pattern["mean_signed_bias_x_px"] = round(float(np.mean(err_x)) * float(screen_width), 2)
        pattern["mean_signed_bias_y_px"] = round(float(np.mean(err_y)) * float(screen_height), 2)

        top_rows = [r for r in valid_records if (r["actual_y"] or 0.5) < 0.40]
        bot_rows = [r for r in valid_records if (r["actual_y"] or 0.5) > 0.60]
        if top_rows and bot_rows:
            pattern["top_mean_error_px"] = round(float(np.mean([float(r["euclidean_error_pixels"]) for r in top_rows])), 2)
            pattern["bottom_mean_error_px"] = round(float(np.mean([float(r["euclidean_error_pixels"]) for r in bot_rows])), 2)

        edge_rows = [r for r in valid_records if min(r["actual_x"] or 0.5, 1.0 - (r["actual_x"] or 0.5)) <= 0.25 or min(r["actual_y"] or 0.5, 1.0 - (r["actual_y"] or 0.5)) <= 0.25]
        center_rows = [r for r in valid_records if r not in edge_rows]
        if edge_rows and center_rows:
            pattern["edge_mean_error_px"] = round(float(np.mean([float(r["euclidean_error_pixels"]) for r in edge_rows])), 2)
            pattern["center_mean_error_px"] = round(float(np.mean([float(r["euclidean_error_pixels"]) for r in center_rows])), 2)

    if valid_d:
        best_d = max(valid_d.items(), key=lambda kv: (kv[1]["target_selection_accuracy_pct"] or 0.0, -(kv[1]["mean_euclidean_pixels"] or 1e9)))
        weakest_d = min(valid_d.items(), key=lambda kv: (kv[1]["target_selection_accuracy_pct"] or 0.0, -(kv[1]["mean_euclidean_pixels"] or 0.0)))
        pattern["best_direction"] = {"direction": best_d[0], "selection_accuracy_pct": best_d[1]["target_selection_accuracy_pct"], "mean_error_px": best_d[1]["mean_euclidean_pixels"]}
        pattern["weakest_direction"] = {"direction": weakest_d[0], "selection_accuracy_pct": weakest_d[1]["target_selection_accuracy_pct"], "mean_error_px": weakest_d[1]["mean_euclidean_pixels"]}

    if valid_r:
        best_r = max(valid_r.items(), key=lambda kv: (kv[1]["target_selection_accuracy_pct"] or 0.0, -(kv[1]["mean_euclidean_pixels"] or 1e9)))
        weakest_r = min(valid_r.items(), key=lambda kv: (kv[1]["target_selection_accuracy_pct"] or 0.0, -(kv[1]["mean_euclidean_pixels"] or 0.0)))
        pattern["best_region"] = {"region": best_r[0], "selection_accuracy_pct": best_r[1]["target_selection_accuracy_pct"], "mean_error_px": best_r[1]["mean_euclidean_pixels"]}
        pattern["weakest_region"] = {"region": weakest_r[0], "selection_accuracy_pct": weakest_r[1]["target_selection_accuracy_pct"], "mean_error_px": weakest_r[1]["mean_euclidean_pixels"]}

    if valid_r9:
        best_r9 = max(valid_r9.items(), key=lambda kv: (kv[1]["target_selection_accuracy_pct"] or 0.0, -(kv[1]["mean_euclidean_pixels"] or 1e9)))
        weakest_r9 = min(valid_r9.items(), key=lambda kv: (kv[1]["target_selection_accuracy_pct"] or 0.0, -(kv[1]["mean_euclidean_pixels"] or 0.0)))
        pattern["best_region_9"] = {"region": best_r9[0], "selection_accuracy_pct": best_r9[1]["target_selection_accuracy_pct"], "mean_error_px": best_r9[1]["mean_euclidean_pixels"]}
        pattern["weakest_region_9"] = {"region": weakest_r9[0], "selection_accuracy_pct": weakest_r9[1]["target_selection_accuracy_pct"], "mean_error_px": weakest_r9[1]["mean_euclidean_pixels"]}

    return {
        "has_valid_metrics": True,
        "normalized_trials": normalized_trials,
        "total_trials": total_trials,
        "successful_trials": n_valid,
        "rejected_trials": n_rej,
        "rejected_trial_pct": round(rejected_pct, 2),
        "successful_selections": successful_selections,
        "failed_selections": failed_selections,
        "target_selection_accuracy_pct": target_selection_acc_pct,
        "target_boundary_description": boundary_desc,
        "screen_diagonal_px": diag_px,
        "threshold_2pct_px": thresh_2_px,
        "threshold_5pct_px": thresh_5_px,
        "threshold_10pct_px": thresh_10_px,
        "trials_within_2pct": within_2,
        "pct_within_2pct": float(within_2 / n_valid * 100.0),
        "trials_within_5pct": within_5,
        "pct_within_5pct": float(within_5 / n_valid * 100.0),
        "trials_within_10pct": within_10,
        "pct_within_10pct": float(within_10 / n_valid * 100.0),
        "mae_x": mae_x,
        "mae_y": mae_y,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "mae_x_px": mae_x_px,
        "mae_y_px": mae_y_px,
        "rmse_x_px": rmse_x_px,
        "rmse_y_px": rmse_y_px,
        "mean_euclidean_norm": mean_e_norm,
        "median_euclidean_norm": median_e_norm,
        "max_euclidean_norm": max_e_norm,
        "p90_norm": p90_norm,
        "mean_error_pixels": mean_e_px,
        "median_error_pixels": median_e_px,
        "max_error_pixels": max_e_px,
        "p90_error_pixels": p90_px,
        "mean_error_pct_diagonal": (mean_e_px / diag_px) * 100.0 if diag_px > 0 else 0.0,
        "median_error_pct_diagonal": (median_e_px / diag_px) * 100.0 if diag_px > 0 else 0.0,
        "max_error_pct_diagonal": (max_e_px / diag_px) * 100.0 if diag_px > 0 else 0.0,
        "target_accuracies": target_accs,
        "direction_accuracies": direction_accs,
        "region_accuracies": region_accs,
        "region_accuracies_9": region_accs_9,
        "pattern_analysis": pattern,
    }
