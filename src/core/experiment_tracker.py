"""
Central Experiment Tracker & Multi-Session Accuracy Aggregation System.

Scans all historical and future calibration/validation experiment records,
reconstructs and verifies metrics directly from raw trial records using the
shared single-source-of-truth engine (`accuracy_metrics.py`), prevents duplicates
by stable (Experiment ID, Trial ID) keys, and generates/updates the centralized
`AABAR_Experiments_Tracker.xlsx` workbook.

Sheets in AABAR_Experiments_Tracker.xlsx:
1. Experiments Summary      (main overview, most critical metrics upfront + chart)
2. All Test Trials          (every recorded validation trial across experiments)
3. Direction Accuracy       (LEFT, RIGHT, CENTER, UP, DOWN per experiment + chart)
4. Screen Region Accuracy   (all 9 regions per experiment + 3x3 grid summary)
5. Experiment Comparison    (validated ranking only; unvalidated excluded cleanly)
6. Data Quality & Validation Status (diagnostics explaining missing accuracy)
"""

import csv
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from .accuracy_metrics import (
    CANONICAL_DIRECTIONS,
    CANONICAL_SCREEN_REGIONS_9,
    compute_validation_metrics_from_trials,
    normalize_trial_record,
)
from .screen_calibration import DEFAULT_9_TARGETS
from .screen_validator import FULL_9_REGION_VALIDATION_TARGETS, DEFAULT_VALIDATION_TARGETS

logger = logging.getLogger(__name__)

TRACKER_FILENAME = "AABAR_Experiments_Tracker.xlsx"

_EMPTY_EXP_NOTICE = "Validation not completed — Accuracy is not available yet."


def _parse_resolution(res_str: Any, fallback_w: int = 1920, fallback_h: int = 1080) -> Tuple[int, int, str]:
    if isinstance(res_str, str) and "x" in res_str.lower():
        parts = res_str.lower().split("x", 1)
        try:
            w = max(1, int(float(parts[0].strip())))
            h = max(1, int(float(parts[1].strip())))
            return w, h, f"{w}x{h}"
        except (TypeError, ValueError):
            pass
    w = max(1, int(fallback_w or 1920))
    h = max(1, int(fallback_h or 1080))
    return w, h, f"{w}x{h}"


def _format_datetime(timestamp: float, exp_id: str) -> str:
    if timestamp and timestamp > 1e8:
        try:
            return datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    # Fallback to timestamp suffix in experiment_YYYYMMDD_HHMMSS if present
    parts = exp_id.split("_")
    if len(parts) >= 3 and len(parts[-2]) == 8 and len(parts[-1]) == 6:
        d, t = parts[-2], parts[-1]
        if d.isdigit() and t.isdigit():
            return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
    return "Unknown"


def _load_profiles_index(profiles_dir: Optional[Path]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Load saved calibration profiles indexed by profile_id for metadata recovery."""
    by_id: Dict[str, Dict[str, Any]] = {}
    all_profiles: List[Dict[str, Any]] = []
    if profiles_dir is None or not profiles_dir.exists():
        return by_id, all_profiles

    for pfile in sorted(profiles_dir.glob("profile_*.json")):
        try:
            with open(pfile, "r", encoding="utf-8") as f:
                pdata = json.load(f)
            if isinstance(pdata, dict):
                pdata["_profile_file"] = str(pfile)
                pid = str(pdata.get("profile_id") or "")
                if pid:
                    by_id[pid] = pdata
                all_profiles.append(pdata)
        except Exception as e:
            logger.debug(f"Could not read profile {pfile}: {e}")
    return by_id, all_profiles


def _read_calibration_csv(calib_csv: Path) -> Dict[str, Any]:
    """Summarize raw calibration_samples.csv without modifying the file."""
    if not calib_csv.exists():
        return {
            "exists": False,
            "total_frames": 0,
            "accepted_frames": 0,
            "rejected_frames": 0,
            "target_ids": set(),
            "session_id": "",
        }
    total = 0
    accepted = 0
    target_ids: Set[int] = set()
    session_id = ""
    try:
        with open(calib_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                total += 1
                if not session_id and row.get("session_id"):
                    session_id = str(row["session_id"]).strip()
                acc_raw = row.get("sample_accepted", row.get("accepted", ""))
                if str(acc_raw).strip().lower() in ("1", "true", "yes"):
                    accepted += 1
                try:
                    if row.get("target_id") not in (None, ""):
                        target_ids.add(int(float(row["target_id"])))
                except (TypeError, ValueError):
                    pass
    except Exception as e:
        logger.debug(f"Error reading calibration CSV {calib_csv}: {e}")
    return {
        "exists": True,
        "total_frames": total,
        "accepted_frames": accepted,
        "rejected_frames": max(0, total - accepted),
        "target_ids": target_ids,
        "session_id": session_id,
    }


def _read_validation_csv(
    val_csv: Path,
    exp_id: str,
    default_session_id: str,
    screen_w: int,
    screen_h: int,
) -> List[Dict[str, Any]]:
    """Load and normalize every trial row from validation_results.csv."""
    if not val_csv.exists():
        return []
    rows: List[Dict[str, Any]] = []
    seen_trial_ids: Set[int] = set()
    try:
        with open(val_csv, "r", encoding="utf-8") as f:
            for idx, raw in enumerate(csv.DictReader(f), start=1):
                rec = normalize_trial_record(
                    raw,
                    default_experiment_id=exp_id,
                    default_session_id=default_session_id,
                    default_screen_width=screen_w,
                    default_screen_height=screen_h,
                )
                if rec["trial_id"] == 0:
                    rec["trial_id"] = idx
                tid = int(rec["trial_id"])
                if tid in seen_trial_ids:
                    continue
                seen_trial_ids.add(tid)
                rows.append(rec)
    except Exception as e:
        logger.debug(f"Error reading validation CSV {val_csv}: {e}")
    return rows


def _expected_target_names_for_mode(
    meta: Dict[str, Any],
    summary_json: Dict[str, Any],
    recorded_trials: List[Dict[str, Any]],
) -> Tuple[int, List[str]]:
    """Determine expected planned trial count and target names if recorded."""
    mode = str(meta.get("validation_grid_mode") or "").lower()
    planned = int(summary_json.get("total_trials") or 0)
    if mode == "interior_5":
        names = [t.name for t in DEFAULT_VALIDATION_TARGETS]
        return max(planned, len(names)), names
    if mode == "full_9":
        names = [t.name for t in FULL_9_REGION_VALIDATION_TARGETS]
        return max(planned, len(names)), names
    if planned >= 9 or len(recorded_trials) >= 9:
        return max(planned, 9), [t.name for t in FULL_9_REGION_VALIDATION_TARGETS]
    if planned == 5 or (0 < len(recorded_trials) <= 5 and any(r.get("target_name") in ("val_top_right", "val_bottom_left") and r.get("actual_x") in (0.3, 0.7) for r in recorded_trials)):
        return max(planned, 5), [t.name for t in DEFAULT_VALIDATION_TARGETS]
    return planned, []


def _inspect_single_experiment(
    exp_dir: Path,
    profiles_by_id: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Collect and validate all artifacts in a single experiment directory cleanly.
    Returns None only if the directory has zero recognizable experiment files.
    """
    exp_dir = Path(exp_dir)
    if not exp_dir.is_dir():
        return None

    acc_json_path = exp_dir / "accuracy_report.json"
    acc_csv_path = exp_dir / "accuracy_summary.csv"
    val_csv_path = exp_dir / "validation_results.csv"
    calib_csv_path = exp_dir / "calibration_samples.csv"
    meta_json_path = exp_dir / "experiment_metadata.json"

    has_any_artifact = any(
        p.exists() for p in (acc_json_path, acc_csv_path, val_csv_path, calib_csv_path, meta_json_path)
    )
    if not has_any_artifact:
        return None

    exp_id = exp_dir.name
    meta: Dict[str, Any] = {}
    if meta_json_path.exists():
        try:
            with open(meta_json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                meta = loaded
        except Exception as e:
            logger.debug(f"Malformed metadata JSON in {exp_dir}: {e}")

    rep_json: Dict[str, Any] = {}
    if acc_json_path.exists():
        try:
            with open(acc_json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                rep_json = loaded
        except Exception as e:
            logger.debug(f"Malformed accuracy JSON in {exp_dir}: {e}")

    acc_csv_row: Dict[str, Any] = {}
    if acc_csv_path.exists():
        try:
            with open(acc_csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                acc_csv_row = dict(rows[0])
        except Exception as e:
            logger.debug(f"Malformed accuracy CSV in {exp_dir}: {e}")

    cal_info = _read_calibration_csv(calib_csv_path)

    prof: Dict[str, Any] = dict(meta.get("profile") or {})
    cal_prof_id = str(
        prof.get("profile_id")
        or rep_json.get("calibration_profile_id")
        or acc_csv_row.get("calibration_profile_id")
        or ""
    )
    if cal_prof_id and cal_prof_id in profiles_by_id:
        # Merge saved profile (e.g. if saved profile later received validation_metrics)
        merged_prof = dict(profiles_by_id[cal_prof_id])
        merged_prof.update({k: v for k, v in prof.items() if v is not None})
        if profiles_by_id[cal_prof_id].get("validation_metrics") and not prof.get("validation_metrics"):
            merged_prof["validation_metrics"] = profiles_by_id[cal_prof_id]["validation_metrics"]
        prof = merged_prof

    session_id = str(cal_info.get("session_id") or meta.get("session_id") or "")

    # Resolution & coordinate convention
    sw_default = int(prof.get("screen_width") or meta.get("screen_width") or 1920)
    sh_default = int(prof.get("screen_height") or meta.get("screen_height") or 1080)
    res_raw = rep_json.get("screen_resolution") or acc_csv_row.get("screen_resolution")
    screen_w, screen_h, resolution_str = _parse_resolution(res_raw, sw_default, sh_default)

    # Timestamps
    ts = float(
        rep_json.get("evaluation_timestamp")
        or meta.get("created_at")
        or prof.get("created_at")
        or 0.0
    )
    if ts <= 0.0:
        try:
            ts = exp_dir.stat().st_mtime
        except Exception:
            ts = 0.0

    model_type = str(
        rep_json.get("model_name")
        or acc_csv_row.get("model_name")
        or prof.get("model_type")
        or meta.get("model_type")
        or "unknown"
    )
    norm_method = str(
        rep_json.get("normalization_method")
        or acc_csv_row.get("normalization_method")
        or prof.get("normalization_method")
        or meta.get("normalization_method")
        or "None"
    )
    coord_conv = str(
        rep_json.get("coordinate_convention")
        or acc_csv_row.get("coordinate_convention")
        or prof.get("coordinate_convention")
        or "legacy_mirrored_target_overlay"
    )

    # Training sample count (never reported as validation!)
    if cal_info["exists"] and cal_info["accepted_frames"] > 0:
        train_samples_count: Optional[int] = int(cal_info["accepted_frames"])
    elif prof.get("calibration_frame_count"):
        train_samples_count = int(prof["calibration_frame_count"])
    elif prof.get("training_sample_count") or rep_json.get("training_sample_count"):
        train_samples_count = int(rep_json.get("training_sample_count") or prof.get("training_sample_count"))
    else:
        train_samples_count = None

    # Threshold configuration
    thresh_cfg = rep_json.get("threshold_analysis") or {}
    t2_frac = float(thresh_cfg.get("threshold_2pct_fraction") or 0.02)
    t5_frac = float(thresh_cfg.get("threshold_5pct_fraction") or 0.05)
    t10_frac = float(thresh_cfg.get("threshold_10pct_fraction") or 0.10)

    # Primary truth: recover metrics directly from stored raw validation trials when present
    raw_trials = _read_validation_csv(val_csv_path, exp_id, session_id, screen_w, screen_h)
    meta_status = str(meta.get("status") or "").upper()
    fail_reason_meta = str(meta.get("failure_reason") or "").strip()

    calc_metrics: Optional[Dict[str, Any]] = None
    if raw_trials:
        calc_metrics = compute_validation_metrics_from_trials(
            raw_trials,
            screen_width=screen_w,
            screen_height=screen_h,
            threshold_2pct=t2_frac,
            threshold_5pct=t5_frac,
            threshold_10pct=t10_frac,
            default_experiment_id=exp_id,
            default_session_id=session_id,
        )

    expected_count, expected_names = _expected_target_names_for_mode(meta, rep_json, raw_trials)
    recorded_names = [str(r.get("target_name") or "") for r in raw_trials]
    missing_targets = [n for n in expected_names if n not in recorded_names] if expected_names else []

    validation_started = bool(raw_trials) or bool(rep_json) or bool(acc_csv_row) or (
        meta_status in ("IN_PROGRESS", "INTERRUPTED_BY_USER", "NO_VALID_TRIALS", "FAILED", "COMPLETED", "INCOMPLETE")
    )

    # Determine status & final validated metrics strictly without fabricating or mixing training data
    validation_completed = False
    validation_status = "NOT_STARTED"
    failure_reason = ""
    missing_metrics_list: List[str] = []
    next_action = ""

    total_trials = 0
    valid_trials = 0
    rejected_trials = 0
    rejected_trial_pct: Optional[float] = None
    successful_selections: Optional[int] = None
    failed_selections: Optional[int] = None
    sel_acc: Optional[float] = None
    p2_acc: Optional[float] = None
    p5_acc: Optional[float] = None
    p10_acc: Optional[float] = None
    mean_err_px: Optional[float] = None
    median_err_px: Optional[float] = None
    max_err_px: Optional[float] = None
    p90_err_px: Optional[float] = None
    mae_x_px: Optional[float] = None
    mae_y_px: Optional[float] = None
    rmse_x_px: Optional[float] = None
    rmse_y_px: Optional[float] = None
    mean_err_pct_diag: Optional[float] = None
    dir_accs: Dict[str, Dict[str, Any]] = {}
    reg_accs: Dict[str, Dict[str, Any]] = {}
    reg_accs_9: Dict[str, Dict[str, Any]] = {}
    pattern: Dict[str, Any] = {}
    target_boundary = ""

    if calc_metrics is not None:
        raw_trials = calc_metrics["normalized_trials"]
        total_trials = max(int(calc_metrics["total_trials"]), expected_count)
        valid_trials = int(calc_metrics["successful_trials"])
        rejected_trials = int(calc_metrics["rejected_trials"])
        rejected_trial_pct = float(calc_metrics["rejected_trial_pct"])
        dir_accs = dict(calc_metrics.get("direction_accuracies") or {})
        reg_accs = dict(calc_metrics.get("region_accuracies") or {})
        reg_accs_9 = dict(calc_metrics.get("region_accuracies_9") or {})
        pattern = dict(calc_metrics.get("pattern_analysis") or {})
        target_boundary = str(calc_metrics.get("target_boundary_description") or "")

        is_interrupted = (
            meta_status == "INTERRUPTED_BY_USER"
            or (rep_json and str(rep_json.get("validation_status", "")).upper() == "INCOMPLETE")
            or (expected_count > 0 and len(raw_trials) < expected_count)
        )

        if valid_trials == 0:
            validation_status = "FAILED"
            reasons = [str(r.get("rejection_reason") or "") for r in raw_trials if r.get("rejection_reason")]
            top_r = reasons[0] if reasons else (fail_reason_meta or "Zero valid validation trials collected")
            failure_reason = f"0 valid trials out of {len(raw_trials)} recorded ({top_r})."
            missing_metrics_list = ["All validation accuracy and pixel error metrics"]
            next_action = "Re-run validation with eyes open & steady fixation: python main.py --validate-screen"
        elif is_interrupted:
            validation_status = "INCOMPLETE"
            failure_reason = (
                fail_reason_meta
                or f"Validation ended early ({len(raw_trials)} of {expected_count or len(raw_trials)} targets collected)."
            )
            # Still compute preliminary observed metrics? Requirement: incomplete validation is NOT completed and ineligible for ranking!
            missing_metrics_list = ["Full-grid completion (partial trials only)"]
            next_action = "Re-run full 9-region validation without pressing 'q' or 't' early."
        else:
            validation_completed = True
            validation_status = "COMPLETED"
            successful_selections = int(calc_metrics["successful_selections"])
            failed_selections = int(calc_metrics["failed_selections"])
            sel_acc = float(calc_metrics["target_selection_accuracy_pct"])
            p2_acc = float(calc_metrics["pct_within_2pct"])
            p5_acc = float(calc_metrics["pct_within_5pct"])
            p10_acc = float(calc_metrics["pct_within_10pct"])
            mean_err_px = float(calc_metrics["mean_error_pixels"])
            median_err_px = float(calc_metrics["median_error_pixels"])
            max_err_px = float(calc_metrics["max_error_pixels"])
            p90_err_px = float(calc_metrics["p90_error_pixels"]) if calc_metrics.get("p90_error_pixels") is not None else None
            mae_x_px = float(calc_metrics["mae_x_px"])
            mae_y_px = float(calc_metrics["mae_y_px"])
            rmse_x_px = float(calc_metrics["rmse_x_px"])
            rmse_y_px = float(calc_metrics["rmse_y_px"])
            mean_err_pct_diag = float(calc_metrics["mean_error_pct_diagonal"])
            next_action = "Completed cleanly — eligible for comparison."
    elif rep_json or acc_csv_row or isinstance(prof.get("validation_metrics"), dict):
        # Summary files present without validation_results.csv (only accept genuine validation records!)
        src = rep_json or (prof.get("validation_metrics") if isinstance(prof.get("validation_metrics"), dict) else {})
        if src:
            v_cnt = int(src.get("successful_trials") or 0)
            t_cnt = int(src.get("total_trials") or 0)
            r_cnt = int(src.get("rejected_trials") or 0)
            v_stat = str(src.get("validation_status") or ("COMPLETED" if v_cnt > 0 else "FAILED")).upper()
            total_trials = t_cnt
            valid_trials = v_cnt
            rejected_trials = r_cnt
            att = v_cnt + r_cnt
            rejected_trial_pct = round(r_cnt / att * 100.0, 2) if att > 0 else 0.0
            if v_cnt > 0 and v_stat == "COMPLETED":
                validation_completed = True
                validation_status = "COMPLETED"
                metrics = src.get("metrics") or {}
                acc_rates = src.get("accuracy_rates") or {}
                thresh = src.get("threshold_analysis") or {}
                successful_selections = acc_rates.get("successful_selections")
                failed_selections = acc_rates.get("failed_selections")
                sel_acc = acc_rates.get("target_selection_accuracy_pct")
                p2_acc = acc_rates.get("accuracy_within_2pct", thresh.get("pct_within_2pct"))
                p5_acc = acc_rates.get("accuracy_within_5pct", thresh.get("pct_within_5pct"))
                p10_acc = acc_rates.get("accuracy_within_10pct", thresh.get("pct_within_10pct"))
                mean_err_px = metrics.get("mean_euclidean_pixels")
                median_err_px = metrics.get("median_euclidean_pixels")
                max_err_px = metrics.get("max_euclidean_pixels")
                p90_err_px = metrics.get("p90_euclidean_pixels")
                mae_x_px = metrics.get("mae_x_pixels")
                mae_y_px = metrics.get("mae_y_pixels")
                rmse_x_px = metrics.get("rmse_x_pixels")
                rmse_y_px = metrics.get("rmse_y_pixels")
                mean_err_pct_diag = metrics.get("mean_error_pct_diagonal")
                dir_accs = dict(src.get("direction_accuracies") or {})
                reg_accs = dict(src.get("region_accuracies") or {})
                reg_accs_9 = dict(src.get("region_accuracies_9") or {})
                pattern = dict(src.get("pattern_analysis") or {})
                target_boundary = str(acc_rates.get("target_boundary") or "")
                missing_metrics_list = ["validation_results.csv raw trial file not saved on disk"]
                next_action = "Summary loaded from report JSON."
            else:
                validation_status = v_stat if v_stat in ("INCOMPLETE", "FAILED") else "FAILED"
                failure_reason = str(src.get("failure_reason") or "Validation did not complete cleanly.")
                missing_metrics_list = ["Validation accuracy and error metrics"]
                next_action = "Re-run full validation to capture valid trials."
        elif acc_csv_row:
            v_cnt = int(float(acc_csv_row.get("successful_trials") or 0))
            t_cnt = int(float(acc_csv_row.get("total_trials") or 0))
            r_cnt = int(float(acc_csv_row.get("rejected_trials") or 0))
            total_trials = t_cnt
            valid_trials = v_cnt
            rejected_trials = r_cnt
            att = v_cnt + r_cnt
            rejected_trial_pct = round(r_cnt / att * 100.0, 2) if att > 0 else 0.0
            if v_cnt > 0:
                validation_completed = True
                validation_status = "COMPLETED"
                sel_acc = float(acc_csv_row["target_selection_accuracy_pct"]) if acc_csv_row.get("target_selection_accuracy_pct") not in (None, "", "N/A") else None
                p2_acc = float(acc_csv_row["pct_within_2pct_diagonal"]) if acc_csv_row.get("pct_within_2pct_diagonal") not in (None, "", "N/A") else None
                p5_acc = float(acc_csv_row["pct_within_5pct_diagonal"]) if acc_csv_row.get("pct_within_5pct_diagonal") not in (None, "", "N/A") else None
                p10_acc = float(acc_csv_row["pct_within_10pct_diagonal"]) if acc_csv_row.get("pct_within_10pct_diagonal") not in (None, "", "N/A") else None
                mean_err_px = float(acc_csv_row["mean_error_pixels"]) if acc_csv_row.get("mean_error_pixels") not in (None, "", "N/A") else None
                median_err_px = float(acc_csv_row["median_error_pixels"]) if acc_csv_row.get("median_error_pixels") not in (None, "", "N/A") else None
                max_err_px = float(acc_csv_row["max_error_pixels"]) if acc_csv_row.get("max_error_pixels") not in (None, "", "N/A") else None
                successful_selections = int(float(acc_csv_row["successful_selections"])) if acc_csv_row.get("successful_selections") not in (None, "", "N/A") else None
                failed_selections = int(float(acc_csv_row["failed_selections"])) if acc_csv_row.get("failed_selections") not in (None, "", "N/A") else None
                missing_metrics_list = ["Direction & Region breakdown (missing validation_results.csv)"]
                next_action = "Summary loaded from accuracy_summary.csv."
    else:
        # Calibration-only or interrupted session with zero validation files
        if meta_status == "NO_VALID_TRIALS":
            validation_status = "FAILED"
            failure_reason = fail_reason_meta or "Validation ran but recorded 0 valid trials."
            missing_metrics_list = ["All validation metrics (0 valid trials)"]
            next_action = "Re-run validation with steady fixation: python main.py --validate-screen"
        elif meta_status == "INTERRUPTED_BY_USER":
            validation_status = "INCOMPLETE"
            failure_reason = fail_reason_meta or "Validation session exited before completing trials."
            missing_metrics_list = ["All validation metrics (interrupted early)"]
            next_action = "Re-run validation to completion."
        elif meta_status == "IN_PROGRESS":
            validation_status = "IN_PROGRESS"
            failure_reason = "Validation started or queued, but process exited before finalizing results."
            missing_metrics_list = ["validation_results.csv and accuracy_summary.csv"]
            prof_hint = Path(prof["_profile_file"]).name if prof.get("_profile_file") else ""
            if prof_hint:
                next_action = f"Validate saved profile: python main.py --profile data/calibration/{prof_hint} --validate-screen"
            else:
                next_action = "Re-run with: python main.py --calibrate-screen --validate-screen"
        else:
            validation_status = "NOT_STARTED"
            failure_reason = "Calibration completed only; independent validation was not started or saved."
            missing_metrics_list = ["All validation trials and accuracy metrics (calibration-only session)"]
            prof_hint = Path(prof["_profile_file"]).name if prof.get("_profile_file") else ""
            if prof_hint and coord_conv == "interface_left0_right1_v1":
                next_action = f"Run: python main.py --profile data/calibration/{prof_hint} --validate-screen"
            else:
                next_action = "Recalibrate and validate: python main.py --calibrate-screen --validate-screen"

    # Best & weakest directions
    valid_dirs = {
        k: v for k, v in dir_accs.items()
        if isinstance(v, dict) and v.get("valid_trials", 0) > 0 and v.get("target_selection_accuracy_pct") is not None
    }
    best_dir = ""
    weak_dir = ""
    if valid_dirs:
        bd = max(valid_dirs.items(), key=lambda kv: (kv[1]["target_selection_accuracy_pct"], -(kv[1].get("mean_euclidean_pixels") or 1e9)))
        wd = min(valid_dirs.items(), key=lambda kv: (kv[1]["target_selection_accuracy_pct"], -(kv[1].get("mean_euclidean_pixels") or 0.0)))
        best_dir = f"{bd[0]} ({bd[1]['target_selection_accuracy_pct']:.0f}% / {bd[1]['mean_euclidean_pixels']:.0f}px)"
        weak_dir = f"{wd[0]} ({wd[1]['target_selection_accuracy_pct']:.0f}% / {wd[1]['mean_euclidean_pixels']:.0f}px)"
    elif pattern.get("weakest_direction"):
        wd_p = pattern["weakest_direction"]
        if isinstance(wd_p, dict) and wd_p.get("direction"):
            weak_dir = str(wd_p["direction"])

    # Training fit error (kept strictly separated for reference only)
    tr_met = prof.get("training_metrics") or meta.get("training_metrics") or {}
    train_fit_px = float(tr_met["mean_euclidean_pixels"]) if isinstance(tr_met, dict) and tr_met.get("mean_euclidean_pixels") is not None else None

    eligible = bool(validation_completed and valid_trials > 0)
    required_collected = (
        "Yes"
        if (validation_completed and (expected_count == 0 or valid_trials + rejected_trials >= expected_count) and not missing_targets)
        else ("Partial" if (raw_trials or valid_trials > 0) else "No")
    )

    return {
        "experiment_id": exp_id,
        "session_id": session_id,
        "path": str(exp_dir),
        "timestamp": ts,
        "date_time": _format_datetime(ts, exp_id),
        "model_type": model_type,
        "normalization_method": norm_method,
        "screen_resolution": resolution_str,
        "screen_width": screen_w,
        "screen_height": screen_h,
        "coordinate_convention": coord_conv,
        "target_boundary": target_boundary,
        "training_sample_count": train_samples_count,
        "calibration_total_frames": cal_info["total_frames"],
        "calibration_rejected_frames": cal_info["rejected_frames"],
        "calibration_targets_covered": len(cal_info["target_ids"]) if cal_info["exists"] else (len(DEFAULT_9_TARGETS) if prof else 0),
        "training_fit_error_px": round(train_fit_px, 2) if train_fit_px is not None else None,
        "validation_started": validation_started,
        "validation_completed": validation_completed,
        "validation_status": validation_status,
        "required_trials_collected": required_collected,
        "missing_targets": missing_targets,
        "missing_metrics": missing_metrics_list,
        "failure_reason": failure_reason,
        "eligible_for_comparison": eligible,
        "recommended_next_action": next_action,
        "total_trials": total_trials,
        "valid_trials": valid_trials,
        "rejected_trials": rejected_trials,
        "rejected_trial_pct": round(rejected_trial_pct, 2) if rejected_trial_pct is not None else None,
        "successful_selections": successful_selections,
        "failed_selections": failed_selections,
        "target_selection_accuracy_pct": round(sel_acc, 2) if sel_acc is not None else None,
        "pct_within_2pct": round(p2_acc, 2) if p2_acc is not None else None,
        "pct_within_5pct": round(p5_acc, 2) if p5_acc is not None else None,
        "pct_within_10pct": round(p10_acc, 2) if p10_acc is not None else None,
        "mean_error_px": round(mean_err_px, 2) if mean_err_px is not None else None,
        "median_error_px": round(median_err_px, 2) if median_err_px is not None else None,
        "max_error_px": round(max_err_px, 2) if max_err_px is not None else None,
        "p90_error_px": round(p90_err_px, 2) if p90_err_px is not None else None,
        "mae_x_px": round(mae_x_px, 2) if mae_x_px is not None else None,
        "mae_y_px": round(mae_y_px, 2) if mae_y_px is not None else None,
        "rmse_x_px": round(rmse_x_px, 2) if rmse_x_px is not None else None,
        "rmse_y_px": round(rmse_y_px, 2) if rmse_y_px is not None else None,
        "mean_error_pct_diagonal": round(mean_err_pct_diag, 2) if mean_err_pct_diag is not None else None,
        "best_direction": best_dir,
        "weakest_direction": weak_dir,
        "direction_accuracies": dir_accs,
        "region_accuracies": reg_accs,
        "region_accuracies_9": reg_accs_9,
        "pattern_analysis": pattern,
        "trials": raw_trials,
        "calibration_profile_id": cal_prof_id,
    }


def load_tracker_dataset(
    experiments_dir: Path,
    profiles_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Scan all experiment directories and profiles into a unified, deduplicated dataset.
    Preserves original files strictly read-only.
    """
    experiments_dir = Path(experiments_dir)
    if profiles_dir is None and experiments_dir.name == "experiments" and (experiments_dir.parent / "calibration").exists():
        profiles_dir = experiments_dir.parent / "calibration"

    profiles_by_id, all_profiles = _load_profiles_index(profiles_dir)

    records_by_id: Dict[str, Dict[str, Any]] = {}
    empty_dirs: List[str] = []

    if experiments_dir.exists():
        for entry in sorted(experiments_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            parsed = _inspect_single_experiment(entry, profiles_by_id)
            if parsed is None:
                empty_dirs.append(entry.name)
                continue
            records_by_id[parsed["experiment_id"]] = parsed

    # Check for standalone orphan profiles not matched to any experiment directory
    linked_prof_ids = {r.get("calibration_profile_id") for r in records_by_id.values() if r.get("calibration_profile_id")}
    for pdata in all_profiles:
        pid = str(pdata.get("profile_id") or "")
        if not pid or pid in linked_prof_ids:
            continue
        synth_id = f"profile_{pid}"
        if synth_id in records_by_id:
            continue
        ts = float(pdata.get("created_at") or 0.0)
        sw = int(pdata.get("screen_width") or 1920)
        sh = int(pdata.get("screen_height") or 1080)
        records_by_id[synth_id] = {
            "experiment_id": synth_id,
            "session_id": "",
            "path": str(pdata.get("_profile_file") or ""),
            "timestamp": ts,
            "date_time": _format_datetime(ts, synth_id),
            "model_type": str(pdata.get("model_type") or "unknown"),
            "normalization_method": str(pdata.get("normalization_method") or "None"),
            "screen_resolution": f"{sw}x{sh}",
            "screen_width": sw,
            "screen_height": sh,
            "coordinate_convention": str(pdata.get("coordinate_convention") or "unknown"),
            "target_boundary": "",
            "training_sample_count": int(pdata.get("calibration_frame_count") or pdata.get("training_sample_count") or 0) or None,
            "calibration_total_frames": 0,
            "calibration_rejected_frames": 0,
            "calibration_targets_covered": 9,
            "training_fit_error_px": None,
            "validation_started": False,
            "validation_completed": False,
            "validation_status": "NOT_STARTED",
            "required_trials_collected": "No",
            "missing_targets": [],
            "missing_metrics": ["All validation metrics (profile saved without linked validation run)"],
            "failure_reason": "Standalone calibration profile without validation trials.",
            "eligible_for_comparison": False,
            "recommended_next_action": f"Run: python main.py --profile {pdata.get('_profile_file')} --validate-screen",
            "total_trials": 0,
            "valid_trials": 0,
            "rejected_trials": 0,
            "rejected_trial_pct": None,
            "successful_selections": None,
            "failed_selections": None,
            "target_selection_accuracy_pct": None,
            "pct_within_2pct": None,
            "pct_within_5pct": None,
            "pct_within_10pct": None,
            "mean_error_px": None,
            "median_error_px": None,
            "max_error_px": None,
            "p90_error_px": None,
            "mae_x_px": None,
            "mae_y_px": None,
            "rmse_x_px": None,
            "rmse_y_px": None,
            "mean_error_pct_diagonal": None,
            "best_direction": "",
            "weakest_direction": "",
            "direction_accuracies": {},
            "region_accuracies": {},
            "region_accuracies_9": {},
            "pattern_analysis": {},
            "trials": [],
            "calibration_profile_id": pid,
        }

    experiments = sorted(records_by_id.values(), key=lambda r: (r["timestamp"], r["experiment_id"]))
    validated = [e for e in experiments if e["eligible_for_comparison"]]
    validated_ranked = sorted(
        validated,
        key=lambda r: (
            -(r["target_selection_accuracy_pct"] if r["target_selection_accuracy_pct"] is not None else -1.0),
            -(r["pct_within_5pct"] if r["pct_within_5pct"] is not None else -1.0),
            r["mean_error_px"] if r["mean_error_px"] is not None else 1e9,
        ),
    )

    all_trials: List[Dict[str, Any]] = []
    seen_trial_keys: Set[Tuple[str, int]] = set()
    for exp in experiments:
        for tr in exp.get("trials") or []:
            key = (str(tr.get("experiment_id") or exp["experiment_id"]), int(tr.get("trial_id") or 0))
            if key in seen_trial_keys:
                continue
            seen_trial_keys.add(key)
            all_trials.append(tr)

    return {
        "experiments": experiments,
        "validated_ranked": validated_ranked,
        "all_trials": all_trials,
        "empty_dirs": empty_dirs,
        "total_dirs_scanned": len(experiments) + len(empty_dirs),
        "completed_validation_count": len(validated),
        "unvalidated_count": len(experiments) - len(validated),
        "profiles_found_count": len(all_profiles),
    }


def _pct_frac(val: Optional[float]) -> Any:
    """Convert 0-100 percentage into Excel 0-1 numeric fraction, or 'N/A' when unavailable."""
    if val is None or not math.isfinite(float(val)):
        return "N/A"
    return round(float(val) / 100.0, 4)


def _num_or_na(val: Optional[float], digits: int = 1) -> Any:
    if val is None or not math.isfinite(float(val)):
        return "N/A"
    return round(float(val), digits)


def write_central_tracker_workbook(
    dataset: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """
    Generate the formatted 6-sheet AABAR_Experiments_Tracker.xlsx workbook
    from the unified tracker dataset.
    """
    import openpyxl
    from openpyxl.chart import BarChart, Reference
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()

    # Palette
    fill_title = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
    fill_header = PatternFill(start_color="1E3A5F", end_color="1E3A5F", fill_type="solid")
    fill_sub = PatternFill(start_color="E2E8F0", end_color="E2E8F0", fill_type="solid")
    fill_ok = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
    fill_warn = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
    fill_bad = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
    fill_gold = PatternFill(start_color="FEF08A", end_color="FEF08A", fill_type="solid")
    fill_zebra = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")

    font_title = Font(name="Calibri", size=13, bold=True, color="FFFFFF")
    font_sub = Font(name="Calibri", size=10, italic=True, color="334155")
    font_header = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    font_body = Font(name="Calibri", size=10, color="0F172A")
    font_bold = Font(name="Calibri", size=10, bold=True, color="0F172A")
    font_na = Font(name="Calibri", size=10, italic=True, color="92400E")

    thin = Side(style="thin", color="CBD5E1")
    border_cell = Border(left=thin, right=thin, top=thin, bottom=thin)

    align_left = Alignment(horizontal="left", vertical="center", wrap_text=False)
    align_center = Alignment(horizontal="center", vertical="center", wrap_text=False)
    align_right = Alignment(horizontal="right", vertical="center", wrap_text=False)
    align_hdr = Alignment(horizontal="center", vertical="center", wrap_text=True)

    def _style_banner(ws: Any, title: str, subtitle: str, ncols: int) -> None:
        ws.views.sheetView[0].showGridLines = True
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
        ws.cell(row=1, column=1, value=title).font = font_title
        ws.cell(row=1, column=1).fill = fill_title
        ws.cell(row=1, column=1).alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws.row_dimensions[1].height = 26

        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
        ws.cell(row=2, column=1, value=subtitle).font = font_sub
        ws.cell(row=2, column=1).fill = fill_sub
        ws.cell(row=2, column=1).alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws.row_dimensions[2].height = 19

    def _write_header(ws: Any, row_idx: int, headers: List[str]) -> None:
        for c_idx, h_text in enumerate(headers, start=1):
            c = ws.cell(row=row_idx, column=c_idx, value=h_text)
            c.fill = fill_header
            c.font = font_header
            c.alignment = align_hdr
            c.border = border_cell
        ws.row_dimensions[row_idx].height = 28

    def _autosize(ws: Any, min_w: int = 12, max_w: int = 44, max_row_scan: int = 400) -> None:
        for col in ws.columns:
            col_idx = col[0].column
            col_letter = get_column_letter(col_idx)
            longest = min_w
            for cell in col[:max_row_scan]:
                if cell.row in (1, 2):
                    continue
                val_s = str(cell.value if cell.value is not None else "")
                longest = max(longest, min(max_w, len(val_s) + 3))
            ws.column_dimensions[col_letter].width = longest

    def _apply_acc_cf(ws: Any, cell_range: str) -> None:
        ws.conditional_formatting.add(
            cell_range,
            CellIsRule(operator="greaterThanOrEqual", formula=["0.8"], fill=fill_ok, font=Font(name="Calibri", size=10, bold=True, color="14532D")),
        )
        ws.conditional_formatting.add(
            cell_range,
            CellIsRule(operator="between", formula=["0.5", "0.7999"], fill=fill_warn, font=Font(name="Calibri", size=10, color="78350F")),
        )
        ws.conditional_formatting.add(
            cell_range,
            CellIsRule(operator="lessThan", formula=["0.5"], fill=fill_bad, font=Font(name="Calibri", size=10, color="7F1D1D")),
        )

    experiments: List[Dict[str, Any]] = list(dataset.get("experiments") or [])
    validated_ranked: List[Dict[str, Any]] = list(dataset.get("validated_ranked") or [])
    all_trials: List[Dict[str, Any]] = list(dataset.get("all_trials") or [])
    empty_dirs: List[str] = list(dataset.get("empty_dirs") or [])

    # =========================================================================
    # SHEET 1: Experiments Summary
    # =========================================================================
    ws1 = wb.active
    ws1.title = "Experiments Summary"
    s1_headers = [
        "Experiment ID",
        "Date and Time",
        "Model Type",
        "Normalization Method",
        "Screen Resolution",
        "Training Sample Count",
        "Validation Trial Count",
        "Successful Target Selections",
        "Failed Target Selections",
        "Rejected Trials",
        "Target Selection Accuracy (%)",
        "Accuracy within 2% (%)",
        "Accuracy within 5% (%)",
        "Accuracy within 10% (%)",
        "Mean Error (pixels)",
        "Median Error (pixels)",
        "Maximum Error (pixels)",
        "Validation Status",
        "Best Performing Direction",
        "Weakest Performing Direction",
        "Status Note / Next Action",
    ]
    _style_banner(
        ws1,
        "AABAR CV — Central Experiment Accuracy Tracker",
        (
            f"Experiments Imported: {len(experiments)} | Completed Validation: {len(validated_ranked)} | "
            f"Unvalidated (N/A): {len(experiments) - len(validated_ranked)} | "
            "Validation Accuracy strictly uses independent validation trials (never training fit)."
        ),
        len(s1_headers),
    )
    _write_header(ws1, 3, s1_headers)
    ws1.freeze_panes = "B4"

    for r_offset, exp in enumerate(experiments):
        r_idx = 4 + r_offset
        has_val = bool(exp["validation_completed"])
        row_vals = [
            exp["experiment_id"],
            exp["date_time"],
            str(exp["model_type"]).upper(),
            exp["normalization_method"],
            exp["screen_resolution"],
            exp["training_sample_count"] if exp["training_sample_count"] is not None else "N/A",
            exp["valid_trials"] if has_val else (exp["total_trials"] if exp["validation_started"] else 0),
            exp["successful_selections"] if has_val and exp["successful_selections"] is not None else "N/A",
            exp["failed_selections"] if has_val and exp["failed_selections"] is not None else "N/A",
            exp["rejected_trials"] if exp["validation_started"] else 0,
            _pct_frac(exp["target_selection_accuracy_pct"]) if has_val else "N/A",
            _pct_frac(exp["pct_within_2pct"]) if has_val else "N/A",
            _pct_frac(exp["pct_within_5pct"]) if has_val else "N/A",
            _pct_frac(exp["pct_within_10pct"]) if has_val else "N/A",
            _num_or_na(exp["mean_error_px"], 1) if has_val else "N/A",
            _num_or_na(exp["median_error_px"], 1) if has_val else "N/A",
            _num_or_na(exp["max_error_px"], 1) if has_val else "N/A",
            exp["validation_status"],
            exp["best_direction"] or "N/A",
            exp["weakest_direction"] or "N/A",
            exp["recommended_next_action"],
        ]
        for c_idx, val in enumerate(row_vals, start=1):
            c = ws1.cell(row=r_idx, column=c_idx, value=val)
            c.border = border_cell
            c.font = font_na if val == "N/A" else font_body
            c.alignment = align_left if c_idx in (1, 19, 20, 21) else (align_center if c_idx in (2, 3, 4, 5, 18) else align_right)
            if not has_val:
                c.fill = fill_bad if exp["validation_status"] == "FAILED" else fill_warn
            elif r_offset % 2 == 1:
                c.fill = fill_zebra

            if c_idx in (11, 12, 13, 14) and isinstance(val, float):
                c.number_format = "0.0%"
                c.font = font_bold if c_idx == 11 else font_body
            elif c_idx in (15, 16, 17) and isinstance(val, (int, float)):
                c.number_format = "0.0"

        stat_cell = ws1.cell(row=r_idx, column=18)
        stat_cell.font = font_bold
        if has_val:
            stat_cell.fill = fill_ok

    last_s1_row = max(3, 3 + len(experiments))
    ws1.auto_filter.ref = f"A3:{get_column_letter(len(s1_headers))}{last_s1_row}"
    if experiments:
        _apply_acc_cf(ws1, f"K4:N{last_s1_row}")
    _autosize(ws1)

    # =========================================================================
    # SHEET 2: All Test Trials
    # =========================================================================
    ws2 = wb.create_sheet(title="All Test Trials")
    s2_headers = [
        "Experiment ID",
        "Session ID",
        "Trial ID",
        "Timestamp",
        "Target ID",
        "Target X",
        "Target Y",
        "Predicted X",
        "Predicted Y",
        "Actual Direction",
        "Predicted Direction",
        "Pixel Error X",
        "Pixel Error Y",
        "Euclidean Error (pixels)",
        "Error (% of screen diagonal)",
        "Inside Target Boundary (Yes/No)",
        "Trial Status",
        "Rejection Reason",
    ]
    _style_banner(
        ws2,
        "AABAR CV — All Recorded Independent Validation Trials",
        f"Total Validated / Attempted Test Trials Recorded: {len(all_trials)} (zero fabricated trials).",
        len(s2_headers),
    )
    _write_header(ws2, 3, s2_headers)
    ws2.freeze_panes = "B4"

    for t_off, tr in enumerate(all_trials):
        r_idx = 4 + t_off
        ts_val = _format_datetime(float(tr["timestamp"] or 0.0), tr["experiment_id"]) if tr.get("timestamp") else "Unknown"
        hit_label = "Yes" if tr.get("target_hit") is True else ("No" if tr.get("target_hit") is False else "N/A")
        row_vals = [
            tr.get("experiment_id", ""),
            tr.get("session_id") or "N/A",
            int(tr.get("trial_id") or 0),
            ts_val,
            int(tr.get("target_id") or 0),
            round(float(tr["actual_x"]), 4) if tr.get("actual_x") is not None else "N/A",
            round(float(tr["actual_y"]), 4) if tr.get("actual_y") is not None else "N/A",
            round(float(tr["predicted_x"]), 4) if tr.get("predicted_x") is not None else "N/A",
            round(float(tr["predicted_y"]), 4) if tr.get("predicted_y") is not None else "N/A",
            tr.get("actual_direction", "UNKNOWN"),
            tr.get("predicted_direction", "UNKNOWN"),
            round(float(tr["pixel_error_x"]), 2) if tr.get("pixel_error_x") is not None else "N/A",
            round(float(tr["pixel_error_y"]), 2) if tr.get("pixel_error_y") is not None else "N/A",
            round(float(tr["euclidean_error_pixels"]), 2) if tr.get("euclidean_error_pixels") is not None else "N/A",
            round(float(tr["error_pct_diagonal"]), 2) if tr.get("error_pct_diagonal") is not None else "N/A",
            hit_label,
            tr.get("trial_status", ""),
            tr.get("rejection_reason") or "",
        ]
        for c_idx, val in enumerate(row_vals, start=1):
            c = ws2.cell(row=r_idx, column=c_idx, value=val)
            c.border = border_cell
            c.font = font_na if val == "N/A" else font_body
            c.alignment = align_left if c_idx in (1, 18) else (align_center if c_idx in (2, 4, 10, 11, 16, 17) else align_right)
            if isinstance(val, float) and c_idx in (6, 7, 8, 9):
                c.number_format = "0.0000"
            elif isinstance(val, float) and c_idx in (12, 13, 14, 15):
                c.number_format = "0.00"
            if hit_label == "Yes" and c_idx == 16:
                c.fill = fill_ok
            elif hit_label == "No" and c_idx == 16:
                c.fill = fill_bad
            if tr.get("trial_status") == "REJECTED" and c_idx == 17:
                c.fill = fill_bad
            elif tr.get("trial_status") == "VALID" and c_idx == 17:
                c.fill = fill_ok

    last_s2_row = max(3, 3 + len(all_trials))
    ws2.auto_filter.ref = f"A3:{get_column_letter(len(s2_headers))}{last_s2_row}"
    _autosize(ws2)

    # =========================================================================
    # SHEET 3: Direction Accuracy
    # =========================================================================
    ws3 = wb.create_sheet(title="Direction Accuracy")
    s3_headers = [
        "Experiment ID",
        "Direction",
        "Total Valid Trials",
        "Successful Predictions",
        "Failed Predictions",
        "Target Selection Accuracy (%)",
        "Direction Match Accuracy (%)",
        "Mean Error (pixels)",
        "Median Error (pixels)",
        "Rejected Trials",
    ]
    _style_banner(
        ws3,
        "AABAR CV — Gaze Direction Accuracy (LEFT / RIGHT / CENTER / UP / DOWN)",
        "Evaluated separately per actual direction. Direction Match Accuracy (%) is tracked distinctly from Target Selection Accuracy (%).",
        len(s3_headers),
    )
    _write_header(ws3, 3, s3_headers)
    ws3.freeze_panes = "B4"

    s3_row = 4
    dir_agg_acc: Dict[str, List[float]] = {d: [] for d in CANONICAL_DIRECTIONS}
    dir_agg_err: Dict[str, List[float]] = {d: [] for d in CANONICAL_DIRECTIONS}

    for exp in experiments:
        has_val = bool(exp["validation_completed"])
        dmap = exp.get("direction_accuracies") or {}
        for d_name in CANONICAL_DIRECTIONS:
            dinfo = dmap.get(d_name) if isinstance(dmap, dict) else None
            if has_val and dinfo and dinfo.get("valid_trials", 0) > 0:
                sel_val = _pct_frac(dinfo.get("target_selection_accuracy_pct"))
                dmatch_val = _pct_frac(dinfo.get("direction_match_accuracy_pct"))
                mean_px = _num_or_na(dinfo.get("mean_euclidean_pixels"), 1)
                med_px = _num_or_na(dinfo.get("median_euclidean_pixels"), 1)
                if isinstance(sel_val, float):
                    dir_agg_acc[d_name].append(sel_val)
                if isinstance(mean_px, (int, float)):
                    dir_agg_err[d_name].append(float(mean_px))
                vals = [
                    exp["experiment_id"],
                    d_name,
                    int(dinfo.get("valid_trials", 0)),
                    int(dinfo.get("successful_selections", 0)),
                    int(dinfo.get("failed_selections", 0)),
                    sel_val,
                    dmatch_val,
                    mean_px,
                    med_px,
                    int(dinfo.get("rejected_trials", 0)),
                ]
            else:
                rej_cnt = int(dinfo.get("rejected_trials", 0)) if dinfo else 0
                vals = [
                    exp["experiment_id"],
                    d_name,
                    0,
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    rej_cnt,
                ]
            for c_idx, val in enumerate(vals, start=1):
                c = ws3.cell(row=s3_row, column=c_idx, value=val)
                c.border = border_cell
                c.font = font_na if val == "N/A" else font_body
                c.alignment = align_left if c_idx == 1 else (align_center if c_idx == 2 else align_right)
                if not has_val:
                    c.fill = fill_warn
                if c_idx in (6, 7) and isinstance(val, float):
                    c.number_format = "0.0%"
                elif c_idx in (8, 9) and isinstance(val, (int, float)):
                    c.number_format = "0.0"
            s3_row += 1

    last_s3_row = max(3, s3_row - 1)
    ws3.auto_filter.ref = f"A3:{get_column_letter(len(s3_headers))}{last_s3_row}"
    if experiments:
        _apply_acc_cf(ws3, f"F4:G{last_s3_row}")

    # Aggregate Direction Summary Table (Columns L-O) for clean Direction Accuracy Chart
    agg_headers = ["Direction", "Validated Sessions", "Avg Target Selection Acc (%)", "Avg Mean Error (px)"]
    for c_off, h_text in enumerate(agg_headers):
        c = ws3.cell(row=3, column=12 + c_off, value=h_text)
        c.fill = fill_header
        c.font = font_header
        c.alignment = align_hdr
        c.border = border_cell
    for d_off, d_name in enumerate(CANONICAL_DIRECTIONS):
        r_idx = 4 + d_off
        acc_list = dir_agg_acc[d_name]
        err_list = dir_agg_err[d_name]
        avg_a = round(sum(acc_list) / len(acc_list), 4) if acc_list else "N/A"
        avg_e = round(sum(err_list) / len(err_list), 1) if err_list else "N/A"
        for c_off, val in enumerate([d_name, len(acc_list), avg_a, avg_e]):
            c = ws3.cell(row=r_idx, column=12 + c_off, value=val)
            c.border = border_cell
            c.font = font_na if val == "N/A" else font_bold if c_off == 0 else font_body
            c.alignment = align_center if c_off == 0 else align_right
            if c_off == 2 and isinstance(val, float):
                c.number_format = "0.0%"
            elif c_off == 3 and isinstance(val, (int, float)):
                c.number_format = "0.0"

    if validated_ranked:
        dir_chart = BarChart()
        dir_chart.type = "col"
        dir_chart.title = "Direction Target Selection Accuracy (%)"
        dir_chart.y_axis.title = "Accuracy"
        dir_chart.y_axis.number_format = "0%"
        dir_chart.y_axis.scaling.min = 0.0
        dir_chart.y_axis.scaling.max = 1.0
        dir_chart.x_axis.title = "Direction"
        data_ref = Reference(ws3, min_col=14, min_row=3, max_row=8)
        cats_ref = Reference(ws3, min_col=12, min_row=4, max_row=8)
        dir_chart.add_data(data_ref, titles_from_data=True)
        dir_chart.set_categories(cats_ref)
        dir_chart.legend = None
        dir_chart.width = 16
        dir_chart.height = 9
        ws3.add_chart(dir_chart, "L10")

    _autosize(ws3)

    # =========================================================================
    # SHEET 4: Screen Region Accuracy (All 9 Calibration Regions)
    # =========================================================================
    ws4 = wb.create_sheet(title="Screen Region Accuracy")
    s4_headers = [
        "Experiment ID",
        "Screen Region",
        "Valid Trials",
        "Successful Selections",
        "Failed Selections",
        "Target Selection Accuracy (%)",
        "Mean Pixel Error (px)",
        "Median Pixel Error (px)",
        "Rejected Trials",
    ]
    _style_banner(
        ws4,
        "AABAR CV — 9-Region Screen Accuracy Breakdown",
        "Performance evaluated separately across all 9 screen regions (Top/Middle/Bottom x Left/Center/Right).",
        len(s4_headers),
    )
    _write_header(ws4, 3, s4_headers)
    ws4.freeze_panes = "B4"

    s4_row = 4
    for exp in experiments:
        has_val = bool(exp["validation_completed"])
        rmap9 = exp.get("region_accuracies_9") or {}
        for reg_name in CANONICAL_SCREEN_REGIONS_9:
            rinfo = rmap9.get(reg_name) if isinstance(rmap9, dict) else None
            if has_val and rinfo and rinfo.get("valid_trials", 0) > 0:
                vals = [
                    exp["experiment_id"],
                    reg_name,
                    int(rinfo.get("valid_trials", 0)),
                    int(rinfo.get("successful_selections", 0)),
                    int(rinfo.get("failed_selections", 0)),
                    _pct_frac(rinfo.get("target_selection_accuracy_pct")),
                    _num_or_na(rinfo.get("mean_euclidean_pixels"), 1),
                    _num_or_na(rinfo.get("median_euclidean_pixels"), 1),
                    int(rinfo.get("rejected_trials", 0)),
                ]
            else:
                vals = [
                    exp["experiment_id"],
                    reg_name,
                    0,
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    int(rinfo.get("rejected_trials", 0)) if rinfo else 0,
                ]
            for c_idx, val in enumerate(vals, start=1):
                c = ws4.cell(row=s4_row, column=c_idx, value=val)
                c.border = border_cell
                c.font = font_na if val == "N/A" else font_body
                c.alignment = align_left if c_idx in (1, 2) else align_right
                if not has_val:
                    c.fill = fill_warn
                if c_idx == 6 and isinstance(val, float):
                    c.number_format = "0.0%"
                elif c_idx in (7, 8) and isinstance(val, (int, float)):
                    c.number_format = "0.0"
            s4_row += 1

    last_s4_row = max(3, s4_row - 1)
    ws4.auto_filter.ref = f"A3:{get_column_letter(len(s4_headers))}{last_s4_row}"
    if experiments:
        _apply_acc_cf(ws4, f"F4:F{last_s4_row}")

    # 3x3 Best Validated Screen Grid Summary on the right (Columns K-N)
    grid_title_exp = validated_ranked[0]["experiment_id"] if validated_ranked else "No Validated Experiment Yet"
    ws4.cell(row=3, column=11, value=f"Best Model 3x3 Grid ({grid_title_exp})").font = font_header
    ws4.cell(row=3, column=11).fill = fill_header
    for c_i, hdr in enumerate(["Left Col", "Center Col", "Right Col"], start=12):
        cell = ws4.cell(row=3, column=c_i, value=hdr)
        cell.fill = fill_header
        cell.font = font_header
        cell.alignment = align_hdr
        cell.border = border_cell

    grid_rows = [
        ("Top Row", ("Top Left", "Top Center", "Top Right")),
        ("Middle Row", ("Middle Left", "Center", "Middle Right")),
        ("Bottom Row", ("Bottom Left", "Bottom Center", "Bottom Right")),
    ]
    best_r9 = validated_ranked[0].get("region_accuracies_9", {}) if validated_ranked else {}
    for idx_r, (row_lbl, cell_names) in enumerate(grid_rows, start=4):
        lbl_cell = ws4.cell(row=idx_r, column=11, value=row_lbl)
        lbl_cell.font = font_bold
        lbl_cell.border = border_cell
        lbl_cell.alignment = align_left
        for idx_c, reg_lbl in enumerate(cell_names, start=12):
            ri = best_r9.get(reg_lbl)
            if ri and ri.get("valid_trials", 0) > 0:
                txt = f"{ri['target_selection_accuracy_pct']:.0f}% ({ri['mean_euclidean_pixels']:.0f}px)"
            else:
                txt = "N/A"
            gc = ws4.cell(row=idx_r, column=idx_c, value=txt)
            gc.border = border_cell
            gc.alignment = align_center
            gc.font = font_na if txt == "N/A" else font_body

    _autosize(ws4)

    # =========================================================================
    # SHEET 5: Experiment Comparison (Validation Only; Never Rank Unvalidated)
    # =========================================================================
    ws5 = wb.create_sheet(title="Experiment Comparison")
    s5_headers = [
        "Validation Rank",
        "Experiment ID",
        "Model Type",
        "Normalization",
        "Resolution",
        "Target Selection Acc (%)",
        "Within 2% (%)",
        "Within 5% (%)",
        "Within 10% (%)",
        "Mean Error (px)",
        "Median Error (px)",
        "LEFT Acc (%)",
        "RIGHT Acc (%)",
        "CENTER Acc (%)",
        "UP Acc (%)",
        "DOWN Acc (%)",
        "Center Region Acc (%)",
        "Edge Regions Acc (%)",
        "Rejected Trials (%)",
        "Ranking Highlight / Notes",
    ]
    _style_banner(
        ws5,
        "AABAR CV — Objective Experiment Comparison (Validated Results Only)",
        "Ranking Criterion: Highest Target Selection Accuracy (%), then Accuracy within 5%, then Lowest Mean Error (px). Unvalidated runs are NEVER ranked.",
        len(s5_headers),
    )
    _write_header(ws5, 3, s5_headers)
    ws5.freeze_panes = "C4"

    s5_row = 4
    for rank_idx, exp in enumerate(validated_ranked, start=1):
        dmap = exp.get("direction_accuracies") or {}
        rmap = exp.get("region_accuracies") or {}
        ctr_r = rmap.get("center_region", {}) if isinstance(rmap, dict) else {}
        edge_r = rmap.get("edge_regions", {}) if isinstance(rmap, dict) else {}
        note = "BEST VALIDATED EXPERIMENT" if rank_idx == 1 else "Eligible validated run"
        vals = [
            rank_idx,
            exp["experiment_id"],
            str(exp["model_type"]).upper(),
            exp["normalization_method"],
            exp["screen_resolution"],
            _pct_frac(exp["target_selection_accuracy_pct"]),
            _pct_frac(exp["pct_within_2pct"]),
            _pct_frac(exp["pct_within_5pct"]),
            _pct_frac(exp["pct_within_10pct"]),
            _num_or_na(exp["mean_error_px"], 1),
            _num_or_na(exp["median_error_px"], 1),
            _pct_frac(dmap.get("LEFT", {}).get("target_selection_accuracy_pct")),
            _pct_frac(dmap.get("RIGHT", {}).get("target_selection_accuracy_pct")),
            _pct_frac(dmap.get("CENTER", {}).get("target_selection_accuracy_pct")),
            _pct_frac(dmap.get("UP", {}).get("target_selection_accuracy_pct")),
            _pct_frac(dmap.get("DOWN", {}).get("target_selection_accuracy_pct")),
            _pct_frac(ctr_r.get("target_selection_accuracy_pct")),
            _pct_frac(edge_r.get("target_selection_accuracy_pct")),
            _pct_frac(exp["rejected_trial_pct"]),
            note,
        ]
        for c_idx, val in enumerate(vals, start=1):
            c = ws5.cell(row=s5_row, column=c_idx, value=val)
            c.border = border_cell
            c.font = font_bold if rank_idx == 1 else (font_na if val == "N/A" else font_body)
            c.alignment = align_left if c_idx in (2, 20) else (align_center if c_idx in (1, 3, 4, 5) else align_right)
            if rank_idx == 1:
                c.fill = fill_gold if c_idx in (1, 20) else fill_ok
            if c_idx in (6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19) and isinstance(val, float):
                c.number_format = "0.0%"
            elif c_idx in (10, 11) and isinstance(val, (int, float)):
                c.number_format = "0.0"
        s5_row += 1

    last_val_s5_row = s5_row - 1
    if validated_ranked:
        _apply_acc_cf(ws5, f"F4:I{last_val_s5_row}")

    # List unvalidated experiments cleanly below without ranking
    unvalidated = [e for e in experiments if not e["eligible_for_comparison"]]
    if unvalidated:
        if s5_row == 4:
            ws5.cell(row=s5_row, column=1, value="--").alignment = align_center
            ws5.cell(row=s5_row, column=2, value="No experiments have completed validation yet.").font = font_na
            ws5.merge_cells(start_row=s5_row, start_column=2, end_row=s5_row, end_column=len(s5_headers))
            s5_row += 1
        s5_row += 1
        ws5.merge_cells(start_row=s5_row, start_column=1, end_row=s5_row, end_column=len(s5_headers))
        sec_hdr = ws5.cell(
            row=s5_row,
            column=1,
            value="Excluded from Comparison Ranking — Validation Incomplete / Not Started (Never Assigned Example or Training Scores)",
        )
        sec_hdr.font = font_bold
        sec_hdr.fill = fill_warn
        s5_row += 1
        for exp in unvalidated:
            excl_vals = [
                "Not Ranked",
                exp["experiment_id"],
                str(exp["model_type"]).upper(),
                exp["normalization_method"],
                exp["screen_resolution"],
                "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
                "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
                _pct_frac(exp["rejected_trial_pct"]) if exp["validation_started"] else "N/A",
                f"[{exp['validation_status']}] {exp['failure_reason'] or exp['recommended_next_action']}",
            ]
            for c_idx, val in enumerate(excl_vals, start=1):
                c = ws5.cell(row=s5_row, column=c_idx, value=val)
                c.border = border_cell
                c.fill = fill_warn if exp["validation_status"] != "FAILED" else fill_bad
                c.font = font_na if val in ("N/A", "Not Ranked") else font_body
                c.alignment = align_left if c_idx in (2, 20) else (align_center if c_idx in (1, 3, 4, 5) else align_right)
            s5_row += 1

    last_s5_row = max(3, s5_row - 1)
    ws5.auto_filter.ref = f"A3:{get_column_letter(len(s5_headers))}{last_s5_row}"

    # Add Target Selection Accuracy Chart onto Sheet 1 anchored cleanly below Sheet 1 rows (reading clean Validated rows on Sheet 5!)
    if validated_ranked:
        comp_chart = BarChart()
        comp_chart.type = "col"
        comp_chart.title = "Target Selection Accuracy Comparison — Completed Experiments (%)"
        comp_chart.y_axis.title = "Target Selection Accuracy"
        comp_chart.y_axis.number_format = "0%"
        comp_chart.y_axis.scaling.min = 0.0
        comp_chart.y_axis.scaling.max = 1.0
        comp_chart.x_axis.title = "Experiment ID"
        data_ref = Reference(ws5, min_col=6, min_row=3, max_row=3 + len(validated_ranked))
        cats_ref = Reference(ws5, min_col=2, min_row=4, max_row=3 + len(validated_ranked))
        comp_chart.add_data(data_ref, titles_from_data=True)
        comp_chart.set_categories(cats_ref)
        comp_chart.legend = None
        comp_chart.width = 18
        comp_chart.height = 10
        ws1.add_chart(comp_chart, f"A{last_s1_row + 3}")
    else:
        note_r = last_s1_row + 3
        ws1.merge_cells(start_row=note_r, start_column=1, end_row=note_r, end_column=10)
        nc = ws1.cell(
            row=note_r,
            column=1,
            value=(
                "Comparison chart will populate automatically once at least 1 experiment completes independent validation "
                "(no placeholder or fake accuracy bars are shown)."
            ),
        )
        nc.font = font_na
        nc.fill = fill_warn

    _autosize(ws5)

    # =========================================================================
    # SHEET 6: Data Quality & Validation
    # =========================================================================
    ws6 = wb.create_sheet(title="Data Quality & Validation")
    s6_headers = [
        "Experiment ID",
        "Validation Started?",
        "Validation Completed?",
        "Required Trials Collected?",
        "Missing Targets",
        "Invalid / Rejected Trials",
        "Missing Metrics",
        "Failure Reason",
        "Eligible for Comparison?",
        "Calibration Frames (Acc / Rej)",
        "Recommended Next Action",
    ]
    _style_banner(
        ws6,
        "AABAR CV — Data Quality & Validation Diagnostics",
        (
            f"Explains missing accuracy per experiment without hiding failures. "
            f"Empty 0-file startup directories excluded from import: {len(empty_dirs)}."
        ),
        len(s6_headers),
    )
    _write_header(ws6, 3, s6_headers)
    ws6.freeze_panes = "B4"

    for r_off, exp in enumerate(experiments):
        r_idx = 4 + r_off
        miss_t_str = ", ".join(exp["missing_targets"]) if exp["missing_targets"] else ("None (full validation missing)" if not exp["validation_completed"] else "None")
        miss_m_str = "; ".join(exp["missing_metrics"]) if exp["missing_metrics"] else "None"
        rej_str = (
            f"{exp['rejected_trials']} / {exp['valid_trials'] + exp['rejected_trials']} trials"
            if exp["validation_started"]
            else "N/A (0 attempted)"
        )
        cal_fr_str = (
            f"{exp['training_sample_count'] or 0} acc / {exp['calibration_rejected_frames']} rej"
            if exp["calibration_total_frames"] > 0
            else (f"{exp['training_sample_count']} samples" if exp["training_sample_count"] else "N/A")
        )
        row_vals = [
            exp["experiment_id"],
            "Yes" if exp["validation_started"] else "No",
            "Yes" if exp["validation_completed"] else "No",
            exp["required_trials_collected"],
            miss_t_str,
            rej_str,
            miss_m_str,
            exp["failure_reason"] or "None",
            "Yes" if exp["eligible_for_comparison"] else "No",
            cal_fr_str,
            exp["recommended_next_action"],
        ]
        for c_idx, val in enumerate(row_vals, start=1):
            c = ws6.cell(row=r_idx, column=c_idx, value=val)
            c.border = border_cell
            c.font = font_bold if c_idx in (1, 9) else font_body
            c.alignment = align_center if c_idx in (2, 3, 4, 6, 9, 10) else align_left
            if c_idx == 9:
                c.fill = fill_ok if exp["eligible_for_comparison"] else (fill_bad if exp["validation_status"] == "FAILED" else fill_warn)
            elif not exp["validation_completed"] and r_off % 2 == 1:
                c.fill = fill_zebra

    last_s6_row = max(3, 3 + len(experiments))
    ws6.auto_filter.ref = f"A3:{get_column_letter(len(s6_headers))}{last_s6_row}"
    _autosize(ws6, max_w=52)

    wb.save(output_path)
    logger.info(f"Updated Central Experiment Tracker workbook: {output_path.resolve()} ({len(experiments)} experiments, {len(validated_ranked)} validated)")
    return output_path


def update_central_tracker(
    experiments_dir: Path,
    tracker_path: Optional[Path] = None,
    profiles_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Scan all historical & new experiment data and regenerate AABAR_Experiments_Tracker.xlsx."""
    experiments_dir = Path(experiments_dir)
    target_xlsx = Path(tracker_path) if tracker_path else (experiments_dir / TRACKER_FILENAME)
    dataset = load_tracker_dataset(experiments_dir=experiments_dir, profiles_dir=profiles_dir)
    written = write_central_tracker_workbook(dataset, target_xlsx)
    return {
        "tracker_path": written,
        "dataset": dataset,
    }


def format_experiment_results_summary(
    exp: Mapping[str, Any],
    tracker_path: Optional[Path] = None,
) -> str:
    """
    Format a prominent, easy-to-read summary for direct display in the CV application CLI.
    Guarantees the exact same metrics as HTML/MD reports and AABAR_Experiments_Tracker.xlsx.
    """
    lines = [
        "=" * 72,
        f" AABAR CV EXPERIMENT RESULTS SUMMARY: {exp['experiment_id']}",
        "=" * 72,
        f" Date & Time:               {exp.get('date_time', 'Unknown')}",
        f" Model / Normalization:     {str(exp.get('model_type', 'unknown')).upper()} | {exp.get('normalization_method', 'None')}",
        f" Screen Resolution:         {exp.get('screen_resolution', 'Unknown')}",
        f" Training Samples:          {exp.get('training_sample_count') if exp.get('training_sample_count') is not None else 'N/A'}",
        f" Validation Status:         [{exp.get('validation_status', 'NOT_STARTED')}]",
        "-" * 72,
    ]
    if exp.get("validation_completed"):
        lines.extend([
            f" Target Selection Accuracy: {exp['target_selection_accuracy_pct']:.1f}% ({exp['successful_selections']}/{exp['valid_trials']} hits inside boundary)",
            f" Accuracy within 2% Diag:   {exp['pct_within_2pct']:.1f}%",
            f" Accuracy within 5% Diag:   {exp['pct_within_5pct']:.1f}%",
            f" Accuracy within 10% Diag:  {exp['pct_within_10pct']:.1f}%",
            f" Mean Error (pixels):       {exp['mean_error_px']:.1f} px ({exp.get('mean_error_pct_diagonal', 0.0):.1f}% diag)",
            f" Median / Max Error (px):   {exp['median_error_px']:.1f} px / {exp['max_error_px']:.1f} px",
            f" Best / Weakest Direction:  {exp.get('best_direction') or 'N/A'} | {exp.get('weakest_direction') or 'N/A'}",
        ])
    else:
        lines.extend([
            f" Target Selection Accuracy: N/A",
            f" Accuracy within 2% / 5% / 10%: N/A",
            f" Mean Error (pixels):       N/A",
            f" Reason:                    {exp.get('failure_reason') or _EMPTY_EXP_NOTICE}",
            f" Next Action:               {exp.get('recommended_next_action')}",
        ])
    if tracker_path is not None:
        lines.append(f" Central Excel Tracker:     {tracker_path}")
    lines.append("=" * 72)
    return "\n".join(lines)
