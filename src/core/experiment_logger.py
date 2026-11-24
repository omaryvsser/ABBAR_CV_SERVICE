"""
Experiment Logger, Simple Accuracy Reports & Comparison Engine for AABAR CV Service (Phase 3).

Persists structured calibration samples, validation trials, evaluation metrics,
human-readable reports (Markdown, HTML, TXT), plots, and experiment metadata into
organized session directories:

data/experiments/experiment_YYYYMMDD_HHMMSS/
  ├── calibration_samples.csv
  ├── validation_results.csv          (when validation trials exist)
  ├── accuracy_summary.csv            (when validation completed)
  ├── experiment_metadata.json
  ├── accuracy_report.json            (when validation completed)
  ├── accuracy_report.md              (Simple, plain-language Markdown summary)
  ├── accuracy_report.html            (User-friendly browser-ready HTML report)
  ├── accuracy_report.txt             (Formatted ASCII summary)
  ├── plot_actual_vs_predicted.png    (Scatter plot & error vectors)
  ├── plot_error_distributions.png    (Error histogram & signed bias scatter)
  ├── plot_per_target_breakdown.png   (Per-target bar chart with thresholds)
  └── experiment_report.xlsx          (Optional, multi-sheet workbook)
"""

import csv
import html
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .calibration_profile import CalibrationProfile
from .screen_calibration import CalibrationSampleRecord
from .screen_validator import AccuracySummary, ValidationTrialRecord

logger = logging.getLogger(__name__)


VALIDATION_NOT_COMPLETED_MSG = "Validation not completed — Accuracy is not available yet."


def _resolve_validation_status_info(
    summary: Optional[AccuracySummary],
    trials: Optional[List[ValidationTrialRecord]],
    metadata_extra: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Determine explicit validation status among:
    COMPLETED, FAILED, INCOMPLETE, IN_PROGRESS, NOT_STARTED
    alongside a clear user-facing explanation reason.
    """
    extra = metadata_extra or {}
    raw_status = str(extra.get("status", "")).upper()
    trial_list = trials or []

    if summary is not None and summary.successful_trials > 0 and getattr(summary, "validation_status", "COMPLETED") == "COMPLETED":
        if summary.rejected_trials > 0:
            reason = (
                f"Validation completed cleanly ({summary.successful_trials}/{summary.total_trials} planned trials valid; "
                f"{summary.rejected_trials} trial(s) rejected)."
            )
        else:
            reason = f"Validation completed cleanly across all {summary.successful_trials} valid trial(s)."
        return "COMPLETED", reason

    if raw_status == "IN_PROGRESS":
        return "IN_PROGRESS", f"Validation currently in progress ({len(trial_list)} trial(s) recorded so far)."

    if summary is not None and summary.validation_status == "INCOMPLETE":
        return "INCOMPLETE", summary.failure_reason or "Validation incomplete: required planned trials were missing."

    if raw_status == "NO_VALID_TRIALS" or (trial_list and all(t.trial_status == "REJECTED" for t in trial_list)):
        reasons = [t.rejection_reason for t in trial_list if t.rejection_reason]
        top_reason = reasons[0] if reasons else "insufficient stable gaze samples or eye blinks"
        return (
            "FAILED",
            f"Validation FAILED across {len(trial_list)} attempted trial(s) with 0 valid fixations ({top_reason}).",
        )

    if raw_status == "INTERRUPTED_BY_USER" or trial_list:
        v_cnt = sum(1 for t in trial_list if t.trial_status == "VALID")
        r_cnt = sum(1 for t in trial_list if t.trial_status == "REJECTED")
        return (
            "INCOMPLETE",
            f"Validation was interrupted before completion ({v_cnt} valid, {r_cnt} rejected trials saved).",
        )

    if extra.get("failure_reason"):
        return "FAILED", str(extra["failure_reason"])

    return (
        "NOT_STARTED",
        "Calibration finished, but independent validation has not been started yet (press 't' in GUI or run --validate-screen).",
    )


def _fmt_pct_threshold(frac: float) -> str:
    """Format threshold fraction (e.g. 0.02 -> '2%', 0.025 -> '2.5%')."""
    pct = frac * 100.0
    if abs(pct - round(pct)) < 1e-6:
        return f"{int(round(pct))}%"
    return f"{pct:.1f}%"


def _extract_training_error_summary(
    accuracy_summary: Optional[AccuracySummary],
    profile: Optional[CalibrationProfile],
    metadata_extra: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Safely extract calibration training fit metrics (distinct from validation accuracy)."""
    if accuracy_summary and accuracy_summary.training_metrics:
        return dict(accuracy_summary.training_metrics)
    if profile and profile.training_metrics:
        return dict(profile.training_metrics)
    if metadata_extra and isinstance(metadata_extra.get("training_metrics"), dict):
        return dict(metadata_extra["training_metrics"])
    return None


def _parse_experiment_record(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Safely read summary metrics and validation status from an experiment folder.

    Returns None only if no recognizable metadata/summary files exist.
    Never treats training_metrics as validation accuracy!
    """
    exp_dir = Path(exp_dir)
    if not exp_dir.is_dir():
        return None

    acc_json = exp_dir / "accuracy_report.json"
    acc_csv = exp_dir / "accuracy_summary.csv"
    meta_json = exp_dir / "experiment_metadata.json"
    calib_csv = exp_dir / "calibration_samples.csv"

    if not (acc_json.exists() or acc_csv.exists() or meta_json.exists()):
        return None

    rec: Dict[str, Any] = {
        "experiment_id": exp_dir.name,
        "path": str(exp_dir),
        "timestamp": 0.0,
        "model": "unknown",
        "normalization": "None",
        "resolution": "unknown",
        "coordinate_convention": "unknown",
        "target_boundary": "",
        "validation_completed": False,
        "validation_status": "NOT_STARTED",
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
        "rmse_x_px": None,
        "rmse_y_px": None,
        "mean_error_pct_diagonal": None,
        "weakest_direction": None,
        "weakest_region": None,
        "status": "Validation not completed",
    }

    meta_raw_status = ""
    if meta_json.exists():
        try:
            with open(meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            rec["timestamp"] = float(meta.get("created_at", 0.0) or 0.0)
            meta_raw_status = str(meta.get("status", "")).upper()
            prof = meta.get("profile") or {}
            if prof:
                rec["model"] = prof.get("model_type", rec["model"])
                rec["normalization"] = prof.get("normalization_method", rec["normalization"])
                rec["coordinate_convention"] = prof.get("coordinate_convention", "legacy_mirrored_target_overlay")
                sw = prof.get("screen_width", meta.get("screen_width", 1920))
                sh = prof.get("screen_height", meta.get("screen_height", 1080))
                rec["resolution"] = f"{sw}x{sh}"
            elif "model_type" in meta:
                rec["model"] = meta.get("model_type", rec["model"])
                rec["normalization"] = meta.get("normalization_method", rec["normalization"])
                sw = meta.get("screen_width", 1920)
                sh = meta.get("screen_height", 1080)
                rec["resolution"] = f"{sw}x{sh}"
        except Exception:
            pass

    if acc_json.exists():
        try:
            with open(acc_json, "r", encoding="utf-8") as f:
                rep = json.load(f)
            valid_trials = int(rep.get("successful_trials", 0) or 0)
            total_trials = int(rep.get("total_trials", 0) or 0)
            rejected_trials = int(rep.get("rejected_trials", 0) or 0)
            rec["model"] = rep.get("model_name", rec["model"])
            rec["normalization"] = rep.get("normalization_method", rec["normalization"])
            rec["resolution"] = rep.get("screen_resolution", rec["resolution"])
            if rep.get("coordinate_convention"):
                rec["coordinate_convention"] = str(rep["coordinate_convention"])
            rec["timestamp"] = float(rep.get("evaluation_timestamp", rec["timestamp"]) or rec["timestamp"])
            rec["total_trials"] = total_trials
            rec["valid_trials"] = valid_trials
            rec["rejected_trials"] = rejected_trials
            attempted = valid_trials + rejected_trials
            rec["rejected_trial_pct"] = (
                float(rep["rejected_trial_pct"])
                if rep.get("rejected_trial_pct") is not None
                else (round(float(rejected_trials / attempted * 100.0), 2) if attempted > 0 else 0.0)
            )

            val_stat = str(rep.get("validation_status", "COMPLETED" if valid_trials > 0 else "FAILED")).upper()
            if valid_trials > 0 and val_stat == "COMPLETED":
                rec["validation_completed"] = True
                rec["validation_status"] = "COMPLETED"
                rec["status"] = "Validated"
                metrics = rep.get("metrics") or {}
                thresh = rep.get("threshold_analysis") or {}
                acc_rates = rep.get("accuracy_rates") or {}
                patt = rep.get("pattern_analysis") or {}

                rec["target_boundary"] = str(acc_rates.get("target_boundary", "") or "")
                wd = patt.get("weakest_direction")
                wr = patt.get("weakest_region")
                if isinstance(wd, dict):
                    rec["weakest_direction"] = wd.get("direction")
                if isinstance(wr, dict):
                    rec["weakest_region"] = wr.get("region")

                rec["mean_error_px"] = float(metrics["mean_euclidean_pixels"]) if metrics.get("mean_euclidean_pixels") is not None else None
                rec["median_error_px"] = float(metrics["median_euclidean_pixels"]) if metrics.get("median_euclidean_pixels") is not None else None
                rec["max_error_px"] = float(metrics["max_euclidean_pixels"]) if metrics.get("max_euclidean_pixels") is not None else None
                rec["p90_error_px"] = float(metrics["p90_euclidean_pixels"]) if metrics.get("p90_euclidean_pixels") is not None else None
                rec["rmse_x_px"] = float(metrics["rmse_x_pixels"]) if metrics.get("rmse_x_pixels") is not None else None
                rec["rmse_y_px"] = float(metrics["rmse_y_pixels"]) if metrics.get("rmse_y_pixels") is not None else None
                rec["mean_error_pct_diagonal"] = (
                    float(metrics["mean_error_pct_diagonal"])
                    if metrics.get("mean_error_pct_diagonal") is not None
                    else None
                )

                rec["pct_within_2pct"] = (
                    float(acc_rates.get("accuracy_within_2pct", thresh.get("pct_within_2pct")))
                    if (acc_rates.get("accuracy_within_2pct") is not None or thresh.get("pct_within_2pct") is not None)
                    else None
                )
                rec["pct_within_5pct"] = (
                    float(acc_rates.get("accuracy_within_5pct", thresh.get("pct_within_5pct")))
                    if (acc_rates.get("accuracy_within_5pct") is not None or thresh.get("pct_within_5pct") is not None)
                    else None
                )
                rec["pct_within_10pct"] = (
                    float(acc_rates.get("accuracy_within_10pct", thresh.get("pct_within_10pct")))
                    if (acc_rates.get("accuracy_within_10pct") is not None or thresh.get("pct_within_10pct") is not None)
                    else None
                )
                if acc_rates.get("target_selection_accuracy_pct") is not None:
                    rec["target_selection_accuracy_pct"] = float(acc_rates["target_selection_accuracy_pct"])
                    rec["successful_selections"] = int(acc_rates.get("successful_selections", 0))
                    rec["failed_selections"] = int(acc_rates.get("failed_selections", 0))
            else:
                rec["validation_status"] = val_stat
                rec["status"] = f"Incomplete ({val_stat})" if val_stat == "INCOMPLETE" else f"Failed ({val_stat})"
        except Exception:
            pass
    elif acc_csv.exists():
        try:
            with open(acc_csv, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                r0 = rows[0]
                valid_trials = int(float(r0.get("successful_trials", 0) or 0))
                total_trials = int(float(r0.get("total_trials", 0) or 0))
                rejected_trials = int(float(r0.get("rejected_trials", 0) or 0))
                rec["model"] = r0.get("model_name", rec["model"])
                rec["normalization"] = r0.get("normalization_method", rec["normalization"])
                rec["resolution"] = r0.get("screen_resolution", rec["resolution"])
                if r0.get("coordinate_convention"):
                    rec["coordinate_convention"] = str(r0["coordinate_convention"])
                rec["target_boundary"] = str(r0.get("target_boundary", "") or "")
                rec["timestamp"] = float(r0.get("evaluation_timestamp", rec["timestamp"]) or rec["timestamp"])
                rec["total_trials"] = total_trials
                rec["valid_trials"] = valid_trials
                rec["rejected_trials"] = rejected_trials
                attempted = valid_trials + rejected_trials
                rec["rejected_trial_pct"] = round(float(rejected_trials / attempted * 100.0), 2) if attempted > 0 else 0.0

                if valid_trials > 0:
                    rec["validation_completed"] = True
                    rec["validation_status"] = "COMPLETED"
                    rec["status"] = "Validated"
                    def _safe_f(val: Any) -> Optional[float]:
                        if val is None or val in ("", "N/A"):
                            return None
                        return float(val)

                    rec["mean_error_px"] = _safe_f(r0.get("mean_error_pixels"))
                    rec["median_error_px"] = _safe_f(r0.get("median_error_pixels"))
                    rec["max_error_px"] = _safe_f(r0.get("max_error_pixels"))
                    rec["p90_error_px"] = _safe_f(r0.get("p90_error_pixels"))
                    rec["rmse_x_px"] = _safe_f(r0.get("RMSE_X_pixels"))
                    rec["rmse_y_px"] = _safe_f(r0.get("RMSE_Y_pixels"))
                    rec["mean_error_pct_diagonal"] = _safe_f(r0.get("mean_error_pct_diagonal"))
                    rec["pct_within_2pct"] = _safe_f(r0.get("pct_within_2pct_diagonal"))
                    rec["pct_within_5pct"] = _safe_f(r0.get("pct_within_5pct_diagonal"))
                    rec["pct_within_10pct"] = _safe_f(r0.get("pct_within_10pct_diagonal"))
                    rec["target_selection_accuracy_pct"] = _safe_f(r0.get("target_selection_accuracy_pct"))
                    if r0.get("successful_selections") not in (None, "", "N/A"):
                        rec["successful_selections"] = int(float(r0["successful_selections"]))
                    if r0.get("failed_selections") not in (None, "", "N/A"):
                        rec["failed_selections"] = int(float(r0["failed_selections"]))
        except Exception:
            pass
    elif meta_raw_status == "NO_VALID_TRIALS":
        rec["validation_status"] = "FAILED"
        rec["status"] = "Failed (0 Valid Trials)"
    elif meta_raw_status == "INTERRUPTED_BY_USER":
        rec["validation_status"] = "INCOMPLETE"
        rec["status"] = "Incomplete (Interrupted)"
    elif calib_csv.exists():
        rec["validation_status"] = "NOT_STARTED"
        rec["status"] = "Calibrated Only (No Validation)"

    if rec["timestamp"] == 0.0:
        try:
            rec["timestamp"] = exp_dir.stat().st_mtime
        except Exception:
            pass

    return rec


def build_experiment_comparison_summary(experiment_dirs: List[Path]) -> Dict[str, Any]:
    """
    Build structured comparison across experiments strictly separated from training metrics.

    Only sessions with completed independent validation (`validation_completed == True`)
    are eligible for accuracy trend analysis and best-model ranking, with explicit protocol
    compatibility warnings when comparing mixed resolutions or coordinate conventions.
    """
    records: List[Dict[str, Any]] = []
    for d in experiment_dirs:
        parsed = _parse_experiment_record(Path(d))
        if parsed is not None:
            records.append(parsed)

    validated = [r for r in records if r.get("validation_completed") and r.get("valid_trials", 0) > 0]
    validated_sorted = sorted(validated, key=lambda r: (r.get("timestamp", 0.0), r["experiment_id"]))

    protocol_warnings: List[str] = []
    if len(validated_sorted) >= 2:
        res_set = {r.get("resolution") for r in validated_sorted if r.get("resolution") not in (None, "unknown")}
        if len(res_set) > 1:
            protocol_warnings.append(
                f"Screen resolutions differ across validated runs ({', '.join(sorted(res_set))}); pixel comparisons may vary."
            )
        conv_set = {r.get("coordinate_convention") for r in validated_sorted if r.get("coordinate_convention") not in (None, "unknown")}
        if len(conv_set) > 1:
            protocol_warnings.append(
                f"Coordinate conventions differ across validated runs ({', '.join(sorted(conv_set))})."
            )
        bnd_set = {r.get("target_boundary") for r in validated_sorted if r.get("target_boundary")}
        if len(bnd_set) > 1:
            protocol_warnings.append(
                "Target button/circle boundaries differ across validated runs."
            )

    latest_vs_previous: Optional[Dict[str, Any]] = None
    best_experiment: Optional[Dict[str, Any]] = None
    best_by_mean_error: Optional[Dict[str, Any]] = None

    if len(validated_sorted) >= 2:
        curr = validated_sorted[-1]
        # Prefer comparing against most recent prior session matching resolution & convention
        compat_prev = [
            r for r in validated_sorted[:-1]
            if (r.get("resolution") == curr.get("resolution") or "unknown" in (r.get("resolution"), curr.get("resolution")))
            and (r.get("coordinate_convention") == curr.get("coordinate_convention") or "unknown" in (r.get("coordinate_convention"), curr.get("coordinate_convention")))
        ]
        prev = compat_prev[-1] if compat_prev else validated_sorted[-2]
        is_compatible_pair = bool(compat_prev)

        prev_mean = prev.get("mean_error_px")
        curr_mean = curr.get("mean_error_px")
        prev_med = prev.get("median_error_px")
        curr_med = curr.get("median_error_px")
        prev_max = prev.get("max_error_px")
        curr_max = curr.get("max_error_px")

        mean_delta_px = (curr_mean - prev_mean) if (curr_mean is not None and prev_mean is not None) else None
        mean_pct_change: Optional[float] = None
        if prev_mean is not None and curr_mean is not None and prev_mean > 1e-6:
            mean_pct_change = ((prev_mean - curr_mean) / prev_mean) * 100.0

        med_pct_change: Optional[float] = None
        if prev_med is not None and curr_med is not None and prev_med > 1e-6:
            med_pct_change = ((prev_med - curr_med) / prev_med) * 100.0

        sel_prev = prev.get("target_selection_accuracy_pct")
        sel_curr = curr.get("target_selection_accuracy_pct")
        sel_pp_change = (sel_curr - sel_prev) if (sel_curr is not None and sel_prev is not None) else None

        p5_prev = prev.get("pct_within_5pct")
        p5_curr = curr.get("pct_within_5pct")
        p5_pp_change = (p5_curr - p5_prev) if (p5_curr is not None and p5_prev is not None) else None

        verdict = "unchanged"
        if sel_pp_change is not None and abs(sel_pp_change) >= 1e-4:
            if sel_pp_change > 0 and (mean_delta_px is None or mean_delta_px <= 2.0):
                verdict = "improved"
            elif sel_pp_change < 0 and (mean_delta_px is None or mean_delta_px >= -2.0):
                verdict = "got worse"
            elif mean_delta_px is not None:
                verdict = "improved" if mean_delta_px < 0 else ("got worse" if mean_delta_px > 0 else "unchanged")
        elif mean_delta_px is not None:
            if mean_delta_px < -0.5:
                verdict = "improved"
            elif mean_delta_px > 0.5:
                verdict = "got worse"
        elif p5_pp_change is not None:
            if p5_pp_change > 0.1:
                verdict = "improved"
            elif p5_pp_change < -0.1:
                verdict = "got worse"

        latest_vs_previous = {
            "latest_id": curr["experiment_id"],
            "previous_id": prev["experiment_id"],
            "compatible_protocol": is_compatible_pair,
            "verdict": verdict,
            "mean_error_delta_px": round(mean_delta_px, 2) if mean_delta_px is not None else None,
            "mean_error_improvement_pct": round(mean_pct_change, 2) if mean_pct_change is not None else None,
            "median_error_improvement_pct": round(med_pct_change, 2) if med_pct_change is not None else None,
            "max_error_delta_px": round(curr_max - prev_max, 2) if (curr_max is not None and prev_max is not None) else None,
            "target_selection_pp_change": round(sel_pp_change, 2) if sel_pp_change is not None else None,
            "within_5pct_pp_change": round(p5_pp_change, 2) if p5_pp_change is not None else None,
        }

    if validated_sorted:
        def _rank_key(r: Dict[str, Any]) -> Tuple[float, float, float]:
            sel = r.get("target_selection_accuracy_pct")
            sel_score = float(sel) if sel is not None else -1.0
            p5 = r.get("pct_within_5pct")
            p5_score = float(p5) if p5 is not None else -1.0
            m_err = r.get("mean_error_px")
            err_score = -float(m_err) if m_err is not None else -1e9
            return (sel_score, p5_score, err_score)

        best_experiment = max(validated_sorted, key=_rank_key)
        with_err = [r for r in validated_sorted if r.get("mean_error_px") is not None]
        if with_err:
            best_by_mean_error = min(with_err, key=lambda r: float(r["mean_error_px"]))

    return {
        "records": records,
        "validated_records": validated_sorted,
        "latest_vs_previous": latest_vs_previous,
        "best_experiment": best_experiment,
        "best_by_mean_error": best_by_mean_error,
        "protocol_warnings": protocol_warnings,
    }


def _generate_plain_explanation(
    summary: Optional[AccuracySummary],
    training_metrics: Optional[Dict[str, Any]],
    comparison_summary: Dict[str, Any],
    current_experiment_id: str,
    trials: Optional[List[ValidationTrialRecord]] = None,
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a plain-language explanation built strictly from measured empirical values."""
    val_status, status_reason = _resolve_validation_status_info(summary, trials, metadata_extra)
    if summary is None or summary.successful_trials <= 0 or val_status != "COMPLETED":
        msg = f"{VALIDATION_NOT_COMPLETED_MSG} Status: [{val_status}] — {status_reason}"
        if training_metrics and training_metrics.get("mean_euclidean_pixels") is not None:
            tr_px = float(training_metrics["mean_euclidean_pixels"])
            msg += (
                f" Calibration training fit residual was {tr_px:.1f} pixels on the calibration grid only, "
                f"which does NOT represent real-world validation accuracy."
            )
        return msg

    t5_lbl = _fmt_pct_threshold(summary.threshold_5pct_frac)
    parts = [
        f"Status: [{val_status}]. The system successfully predicted gaze positions inside the target boundaries in "
        f"{summary.successful_selections} out of {summary.successful_trials} valid trials "
        f"({summary.target_selection_accuracy_pct:.1f}%).",
        f"Its average prediction error was {summary.mean_error_pixels:.1f} pixels "
        f"(median: {summary.median_error_pixels:.1f} px, {summary.mean_error_pct_diagonal:.1f}% of the screen diagonal), and "
        f"{summary.pct_within_5pct:.1f}% of valid trials fell within {t5_lbl} of the screen diagonal "
        f"({summary.trials_within_5pct} out of {summary.successful_trials}).",
    ]
    wd = summary.pattern_analysis.get("weakest_direction") if summary.pattern_analysis else None
    wr = summary.pattern_analysis.get("weakest_region") if summary.pattern_analysis else None
    if isinstance(wd, dict) and isinstance(wr, dict):
        parts.append(
            f"Weakest direction observed: {wd.get('direction')} (mean error {wd.get('mean_pixel_error', 0.0):.1f} px); "
            f"weakest screen region: {wr.get('region')} (mean error {wr.get('mean_pixel_error', 0.0):.1f} px)."
        )
    if summary.rejected_trials > 0:
        parts.append(
            f"{summary.rejected_trials} out of {summary.total_trials} attempted trial(s) ({summary.rejected_trial_pct:.1f}%) "
            f"were rejected due to blinks, head movement, or unstable gaze and were excluded from accuracy metrics."
        )

    validated = comparison_summary.get("validated_records", [])
    prev_list = [r for r in validated if r["experiment_id"] != current_experiment_id]
    if prev_list:
        prev = prev_list[-1]
        prev_id = prev["experiment_id"]
        prev_mean = prev.get("mean_error_px")
        prev_sel = prev.get("target_selection_accuracy_pct")

        trend_phrase = "performed similarly to"
        if prev_sel is not None and abs(summary.target_selection_accuracy_pct - prev_sel) >= 0.5:
            trend_phrase = "performed better than" if summary.target_selection_accuracy_pct > prev_sel else "performed worse than"
        elif prev_mean is not None:
            if summary.mean_error_pixels < prev_mean - 0.5:
                trend_phrase = "performed better than"
            elif summary.mean_error_pixels > prev_mean + 0.5:
                trend_phrase = "performed worse than"

        details: List[str] = []
        if prev_mean is not None:
            if prev_mean > 1e-6:
                rel_pct = ((prev_mean - summary.mean_error_pixels) / prev_mean) * 100.0
                direction = "error reduction" if rel_pct >= 0 else "error increase"
                details.append(
                    f"average error changed from {prev_mean:.1f} px to {summary.mean_error_pixels:.1f} px "
                    f"({abs(rel_pct):.1f}% {direction})"
                )
            else:
                details.append(
                    f"average error changed from {prev_mean:.1f} px to {summary.mean_error_pixels:.1f} px"
                )
        if prev_sel is not None:
            diff_pp = summary.target_selection_accuracy_pct - prev_sel
            details.append(
                f"target selection accuracy changed from {prev_sel:.1f}% to "
                f"{summary.target_selection_accuracy_pct:.1f}% ({diff_pp:+.1f} percentage points)"
            )
        detail_txt = f" ({'; '.join(details)})" if details else ""
        parts.append(
            f"This experiment {trend_phrase} the previous validated experiment ({prev_id}) "
            f"according to the independent validation results{detail_txt}."
        )
    else:
        parts.append(
            "No previous completed validation experiments are available yet for historical comparison."
        )

    return " ".join(parts)


class ExperimentLogger:
    """
    Manages structured file logging, user-friendly reports (MD, HTML, TXT),
    diagnostic plots, and experiment comparisons for calibration and validation sessions.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path("data/experiments")
        self.timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.experiment_id = experiment_id or f"experiment_{self.timestamp_str}"
        self.experiment_dir = self.base_dir / self.experiment_id

    def log_experiment(
        self,
        calibration_samples: List[CalibrationSampleRecord],
        validation_results: Optional[List[ValidationTrialRecord]] = None,
        accuracy_summary: Optional[AccuracySummary] = None,
        profile: Optional[CalibrationProfile] = None,
        metadata_extra: Optional[Dict[str, Any]] = None,
        export_excel: bool = True,
        generate_plots: bool = True,
        update_tracker: bool = True,
    ) -> Dict[str, Path]:
        """
        Write all experiment data files, user-friendly reports, and plots to session directory.

        Returns:
            Dictionary mapping artifact names to their created file paths.
        """
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        created_files: Dict[str, Path] = {}

        # 1. calibration_samples.csv
        calib_csv = self._write_calibration_samples(calibration_samples)
        created_files["calibration_samples_csv"] = calib_csv

        # 2. validation_results.csv
        if validation_results:
            val_csv = self._write_validation_results(validation_results)
            created_files["validation_results_csv"] = val_csv

        # 3. accuracy_summary.csv & accuracy_report.json (only when validation produced results)
        if accuracy_summary:
            acc_csv = self._write_accuracy_summary(accuracy_summary)
            created_files["accuracy_summary_csv"] = acc_csv

            acc_json = self._write_accuracy_report_json(accuracy_summary)
            created_files["accuracy_report_json"] = acc_json

        # 4. experiment_metadata.json
        meta_json = self._write_metadata_json(profile, metadata_extra)
        created_files["metadata_json"] = meta_json

        # 5. Diagnostic accuracy plots (before HTML/MD reports so images can be referenced)
        plot_paths: List[Path] = []
        if generate_plots and accuracy_summary and validation_results:
            plot_paths = self._generate_accuracy_plots(accuracy_summary, validation_results)
            for p in plot_paths:
                created_files[p.stem] = p

        # Gather prior experiments from base_dir safely (read-only) for comparison
        sibling_dirs: List[Path] = []
        if self.base_dir.exists():
            sibling_dirs = sorted([d for d in self.base_dir.iterdir() if d.is_dir()])
        if self.experiment_dir not in sibling_dirs:
            sibling_dirs.append(self.experiment_dir)
        comparison_summary = build_experiment_comparison_summary(sibling_dirs)

        # 6. Always generate Simple Human-Readable Reports (MD, HTML, TXT)
        acc_txt = self._write_accuracy_report_text(
            accuracy_summary,
            validation_results or [],
            calibration_samples=calibration_samples,
            profile=profile,
            metadata_extra=metadata_extra,
            comparison_summary=comparison_summary,
        )
        created_files["accuracy_report_text"] = acc_txt

        acc_md = self._write_accuracy_report_markdown(
            accuracy_summary,
            validation_results or [],
            calibration_samples=calibration_samples,
            profile=profile,
            metadata_extra=metadata_extra,
            comparison_summary=comparison_summary,
            plot_paths=plot_paths,
        )
        created_files["accuracy_report_md"] = acc_md

        acc_html = self._write_accuracy_report_html(
            accuracy_summary,
            validation_results or [],
            calibration_samples=calibration_samples,
            profile=profile,
            metadata_extra=metadata_extra,
            comparison_summary=comparison_summary,
            plot_paths=plot_paths,
        )
        created_files["accuracy_report_html"] = acc_html

        # 7. Optional Excel workbook export + Centralized Tracker Sync
        if export_excel:
            xlsx_path = self._export_excel_workbook(
                calibration_samples=calibration_samples,
                validation_results=validation_results or [],
                accuracy_summary=accuracy_summary,
                metadata_extra=metadata_extra,
            )
            if xlsx_path:
                created_files["excel_workbook"] = xlsx_path
            if update_tracker:
                try:
                    from .experiment_tracker import update_central_tracker
                    profiles_dir = (
                        self.base_dir.parent / "calibration"
                        if self.base_dir.name == "experiments" and (self.base_dir.parent / "calibration").exists()
                        else None
                    )
                    sync_res = update_central_tracker(
                        experiments_dir=self.base_dir,
                        profiles_dir=profiles_dir,
                    )
                    created_files["central_tracker_xlsx"] = sync_res["tracker_path"]
                except Exception as e:
                    logger.warning(f"Failed to update central AABAR_Experiments_Tracker.xlsx: {e}")

        logger.info(f"Experiment session [{self.experiment_id}] logged to: {self.experiment_dir.resolve()}")
        return created_files

    def _resolve_overview_info(
        self,
        summary: Optional[AccuracySummary],
        calibration_samples: List[CalibrationSampleRecord],
        profile: Optional[CalibrationProfile],
        metadata_extra: Optional[Dict[str, Any]],
        trials: List[ValidationTrialRecord],
    ) -> Dict[str, Any]:
        """Collect normalized overview fields safely across calibration-only and validated sessions."""
        ts = summary.evaluation_timestamp if summary else (profile.created_at if profile else time.time())
        date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

        if summary:
            res_str = summary.screen_resolution
            diag_px = summary.screen_diagonal_px
            model_str = summary.model_name.upper()
            norm_method = summary.normalization_method
            train_targets = summary.training_sample_count
        else:
            sw = profile.screen_width if profile else int((metadata_extra or {}).get("screen_width", 1920))
            sh = profile.screen_height if profile else int((metadata_extra or {}).get("screen_height", 1080))
            res_str = f"{sw}x{sh}"
            diag_px = math.hypot(sw, sh)
            model_str = (
                profile.model_type.upper()
                if profile
                else str((metadata_extra or {}).get("model_type", "unknown")).upper()
            )
            norm_method = (
                profile.normalization_method
                if profile
                else str((metadata_extra or {}).get("normalization_method", "StandardScaler"))
            )
            train_targets = profile.training_sample_count if profile else 0

        accepted_frames = sum(1 for s in calibration_samples if s.accepted)
        if accepted_frames == 0 and profile and profile.calibration_frame_count > 0:
            accepted_frames = profile.calibration_frame_count

        if train_targets > 0 and accepted_frames > 0:
            train_sample_desc = f"{train_targets} targets ({accepted_frames} accepted frames)"
        elif train_targets > 0:
            train_sample_desc = f"{train_targets} target point(s)"
        elif accepted_frames > 0:
            train_sample_desc = f"{accepted_frames} accepted frames"
        else:
            train_sample_desc = "0 (loaded external profile)" if profile else "0"

        val_status, val_reason = _resolve_validation_status_info(summary, trials, metadata_extra)

        if summary and summary.successful_trials > 0:
            val_desc = (
                f"{summary.successful_trials} valid / {summary.total_trials} planned "
                f"({summary.rejected_trials} rejected, {summary.rejected_trial_pct:.1f}% rej rate)"
            )
        elif trials:
            v_cnt = sum(1 for t in trials if t.trial_status == "VALID")
            r_cnt = sum(1 for t in trials if t.trial_status == "REJECTED")
            val_desc = f"{v_cnt} valid / {len(trials)} attempted ({r_cnt} rejected — {val_status})"
        else:
            val_desc = f"0 ({val_status}: Validation not started)"

        return {
            "experiment_id": self.experiment_id,
            "date_time": date_str,
            "screen_resolution": res_str,
            "screen_diagonal_px": diag_px,
            "model": model_str,
            "normalization_method": norm_method,
            "training_samples_desc": train_sample_desc,
            "validation_trials_desc": val_desc,
            "validation_status": val_status,
            "validation_status_reason": val_reason,
        }

    def _write_calibration_samples(self, samples: List[CalibrationSampleRecord]) -> Path:
        filepath = self.experiment_dir / "calibration_samples.csv"
        headers = [
            "experiment_id",
            "session_id",
            "target_id",
            "target_x",
            "target_y",
            "timestamp",
            "raw_x",
            "raw_y",
            "r_iris_x",
            "r_iris_y",
            "l_iris_x",
            "l_iris_y",
            "r_openness",
            "l_openness",
            "yaw",
            "pitch",
            "roll",
            "tracking_quality",
            "confidence",
            "sample_accepted",
            "rejection_reason",
        ]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for s in samples:
                writer.writerow(s.to_csv_dict())
        return filepath

    def _write_validation_results(self, trials: List[ValidationTrialRecord]) -> Path:
        filepath = self.experiment_dir / "validation_results.csv"
        headers = [
            "experiment_id",
            "session_id",
            "calibration_profile_id",
            "model_name",
            "model_version",
            "trial_id",
            "target_id",
            "target_name",
            "actual_x",
            "actual_y",
            "predicted_x",
            "predicted_y",
            "error_x",
            "error_y",
            "abs_error_x",
            "abs_error_y",
            "pixel_error_x",
            "pixel_error_y",
            "euclidean_error_normalized",
            "euclidean_error_pixels",
            "error_pct_diagonal",
            "target_hit",
            "target_boundary",
            "screen_width",
            "screen_height",
            "tracking_quality",
            "accepted_sample_count",
            "rejected_sample_count",
            "trial_duration_sec",
            "trial_status",
            "rejection_reason",
            "timestamp",
        ]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for t in trials:
                writer.writerow(t.to_csv_dict())
        return filepath

    def _write_accuracy_summary(self, summary: AccuracySummary) -> Path:
        filepath = self.experiment_dir / "accuracy_summary.csv"
        headers = [
            "experiment_id",
            "calibration_profile_id",
            "model_name",
            "model_configuration",
            "normalization_method",
            "training_sample_count",
            "total_trials",
            "successful_trials",
            "rejected_trials",
            "rejected_trial_pct",
            "validation_status",
            "failure_reason",
            "coordinate_convention",
            "successful_selections",
            "failed_selections",
            "target_selection_accuracy_pct",
            "target_boundary",
            "MAE_X",
            "MAE_Y",
            "MAE_X_pixels",
            "MAE_Y_pixels",
            "RMSE_X",
            "RMSE_Y",
            "RMSE_X_pixels",
            "RMSE_Y_pixels",
            "mean_euclidean_error",
            "median_euclidean_error",
            "p90_euclidean_error",
            "maximum_error",
            "mean_error_pixels",
            "mean_error_pct_diagonal",
            "median_error_pixels",
            "median_error_pct_diagonal",
            "p90_error_pixels",
            "max_error_pixels",
            "max_error_pct_diagonal",
            "screen_diagonal_pixels",
            "pct_within_2pct_diagonal",
            "pct_within_5pct_diagonal",
            "pct_within_10pct_diagonal",
            "threshold_strict_pct",
            "threshold_moderate_pct",
            "threshold_relaxed_pct",
            "screen_resolution",
            "evaluation_timestamp",
        ]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerow(summary.to_csv_dict())
        return filepath

    def _write_accuracy_report_json(self, summary: AccuracySummary) -> Path:
        filepath = self.experiment_dir / "accuracy_report.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary.to_report_dict(), f, indent=2)
        return filepath

    def _write_accuracy_report_text(
        self,
        summary: Optional[AccuracySummary],
        trials: List[ValidationTrialRecord],
        calibration_samples: Optional[List[CalibrationSampleRecord]] = None,
        profile: Optional[CalibrationProfile] = None,
        metadata_extra: Optional[Dict[str, Any]] = None,
        comparison_summary: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write formatted human-readable plain-text summary report."""
        filepath = self.experiment_dir / "accuracy_report.txt"
        overview = self._resolve_overview_info(
            summary, calibration_samples or [], profile, metadata_extra, trials
        )
        train_metrics = _extract_training_error_summary(summary, profile, metadata_extra)
        comp = comparison_summary or {"records": [], "validated_records": []}
        explanation = _generate_plain_explanation(
            summary, train_metrics, comp, self.experiment_id, trials=trials, metadata_extra=metadata_extra
        )

        lines = [
            "=" * 80,
            "AABAR CV - Experiment Accuracy & Gaze Evaluation Report",
            "=" * 80,
            "",
            "--- EXPERIMENT OVERVIEW ---",
            f"Experiment ID:           {overview['experiment_id']}",
            f"Date & Time:             {overview['date_time']}",
            f"Screen Resolution:       {overview['screen_resolution']} (Diagonal: {overview['screen_diagonal_px']:.1f} px)",
            f"Calibration Model:       {overview['model']}",
            f"Normalization Method:    {overview['normalization_method']}",
            f"Training Samples:        {overview['training_samples_desc']}",
            f"Validation Status:       {overview['validation_status']}",
            f"Validation Trials:       {overview['validation_trials_desc']}",
            "",
        ]

        if summary and summary.successful_trials > 0:
            t2_lbl = _fmt_pct_threshold(summary.threshold_2pct_frac)
            t5_lbl = _fmt_pct_threshold(summary.threshold_5pct_frac)
            t10_lbl = _fmt_pct_threshold(summary.threshold_10pct_frac)
            lines.extend([
                "--- MAIN RESULTS (INDEPENDENT VALIDATION DATA ONLY) ---",
                f"Accuracy within {t2_lbl:<5}:  {summary.pct_within_2pct:5.1f}% ({summary.trials_within_2pct}/{summary.successful_trials} trials, <= {summary.threshold_2pct_px:.1f} px)",
                f"Accuracy within {t5_lbl:<5}:  {summary.pct_within_5pct:5.1f}% ({summary.trials_within_5pct}/{summary.successful_trials} trials, <= {summary.threshold_5pct_px:.1f} px)",
                f"Accuracy within {t10_lbl:<5}:  {summary.pct_within_10pct:5.1f}% ({summary.trials_within_10pct}/{summary.successful_trials} trials, <= {summary.threshold_10pct_px:.1f} px)",
                f"Target Selection Accuracy: {summary.target_selection_accuracy_pct:5.1f}% ({summary.successful_selections} hits, {summary.failed_selections} misses / {summary.successful_trials} valid trials)",
                f"Average Prediction Error:  {summary.mean_error_pixels:5.1f} pixels ({summary.mean_error_pct_diagonal:.1f}% of screen diagonal)",
                f"Rejected Trial Rate:       {summary.rejected_trial_pct:5.1f}% ({summary.rejected_trials}/{summary.successful_trials + summary.rejected_trials} attempted)",
                f"Target Boundary Tested:    {summary.target_boundary_description or 'Rendered Target Circle'}",
                "",
                "--- DETAILED PREDICTION ERROR METRICS ---",
                f"Mean Euclidean Error:    {summary.mean_error_pixels:.1f} px ({summary.mean_error_pct_diagonal:.1f}% diagonal | {summary.mean_euclidean_error:.4f} norm)",
                f"Median Euclidean Error:  {summary.median_error_pixels:.1f} px ({summary.median_error_pct_diagonal:.1f}% diagonal | {summary.median_euclidean_error:.4f} norm)",
                f"Maximum Euclidean Error: {summary.max_error_pixels:.1f} px ({summary.max_error_pct_diagonal:.1f}% diagonal | {summary.maximum_error:.4f} norm)",
                f"90th Percentile Error:   {summary.p90_error_pixels:.1f} px" if summary.p90_error_pixels is not None else "90th Percentile Error:   N/A (< 5 trials)",
                f"X-Axis Error (MAE / RMSE): MAE_X = {summary.mae_x_px:.1f} px ({summary.mae_x:.4f}) | RMSE_X = {summary.rmse_x_px:.1f} px ({summary.rmse_x:.4f})",
                f"Y-Axis Error (MAE / RMSE): MAE_Y = {summary.mae_y_px:.1f} px ({summary.mae_y:.4f}) | RMSE_Y = {summary.rmse_y_px:.1f} px ({summary.rmse_y:.4f})",
                f"Valid vs Rejected Trials:  Valid = {summary.successful_trials} | Rejected/Invalid = {summary.rejected_trials} | Successful Hits = {summary.successful_selections} | Failed Selections = {summary.failed_selections}",
                "",
            ])
        else:
            lines.extend([
                "--- MAIN RESULTS ---",
                f"{VALIDATION_NOT_COMPLETED_MSG}",
                f"Status Detail [{overview['validation_status']}]: {overview['validation_status_reason']}",
                "",
            ])

        lines.extend([
            "--- MODEL TRAINING PERFORMANCE VS INDEPENDENT VALIDATION ---",
        ])
        if train_metrics and train_metrics.get("mean_euclidean_pixels") is not None:
            lines.append(
                f"Calibration Training Error (Fit Only): Mean = {train_metrics['mean_euclidean_pixels']:.1f} px | "
                f"MAE_X = {train_metrics.get('mae_x', 0.0):.4f} | MAE_Y = {train_metrics.get('mae_y', 0.0):.4f}"
            )
        else:
            lines.append("Calibration Training Error (Fit Only): Not recorded")
        if summary and summary.successful_trials > 0:
            lines.append(
                f"Independent Validation Accuracy:       Target Hit = {summary.target_selection_accuracy_pct:.1f}% | "
                f"Within 5% = {summary.pct_within_5pct:.1f}% | Mean Error = {summary.mean_error_pixels:.1f} px"
            )
        else:
            lines.append(f"Independent Validation Accuracy:       {VALIDATION_NOT_COMPLETED_MSG} ({overview['validation_status']})")
        lines.append("")

        if summary and summary.direction_accuracies:
            lines.extend([
                "--- DIRECTION ACCURACY (LEFT / RIGHT / CENTER / UP / DOWN) ---",
                f"  {'Direction':<12} | {'Valid/Rej':<9} | {'Sel Acc':<8} | {'Within 5%':<10} | {'Mean Err (px)':<13} | {'Bias (X, Y) px'}",
                "  " + "-" * 72,
            ])
            for dname, dinfo in summary.direction_accuracies.items():
                vc, rc = dinfo.get("valid_trials", 0), dinfo.get("rejected_trials", 0)
                if vc > 0:
                    lines.append(
                        f"  {dname:<12} | {vc:>4}/{rc:<4} | {dinfo['target_selection_accuracy_pct']:5.1f}%  | "
                        f"{dinfo['pct_within_5pct']:6.1f}%   | {dinfo['mean_euclidean_pixels']:<13.1f} | "
                        f"({dinfo['signed_bias_x_px']:+.1f}, {dinfo['signed_bias_y_px']:+.1f})"
                    )
                else:
                    lines.append(f"  {dname:<12} |    0/{rc:<4} | N/A      | N/A        | REJECTED      | N/A")
            lines.append("")

        if summary and summary.region_accuracies:
            lines.extend([
                "--- SCREEN REGION ACCURACY (ROWS / COLUMNS / EDGES) ---",
                f"  {'Region':<16} | {'Valid/Rej':<9} | {'Sel Acc':<8} | {'Within 5%':<10} | {'Mean Err (px)':<13} | {'Bias (X, Y) px'}",
                "  " + "-" * 76,
            ])
            for rname, rinfo in summary.region_accuracies.items():
                vc, rc = rinfo.get("valid_trials", 0), rinfo.get("rejected_trials", 0)
                if vc > 0:
                    lines.append(
                        f"  {rname:<16} | {vc:>4}/{rc:<4} | {rinfo['target_selection_accuracy_pct']:5.1f}%  | "
                        f"{rinfo['pct_within_5pct']:6.1f}%   | {rinfo['mean_euclidean_pixels']:<13.1f} | "
                        f"({rinfo['signed_bias_x_px']:+.1f}, {rinfo['signed_bias_y_px']:+.1f})"
                    )
                else:
                    lines.append(f"  {rname:<16} |    0/{rc:<4} | N/A      | N/A        | REJECTED      | N/A")
            lines.append("")

        if summary and summary.target_accuracies:
            lines.extend([
                "--- PER-TARGET PERFORMANCE ---",
                f"  {'Target Name':<18} | {'Valid/Rej':<9} | {'Hits/Valid':<10} | {'Sel Acc':<8} | {'Mean Err (px)':<13} | {'Bias (X, Y) px'}",
                "  " + "-" * 76,
            ])
            for tname, tinfo in summary.target_accuracies.items():
                v_cnt = tinfo.get("valid_trials", 0)
                r_cnt = tinfo.get("rejected_trials", 0)
                if v_cnt > 0:
                    hits = tinfo.get("successful_selections", 0)
                    sel_pct = tinfo.get("target_selection_accuracy_pct", 0.0)
                    sel_str = f"{sel_pct:.1f}%" if sel_pct is not None else "N/A"
                    bias_str = f"({tinfo.get('signed_bias_x_px', 0.0):+.1f}, {tinfo.get('signed_bias_y_px', 0.0):+.1f})"
                    lines.append(
                        f"  {tname:<18} | {v_cnt:>4}/{r_cnt:<4} | {hits:>4}/{v_cnt:<5} | {sel_str:<8} | "
                        f"{tinfo.get('mean_euclidean_pixels', 0.0):<13.1f} | {bias_str}"
                    )
                else:
                    lines.append(
                        f"  {tname:<18} |    0/{r_cnt:<4} | 0/0        | N/A      | REJECTED      | N/A"
                    )
            lines.append("")

        lines.extend([
            "--- SIMPLE EXPLANATION ---",
            explanation,
            "",
            "=" * 80,
        ])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return filepath

    def _write_accuracy_report_markdown(
        self,
        summary: Optional[AccuracySummary],
        trials: List[ValidationTrialRecord],
        calibration_samples: List[CalibrationSampleRecord],
        profile: Optional[CalibrationProfile],
        metadata_extra: Optional[Dict[str, Any]],
        comparison_summary: Dict[str, Any],
        plot_paths: List[Path],
    ) -> Path:
        """Write a clean, beginner-friendly Markdown report for every experiment."""
        filepath = self.experiment_dir / "accuracy_report.md"
        overview = self._resolve_overview_info(
            summary, calibration_samples, profile, metadata_extra, trials
        )
        train_metrics = _extract_training_error_summary(summary, profile, metadata_extra)
        explanation = _generate_plain_explanation(
            summary, train_metrics, comparison_summary, self.experiment_id, trials=trials, metadata_extra=metadata_extra
        )

        lines: List[str] = [
            f"# Gaze Tracking Experiment Report: `{overview['experiment_id']}`",
            "",
            "## Experiment Overview",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| **Experiment ID** | `{overview['experiment_id']}` |",
            f"| **Date and Time** | {overview['date_time']} |",
            f"| **Screen Resolution** | {overview['screen_resolution']} (Diagonal: {overview['screen_diagonal_px']:.1f} px) |",
            f"| **Calibration Model** | {overview['model']} |",
            f"| **Normalization Method** | {overview['normalization_method']} |",
            f"| **Number of Training Samples** | {overview['training_samples_desc']} |",
            f"| **Validation Status** | `{overview['validation_status']}` |",
            f"| **Number of Validation Trials** | {overview['validation_trials_desc']} |",
            "",
            "## Main Results",
            "",
        ]

        if summary and summary.successful_trials > 0:
            t2_lbl = _fmt_pct_threshold(summary.threshold_2pct_frac)
            t5_lbl = _fmt_pct_threshold(summary.threshold_5pct_frac)
            t10_lbl = _fmt_pct_threshold(summary.threshold_10pct_frac)
            lines.extend([
                "> **Accuracy percentages** below measure how often gaze predictions landed successfully within strict screen tolerances or inside target boundaries (higher is better). **Average Error** measures distance from the true target (lower is better).",
                "",
                "| Result | Value |",
                "|---|---:|",
                f"| **Accuracy within {t2_lbl}** | **{summary.pct_within_2pct:.1f}%** ({summary.trials_within_2pct}/{summary.successful_trials} trials, within {summary.threshold_2pct_px:.1f} px) |",
                f"| **Accuracy within {t5_lbl}** | **{summary.pct_within_5pct:.1f}%** ({summary.trials_within_5pct}/{summary.successful_trials} trials, within {summary.threshold_5pct_px:.1f} px) |",
                f"| **Accuracy within {t10_lbl}** | **{summary.pct_within_10pct:.1f}%** ({summary.trials_within_10pct}/{summary.successful_trials} trials, within {summary.threshold_10pct_px:.1f} px) |",
                f"| **Target Selection Accuracy** | **{summary.target_selection_accuracy_pct:.1f}%** ({summary.successful_selections}/{summary.successful_trials} hits) |",
                f"| **Average Error (pixels)** | {summary.mean_error_pixels:.1f} px |",
                f"| **Average Error (% of screen diagonal)** | {summary.mean_error_pct_diagonal:.1f}% |",
                f"| **Rejected Trial Rate** | {summary.rejected_trial_pct:.1f}% ({summary.rejected_trials}/{summary.successful_trials + summary.rejected_trials}) |",
                "",
            ])
        else:
            lines.extend([
                f"> **Status (`{overview['validation_status']}`):** {VALIDATION_NOT_COMPLETED_MSG}",
                f"> **Reason:** {overview['validation_status_reason']}",
                "",
                "| Result | Value |",
                "|---|---:|",
                "| Accuracy within 2% | Not available (Validation not completed) |",
                "| Accuracy within 5% | Not available (Validation not completed) |",
                "| Accuracy within 10% | Not available (Validation not completed) |",
                "| Target Selection Accuracy | Not available (Validation not completed) |",
                "| Average Error (pixels) | Not available (Validation not completed) |",
                "| Average Error (% of screen diagonal) | Not available (Validation not completed) |",
                "",
            ])

        lines.extend([
            "## Detailed Results",
            "",
        ])

        if summary and summary.successful_trials > 0:
            p90_str = f"{summary.p90_error_pixels:.1f} px" if summary.p90_error_pixels is not None else "N/A (< 5 trials)"
            lines.extend([
                "### Prediction Error Breakdown (Independent Validation Trials)",
                "",
                "| Error Measurement | Pixels | Screen Diagonal / Normalized |",
                "|---|---:|---:|",
                f"| **X-Axis MAE** | {summary.mae_x_px:.1f} px | {summary.mae_x:.4f} norm |",
                f"| **Y-Axis MAE** | {summary.mae_y_px:.1f} px | {summary.mae_y:.4f} norm |",
                f"| **X-Axis RMSE** | {summary.rmse_x_px:.1f} px | {summary.rmse_x:.4f} norm |",
                f"| **Y-Axis RMSE** | {summary.rmse_y_px:.1f} px | {summary.rmse_y:.4f} norm |",
                f"| **Median Error** | {summary.median_error_pixels:.1f} px | {summary.median_error_pct_diagonal:.1f}% diag ({summary.median_euclidean_error:.4f} norm) |",
                f"| **Maximum Error** | {summary.max_error_pixels:.1f} px | {summary.max_error_pct_diagonal:.1f}% diag ({summary.maximum_error:.4f} norm) |",
                f"| **90th Percentile Error** | {p90_str} | — |",
                "",
                "### Target Selection & Trial Counts",
                "",
                f"- **Target Boundary Evaluated:** {summary.target_boundary_description or 'Rendered Target Circle'}",
                f"- **Successful Selections (Hits):** {summary.successful_selections}",
                f"- **Failed Selections (Misses):** {summary.failed_selections}",
                f"- **Total Valid Trials:** {summary.successful_trials}",
                f"- **Rejected / Invalid Trials:** {summary.rejected_trials} ({summary.rejected_trial_pct:.1f}%)",
                "",
            ])

            if summary.direction_accuracies:
                lines.extend([
                    "### Direction Accuracy Breakdown (LEFT / RIGHT / CENTER / UP / DOWN)",
                    "",
                    "| Direction | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |",
                    "|---|---:|---:|---:|---:|---:|",
                ])
                for dname, dinfo in summary.direction_accuracies.items():
                    vc, rc = dinfo.get("valid_trials", 0), dinfo.get("rejected_trials", 0)
                    if vc > 0:
                        lines.append(
                            f"| `{dname}` | {vc} / {rc} | **{dinfo['target_selection_accuracy_pct']:.1f}%** | "
                            f"{dinfo['pct_within_5pct']:.1f}% | {dinfo['mean_euclidean_pixels']:.1f} px | "
                            f"({dinfo['signed_bias_x_px']:+.1f}, {dinfo['signed_bias_y_px']:+.1f}) |"
                        )
                    else:
                        lines.append(f"| `{dname}` | 0 / {rc} | N/A | N/A | REJECTED | N/A |")
                lines.append("")

            if summary.region_accuracies:
                lines.extend([
                    "### Screen Region Accuracy Breakdown (Rows, Columns & Edges)",
                    "",
                    "| Screen Region | Valid / Rej | Selection Accuracy | Within 5% | Mean Error (px) | Bias (X, Y) px |",
                    "|---|---:|---:|---:|---:|---:|",
                ])
                for rname, rinfo in summary.region_accuracies.items():
                    vc, rc = rinfo.get("valid_trials", 0), rinfo.get("rejected_trials", 0)
                    if vc > 0:
                        lines.append(
                            f"| `{rname}` | {vc} / {rc} | **{rinfo['target_selection_accuracy_pct']:.1f}%** | "
                            f"{rinfo['pct_within_5pct']:.1f}% | {rinfo['mean_euclidean_pixels']:.1f} px | "
                            f"({rinfo['signed_bias_x_px']:+.1f}, {rinfo['signed_bias_y_px']:+.1f}) |"
                        )
                    else:
                        lines.append(f"| `{rname}` | 0 / {rc} | N/A | N/A | REJECTED | N/A |")
                lines.append("")

            if summary.target_accuracies:
                lines.extend([
                    "### Per-Target Performance",
                    "",
                    "| Target Point | Valid / Rejected | Hits / Valid | Target Selection Accuracy | Mean Error (px) | Bias (X, Y) px |",
                    "|---|---:|---:|---:|---:|---:|",
                ])
                for tname, tinfo in summary.target_accuracies.items():
                    v_cnt = tinfo.get("valid_trials", 0)
                    r_cnt = tinfo.get("rejected_trials", 0)
                    if v_cnt > 0:
                        hits = tinfo.get("successful_selections", 0)
                        sel_pct = tinfo.get("target_selection_accuracy_pct", 0.0)
                        bias = f"({tinfo.get('signed_bias_x_px', 0.0):+.1f}, {tinfo.get('signed_bias_y_px', 0.0):+.1f})"
                        lines.append(
                            f"| `{tname}` | {v_cnt} / {r_cnt} | {hits} / {v_cnt} | **{sel_pct:.1f}%** | "
                            f"{tinfo.get('mean_euclidean_pixels', 0.0):.1f} px | {bias} |"
                        )
                    else:
                        lines.append(f"| `{tname}` | 0 / {r_cnt} | 0 / 0 | N/A | REJECTED | N/A |")
                lines.append("")

        lines.extend([
            "### Training Performance vs. Independent Validation Performance",
            "",
            "> **Note:** Training error only measures how tightly the model fits the 9 calibration points and must NOT be interpreted as real-world validation accuracy.",
            "",
            "| Evaluation Split | Purpose | Mean Error (px) | Validation Accuracy |",
            "|---|---|---:|---:|",
        ])
        tr_mean_str = (
            f"{train_metrics['mean_euclidean_pixels']:.1f} px"
            if (train_metrics and train_metrics.get("mean_euclidean_pixels") is not None)
            else "Not recorded"
        )
        lines.append(f"| **Calibration Training Set** | Model fitting residual (training fit only) | {tr_mean_str} | N/A (training split) |")
        if summary and summary.successful_trials > 0:
            lines.append(
                f"| **Independent Validation Set** | Unseen test targets (real accuracy) | "
                f"{summary.mean_error_pixels:.1f} px ({summary.mean_error_pct_diagonal:.1f}% diag) | "
                f"{summary.target_selection_accuracy_pct:.1f}% hits / {summary.pct_within_5pct:.1f}% within 5% |"
            )
        else:
            lines.append(f"| **Independent Validation Set** | Unseen test targets | N/A | {VALIDATION_NOT_COMPLETED_MSG} |")
        lines.append("")

        lines.extend([
            "## Simple Explanation",
            "",
            explanation,
            "",
            "## Comparison with Previous Experiments",
            "",
            compare_experiments(
                [Path(r["path"]) for r in comparison_summary.get("records", [])]
            ),
            "",
        ])

        if plot_paths:
            lines.extend(["## Diagnostic Plots", ""])
            for p in plot_paths:
                lines.append(f"![{p.stem}]({p.name})")
            lines.append("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return filepath

    def _write_accuracy_report_html(
        self,
        summary: Optional[AccuracySummary],
        trials: List[ValidationTrialRecord],
        calibration_samples: List[CalibrationSampleRecord],
        profile: Optional[CalibrationProfile],
        metadata_extra: Optional[Dict[str, Any]],
        comparison_summary: Dict[str, Any],
        plot_paths: List[Path],
    ) -> Path:
        """Write a clean, standalone, beginner-friendly HTML report."""
        filepath = self.experiment_dir / "accuracy_report.html"
        overview = self._resolve_overview_info(
            summary, calibration_samples, profile, metadata_extra, trials
        )
        train_metrics = _extract_training_error_summary(summary, profile, metadata_extra)
        explanation = _generate_plain_explanation(
            summary, train_metrics, comparison_summary, self.experiment_id, trials=trials, metadata_extra=metadata_extra
        )

        def esc(v: Any) -> str:
            return html.escape(str(v))

        # Main Results section
        if summary and summary.successful_trials > 0:
            t2_lbl = _fmt_pct_threshold(summary.threshold_2pct_frac)
            t5_lbl = _fmt_pct_threshold(summary.threshold_5pct_frac)
            t10_lbl = _fmt_pct_threshold(summary.threshold_10pct_frac)
            main_results_html = f"""
            <div class="cards">
              <div class="card highlight">
                <div class="card-label">Target Selection Accuracy</div>
                <div class="card-val">{summary.target_selection_accuracy_pct:.1f}%</div>
                <div class="card-sub">{summary.successful_selections} of {summary.successful_trials} valid hits</div>
              </div>
              <div class="card">
                <div class="card-label">Accuracy within {esc(t2_lbl)}</div>
                <div class="card-val">{summary.pct_within_2pct:.1f}%</div>
                <div class="card-sub">{summary.trials_within_2pct}/{summary.successful_trials} trials (&le; {summary.threshold_2pct_px:.1f} px)</div>
              </div>
              <div class="card">
                <div class="card-label">Accuracy within {esc(t5_lbl)}</div>
                <div class="card-val">{summary.pct_within_5pct:.1f}%</div>
                <div class="card-sub">{summary.trials_within_5pct}/{summary.successful_trials} trials (&le; {summary.threshold_5pct_px:.1f} px)</div>
              </div>
              <div class="card">
                <div class="card-label">Accuracy within {esc(t10_lbl)}</div>
                <div class="card-val">{summary.pct_within_10pct:.1f}%</div>
                <div class="card-sub">{summary.trials_within_10pct}/{summary.successful_trials} trials (&le; {summary.threshold_10pct_px:.1f} px)</div>
              </div>
              <div class="card">
                <div class="card-label">Mean Error (pixels)</div>
                <div class="card-val">{summary.mean_error_pixels:.1f} px</div>
                <div class="card-sub">{summary.mean_error_pct_diagonal:.1f}% of screen diagonal</div>
              </div>
              <div class="card">
                <div class="card-label">Validation Status</div>
                <div class="card-val">{esc(overview['validation_status'])}</div>
                <div class="card-sub">{summary.rejected_trials}/{summary.successful_trials + summary.rejected_trials} trials rejected ({summary.rejected_trial_pct:.1f}%)</div>
              </div>
            </div>
            <table>
              <thead><tr><th>Result</th><th class="num">Value</th></tr></thead>
              <tbody>
                <tr><td>Validation Status</td><td class="num"><strong>{esc(overview['validation_status'])}</strong></td></tr>
                <tr><td>Accuracy within {esc(t2_lbl)} (within {summary.threshold_2pct_px:.1f} px)</td><td class="num"><strong>{summary.pct_within_2pct:.1f}%</strong> ({summary.trials_within_2pct}/{summary.successful_trials})</td></tr>
                <tr><td>Accuracy within {esc(t5_lbl)} (within {summary.threshold_5pct_px:.1f} px)</td><td class="num"><strong>{summary.pct_within_5pct:.1f}%</strong> ({summary.trials_within_5pct}/{summary.successful_trials})</td></tr>
                <tr><td>Accuracy within {esc(t10_lbl)} (within {summary.threshold_10pct_px:.1f} px)</td><td class="num"><strong>{summary.pct_within_10pct:.1f}%</strong> ({summary.trials_within_10pct}/{summary.successful_trials})</td></tr>
                <tr><td>Target Selection Accuracy ({esc(summary.target_boundary_description or 'Target boundary')})</td><td class="num"><strong>{summary.target_selection_accuracy_pct:.1f}%</strong> ({summary.successful_selections}/{summary.successful_trials})</td></tr>
                <tr><td>Mean Error (pixels)</td><td class="num">{summary.mean_error_pixels:.1f} px</td></tr>
                <tr><td>Average Error (% of screen diagonal)</td><td class="num">{summary.mean_error_pct_diagonal:.1f}%</td></tr>
              </tbody>
            </table>
            """
        else:
            main_results_html = f"""
            <div class="notice warn"><strong>Status [{esc(overview['validation_status'])}]:</strong> {esc(VALIDATION_NOT_COMPLETED_MSG)}<br/>{esc(overview['validation_status_reason'])}</div>
            <div class="cards">
              <div class="card"><div class="card-label">Target Selection Accuracy</div><div class="card-val">N/A</div><div class="card-sub">Validation required</div></div>
              <div class="card"><div class="card-label">Accuracy within 2%</div><div class="card-val">N/A</div><div class="card-sub">Validation required</div></div>
              <div class="card"><div class="card-label">Accuracy within 5%</div><div class="card-val">N/A</div><div class="card-sub">Validation required</div></div>
              <div class="card"><div class="card-label">Accuracy within 10%</div><div class="card-val">N/A</div><div class="card-sub">Validation required</div></div>
              <div class="card"><div class="card-label">Mean Error (pixels)</div><div class="card-val">N/A</div><div class="card-sub">Validation required</div></div>
              <div class="card"><div class="card-label">Validation Status</div><div class="card-val">{esc(overview['validation_status'])}</div><div class="card-sub">{esc(overview['validation_status_reason'])}</div></div>
            </div>
            <table>
              <thead><tr><th>Result</th><th class="num">Value</th></tr></thead>
              <tbody>
                <tr><td>Accuracy within 2%</td><td class="num">Not available</td></tr>
                <tr><td>Accuracy within 5%</td><td class="num">Not available</td></tr>
                <tr><td>Accuracy within 10%</td><td class="num">Not available</td></tr>
                <tr><td>Target Selection Accuracy</td><td class="num">Not available</td></tr>
                <tr><td>Average Error (pixels)</td><td class="num">Not available</td></tr>
                <tr><td>Average Error (% of screen diagonal)</td><td class="num">Not available</td></tr>
              </tbody>
            </table>
            """

        # Detailed Results HTML
        detailed_html = ""
        if summary and summary.successful_trials > 0:
            dir_rows = ""
            for dname, dinfo in summary.direction_accuracies.items():
                vc, rc = dinfo.get("valid_trials", 0), dinfo.get("rejected_trials", 0)
                if vc > 0:
                    dir_rows += (
                        f"<tr><td><code>{esc(dname)}</code></td><td class='num'>{vc} / {rc}</td>"
                        f"<td class='num'><strong>{dinfo['target_selection_accuracy_pct']:.1f}%</strong></td>"
                        f"<td class='num'>{dinfo['pct_within_5pct']:.1f}%</td>"
                        f"<td class='num'>{dinfo['mean_euclidean_pixels']:.1f} px</td>"
                        f"<td class='num'>({dinfo['signed_bias_x_px']:+.1f}, {dinfo['signed_bias_y_px']:+.1f})</td></tr>"
                    )
                else:
                    dir_rows += f"<tr><td><code>{esc(dname)}</code></td><td class='num'>0 / {rc}</td><td class='num'>N/A</td><td class='num'>N/A</td><td class='num'>REJECTED</td><td class='num'>N/A</td></tr>"

            reg_rows = ""
            for rname, rinfo in summary.region_accuracies.items():
                vc, rc = rinfo.get("valid_trials", 0), rinfo.get("rejected_trials", 0)
                if vc > 0:
                    reg_rows += (
                        f"<tr><td><code>{esc(rname)}</code></td><td class='num'>{vc} / {rc}</td>"
                        f"<td class='num'><strong>{rinfo['target_selection_accuracy_pct']:.1f}%</strong></td>"
                        f"<td class='num'>{rinfo['pct_within_5pct']:.1f}%</td>"
                        f"<td class='num'>{rinfo['mean_euclidean_pixels']:.1f} px</td>"
                        f"<td class='num'>({rinfo['signed_bias_x_px']:+.1f}, {rinfo['signed_bias_y_px']:+.1f})</td></tr>"
                    )
                else:
                    reg_rows += f"<tr><td><code>{esc(rname)}</code></td><td class='num'>0 / {rc}</td><td class='num'>N/A</td><td class='num'>N/A</td><td class='num'>REJECTED</td><td class='num'>N/A</td></tr>"

            per_target_rows = ""
            for tname, tinfo in summary.target_accuracies.items():
                v_cnt = tinfo.get("valid_trials", 0)
                r_cnt = tinfo.get("rejected_trials", 0)
                if v_cnt > 0:
                    hits = tinfo.get("successful_selections", 0)
                    sel_pct = tinfo.get("target_selection_accuracy_pct", 0.0)
                    per_target_rows += (
                        f"<tr><td><code>{esc(tname)}</code></td><td class='num'>{v_cnt} / {r_cnt}</td>"
                        f"<td class='num'>{hits} / {v_cnt}</td><td class='num'><strong>{sel_pct:.1f}%</strong></td>"
                        f"<td class='num'>{tinfo.get('mean_euclidean_pixels', 0.0):.1f} px</td></tr>"
                    )
                else:
                    per_target_rows += (
                        f"<tr><td><code>{esc(tname)}</code></td><td class='num'>0 / {r_cnt}</td>"
                        f"<td class='num'>0 / 0</td><td class='num'>N/A</td><td class='num'>REJECTED</td></tr>"
                    )
            detailed_html = f"""
            <h3>Prediction Error Breakdown (Lower is Better)</h3>
            <table>
              <thead><tr><th>Error Metric</th><th class="num">Pixels</th><th class="num">Diagonal % / Normalized</th></tr></thead>
              <tbody>
                <tr><td>X-Axis MAE / RMSE</td><td class="num">{summary.mae_x_px:.1f} px / {summary.rmse_x_px:.1f} px</td><td class="num">{summary.mae_x:.4f} / {summary.rmse_x:.4f} norm</td></tr>
                <tr><td>Y-Axis MAE / RMSE</td><td class="num">{summary.mae_y_px:.1f} px / {summary.rmse_y_px:.1f} px</td><td class="num">{summary.mae_y:.4f} / {summary.rmse_y:.4f} norm</td></tr>
                <tr><td>Median Error</td><td class="num">{summary.median_error_pixels:.1f} px</td><td class="num">{summary.median_error_pct_diagonal:.1f}% diag</td></tr>
                <tr><td>Maximum Error</td><td class="num">{summary.max_error_pixels:.1f} px</td><td class="num">{summary.max_error_pct_diagonal:.1f}% diag</td></tr>
                <tr><td>Selections (Hits / Fails) &amp; Trials (Valid / Rejected)</td><td class="num" colspan="2">{summary.successful_selections} Hits, {summary.failed_selections} Fails | {summary.successful_trials} Valid, {summary.rejected_trials} Rejected ({summary.rejected_trial_pct:.1f}%)</td></tr>
              </tbody>
            </table>
            {"<h3>Direction Accuracy (LEFT / RIGHT / CENTER / UP / DOWN)</h3><table><thead><tr><th>Direction</th><th class='num'>Valid / Rej</th><th class='num'>Selection Acc</th><th class='num'>Within 5%</th><th class='num'>Mean Error</th><th class='num'>Bias (X, Y) px</th></tr></thead><tbody>" + dir_rows + "</tbody></table>" if dir_rows else ""}
            {"<h3>Screen Region Accuracy (Rows, Columns &amp; Edges)</h3><table><thead><tr><th>Region</th><th class='num'>Valid / Rej</th><th class='num'>Selection Acc</th><th class='num'>Within 5%</th><th class='num'>Mean Error</th><th class='num'>Bias (X, Y) px</th></tr></thead><tbody>" + reg_rows + "</tbody></table>" if reg_rows else ""}
            <h3>Per-Target Performance</h3>
            <table>
              <thead><tr><th>Target</th><th class="num">Valid / Rejected</th><th class="num">Hits / Valid</th><th class="num">Selection Accuracy</th><th class="num">Mean Error (px)</th></tr></thead>
              <tbody>{per_target_rows}</tbody>
            </table>
            """

        tr_mean_str = (
            f"{train_metrics['mean_euclidean_pixels']:.1f} px"
            if (train_metrics and train_metrics.get("mean_euclidean_pixels") is not None)
            else "Not recorded"
        )
        val_perf_str = (
            f"{summary.target_selection_accuracy_pct:.1f}% hits | {summary.pct_within_5pct:.1f}% within 5% | Mean Error: {summary.mean_error_pixels:.1f} px"
            if (summary and summary.successful_trials > 0)
            else VALIDATION_NOT_COMPLETED_MSG
        )

        # Comparison rows HTML
        comp_rows = ""
        for r in comparison_summary.get("records", []):
            is_cur = " style='background:#eef6ff;'" if r["experiment_id"] == self.experiment_id else ""
            if r["validation_completed"]:
                sel_s = f"{r['target_selection_accuracy_pct']:.1f}%" if r["target_selection_accuracy_pct"] is not None else "N/A"
                p2_s = f"{r['pct_within_2pct']:.1f}%" if r["pct_within_2pct"] is not None else "N/A"
                p5_s = f"{r['pct_within_5pct']:.1f}%" if r["pct_within_5pct"] is not None else "N/A"
                p10_s = f"{r['pct_within_10pct']:.1f}%" if r["pct_within_10pct"] is not None else "N/A"
                mean_s = f"{r['mean_error_px']:.1f} px" if r["mean_error_px"] is not None else "N/A"
                med_s = f"{r['median_error_px']:.1f} px" if r["median_error_px"] is not None else "N/A"
                max_s = f"{r['max_error_px']:.1f} px" if r["max_error_px"] is not None else "N/A"
                trials_s = f"{r['valid_trials']} / {r['rejected_trials']}"
            else:
                sel_s = p2_s = p5_s = p10_s = mean_s = med_s = max_s = "N/A"
                trials_s = "Not validated"
            comp_rows += (
                f"<tr{is_cur}><td><code>{esc(r['experiment_id'])}</code></td><td>{esc(r['model'].upper())}</td>"
                f"<td>{esc(r['normalization'])}</td><td class='num'>{sel_s}</td>"
                f"<td class='num'>{p2_s}</td><td class='num'>{p5_s}</td><td class='num'>{p10_s}</td>"
                f"<td class='num'>{mean_s}</td><td class='num'>{med_s}</td><td class='num'>{max_s}</td>"
                f"<td class='num'>{trials_s}</td><td>{esc(r['status'])}</td></tr>"
            )

        plots_html = ""
        if plot_paths:
            imgs = "".join(
                f"<div class='plot'><img src='{esc(p.name)}' alt='{esc(p.stem)}'/></div>"
                for p in plot_paths
            )
            plots_html = f"<h2>Diagnostic Plots</h2><div class='plots'>{imgs}</div>"

        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>AABAR CV Report — {esc(overview['experiment_id'])}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 2rem auto; max-width: 1020px; padding: 0 1.25rem; color: #1e293b; background: #f8fafc; line-height: 1.55; }}
  h1, h2, h3 {{ color: #0f172a; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin: 1rem 0 1.5rem; }}
  .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem 1.2rem; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
  .card.highlight {{ border-color: #2563eb; background: #eff6ff; }}
  .card-label {{ font-size: 0.85rem; color: #475569; font-weight: 600; text-transform: uppercase; }}
  .card-val {{ font-size: 1.85rem; font-weight: 700; color: #0f172a; margin: 0.25rem 0; }}
  .card-sub {{ font-size: 0.85rem; color: #64748b; }}
  table {{ width: 100%; border-collapse: collapse; background: #ffffff; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0; margin-bottom: 1.5rem; font-size: 0.92rem; }}
  th, td {{ padding: 0.65rem 0.85rem; border-bottom: 1px solid #e2e8f0; text-align: left; }}
  th {{ background: #1e293b; color: #ffffff; font-weight: 600; }}
  td.num, th.num {{ text-align: right; }}
  .notice {{ padding: 0.9rem 1.1rem; border-radius: 8px; margin: 1rem 0; font-weight: 600; }}
  .notice.warn {{ background: #fef3c7; border: 1px solid #f59e0b; color: #92400e; }}
  .notice.info {{ background: #e0f2fe; border: 1px solid #0284c7; color: #0c4a6e; font-weight: 400; }}
  .plots {{ display: grid; grid-template-columns: 1fr; gap: 1rem; }}
  .plot img {{ max-width: 100%; border-radius: 8px; border: 1px solid #cbd5e1; }}
</style>
</head>
<body>
  <h1>AABAR Gaze Tracking Experiment Report</h1>

  <h2>Experiment Overview</h2>
  <table>
    <tbody>
      <tr><th>Experiment ID</th><td><code>{esc(overview['experiment_id'])}</code></td></tr>
      <tr><th>Date and Time</th><td>{esc(overview['date_time'])}</td></tr>
      <tr><th>Screen Resolution</th><td>{esc(overview['screen_resolution'])} (Diagonal: {overview['screen_diagonal_px']:.1f} px)</td></tr>
      <tr><th>Calibration Model</th><td>{esc(overview['model'])}</td></tr>
      <tr><th>Normalization Method</th><td>{esc(overview['normalization_method'])}</td></tr>
      <tr><th>Number of Training Samples</th><td>{esc(overview['training_samples_desc'])}</td></tr>
      <tr><th>Number of Validation Trials</th><td>{esc(overview['validation_trials_desc'])}</td></tr>
    </tbody>
  </table>

  <h2>Main Results</h2>
  {main_results_html}

  <h2>Detailed Results</h2>
  {detailed_html}
  <h3>Model Training Fit vs. Independent Validation</h3>
  <table>
    <tbody>
      <tr><th>Calibration Training Error (Fit Only — Not Validation Accuracy)</th><td class="num">{esc(tr_mean_str)}</td></tr>
      <tr><th>Independent Validation Performance</th><td class="num">{esc(val_perf_str)}</td></tr>
    </tbody>
  </table>

  <h2>Simple Explanation</h2>
  <div class="notice info">{esc(explanation)}</div>

  <h2>Comparison with Previous Experiments</h2>
  <table>
    <thead>
      <tr>
        <th>Experiment ID</th><th>Model</th><th>Norm</th>
        <th class="num">Selection Acc</th><th class="num">Within 2%</th><th class="num">Within 5%</th><th class="num">Within 10%</th>
        <th class="num">Mean Err</th><th class="num">Median Err</th><th class="num">Max Err</th>
        <th class="num">Valid / Rej</th><th>Status</th>
      </tr>
    </thead>
    <tbody>{comp_rows}</tbody>
  </table>

  {plots_html}
</body>
</html>
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_doc)
        return filepath

    def _generate_accuracy_plots(
        self,
        summary: AccuracySummary,
        trials: List[ValidationTrialRecord],
    ) -> List[Path]:
        """Generate diagnostic accuracy plots using Matplotlib (Agg backend)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            logger.warning(f"Matplotlib not available or failed to load: {e}. Skipping plot generation.")
            return []

        created_plots: List[Path] = []
        valid_trials = [r for r in trials if r.trial_status == "VALID" and r.error_x is not None]
        if not valid_trials:
            logger.warning("No valid trials available to generate accuracy plots.")
            return []

        # 1. plot_actual_vs_predicted.png
        try:
            fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150)
            fig.patch.set_facecolor("#1e1e24")
            ax.set_facecolor("#121216")

            ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="#555566", linestyle="--", linewidth=1.5, label="Screen Boundary")

            act_x = [r.actual_x for r in valid_trials]
            act_y = [r.actual_y for r in valid_trials]
            pred_x = [r.predicted_x for r in valid_trials]
            pred_y = [r.predicted_y for r in valid_trials]

            for r in valid_trials:
                ax.annotate(
                    "",
                    xy=(r.predicted_x, r.predicted_y),
                    xytext=(r.actual_x, r.actual_y),
                    arrowprops=dict(arrowstyle="->", color="#ff5555", lw=1.5, alpha=0.85),
                )

            ax.scatter(act_x, act_y, color="#00d4ff", s=110, edgecolors="white", linewidths=1.2, zorder=5, label="Target (Ground Truth)")
            ax.scatter(pred_x, pred_y, color="#ffaa00", s=70, marker="D", edgecolors="black", linewidths=0.8, zorder=6, label="Predicted Gaze")

            seen = set()
            for r in valid_trials:
                if r.target_name not in seen:
                    seen.add(r.target_name)
                    ax.text(r.actual_x, r.actual_y - 0.035, r.target_name, color="#00d4ff", fontsize=8, ha="center", va="top")

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.invert_yaxis()
            ax.set_xlabel("Normalized Screen X (0.0 = Left, 1.0 = Right)", color="#cccccc", fontsize=9)
            ax.set_ylabel("Normalized Screen Y (0.0 = Top, 1.0 = Bottom)", color="#cccccc", fontsize=9)
            ax.tick_params(colors="#888899", labelsize=8)
            ax.grid(True, color="#2a2a35", linestyle=":", alpha=0.7)

            title_str = (
                f"Actual vs Predicted Screen Gaze ({summary.model_name.upper()} | {summary.normalization_method})\n"
                f"Target Selection Accuracy: {summary.target_selection_accuracy_pct:.1f}% | "
                f"Mean Error: {summary.mean_error_pixels:.1f} px ({summary.mean_error_pct_diagonal:.1f}% diag)"
            )
            ax.set_title(title_str, color="white", fontsize=11, fontweight="bold", pad=12)
            ax.legend(facecolor="#22222a", edgecolor="#444455", labelcolor="#dddddd", fontsize=8, loc="upper right")

            plot_path1 = self.experiment_dir / "plot_actual_vs_predicted.png"
            plt.tight_layout()
            plt.savefig(plot_path1, facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)
            created_plots.append(plot_path1)
        except Exception as e:
            logger.warning(f"Failed to generate plot_actual_vs_predicted.png: {e}")

        # 2. plot_error_distributions.png
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
            fig.patch.set_facecolor("#1e1e24")
            ax1.set_facecolor("#121216")
            ax2.set_facecolor("#121216")

            e_px = [r.euclidean_error_pixels for r in valid_trials if r.euclidean_error_pixels is not None]
            ax1.hist(e_px, bins=max(5, len(e_px) // 2), color="#00bcd4", edgecolor="#1e1e24", alpha=0.75, density=False)
            ax1.axvline(summary.mean_error_pixels, color="#ffaa00", linestyle="--", linewidth=1.8, label=f"Mean ({summary.mean_error_pixels:.1f} px)")
            ax1.axvline(summary.median_error_pixels, color="#00ff88", linestyle=":", linewidth=1.8, label=f"Median ({summary.median_error_pixels:.1f} px)")
            ax1.axvline(summary.threshold_2pct_px, color="#ff4444", linestyle="-.", linewidth=1.2, alpha=0.8, label=f"{_fmt_pct_threshold(summary.threshold_2pct_frac)} Diag ({summary.threshold_2pct_px:.0f} px)")
            ax1.axvline(summary.threshold_5pct_px, color="#ff8800", linestyle="-.", linewidth=1.2, alpha=0.8, label=f"{_fmt_pct_threshold(summary.threshold_5pct_frac)} Diag ({summary.threshold_5pct_px:.0f} px)")

            ax1.set_title("Euclidean Pixel Error Distribution", color="white", fontsize=10, fontweight="bold")
            ax1.set_xlabel("Euclidean Error (Pixels)", color="#cccccc", fontsize=9)
            ax1.set_ylabel("Frequency (Trials)", color="#cccccc", fontsize=9)
            ax1.tick_params(colors="#888899", labelsize=8)
            ax1.grid(True, color="#2a2a35", linestyle=":", alpha=0.7)
            ax1.legend(facecolor="#22222a", edgecolor="#444455", labelcolor="#dddddd", fontsize=7.5)

            err_x_px = [r.error_x * summary.screen_width for r in valid_trials if r.error_x is not None]
            err_y_px = [r.error_y * summary.screen_height for r in valid_trials if r.error_y is not None]

            ax2.scatter(err_x_px, err_y_px, color="#ff7700", s=60, edgecolors="white", alpha=0.85)
            ax2.axhline(0, color="#555566", linestyle="--", linewidth=1.0)
            ax2.axvline(0, color="#555566", linestyle="--", linewidth=1.0)

            mean_bias_x = np.mean(err_x_px)
            mean_bias_y = np.mean(err_y_px)
            ax2.scatter([mean_bias_x], [mean_bias_y], color="#00ffcc", s=130, marker="X", label=f"Mean Bias ({mean_bias_x:+.1f}, {mean_bias_y:+.1f}) px")

            ax2.set_title("Signed Error Scatter (Spatial Bias)", color="white", fontsize=10, fontweight="bold")
            ax2.set_xlabel("Horizontal Error (Pixels, + = Right)", color="#cccccc", fontsize=9)
            ax2.set_ylabel("Vertical Error (Pixels, + = Down)", color="#cccccc", fontsize=9)
            ax2.tick_params(colors="#888899", labelsize=8)
            ax2.grid(True, color="#2a2a35", linestyle=":", alpha=0.7)
            ax2.legend(facecolor="#22222a", edgecolor="#444455", labelcolor="#dddddd", fontsize=7.5)

            plot_path2 = self.experiment_dir / "plot_error_distributions.png"
            plt.tight_layout()
            plt.savefig(plot_path2, facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)
            created_plots.append(plot_path2)
        except Exception as e:
            logger.warning(f"Failed to generate plot_error_distributions.png: {e}")

        # 3. plot_per_target_breakdown.png
        try:
            valid_targets = {k: v for k, v in summary.target_accuracies.items() if v.get("valid_trials", 0) > 0}
            if valid_targets:
                fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
                fig.patch.set_facecolor("#1e1e24")
                ax.set_facecolor("#121216")

                tnames = list(valid_targets.keys())
                terrs = [valid_targets[k]["mean_euclidean_pixels"] for k in tnames]

                bars = ax.bar(tnames, terrs, color="#00bcd4", width=0.55, edgecolor="#222230", alpha=0.85)

                ax.axhline(summary.threshold_2pct_px, color="#00ff88", linestyle="--", linewidth=1.2, label=f"{_fmt_pct_threshold(summary.threshold_2pct_frac)} Diag ({summary.threshold_2pct_px:.0f} px)")
                ax.axhline(summary.threshold_5pct_px, color="#ffaa00", linestyle="--", linewidth=1.2, label=f"{_fmt_pct_threshold(summary.threshold_5pct_frac)} Diag ({summary.threshold_5pct_px:.0f} px)")
                ax.axhline(summary.threshold_10pct_px, color="#ff4444", linestyle="--", linewidth=1.2, label=f"{_fmt_pct_threshold(summary.threshold_10pct_frac)} Diag ({summary.threshold_10pct_px:.0f} px)")

                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f} px", ha="center", va="bottom", color="#ffffff", fontsize=8)

                ax.set_title("Per-Target Mean Euclidean Error (Pixels)", color="white", fontsize=11, fontweight="bold", pad=10)
                ax.set_ylabel("Mean Error (Pixels)", color="#cccccc", fontsize=9)
                ax.tick_params(colors="#888899", labelsize=8)
                ax.set_xticks(range(len(tnames)))
                ax.set_xticklabels(tnames, rotation=15, ha="right", color="#dddddd")
                ax.grid(True, axis="y", color="#2a2a35", linestyle=":", alpha=0.7)
                ax.legend(facecolor="#22222a", edgecolor="#444455", labelcolor="#dddddd", fontsize=8)

                plot_path3 = self.experiment_dir / "plot_per_target_breakdown.png"
                plt.tight_layout()
                plt.savefig(plot_path3, facecolor=fig.get_facecolor(), edgecolor="none")
                plt.close(fig)
                created_plots.append(plot_path3)
        except Exception as e:
            logger.warning(f"Failed to generate plot_per_target_breakdown.png: {e}")

        return created_plots

    def _write_metadata_json(
        self,
        profile: Optional[CalibrationProfile],
        extra: Optional[Dict[str, Any]],
    ) -> Path:
        filepath = self.experiment_dir / "experiment_metadata.json"
        meta: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "created_at": time.time(),
            "timestamp_str": self.timestamp_str,
            "notice": (
                "Calibration metrics represent training fit on calibration targets only. "
                "Validation metrics represent empirical accuracy and error on independent test targets."
            ),
            "profile": profile.to_dict() if profile else None,
        }
        if extra:
            meta.update(extra)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return filepath

    def _export_excel_workbook(
        self,
        calibration_samples: List[CalibrationSampleRecord],
        validation_results: List[ValidationTrialRecord],
        accuracy_summary: Optional[AccuracySummary],
        metadata_extra: Optional[Dict[str, Any]],
    ) -> Optional[Path]:
        """Create a multi-sheet formatted Excel report if openpyxl is available."""
        try:
            import openpyxl
            from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        except ImportError:
            logger.info("Optional package 'openpyxl' not installed. Skipping Excel export (CSVs written).")
            return None

        filepath = self.experiment_dir / "experiment_report.xlsx"
        wb = openpyxl.Workbook()

        header_fill = PatternFill(start_color="1A365D", end_color="1A365D", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        thin_border = Border(
            left=Side(style="thin", color="D1D5DB"),
            right=Side(style="thin", color="D1D5DB"),
            top=Side(style="thin", color="D1D5DB"),
            bottom=Side(style="thin", color="D1D5DB"),
        )

        def style_header(row: Any) -> None:
            for cell in row:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")

        def auto_column_width(ws: Any) -> None:
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                col_letter = openpyxl.utils.get_column_letter(col[0].column)
                ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

        ws_meta = wb.active
        ws_meta.title = "Overview & Metadata"
        ws_meta.views.sheetView[0].showGridLines = True

        meta_rows = [
            ("AABAR CV - Calibration & Validation Report", ""),
            ("Experiment ID", self.experiment_id),
            ("Date & Time", time.strftime("%Y-%m-%d %H:%M:%S")),
            ("Data Leakage Prevention", "Strict Separation: 9 Training Points vs Independent Validation Points"),
            ("Calibration Samples Count", len(calibration_samples)),
            ("Validation Trials Count", len(validation_results)),
            ("Model Architecture", accuracy_summary.model_name.upper() if accuracy_summary else "N/A"),
            ("Normalization Method", accuracy_summary.normalization_method if accuracy_summary else "N/A"),
            ("Screen Resolution", accuracy_summary.screen_resolution if accuracy_summary else "N/A"),
            ("Target Selection Accuracy", f"{accuracy_summary.target_selection_accuracy_pct:.1f}%" if accuracy_summary else VALIDATION_NOT_COMPLETED_MSG),
            ("Accuracy Within 5% Diagonal", f"{accuracy_summary.pct_within_5pct:.1f}%" if accuracy_summary else VALIDATION_NOT_COMPLETED_MSG),
            ("Mean Prediction Error (Pixels)", f"{accuracy_summary.mean_error_pixels:.1f} px" if accuracy_summary else "N/A"),
        ]
        for r_idx, (k, v) in enumerate(meta_rows, start=1):
            ws_meta.cell(row=r_idx, column=1, value=k)
            ws_meta.cell(row=r_idx, column=2, value=v)
            if r_idx == 1:
                ws_meta.cell(row=r_idx, column=1).font = Font(name="Calibri", size=14, bold=True, color="1A365D")
            else:
                ws_meta.cell(row=r_idx, column=1).font = Font(name="Calibri", size=11, bold=True)
                ws_meta.cell(row=r_idx, column=2).font = Font(name="Calibri", size=11)
        auto_column_width(ws_meta)

        if accuracy_summary:
            ws_acc = wb.create_sheet(title="Results Summary")
            ws_acc.views.sheetView[0].showGridLines = True

            ws_acc.append(["Metric", "Value", "Unit", "Category / Description"])
            style_header(ws_acc[1])

            acc_items = [
                ("Total Trials Attempted", accuracy_summary.total_trials, "trials", "Validation Volume"),
                ("Successful Valid Trials", accuracy_summary.successful_trials, "trials", "Validation Volume"),
                ("Rejected Trials", accuracy_summary.rejected_trials, "trials", "Validation Volume"),
                ("Target Selection Accuracy", f"{accuracy_summary.target_selection_accuracy_pct:.1f}%", "%", f"Accuracy ({accuracy_summary.successful_selections}/{accuracy_summary.successful_trials} inside boundary)"),
                ("Accuracy Within 2% Diagonal", f"{accuracy_summary.pct_within_2pct:.1f}%", "%", f"Accuracy (Error <= {accuracy_summary.threshold_2pct_px:.1f} px)"),
                ("Accuracy Within 5% Diagonal", f"{accuracy_summary.pct_within_5pct:.1f}%", "%", f"Accuracy (Error <= {accuracy_summary.threshold_5pct_px:.1f} px)"),
                ("Accuracy Within 10% Diagonal", f"{accuracy_summary.pct_within_10pct:.1f}%", "%", f"Accuracy (Error <= {accuracy_summary.threshold_10pct_px:.1f} px)"),
                ("Mean Error (Pixels)", round(accuracy_summary.mean_error_pixels, 1), "pixels", "Prediction Error"),
                ("Mean Error (% Diagonal)", round(accuracy_summary.mean_error_pct_diagonal, 2), "% diag", "Prediction Error"),
                ("Median Error (Pixels)", round(accuracy_summary.median_error_pixels, 1), "pixels", "Prediction Error"),
                ("Maximum Error (Pixels)", round(accuracy_summary.max_error_pixels, 1), "pixels", "Prediction Error"),
                ("Horizontal MAE", round(accuracy_summary.mae_x_px, 1), "pixels", "Prediction Error X"),
                ("Vertical MAE", round(accuracy_summary.mae_y_px, 1), "pixels", "Prediction Error Y"),
                ("Horizontal RMSE", round(accuracy_summary.rmse_x_px, 1), "pixels", "Prediction Error X"),
                ("Vertical RMSE", round(accuracy_summary.rmse_y_px, 1), "pixels", "Prediction Error Y"),
            ]
            for row in acc_items:
                ws_acc.append(list(row))
            for row in ws_acc.iter_rows(min_row=2):
                for cell in row:
                    cell.border = thin_border
            auto_column_width(ws_acc)

        if validation_results:
            ws_val = wb.create_sheet(title="Validation Trials")
            ws_val.views.sheetView[0].showGridLines = True

            val_headers = [
                "Trial ID", "Target Name", "Actual X", "Actual Y", "Predicted X", "Predicted Y",
                "Pixel Err X", "Pixel Err Y", "Euclidean (Pixels)", "Err % Diag", "Target Hit",
                "Accepted Samples", "Rejected Samples", "Duration (s)", "Status", "Rejection Reason",
            ]
            ws_val.append(val_headers)
            style_header(ws_val[1])

            for t in validation_results:
                ws_val.append([
                    t.trial_id,
                    t.target_name,
                    t.actual_x,
                    t.actual_y,
                    t.predicted_x if t.predicted_x is not None else "",
                    t.predicted_y if t.predicted_y is not None else "",
                    round(t.pixel_error_x, 2) if t.pixel_error_x is not None else "",
                    round(t.pixel_error_y, 2) if t.pixel_error_y is not None else "",
                    round(t.euclidean_error_pixels, 2) if t.euclidean_error_pixels is not None else "",
                    round(t.error_pct_diagonal, 2) if t.error_pct_diagonal is not None else "",
                    "HIT" if t.target_hit is True else ("MISS" if t.target_hit is False else ""),
                    t.accepted_sample_count,
                    t.rejected_sample_count,
                    t.trial_duration_sec,
                    t.trial_status,
                    t.rejection_reason or "",
                ])
            for row in ws_val.iter_rows(min_row=2):
                for cell in row:
                    cell.border = thin_border
            auto_column_width(ws_val)

        if accuracy_summary and accuracy_summary.direction_accuracies:
            ws_dir = wb.create_sheet(title="Direction & Region Accuracy")
            ws_dir.views.sheetView[0].showGridLines = True
            ws_dir.append(["Breakdown Type", "Name", "Valid", "Rejected", "Selection Acc (%)", "Within 5% (%)", "Mean Error (px)", "Median Error (px)", "Bias X (px)", "Bias Y (px)"])
            style_header(ws_dir[1])
            for cat, dmap in [("Direction", accuracy_summary.direction_accuracies), ("Screen Region", accuracy_summary.region_accuracies)]:
                for k, v in dmap.items():
                    ws_dir.append([
                        cat, k, v.get("valid_trials", 0), v.get("rejected_trials", 0),
                        v.get("target_selection_accuracy_pct", "N/A"),
                        v.get("pct_within_5pct", "N/A"),
                        v.get("mean_euclidean_pixels", "N/A"),
                        v.get("median_euclidean_pixels", "N/A"),
                        v.get("signed_bias_x_px", "N/A"),
                        v.get("signed_bias_y_px", "N/A"),
                    ])
            for row in ws_dir.iter_rows(min_row=2):
                for cell in row:
                    cell.border = thin_border
            auto_column_width(ws_dir)

        wb.save(filepath)
        return filepath


def compare_experiments(
    experiment_dirs: List[Path],
    output_format: str = "markdown",
) -> str:
    """
    Compare multiple calibration/validation experiment sessions side-by-side.

    Strictly compares independent validation metrics only and flags incompatible
    resolutions, target boundaries, or coordinate conventions explicitly.
    """
    comp = build_experiment_comparison_summary(experiment_dirs)
    records = comp["records"]

    if not records:
        return "No experiment directories found with session metadata or accuracy summary."

    header = (
        "| Experiment ID | Model | Norm | Sel Acc | Within 2% | Within 5% | Within 10% | Mean Error | Median | Rej % | Weakest Dir / Region | Status |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |"
    )
    lines = [header]
    for r in records:
        if r["validation_completed"]:
            sel_s = f"{r['target_selection_accuracy_pct']:.1f}%" if r["target_selection_accuracy_pct"] is not None else "N/A"
            p2_s = f"{r['pct_within_2pct']:.1f}%" if r["pct_within_2pct"] is not None else "N/A"
            p5_s = f"{r['pct_within_5pct']:.1f}%" if r["pct_within_5pct"] is not None else "N/A"
            p10_s = f"{r['pct_within_10pct']:.1f}%" if r["pct_within_10pct"] is not None else "N/A"
            mean_s = f"**{r['mean_error_px']:.1f} px**" if r["mean_error_px"] is not None else "N/A"
            med_s = f"{r['median_error_px']:.1f} px" if r["median_error_px"] is not None else "N/A"
            rej_s = f"{r['rejected_trial_pct']:.1f}%" if r.get("rejected_trial_pct") is not None else "0.0%"
            wd = r.get("weakest_direction") or "--"
            wr = r.get("weakest_region") or "--"
            weak_s = f"{wd} / {wr}"
        else:
            sel_s = p2_s = p5_s = p10_s = mean_s = med_s = "N/A"
            rej_s = f"{r['rejected_trial_pct']:.1f}%" if r.get("rejected_trial_pct") is not None else "N/A"
            weak_s = "--"

        lines.append(
            f"| `{r['experiment_id']}` | **{r['model'].upper()}** | {r['normalization']} | "
            f"{sel_s} | {p2_s} | {p5_s} | {p10_s} | "
            f"{mean_s} | {med_s} | {rej_s} | {weak_s} | {r['status']} |"
        )

    lines.append("")
    lvp = comp.get("latest_vs_previous")
    best = comp.get("best_experiment")
    best_err = comp.get("best_by_mean_error")
    warnings_list = comp.get("protocol_warnings") or []

    for wmsg in warnings_list:
        lines.append(f"- **Compatibility Warning:** {wmsg}")

    if best is not None:
        best_sel = (
            f"{best['target_selection_accuracy_pct']:.1f}% target selection, "
            if best["target_selection_accuracy_pct"] is not None
            else ""
        )
        best_err_s = f"{best['mean_error_px']:.1f} px mean error" if best["mean_error_px"] is not None else ""
        weak_bits = []
        if best.get("weakest_direction"):
            weak_bits.append(f"weakest dir: {best['weakest_direction']}")
        if best.get("weakest_region"):
            weak_bits.append(f"weakest region: {best['weakest_region']}")
        weak_note = f" ({', '.join(weak_bits)})" if weak_bits else ""
        lines.append(
            f"- **Best Validated Experiment:** `{best['experiment_id']}` ({best_sel}{best_err_s}){weak_note}."
        )
        if best_err is not None and best_err["experiment_id"] != best["experiment_id"]:
            lines.append(
                f"- **Lowest Mean Pixel Error:** `{best_err['experiment_id']}` ({best_err['mean_error_px']:.1f} px)."
            )
    else:
        lines.append("- **Best Validated Experiment:** Not available — no completed validation experiments found.")

    if lvp is not None:
        verdict = lvp["verdict"]
        delta_parts: List[str] = []
        if lvp["mean_error_improvement_pct"] is not None:
            pct_imp = lvp["mean_error_improvement_pct"]
            lbl = "error reduction" if pct_imp >= 0 else "error increase"
            delta_parts.append(f"Mean Error {lvp['mean_error_delta_px']:+.1f} px ({abs(pct_imp):.1f}% {lbl})")
        elif lvp["mean_error_delta_px"] is not None:
            delta_parts.append(f"Mean Error {lvp['mean_error_delta_px']:+.1f} px (relative % N/A from 0 px baseline)")
        if lvp["target_selection_pp_change"] is not None:
            delta_parts.append(f"Target Selection Accuracy {lvp['target_selection_pp_change']:+.1f} percentage points")
        if lvp["within_5pct_pp_change"] is not None:
            delta_parts.append(f"Within 5% Accuracy {lvp['within_5pct_pp_change']:+.1f} percentage points")
        compat_tag = "" if lvp.get("compatible_protocol", True) else " [Warning: protocol/resolution mismatch]"
        summary_str = "; ".join(delta_parts)
        lines.append(
            f"- **Latest vs. Previous Validated (`{lvp['latest_id']}` vs `{lvp['previous_id']}`):** "
            f"**{verdict.upper()}** ({summary_str}){compat_tag}."
        )
    elif len(comp.get("validated_records", [])) == 1:
        lines.append("- **Trend:** Only 1 completed validation experiment available; run another validation session to compare improvement.")

    return "\n".join(lines)
