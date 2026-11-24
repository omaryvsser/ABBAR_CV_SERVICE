"""
Tests for Central Experiment Tracker (`AABAR_Experiments_Tracker.xlsx`),
Historical Data Recovery, Deduplication, Single-Source Accuracy Consistency,
and Results Summary Visibility.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List

import openpyxl
import pytest

from src.core.calibration_profile import CalibrationProfile
from src.core.experiment_logger import ExperimentLogger, VALIDATION_NOT_COMPLETED_MSG
from src.core.experiment_tracker import (
    CANONICAL_DIRECTIONS,
    CANONICAL_SCREEN_REGIONS_9,
    TRACKER_FILENAME,
    format_experiment_results_summary,
    load_tracker_dataset,
    update_central_tracker,
)
from src.core.screen_calibration import CalibrationSampleRecord
from src.core.screen_validator import (
    FULL_9_REGION_VALIDATION_TARGETS,
    AccuracySummary,
    ScreenValidator,
    ValidationTrialRecord,
)
from tests.conftest import create_synthetic_landmarks
from src.core.eye_features import extract_eye_features
from src.core.face_landmarker import FaceLandmarkerDetectionResult
from src.core.gaze_mapper import LinearGazeMapper
from src.core.gaze_output import ScreenGazeOutput
from src.utils.fps_counter import FPSMetrics
from src.visualization.visualizer import CVVisualizer
import numpy as np


def _write_calibration_csv(exp_dir: Path, exp_id: str, session_id: str = "sess_hist", count: int = 18) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    headers = ["experiment_id", "session_id", "target_id", "target_x", "target_y", "sample_accepted"]
    with open(exp_dir / "calibration_samples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for i in range(count):
            w.writerow({
                "experiment_id": exp_id,
                "session_id": session_id,
                "target_id": i % 9,
                "target_x": 0.5,
                "target_y": 0.5,
                "sample_accepted": 1,
            })


def _make_9_region_trials(
    exp_id: str,
    hit_indices: List[int],
    reject_indices: List[int] = (),  # type: ignore[assignment]
    session_id: str = "sess_val",
) -> List[ValidationTrialRecord]:
    records: List[ValidationTrialRecord] = []
    for idx, tgt in enumerate(FULL_9_REGION_VALIDATION_TARGETS):
        t_id = idx + 1
        if idx in reject_indices:
            records.append(
                ValidationTrialRecord(
                    experiment_id=exp_id,
                    session_id=session_id,
                    trial_id=t_id,
                    target_id=tgt.target_id,
                    target_name=tgt.name,
                    actual_x=tgt.x,
                    actual_y=tgt.y,
                    predicted_x=None,
                    predicted_y=None,
                    error_x=None,
                    error_y=None,
                    euclidean_error_norm=None,
                    euclidean_error_pixels=None,
                    screen_width=1920,
                    screen_height=1080,
                    tracking_quality="POOR",
                    accepted_sample_count=1,
                    rejected_sample_count=15,
                    trial_status="REJECTED",
                    rejection_reason="BLINK_DETECTED",
                    timestamp=1000.0 + idx,
                )
            )
            continue

        # Offset: hits inside 22 display px circle (~20 px error on 1920x1080), misses ~160 px off
        dx = 0.008 if idx in hit_indices else 0.080
        dy = 0.005 if idx in hit_indices else 0.050
        px = min(1.0, tgt.x + dx)
        py = min(1.0, tgt.y + dy)
        hit, b_spec = tgt.contains_point(px, py, 1920, 1080, 1280, 720)
        records.append(
            ValidationTrialRecord(
                experiment_id=exp_id,
                session_id=session_id,
                trial_id=t_id,
                target_id=tgt.target_id,
                target_name=tgt.name,
                actual_x=tgt.x,
                actual_y=tgt.y,
                predicted_x=px,
                predicted_y=py,
                error_x=px - tgt.x,
                error_y=py - tgt.y,
                euclidean_error_norm=float(np.hypot(px - tgt.x, py - tgt.y)),
                euclidean_error_pixels=float(np.hypot((px - tgt.x) * 1920, (py - tgt.y) * 1080)),
                screen_width=1920,
                screen_height=1080,
                tracking_quality="EXCELLENT",
                accepted_sample_count=20,
                rejected_sample_count=0,
                target_hit=hit,
                target_boundary=b_spec,
                trial_status="VALID",
                timestamp=1000.0 + idx,
            )
        )
    return records


def test_historical_recovery_empty_dirs_and_no_fake_training_accuracy(tmp_path: Path) -> None:
    """
    Historical legacy experiments (calibration_samples + metadata with training_metrics only)
    are recovered with N/A validation accuracy, empty startup folders are skipped cleanly,
    and historical files remain completely untouched.
    """
    exp_root = tmp_path / "experiments"
    prof_root = tmp_path / "calibration"
    exp_root.mkdir()
    prof_root.mkdir()

    # Empty startup dir (0 files) -> skipped, zero fake row
    (exp_root / "experiment_20261123_215714").mkdir()

    # Legacy historical session matching 20261123_221716 format (training_metrics only, NO validation)
    hist_dir = exp_root / "experiment_20261123_221716"
    _write_calibration_csv(hist_dir, "experiment_20261123_221716", session_id="sess_legacy", count=270)
    hist_meta = {
        "experiment_id": "experiment_20261123_221716",
        "created_at": 1795465056.21,
        "profile": {
            "profile_id": "user_default_1795465056",
            "model_type": "polynomial",
            "screen_width": 1920,
            "screen_height": 1080,
            "normalization_method": "None",
            "training_sample_count": 9,
            "training_metrics": {"mean_euclidean_pixels": 0.9},
        },
    }
    meta_bytes_before = json.dumps(hist_meta, indent=2)
    (hist_dir / "experiment_metadata.json").write_text(meta_bytes_before, encoding="utf-8")

    # Raw validation CSV WITHOUT summary JSON/CSV -> metrics MUST be recalculated from raw trials
    raw_only_dir = exp_root / "experiment_20261124_231000"
    _write_calibration_csv(raw_only_dir, "experiment_20261124_231000", count=180)
    (raw_only_dir / "experiment_metadata.json").write_text(
        json.dumps({"experiment_id": "experiment_20261124_231000", "created_at": 1795554000.0, "model_type": "compact_linear", "normalization_method": "StandardScaler", "validation_grid_mode": "full_9"}),
        encoding="utf-8",
    )
    trials = _make_9_region_trials("experiment_20261124_231000", hit_indices=[0, 1, 2, 3, 4, 5, 6, 7], reject_indices=[8])
    with open(raw_only_dir / "validation_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trials[0].to_csv_dict().keys()))
        w.writeheader()
        for t in trials:
            w.writerow(t.to_csv_dict())

    res = update_central_tracker(experiments_dir=exp_root, profiles_dir=prof_root)
    ds = res["dataset"]
    tracker_path = res["tracker_path"]

    assert tracker_path.exists()
    assert tracker_path.name == TRACKER_FILENAME
    assert len(ds["empty_dirs"]) == 1
    assert len(ds["experiments"]) == 2
    assert ds["completed_validation_count"] == 1
    assert ds["unvalidated_count"] == 1

    # Original historical file completely unmodified
    assert (hist_dir / "experiment_metadata.json").read_text(encoding="utf-8") == meta_bytes_before

    hist_rec = next(e for e in ds["experiments"] if e["experiment_id"] == "experiment_20261123_221716")
    assert hist_rec["validation_completed"] is False
    assert hist_rec["validation_status"] == "NOT_STARTED"
    assert hist_rec["target_selection_accuracy_pct"] is None
    assert hist_rec["mean_error_px"] is None
    assert hist_rec["training_fit_error_px"] == 0.9  # Kept strictly separate!
    assert hist_rec["training_sample_count"] == 270

    raw_rec = next(e for e in ds["experiments"] if e["experiment_id"] == "experiment_20261124_231000")
    assert raw_rec["validation_completed"] is True
    assert raw_rec["valid_trials"] == 8
    assert raw_rec["rejected_trials"] == 1
    # All 8 valid trials had small offsets inside boundary -> 100.0%
    assert raw_rec["target_selection_accuracy_pct"] == 100.0
    assert raw_rec["pct_within_2pct"] == 100.0


def test_tracker_deduplication_idempotency_and_6_sheets_structure(tmp_path: Path) -> None:
    """
    Generating/refreshing AABAR_Experiments_Tracker.xlsx multiple times produces zero duplicate rows,
    creates all 6 required sheets with exact columns, frozen panes, filters, formatting, and charts.
    """
    exp_root = tmp_path / "experiments"
    # Experiment 1: Completed full 9-region (7 hits, 1 miss, 1 rejected -> 7/8 = 87.5%)
    log1 = ExperimentLogger(base_dir=exp_root, experiment_id="experiment_20261124_230100")
    trials1 = _make_9_region_trials("experiment_20261124_230100", hit_indices=[0, 1, 2, 3, 4, 5, 6], reject_indices=[8])
    val1 = ScreenValidator(targets=FULL_9_REGION_VALIDATION_TARGETS, screen_width=1920, screen_height=1080, session_id="sess_a")
    val1.experiment_id = "experiment_20261124_230100"
    val1.mapper = LinearGazeMapper()
    val1._trial_sequence = list(FULL_9_REGION_VALIDATION_TARGETS)
    val1._trial_records = trials1
    sum1 = val1._finalize_validation()
    log1.log_experiment(
        calibration_samples=[],
        validation_results=trials1,
        accuracy_summary=sum1,
        metadata_extra={"screen_width": 1920, "screen_height": 1080, "validation_grid_mode": "full_9", "status": "COMPLETED"},
        export_excel=True,
        generate_plots=False,
    )

    # Experiment 2: Incomplete validation (partial 2 trials)
    log2 = ExperimentLogger(base_dir=exp_root, experiment_id="experiment_20261124_230200")
    trials2 = _make_9_region_trials("experiment_20261124_230200", hit_indices=[0])[:2]
    log2.log_experiment(
        calibration_samples=[],
        validation_results=trials2,
        accuracy_summary=None,
        metadata_extra={"screen_width": 1920, "screen_height": 1080, "validation_grid_mode": "full_9", "status": "INTERRUPTED_BY_USER"},
        export_excel=True,
        generate_plots=False,
    )

    # Experiment 3: Failed (all rejected trials)
    log3 = ExperimentLogger(base_dir=exp_root, experiment_id="experiment_20261124_230300")
    trials3 = _make_9_region_trials("experiment_20261124_230300", hit_indices=[], reject_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    log3.log_experiment(
        calibration_samples=[],
        validation_results=trials3,
        accuracy_summary=None,
        metadata_extra={"screen_width": 1920, "screen_height": 1080, "validation_grid_mode": "full_9", "status": "NO_VALID_TRIALS"},
        export_excel=True,
        generate_plots=False,
    )

    # Re-run update_central_tracker explicitly twice to prove zero duplicates
    update_central_tracker(experiments_dir=exp_root)
    res = update_central_tracker(experiments_dir=exp_root)
    tracker_path = res["tracker_path"]

    wb = openpyxl.load_workbook(tracker_path)
    assert wb.sheetnames == [
        "Experiments Summary",
        "All Test Trials",
        "Direction Accuracy",
        "Screen Region Accuracy",
        "Experiment Comparison",
        "Data Quality & Validation",
    ]

    # Sheet 1 verification
    ws1 = wb["Experiments Summary"]
    assert ws1.freeze_panes == "B4"
    assert ws1.auto_filter.ref is not None
    assert len(ws1._charts) == 1  # Target Selection Accuracy comparison chart!
    s1_ids = [ws1.cell(row=r, column=1).value for r in range(4, 7)]
    assert s1_ids == ["experiment_20261124_230100", "experiment_20261124_230200", "experiment_20261124_230300"]
    # Completed row accuracy fraction (87.5% -> 0.875) & incomplete/failed -> N/A
    assert ws1.cell(row=4, column=11).value == pytest.approx(0.875, abs=1e-4)
    assert ws1.cell(row=4, column=11).number_format == "0.0%"
    assert ws1.cell(row=4, column=18).value == "COMPLETED"
    assert ws1.cell(row=5, column=11).value == "N/A"
    assert ws1.cell(row=5, column=18).value == "INCOMPLETE"
    assert ws1.cell(row=6, column=11).value == "N/A"
    assert ws1.cell(row=6, column=18).value == "FAILED"

    # Sheet 2 verification (9 + 2 + 9 = 20 unique trials, zero duplicates)
    ws2 = wb["All Test Trials"]
    trial_keys = [(ws2.cell(row=r, column=1).value, ws2.cell(row=r, column=3).value) for r in range(4, ws2.max_row + 1)]
    assert len(trial_keys) == 20
    assert len(set(trial_keys)) == 20
    # Verify actual/predicted directions and boundary labels populated cleanly
    assert ws2.cell(row=4, column=10).value == "UP_LEFT"
    assert ws2.cell(row=4, column=16).value == "Yes"

    # Sheet 3 verification (3 experiments * 5 directions = 15 rows + direction chart)
    ws3 = wb["Direction Accuracy"]
    assert len(ws3._charts) == 1
    assert ws3.cell(row=4, column=2).value == "LEFT"
    assert ws3.cell(row=8, column=2).value == "DOWN"

    # Sheet 4 verification (all 9 calibration regions populated)
    ws4 = wb["Screen Region Accuracy"]
    regions_seen = [ws4.cell(row=r, column=2).value for r in range(4, 13)]
    assert regions_seen == list(CANONICAL_SCREEN_REGIONS_9)

    # Sheet 5 verification: only exp1 ranked (#1), exp2/3 explicitly excluded
    ws5 = wb["Experiment Comparison"]
    assert ws5.cell(row=4, column=1).value == 1
    assert ws5.cell(row=4, column=2).value == "experiment_20261124_230100"

    # Sheet 6 diagnostics verification
    ws6 = wb["Data Quality & Validation"]
    assert ws6.cell(row=4, column=9).value == "Yes"
    assert ws6.cell(row=5, column=9).value == "No"
    assert ws6.cell(row=6, column=9).value == "No"


def test_single_source_consistency_across_json_html_md_cli_excel_and_ui(tmp_path: Path, dummy_bgr_frame) -> None:
    """
    Verify exact numerical equality between ScreenValidator AccuracySummary, accuracy_report.json,
    HTML, Markdown, TXT, CLI summary output, AABAR_Experiments_Tracker.xlsx, and OpenCV HUD panel.
    """
    exp_root = tmp_path / "experiments"
    exp_id = "experiment_20261124_231500"
    logger = ExperimentLogger(base_dir=exp_root, experiment_id=exp_id)
    trials = _make_9_region_trials(exp_id, hit_indices=[0, 1, 2, 3, 4, 5], reject_indices=[])
    val = ScreenValidator(targets=FULL_9_REGION_VALIDATION_TARGETS, screen_width=1920, screen_height=1080)
    val.experiment_id = exp_id
    val.mapper = LinearGazeMapper()
    val._trial_sequence = list(FULL_9_REGION_VALIDATION_TARGETS)
    val._trial_records = trials
    summary = val._finalize_validation()

    created = logger.log_experiment(
        calibration_samples=[],
        validation_results=trials,
        accuracy_summary=summary,
        metadata_extra={"screen_width": 1920, "screen_height": 1080, "validation_grid_mode": "full_9", "status": "COMPLETED"},
        export_excel=True,
        generate_plots=False,
    )

    rep_json = json.loads((logger.experiment_dir / "accuracy_report.json").read_text(encoding="utf-8"))
    html_txt = (logger.experiment_dir / "accuracy_report.html").read_text(encoding="utf-8")
    md_txt = (logger.experiment_dir / "accuracy_report.md").read_text(encoding="utf-8")

    # Expect 6/9 hits = 66.7%
    expected_sel = round(summary.target_selection_accuracy_pct, 1)
    expected_p5 = round(summary.pct_within_5pct, 1)
    expected_mean = round(summary.mean_error_pixels, 1)

    assert round(rep_json["accuracy_rates"]["target_selection_accuracy_pct"], 1) == expected_sel
    assert f"{expected_sel:.1f}%" in html_txt
    assert f"{expected_p5:.1f}%" in html_txt
    assert f"{expected_mean:.1f} px" in html_txt
    assert f"{expected_sel:.1f}%" in md_txt

    # Tracker matches JSON identically
    ds = load_tracker_dataset(experiments_dir=exp_root)
    exp_rec = ds["experiments"][0]
    assert round(exp_rec["target_selection_accuracy_pct"], 1) == expected_sel
    assert round(exp_rec["pct_within_5pct"], 1) == expected_p5
    assert round(exp_rec["mean_error_px"], 1) == expected_mean

    wb = openpyxl.load_workbook(created["central_tracker_xlsx"])
    ws1 = wb["Experiments Summary"]
    assert round(ws1.cell(row=4, column=11).value * 100.0, 1) == expected_sel
    assert round(ws1.cell(row=4, column=15).value, 1) == expected_mean

    # CLI formatter matches identically
    cli_summary = format_experiment_results_summary(exp_rec, tracker_path=created["central_tracker_xlsx"])
    assert f"{expected_sel:.1f}%" in cli_summary
    assert f"{expected_mean:.1f} px" in cli_summary
    assert "COMPLETED" in cli_summary

    # Visualizer post-session results panel renders cleanly without crashes
    viz = CVVisualizer()
    lms = create_synthetic_landmarks()
    feats = extract_eye_features(lms)
    det = FaceLandmarkerDetectionResult(
        has_face=True, all_landmarks=lms, left_eye_landmarks=None, right_eye_landmarks=None,
        left_iris_landmarks=None, right_iris_landmarks=None, left_iris_center=None,
        right_iris_center=None, timestamp_ms=100, inference_time_ms=1.0,
    )
    fps = FPSMetrics(fps=30.0, frame_time_ms=33.3, detection_latency_ms=1.0, total_frames=1, uptime_seconds=0.1)
    sg = ScreenGazeOutput(gaze_x=0.5, gaze_y=0.5, pixel_x=960, pixel_y=540, is_calibrated=True, tracking_quality="EXCELLENT", confidence=0.95)
    rendered = viz.render(frame=dummy_bgr_frame, detection_result=det, eye_features=feats, fps_metrics=fps, screen_gaze=sg, screen_validator=val)
    assert rendered.shape == dummy_bgr_frame.shape


def test_automatic_tracker_update_after_new_experiment_and_direction_region_math(tmp_path: Path) -> None:
    """
    Logging a new experiment automatically appends/updates AABAR_Experiments_Tracker.xlsx
    while preserving prior runs and computing exact direction & 9-region accuracy cells.
    """
    exp_root = tmp_path / "experiments"

    # Run 1: Calibration only (NOT_STARTED)
    log1 = ExperimentLogger(base_dir=exp_root, experiment_id="experiment_20261124_232000")
    log1.log_experiment(
        calibration_samples=[],
        validation_results=[],
        accuracy_summary=None,
        metadata_extra={"screen_width": 1920, "screen_height": 1080, "status": "NOT_STARTED"},
        export_excel=True,
        generate_plots=False,
    )
    tracker_path = exp_root / TRACKER_FILENAME
    assert tracker_path.exists()
    wb1 = openpyxl.load_workbook(tracker_path)
    assert [wb1["Experiments Summary"].cell(row=4, column=1).value] == ["experiment_20261124_232000"]
    assert wb1["Experiments Summary"].cell(row=4, column=11).value == "N/A"

    # Run 2: Complete 9-region run hitting all LEFT/CENTER targets and missing RIGHT targets (targets 2, 5, 8)
    # FULL_9_REGION_VALIDATION_TARGETS indices:
    # 0:top_left(L,U), 1:top_center(U), 2:top_right(R,U),
    # 3:mid_left(L), 4:center(C), 5:mid_right(R),
    # 6:bot_left(L,D), 7:bot_center(D), 8:bot_right(R,D)
    log2 = ExperimentLogger(base_dir=exp_root, experiment_id="experiment_20261124_232100")
    trials2 = _make_9_region_trials("experiment_20261124_232100", hit_indices=[0, 1, 3, 4, 6, 7])
    val2 = ScreenValidator(targets=FULL_9_REGION_VALIDATION_TARGETS, screen_width=1920, screen_height=1080)
    val2.experiment_id = "experiment_20261124_232100"
    val2.mapper = LinearGazeMapper()
    val2._trial_sequence = list(FULL_9_REGION_VALIDATION_TARGETS)
    val2._trial_records = trials2
    sum2 = val2._finalize_validation()
    log2.log_experiment(
        calibration_samples=[],
        validation_results=trials2,
        accuracy_summary=sum2,
        metadata_extra={"screen_width": 1920, "screen_height": 1080, "validation_grid_mode": "full_9", "status": "COMPLETED"},
        export_excel=True,
        generate_plots=False,
    )

    wb2 = openpyxl.load_workbook(tracker_path)
    ws1 = wb2["Experiments Summary"]
    assert [ws1.cell(row=r, column=1).value for r in (4, 5)] == ["experiment_20261124_232000", "experiment_20261124_232100"]
    # 6/9 hits = 66.7% -> 0.6667
    assert ws1.cell(row=5, column=11).value == pytest.approx(6.0 / 9.0, abs=1e-3)

    # Verify Sheet 3 Direction Accuracy exact math for Run 2 (rows 9..13)
    ws3 = wb2["Direction Accuracy"]
    dir_rows = {ws3.cell(row=r, column=2).value: ws3.cell(row=r, column=6).value for r in range(9, 14)}
    assert dir_rows["LEFT"] == pytest.approx(1.0, abs=1e-4)       # 3/3 hits
    assert dir_rows["RIGHT"] == pytest.approx(0.0, abs=1e-4)      # 0/3 hits
    assert dir_rows["CENTER"] == pytest.approx(1.0, abs=1e-4)     # 1/1 hit
    assert dir_rows["UP"] == pytest.approx(2.0 / 3.0, abs=1e-3)   # 2/3 hits
    assert dir_rows["DOWN"] == pytest.approx(2.0 / 3.0, abs=1e-3) # 2/3 hits

    # Verify Sheet 4 9-Region Accuracy exact math for Run 2 (rows 13..21)
    ws4 = wb2["Screen Region Accuracy"]
    reg_rows = {ws4.cell(row=r, column=2).value: ws4.cell(row=r, column=6).value for r in range(13, 22)}
    assert reg_rows["Top Left"] == pytest.approx(1.0, abs=1e-4)
    assert reg_rows["Middle Right"] == pytest.approx(0.0, abs=1e-4)
    assert reg_rows["Center"] == pytest.approx(1.0, abs=1e-4)
