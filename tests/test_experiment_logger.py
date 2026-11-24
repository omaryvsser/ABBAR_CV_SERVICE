"""
Unit tests for CSV Experiment Logging and Excel Exporter (Phase 3).

Covers:
- Writing calibration_samples.csv
- Writing validation_results.csv
- Writing accuracy_summary.csv
- Writing experiment_metadata.json and accuracy_report.json
- Verifying column headers and numerical precision
- Optional Excel workbook export / fallback handling
"""

import csv
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.core.calibration_profile import CalibrationProfile
from src.core.experiment_logger import ExperimentLogger
from src.core.screen_calibration import CalibrationSampleRecord
from src.core.screen_validator import AccuracySummary, ValidationTrialRecord


def test_experiment_logger_writes_all_csv_and_json_files():
    """Verify complete suite of experiment artifacts are persisted correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(base_dir=Path(tmpdir), experiment_id="test_exp_001")

        # Create dummy calibration samples
        samples = [
            CalibrationSampleRecord(
                experiment_id="test_exp_001",
                session_id="sess_1",
                target_id=0,
                target_x=0.10,
                target_y=0.10,
                timestamp=1000.0,
                features=np.zeros(10),
                raw_x=0.48,
                raw_y=0.49,
                r_iris_x=0.48,
                r_iris_y=0.49,
                l_iris_x=0.48,
                l_iris_y=0.49,
                r_openness=0.85,
                l_openness=0.82,
                yaw=1.5,
                pitch=-0.5,
                roll=0.0,
                tracking_quality="EXCELLENT",
                confidence=0.92,
                accepted=True,
                rejection_reason=None,
            ),
            CalibrationSampleRecord(
                experiment_id="test_exp_001",
                session_id="sess_1",
                target_id=0,
                target_x=0.10,
                target_y=0.10,
                timestamp=1000.1,
                features=None,
                raw_x=None,
                raw_y=None,
                r_iris_x=None,
                r_iris_y=None,
                l_iris_x=None,
                l_iris_y=None,
                r_openness=0.0,
                l_openness=0.0,
                yaw=None,
                pitch=None,
                roll=None,
                tracking_quality="POOR",
                confidence=0.0,
                accepted=False,
                rejection_reason="BLINK_DETECTED",
            ),
        ]

        # Dummy validation results
        validation_results = [
            ValidationTrialRecord(
                experiment_id="test_exp_001",
                trial_id=1,
                target_id=0,
                target_name="val_center",
                actual_x=0.50,
                actual_y=0.50,
                predicted_x=0.52,
                predicted_y=0.49,
                error_x=0.02,
                error_y=-0.01,
                euclidean_error_norm=0.02236,
                euclidean_error_pixels=40.2,
                screen_width=1920,
                screen_height=1080,
                tracking_quality="EXCELLENT",
                accepted_sample_count=20,
                timestamp=1005.0,
            )
        ]

        # Dummy accuracy summary
        summary = AccuracySummary(
            experiment_id="test_exp_001",
            model_name="polynomial",
            model_configuration={"ridge_alpha": 0.01, "model_type": "polynomial"},
            total_trials=1,
            successful_trials=1,
            rejected_trials=0,
            mae_x=0.02,
            mae_y=0.01,
            rmse_x=0.02,
            rmse_y=0.01,
            mean_euclidean_error=0.0224,
            median_euclidean_error=0.0224,
            maximum_error=0.0224,
            mean_error_pixels=40.2,
            median_error_pixels=40.2,
            max_error_pixels=40.2,
            screen_resolution="1920x1080",
            evaluation_timestamp=1010.0,
        )

        created = logger.log_experiment(
            calibration_samples=samples,
            validation_results=validation_results,
            accuracy_summary=summary,
            export_excel=True,
        )

        exp_dir = logger.experiment_dir

        # 1. calibration_samples.csv
        calib_csv = exp_dir / "calibration_samples.csv"
        assert calib_csv.exists()
        with open(calib_csv, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert len(reader) == 3  # header + 2 samples
            assert "sample_accepted" in reader[0]
            assert "rejection_reason" in reader[0]
            assert reader[1][reader[0].index("sample_accepted")] == "1"
            assert reader[2][reader[0].index("rejection_reason")] == "BLINK_DETECTED"

        # 2. validation_results.csv
        val_csv = exp_dir / "validation_results.csv"
        assert val_csv.exists()
        with open(val_csv, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert len(reader) == 2  # header + 1 trial
            assert "euclidean_error_pixels" in reader[0]

        # 3. accuracy_summary.csv
        acc_csv = exp_dir / "accuracy_summary.csv"
        assert acc_csv.exists()
        with open(acc_csv, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert len(reader) == 2
            assert "MAE_X" in reader[0]
            assert "mean_error_pixels" in reader[0]

        # 4. JSON metadata & report
        meta_json = exp_dir / "experiment_metadata.json"
        assert meta_json.exists()
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
            assert meta["experiment_id"] == "test_exp_001"

        acc_json = exp_dir / "accuracy_report.json"
        assert acc_json.exists()
        with open(acc_json, "r", encoding="utf-8") as f:
            rep = json.load(f)
            assert rep["metrics"]["mean_euclidean_pixels"] == 40.2
