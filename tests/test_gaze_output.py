"""Unit tests for Structured Gaze Tracking Output Schema."""
import json
import pytest
from src.core.gaze_output import (
    EyesOutput,
    GazeFrameResult,
    GazeOutput,
    HeadPoseOutput,
    SingleEyeOutput,
)


def test_gaze_output_serialization():
    """Verify structure and JSON serialization match project specifications."""
    left_eye = SingleEyeOutput(iris_x_ratio=0.51, iris_y_ratio=0.47, openness=0.92, valid=True)
    right_eye = SingleEyeOutput(iris_x_ratio=0.49, iris_y_ratio=0.50, openness=0.90, valid=True)
    eyes = EyesOutput(left=left_eye, right=right_eye)

    gaze = GazeOutput(
        raw_x=0.52,
        raw_y=0.48,
        smoothed_x=0.51,
        smoothed_y=0.47,
        direction="CENTER",
        confidence=0.88,
    )

    head_pose = HeadPoseOutput(yaw=-1.2, pitch=3.4, roll=0.5)

    result = GazeFrameResult(
        timestamp=1700000000.123,
        face_detected=True,
        gaze=gaze,
        eyes=eyes,
        head_pose=head_pose,
    )

    d = result.to_dict()
    assert d["face_detected"] is True
    assert d["gaze"]["direction"] == "CENTER"
    assert d["gaze"]["raw_x"] == 0.52
    assert d["gaze"]["smoothed_x"] == 0.51
    assert d["eyes"]["left"]["openness"] == 0.92
    assert d["head_pose"]["yaw"] == -1.2

    # Verify JSON serialization
    json_str = result.to_json(indent=2)
    parsed = json.loads(json_str)
    assert parsed["face_detected"] is True
    assert parsed["gaze"]["direction"] == "CENTER"
