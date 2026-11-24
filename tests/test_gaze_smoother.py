"""Unit tests for Saccade-Adaptive Gaze Smoothing."""
import pytest
from src.core.gaze_smoother import GazeSmoother


def test_first_observation_initialization():
    """Verify first frame directly sets smoothed gaze without delay."""
    smoother = GazeSmoother(alpha=0.25)
    sx, sy = smoother.update(0.52, 0.48, is_valid=True)
    assert sx == pytest.approx(0.52)
    assert sy == pytest.approx(0.48)


def test_steady_state_ema_smoothing():
    """Verify mathematical correctness of Exponential Moving Average."""
    alpha = 0.30
    smoother = GazeSmoother(alpha=alpha, saccade_threshold=0.50)  # High threshold to prevent saccade trigger

    # Initial frame
    smoother.update(0.50, 0.50, is_valid=True)

    # Frame 2: input 0.55, expected: 0.30 * 0.55 + 0.70 * 0.50 = 0.515
    sx, sy = smoother.update(0.55, 0.55, is_valid=True)
    assert sx == pytest.approx(0.515)
    assert sy == pytest.approx(0.515)


def test_saccade_acceleration():
    """Verify that a large saccadic jump accelerates alpha for fast response."""
    smoother = GazeSmoother(
        alpha=0.20,
        saccade_threshold=0.15,
        saccade_alpha=0.80,
    )
    smoother.update(0.50, 0.50, is_valid=True)

    # Sudden jump from 0.50 to 0.80 (delta = 0.30 > 0.15)
    # Expected: 0.80 * 0.80 + 0.20 * 0.50 = 0.64 + 0.10 = 0.74
    sx, _ = smoother.update(0.80, 0.50, is_valid=True)
    assert sx == pytest.approx(0.74)


def test_invalid_frame_blink_holding():
    """Verify that brief invalid frames (blinks) hold previous smoothed coordinates."""
    smoother = GazeSmoother(max_invalid_frames=3)
    smoother.update(0.52, 0.48, is_valid=True)

    # Frame 1 of blink: should hold (0.52, 0.48)
    sx1, sy1 = smoother.update(None, None, is_valid=False)
    assert sx1 == pytest.approx(0.52)
    assert sy1 == pytest.approx(0.48)

    # Frame 2 of blink
    sx2, sy2 = smoother.update(0.20, 0.20, is_valid=False)
    assert sx2 == pytest.approx(0.52)
    assert sy2 == pytest.approx(0.48)

    # Recovery on frame 3: valid input again
    sx3, _ = smoother.update(0.53, 0.48, is_valid=True)
    assert sx3 is not None


def test_exceeding_max_invalid_frames_resets():
    """Verify smoother resets to None when invalid frames exceed tolerance."""
    smoother = GazeSmoother(max_invalid_frames=2)
    smoother.update(0.50, 0.50, is_valid=True)

    smoother.update(None, None, is_valid=False)  # Frame 1: hold
    smoother.update(None, None, is_valid=False)  # Frame 2: hold
    sx3, sy3 = smoother.update(None, None, is_valid=False)  # Frame 3: exceeds 2 -> reset

    assert sx3 is None
    assert sy3 is None
    assert smoother.smoothed_x is None


def test_manual_reset():
    """Verify manual reset clears state."""
    smoother = GazeSmoother()
    smoother.update(0.55, 0.45, is_valid=True)
    assert smoother.smoothed_x is not None

    smoother.reset()
    assert smoother.smoothed_x is None
    assert smoother.smoothed_y is None
