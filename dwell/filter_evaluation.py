"""
Quantitative Evaluation & Comparison of Gaze Smoothing Filters.

Compares:
1. PassThrough (None / Raw)
2. Saccade-Adaptive Exponential Moving Average (EMA)
3. One Euro Adaptive Filter (1-Euro)

Runs on identical deterministic synthetic sequences:
- Stationary Fixation with realistic sensor noise / tremor (sigma = 0.015, ~29 px)
- Step Saccade Jump (0.20 -> 0.70) to evaluate dynamic response latency

Measures:
- Jitter Standard Deviation (RMS error during fixation - lower is better)
- Peak-to-Peak Jitter Window (max - min excursion during fixation - lower is better)
- 90% Step Settling Latency (ms to reach 90% of target - lower is better)
- Overshoot / Ringing error (lower is better)
"""

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Tuple

from dwell.gaze_filter import (
    BaseGazeFilter,
    EMAGazeFilter,
    OneEuroGazeFilter,
    PassThroughGazeFilter,
)


@dataclass
class FilterBenchmarkMetrics:
    filter_name: str
    fixation_std_dev_px: float
    fixation_peak_to_peak_px: float
    saccade_latency_90_ms: float
    overshoot_px: float


def evaluate_filter_on_fixation(
    filt: BaseGazeFilter,
    ground_truth_x: float = 0.50,
    ground_truth_y: float = 0.50,
    noise_sigma: float = 0.015,
    sample_count: int = 150,
    dt_ms: float = 33.33,
    screen_width: int = 1920,
) -> Tuple[float, float]:
    """Measure RMS jitter (px) and peak-to-peak window (px) during stationary fixation."""
    filt.reset()
    rng = random.Random(1337)  # Fixed seed for identical inputs across all filters

    filtered_xs: List[float] = []
    t = 1000.0

    for _ in range(sample_count):
        raw_x = ground_truth_x + rng.gauss(0.0, noise_sigma)
        raw_y = ground_truth_y + rng.gauss(0.0, noise_sigma)
        fx, fy = filt.update(raw_x, raw_y, timestamp=t, is_valid=True)
        if fx is not None:
            filtered_xs.append(fx * screen_width)
        t += dt_ms / 1000.0

    # Discard initial 10 warmup frames
    warm_xs = filtered_xs[10:]
    mean_x = sum(warm_xs) / len(warm_xs)
    variance = sum((x - mean_x) ** 2 for x in warm_xs) / len(warm_xs)
    std_dev_px = math.sqrt(variance)
    ptp_px = max(warm_xs) - min(warm_xs)

    return std_dev_px, ptp_px


def evaluate_filter_on_saccade(
    filt: BaseGazeFilter,
    start_x: float = 0.20,
    end_x: float = 0.70,
    dt_ms: float = 33.33,
    screen_width: int = 1920,
) -> Tuple[float, float]:
    """Measure latency (ms to reach 90% of step change) and overshoot (px)."""
    filt.reset()
    target_90_x = start_x + 0.90 * (end_x - start_x)
    step_time_ms: Optional[float] = None
    max_val_x = end_x

    t = 1000.0
    # 30 frames at start_x
    for _ in range(30):
        filt.update(start_x, 0.50, timestamp=t, is_valid=True)
        t += dt_ms / 1000.0

    step_start_t = t
    reached_90 = False

    # 40 frames at end_x
    for _ in range(40):
        fx, fy = filt.update(end_x, 0.50, timestamp=t, is_valid=True)
        if fx is not None:
            max_val_x = max(max_val_x, fx)
            if not reached_90 and fx >= target_90_x:
                step_time_ms = (t - step_start_t) * 1000.0
                reached_90 = True
        t += dt_ms / 1000.0

    latency_ms = step_time_ms if step_time_ms is not None else 999.0
    overshoot_px = max(0.0, (max_val_x - end_x) * screen_width)
    return latency_ms, overshoot_px


def run_filter_comparison() -> List[FilterBenchmarkMetrics]:
    """Execute head-to-head comparison across all three filters."""
    filters = [
        PassThroughGazeFilter(),
        EMAGazeFilter(alpha=0.35, saccade_threshold=0.08, saccade_alpha=0.85),
        OneEuroGazeFilter(min_cutoff=1.2, beta=12.0, d_cutoff=1.0),
    ]

    results: List[FilterBenchmarkMetrics] = []
    for f in filters:
        std_px, ptp_px = evaluate_filter_on_fixation(f)
        lat_ms, over_px = evaluate_filter_on_saccade(f)
        results.append(
            FilterBenchmarkMetrics(
                filter_name=f.filter_type,
                fixation_std_dev_px=round(std_px, 2),
                fixation_peak_to_peak_px=round(ptp_px, 2),
                saccade_latency_90_ms=round(lat_ms, 1),
                overshoot_px=round(over_px, 2),
            )
        )
    return results


if __name__ == "__main__":
    metrics = run_filter_comparison()
    print("=" * 80)
    print(" GAZE FILTER QUANTITATIVE COMPARISON (1920x1080 DISPLAY)")
    print("=" * 80)
    print(f"{'Filter Algorithm':<15} | {'Fixation StdDev':<16} | {'Peak-to-Peak Window':<20} | {'90% Step Latency':<16} | {'Overshoot':<10}")
    print("-" * 80)
    for m in metrics:
        print(f"{m.filter_name:<15} | {m.fixation_std_dev_px:>13.2f} px | {m.fixation_peak_to_peak_px:>17.2f} px | {m.saccade_latency_90_ms:>13.1f} ms | {m.overshoot_px:>7.2f} px")
    print("=" * 80)
