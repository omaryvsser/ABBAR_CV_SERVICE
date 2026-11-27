"""
Deterministic Evaluation Suite for Phase 4 Dwell Selection Engine.

Runs 10 standardized synthetic scenarios using a controllable clock to evaluate:
1. Stable gaze inside one target
2. Gaze jitter near target boundaries
3. Short saccades outside a target
4. Long exits from a target
5. Blinks and temporary tracking loss
6. Switching between adjacent targets
7. Overlapping targets
8. Repeated selection attempts
9. NaN and infinite coordinates
10. Sudden gaze jumps / saccadic traversals

Measures:
- Correct selections
- False selections
- Missed selections
- Average selection time (ms)
- Target switching accuracy (%)
- Recovery after tracking loss (%)
"""

from dataclasses import dataclass, field
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from dwell.dwell_config import DwellConfig, DwellUserSkill
from dwell.dwell_engine import DwellEngine
from dwell.dwell_state import DwellState, SelectionEvent
from dwell.target_manager import DwellTarget, TargetManager


class DeterministicClock:
    """Controllable clock for instant, deterministic synthetic evaluation."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.time = start_time

    def __call__(self) -> float:
        return self.time

    def advance(self, dt_seconds: float) -> None:
        self.time += dt_seconds

    def advance_ms(self, dt_ms: float) -> None:
        self.time += dt_ms / 1000.0


@dataclass
class ScenarioResult:
    """Outcome of a single evaluation scenario."""
    scenario_name: str
    passed: bool
    correct_selections: int = 0
    false_selections: int = 0
    missed_selections: int = 0
    selection_time_ms: Optional[float] = None
    target_switching_correct: bool = True
    tracking_recovery_correct: bool = True
    details: str = ""


@dataclass
class BenchmarkReport:
    """Aggregated report across all scenarios."""
    engine_name: str
    scenario_results: List[ScenarioResult] = field(default_factory=list)
    total_scenarios: int = 0
    passed_scenarios: int = 0
    total_correct_selections: int = 0
    total_false_selections: int = 0
    total_missed_selections: int = 0
    avg_selection_time_ms: float = 0.0
    target_switching_accuracy_pct: float = 0.0
    tracking_recovery_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "total_scenarios": self.total_scenarios,
            "passed_scenarios": self.passed_scenarios,
            "pass_rate_pct": round((self.passed_scenarios / max(1, self.total_scenarios)) * 100.0, 1),
            "correct_selections": self.total_correct_selections,
            "false_selections": self.total_false_selections,
            "missed_selections": self.total_missed_selections,
            "avg_selection_time_ms": round(self.avg_selection_time_ms, 1),
            "target_switching_accuracy_pct": round(self.target_switching_accuracy_pct, 1),
            "tracking_recovery_pct": round(self.tracking_recovery_pct, 1),
            "scenarios": [
                {
                    "name": s.scenario_name,
                    "passed": s.passed,
                    "correct": s.correct_selections,
                    "false": s.false_selections,
                    "missed": s.missed_selections,
                    "time_ms": round(s.selection_time_ms, 1) if s.selection_time_ms else None,
                    "details": s.details,
                }
                for s in self.scenario_results
            ],
        }


def run_dwell_benchmark(
    config: Optional[DwellConfig] = None,
    engine_name: str = "Standard DwellEngine",
) -> BenchmarkReport:
    """
    Execute all 10 evaluation scenarios against a DwellEngine configured with `config`.
    """
    results: List[ScenarioResult] = []

    base_cfg = config or DwellConfig()

    def make_engine(clk: DeterministicClock, tm: Optional[TargetManager] = None) -> DwellEngine:
        if tm is None:
            tm = TargetManager()
            tm.add_target(DwellTarget("t1", "YES", 0.10, 0.10, 0.35, 0.35, priority=1))
            tm.add_target(DwellTarget("t2", "NO", 0.55, 0.10, 0.35, 0.35, priority=1))
        return DwellEngine(config=base_cfg, target_manager=tm, clock=clk)

    # --- Scenario 1: Stable Gaze Inside One Target ---
    clk = DeterministicClock(1000.0)
    eng = make_engine(clk)
    selections: List[SelectionEvent] = []
    eng.on_selection(lambda e: selections.append(e))

    # 1500 ms fixation at center of t1 (0.275, 0.275), 33ms steps (30 FPS)
    first_sel_time: Optional[float] = None
    for _ in range(45):
        clk.advance_ms(33.33)
        snap = eng.update(0.275, 0.275)
        if snap.selection_event and first_sel_time is None:
            first_sel_time = clk.time - 1000.0

    s1_passed = len(selections) == 1 and selections[0].target_id == "t1"
    results.append(
        ScenarioResult(
            scenario_name="1. Stable Gaze Inside One Target",
            passed=s1_passed,
            correct_selections=1 if s1_passed else 0,
            false_selections=max(0, len(selections) - 1),
            missed_selections=0 if s1_passed else 1,
            selection_time_ms=(first_sel_time * 1000.0) if first_sel_time else None,
            details=f"Selections: {len(selections)}, Time: {round(first_sel_time * 1000, 1) if first_sel_time else 'N/A'}ms",
        )
    )

    # --- Scenario 2: Gaze Jitter Near Target Boundaries ---
    # Target t1 boundary is x in [0.10, 0.45], y in [0.10, 0.45].
    # Jitter oscillates around right border x=0.45: x in [0.444, 0.456] (+/- 0.006, ~11 px on 1920 display)
    clk = DeterministicClock(1000.0)
    eng = make_engine(clk)
    selections.clear()
    eng.on_selection(lambda e: selections.append(e))

    # Fixation intended for 1400 ms with high-frequency boundary chatter (every 33ms oscillating in/out)
    rng = random.Random(42)
    s2_sel_time: Optional[float] = None
    for i in range(45):
        clk.advance_ms(33.33)
        jx = 0.446 if (i % 2 == 0) else 0.456
        jy = 0.275 + rng.uniform(-0.005, 0.005)
        snap = eng.update(jx, jy)
        if snap.selection_event and s2_sel_time is None:
            s2_sel_time = clk.time - 1000.0

    s2_passed = len(selections) == 1 and selections[0].target_id == "t1"
    results.append(
        ScenarioResult(
            scenario_name="2. Gaze Jitter Near Target Boundaries",
            passed=s2_passed,
            correct_selections=1 if s2_passed else 0,
            false_selections=max(0, len(selections) - 1),
            missed_selections=0 if s2_passed else 1,
            selection_time_ms=(s2_sel_time * 1000.0) if s2_sel_time else None,
            details=f"Selections: {len(selections)}, Boundary chatter handled: {s2_passed}",
        )
    )

    # --- Scenario 3: Short Saccades Outside Target (< tolerance) ---
    clk = DeterministicClock(1000.0)
    eng = make_engine(clk)
    selections.clear()
    eng.on_selection(lambda e: selections.append(e))

    # 600ms inside t1
    for _ in range(18):
        clk.advance_ms(33.33)
        eng.update(0.275, 0.275)

    # 100ms saccade excursion outside (at 0.02, 0.02) -> under 150ms tolerance
    for _ in range(3):
        clk.advance_ms(33.33)
        eng.update(0.02, 0.02)

    # 700ms back inside t1
    s3_sel_time: Optional[float] = None
    for _ in range(21):
        clk.advance_ms(33.33)
        snap = eng.update(0.275, 0.275)
        if snap.selection_event and s3_sel_time is None:
            s3_sel_time = clk.time - 1000.0

    s3_passed = len(selections) == 1 and selections[0].target_id == "t1"
    results.append(
        ScenarioResult(
            scenario_name="3. Short Saccades Outside Target (< tolerance)",
            passed=s3_passed,
            correct_selections=1 if s3_passed else 0,
            false_selections=max(0, len(selections) - 1),
            missed_selections=0 if s3_passed else 1,
            selection_time_ms=(s3_sel_time * 1000.0) if s3_sel_time else None,
            details=f"Selections: {len(selections)}, Short saccade survived: {s3_passed}",
        )
    )

    # --- Scenario 4: Long Exits from Target (> tolerance) ---
    clk = DeterministicClock(1000.0)
    eng = make_engine(clk)
    selections.clear()
    eng.on_selection(lambda e: selections.append(e))

    # 600ms inside t1
    for _ in range(18):
        clk.advance_ms(33.33)
        eng.update(0.275, 0.275)

    # 300ms outside (> 150ms tolerance) -> should abort dwell
    for _ in range(9):
        clk.advance_ms(33.33)
        eng.update(0.02, 0.02)

    # 400ms inside t1 -> total inside is 600+400=1000ms (< 1200ms fresh dwell)
    for _ in range(12):
        clk.advance_ms(33.33)
        eng.update(0.275, 0.275)

    s4_passed = len(selections) == 0 and eng.state == DwellState.DWELLING and eng.accumulated_dwell_ms < 600.0
    results.append(
        ScenarioResult(
            scenario_name="4. Long Exits from Target (> tolerance)",
            passed=s4_passed,
            correct_selections=0,
            false_selections=len(selections),
            missed_selections=0,
            selection_time_ms=None,
            details=f"Selections: {len(selections)} (expected 0), Reset observed: {s4_passed}",
        )
    )

    # --- Scenario 5: Blinks and Temporary Tracking Loss ---
    clk = DeterministicClock(1000.0)
    eng = make_engine(clk)
    selections.clear()
    eng.on_selection(lambda e: selections.append(e))

    # 600ms inside t1
    for _ in range(18):
        clk.advance_ms(33.33)
        eng.update(0.275, 0.275)

    # 150ms blink (None coordinates, < 250ms lost tracking tolerance)
    for _ in range(5):
        clk.advance_ms(30.0)
        eng.update(None, None)

    # 700ms back inside t1
    s5_sel_time: Optional[float] = None
    for _ in range(21):
        clk.advance_ms(33.33)
        snap = eng.update(0.275, 0.275)
        if snap.selection_event and s5_sel_time is None:
            s5_sel_time = clk.time - 1000.0

    s5_passed = len(selections) == 1 and selections[0].target_id == "t1"
    results.append(
        ScenarioResult(
            scenario_name="5. Blinks and Temporary Tracking Loss",
            passed=s5_passed,
            correct_selections=1 if s5_passed else 0,
            false_selections=max(0, len(selections) - 1),
            missed_selections=0 if s5_passed else 1,
            selection_time_ms=(s5_sel_time * 1000.0) if s5_sel_time else None,
            tracking_recovery_correct=s5_passed,
            details=f"Selections: {len(selections)}, Blink recovered: {s5_passed}",
        )
    )

    # --- Scenario 6: Switching Between Adjacent Targets ---
    # Target t1 at [0.10, 0.10, 0.35, 0.35], Target t2 at [0.46, 0.10, 0.35, 0.35] (tight 0.01 gap)
    clk = DeterministicClock(1000.0)
    tm6 = TargetManager()
    tm6.add_target(DwellTarget("t1", "YES", 0.10, 0.10, 0.35, 0.35, priority=1))
    tm6.add_target(DwellTarget("t2", "NO", 0.46, 0.10, 0.35, 0.35, priority=1))
    eng6 = make_engine(clk, tm6)
    selections.clear()
    eng6.on_selection(lambda e: selections.append(e))

    # Gaze dwells on t1 for 600ms (halfway)
    for _ in range(18):
        clk.advance_ms(33.33)
        eng6.update(0.25, 0.25)

    # Brief 1-frame jitter across boundary into t2 (0.47, 0.25) then back into t1
    clk.advance_ms(33.33)
    eng6.update(0.47, 0.25)

    # Returns to t1 and stays for 700ms (completing 1300ms total on t1)
    s6_sel_time: Optional[float] = None
    for _ in range(21):
        clk.advance_ms(33.33)
        snap = eng6.update(0.25, 0.25)
        if snap.selection_event and s6_sel_time is None:
            s6_sel_time = clk.time - 1000.0

    s6_passed = len(selections) == 1 and selections[0].target_id == "t1"
    results.append(
        ScenarioResult(
            scenario_name="6. Switching Between Adjacent Targets",
            passed=s6_passed,
            correct_selections=1 if s6_passed else 0,
            false_selections=max(0, len(selections) - 1),
            missed_selections=0 if s6_passed else 1,
            target_switching_correct=s6_passed,
            selection_time_ms=(s6_sel_time * 1000.0) if s6_sel_time else None,
            details=f"Selections: {len(selections)} (expected 1 on t1), 1-frame jitter survived: {s6_passed}",
        )
    )

    # --- Scenario 7: Overlapping Targets Resolution ---
    clk = DeterministicClock(1000.0)
    tm7 = TargetManager()
    tm7.add_target(DwellTarget("parent", "PARENT", 0.1, 0.1, 0.8, 0.8, priority=0))
    tm7.add_target(DwellTarget("child", "CHILD", 0.3, 0.3, 0.2, 0.2, priority=5))
    eng7 = make_engine(clk, tm7)
    selections.clear()
    eng7.on_selection(lambda e: selections.append(e))

    for _ in range(40):
        clk.advance_ms(33.33)
        eng7.update(0.35, 0.35)

    s7_passed = len(selections) == 1 and selections[0].target_id == "child"
    results.append(
        ScenarioResult(
            scenario_name="7. Overlapping Targets (Priority Resolution)",
            passed=s7_passed,
            correct_selections=1 if s7_passed else 0,
            false_selections=max(0, len(selections) - 1),
            missed_selections=0 if s7_passed else 1,
            details=f"Selected: {selections[0].target_id if selections else 'None'} (expected child)",
        )
    )

    # --- Scenario 8: Repeated Selection Attempts (Lockout & Retrigger) ---
    clk = DeterministicClock(1000.0)
    eng8 = make_engine(clk)
    selections.clear()
    eng8.on_selection(lambda e: selections.append(e))

    # 1. Trigger selection (1300ms)
    for _ in range(40):
        clk.advance_ms(33.33)
        eng8.update(0.275, 0.275)
    first_count = len(selections)

    # 2. Stay staring at same target during display (250ms) and cooldown (800ms) = 1050ms
    for _ in range(32):
        clk.advance_ms(33.33)
        eng8.update(0.275, 0.275)

    stare_count = len(selections)

    # 3. Look away to empty space for 200ms
    for _ in range(6):
        clk.advance_ms(33.33)
        eng8.update(0.02, 0.02)

    # 4. Return to target for 1300ms -> should trigger second selection
    for _ in range(40):
        clk.advance_ms(33.33)
        eng8.update(0.275, 0.275)
    final_count = len(selections)

    s8_passed = first_count == 1 and stare_count == 1 and final_count == 2
    results.append(
        ScenarioResult(
            scenario_name="8. Repeated Selection Attempts (Lockout & Retrigger)",
            passed=s8_passed,
            correct_selections=2 if s8_passed else final_count,
            false_selections=1 if stare_count > 1 else 0,
            missed_selections=1 if final_count < 2 else 0,
            details=f"First: {first_count}, During lockout stare: {stare_count}, After lookaway: {final_count}",
        )
    )

    # --- Scenario 9: NaN, Infinite, and Out-of-Bounds Coordinates ---
    clk = DeterministicClock(1000.0)
    eng9 = make_engine(clk)
    selections.clear()
    eng9.on_selection(lambda e: selections.append(e))

    abnormal_inputs = [
        (float("nan"), float("nan")),
        (float("inf"), 0.5),
        (0.5, float("-inf")),
        (-5.0, 0.5),
        (0.5, 10.0),
        (None, 0.5),
        (0.5, None),
    ]
    no_exception = True
    try:
        for gx, gy in abnormal_inputs:
            clk.advance_ms(33.33)
            snap = eng9.update(gx, gy)
            assert snap.state in (DwellState.IDLE, DwellState.LOST_TRACKING)
            assert not snap.is_tracking_valid
    except Exception as e:
        no_exception = False

    s9_passed = no_exception and len(selections) == 0
    results.append(
        ScenarioResult(
            scenario_name="9. NaN, Infinite, and Out-of-Bounds Coordinates",
            passed=s9_passed,
            correct_selections=0,
            false_selections=len(selections),
            missed_selections=0,
            tracking_recovery_correct=s9_passed,
            details=f"No crashes: {no_exception}, False triggers: {len(selections)}",
        )
    )

    # --- Scenario 10: Sudden Gaze Jumps / Saccadic Traversals ---
    # Fast saccade sweeping from (0.02, 0.275) to (0.85, 0.275) passing right through t1.
    # Total traversal time is 2 frames (60ms) moving at 4.15 screen units / sec (> 1.8 threshold).
    clk = DeterministicClock(1000.0)
    eng10 = make_engine(clk)
    selections.clear()
    eng10.on_selection(lambda e: selections.append(e))

    # Frame 1: empty space
    eng10.update(0.02, 0.275)

    # Fast sweep across t1 (which spans x=0.10..0.45)
    clk.advance_ms(30.0)
    snap_sweep1 = eng10.update(0.25, 0.275)  # inside t1, high speed saccade

    clk.advance_ms(30.0)
    snap_sweep2 = eng10.update(0.40, 0.275)  # still inside t1, high speed

    clk.advance_ms(30.0)
    snap_sweep3 = eng10.update(0.85, 0.275)  # out of t1

    accumulated_ms = eng10.accumulated_dwell_ms
    s10_passed = len(selections) == 0 and accumulated_ms == 0.0
    results.append(
        ScenarioResult(
            scenario_name="10. Sudden Gaze Jumps / Saccadic Traversals",
            passed=s10_passed,
            correct_selections=0,
            false_selections=len(selections),
            missed_selections=0,
            details=f"Accumulated dwell time on sweep: {round(accumulated_ms, 1)}ms (target: 0.0ms)",
        )
    )

    # Aggregate
    total_scenarios = len(results)
    passed_scenarios = sum(1 for r in results if r.passed)
    tot_correct = sum(r.correct_selections for r in results)
    tot_false = sum(r.false_selections for r in results)
    tot_missed = sum(r.missed_selections for r in results)

    valid_times = [r.selection_time_ms for r in results if r.selection_time_ms is not None]
    avg_time = (sum(valid_times) / len(valid_times)) if valid_times else 0.0

    switching_checks = [r.target_switching_correct for r in results if "Adjacent" in r.scenario_name]
    switch_acc = (sum(1 for c in switching_checks if c) / len(switching_checks) * 100.0) if switching_checks else 100.0

    recovery_checks = [r.tracking_recovery_correct for r in results if "Tracking" in r.scenario_name or "Blink" in r.scenario_name]
    rec_acc = (sum(1 for c in recovery_checks if c) / len(recovery_checks) * 100.0) if recovery_checks else 100.0

    return BenchmarkReport(
        engine_name=engine_name,
        scenario_results=results,
        total_scenarios=total_scenarios,
        passed_scenarios=passed_scenarios,
        total_correct_selections=tot_correct,
        total_false_selections=tot_false,
        total_missed_selections=tot_missed,
        avg_selection_time_ms=avg_time,
        target_switching_accuracy_pct=switch_acc,
        tracking_recovery_pct=rec_acc,
    )


def run_comparative_benchmark() -> Tuple[BenchmarkReport, BenchmarkReport]:
    """Execute baseline vs enhanced comparative benchmark."""
    # 1. Baseline: Raw coordinates, zero hysteresis, zero switch confirmation, no fixation filter
    baseline_cfg = DwellConfig(
        dwell_duration_ms=1200.0,
        cooldown_duration_ms=800.0,
        filter_type="none",
        hysteresis_enabled=False,
        exit_margin=0.0,
        switch_confirmation_ms=0.0,
        fixation_detection_enabled=False,
    )
    rep_baseline = run_dwell_benchmark(config=baseline_cfg, engine_name="Baseline (Raw / No Hysteresis)")

    # 2. Enhanced: One-Euro filter + Target Hysteresis (4% exit margin) + 80ms Switch Confirmation + Fixation Detection
    enhanced_cfg = DwellConfig(
        dwell_duration_ms=1200.0,
        cooldown_duration_ms=800.0,
        filter_type="one_euro",
        hysteresis_enabled=True,
        exit_margin=0.04,
        switch_confirmation_ms=80.0,
        fixation_detection_enabled=True,
        saccade_velocity_threshold=1.8,
    )
    rep_enhanced = run_dwell_benchmark(config=enhanced_cfg, engine_name="Enhanced (1-Euro + Hysteresis + Fixation)")

    return rep_baseline, rep_enhanced


if __name__ == "__main__":
    rep_base, rep_enh = run_comparative_benchmark()

    for report in (rep_base, rep_enh):
        print("=" * 72)
        print(f" BENCHMARK REPORT: {report.engine_name}")
        print("=" * 72)
        print(f" Total Scenarios:  {report.total_scenarios}")
        print(f" Passed Scenarios: {report.passed_scenarios}/{report.total_scenarios} ({round(report.passed_scenarios / report.total_scenarios * 100, 1)}%)")
        print(f" Correct Selects:  {report.total_correct_selections}")
        print(f" False Selects:    {report.total_false_selections}")
        print(f" Missed Selects:   {report.total_missed_selections}")
        print(f" Avg Select Time:  {round(report.avg_selection_time_ms, 1)} ms")
        print(f" Switch Accuracy:  {round(report.target_switching_accuracy_pct, 1)} %")
        print(f" Recovery Rate:    {round(report.tracking_recovery_pct, 1)} %")
        print("-" * 72)
        for r in report.scenario_results:
            status = "[PASS]" if r.passed else "[FAIL]"
            print(f" {status} {r.scenario_name}")
            print(f"        {r.details}")
        print("=" * 72)
        print()
