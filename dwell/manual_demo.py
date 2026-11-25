#!/usr/bin/env python3
"""
AABAR - Computer Vision Service | Phase 4 Dwell Selection Engine Demo.

Provides an interactive testing tool for manual evaluation of the Dwell Selection
State Machine. Supports both:
  1. Mouse-driven gaze simulation (--mouse, default): hover over AAC buttons to dwell and trigger selections.
  2. Live webcam gaze tracking (--camera): use eye tracking from calibrated/uncalibrated gaze pipeline.

Usage:
    python -m dwell.manual_demo --mouse
    python -m dwell.manual_demo --camera
    python -m dwell.manual_demo --test  # Headless self-check
"""

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

# Ensure cv-service root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure MPLCONFIGDIR is set to writable directory
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_aabar"

from dwell.dwell_config import DwellConfig
from dwell.dwell_engine import DwellEngine
from dwell.dwell_state import DwellSnapshot, DwellState, SelectionEvent
from dwell.target_manager import DwellTarget, TargetManager
from src.config import load_config
from src.visualization.visualizer import CVVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AABAR-DwellDemo")


class MouseGazeTracker:
    """Tracks mouse position and translates window pixels to normalized [0, 1] gaze coordinates."""

    def __init__(self, visualizer: CVVisualizer, width: int = 1280, height: int = 720) -> None:
        self.visualizer = visualizer
        self.width = width
        self.height = height
        self.mouse_x: int = width // 2
        self.mouse_y: int = height // 2
        self.is_tracking_lost: bool = False
        self.is_active: bool = True

    def on_mouse_event(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """OpenCV mouse callback."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click toggles simulated tracking loss
            self.is_tracking_lost = not self.is_tracking_lost
            logger.info(f"Simulated tracking loss toggled: {self.is_tracking_lost}")

    def get_normalized_gaze(self) -> tuple[Optional[float], Optional[float]]:
        """Compute normalized gaze coordinates based on current mouse position."""
        if self.is_tracking_lost:
            return None, None

        margin_x, top_y, bottom_y = self.visualizer.get_content_bounds(self.width, self.height)
        usable_w = max(1, self.width - 2 * margin_x)
        usable_h = max(1, bottom_y - top_y)

        norm_x = (self.mouse_x - margin_x) / float(usable_w)
        norm_y = (self.mouse_y - top_y) / float(usable_h)

        # Allow slight overshoot for boundary testing, clamp to [-0.1, 1.1]
        norm_x = max(-0.1, min(1.1, norm_x))
        norm_y = max(-0.1, min(1.1, norm_y))
        return norm_x, norm_y


def run_mouse_demo(
    dwell_ms: int = 1200,
    cooldown_ms: int = 800,
    width: int = 1280,
    height: int = 720,
) -> int:
    """Run interactive manual dwell selection demo using mouse as gaze source."""
    print("\n" + "=" * 70)
    print(" AABAR CV — Phase 4 Dwell Selection Engine [Manual Mouse Demo]")
    print("=" * 70)
    print(f" Dwell Duration:     {dwell_ms} ms")
    print(f" Cooldown Duration:  {cooldown_ms} ms")
    print(f" Window Resolution:  {width}x{height}")
    print("\n Instructions:")
    print("  - Hover the mouse cursor over any AAC button to initiate dwelling.")
    print("  - Maintain position inside the card for 1200 ms to trigger a SELECTION.")
    print("  - Notice the green selection flash and refractory COOLDOWN period.")
    print("  - Leave the card to reset or hover another card to dwell immediately.")
    print("\n Keyboard Controls:")
    print("  '1' - '4' : Toggle Buttons (1:YES, 2:NO, 3:HELP, 4:REST) Enabled/Disabled")
    print("  'r' / 'R' : Manually reset dwell state to IDLE")
    print("  't' / 'T' : Toggle simulated tracking loss (or Right-Click)")
    print("  'd' / 'D' : Toggle Dwell Engine Enabled/Disabled")
    print("  '+' / '-' : Adjust dwell duration (+/- 200 ms)")
    print("  'q' / ESC : Exit Demo")
    print("=" * 70 + "\n")

    cfg = load_config()
    cfg.visualization.enabled = True
    visualizer = CVVisualizer(config=cfg.visualization)

    target_manager = TargetManager.create_sample_grid(rows=2, cols=2)
    dwell_config = DwellConfig(
        dwell_duration_ms=dwell_ms,
        cooldown_duration_ms=cooldown_ms,
        stability_filter_enabled=True,
        pause_on_lost_tracking=True,
        allow_repeat_without_leaving=False,
    )
    engine = DwellEngine(config=dwell_config, target_manager=target_manager)

    selection_history: List[SelectionEvent] = []

    def handle_selection(evt: SelectionEvent) -> None:
        selection_history.append(evt)
        print(
            f" [SELECTION #{len(selection_history)}] >>> '{evt.target_label}' ({evt.target_id}) <<< "
            f"at ({evt.gaze_x:.3f}, {evt.gaze_y:.3f}) | Duration: {evt.dwell_duration_ms}ms"
        )

    engine.on_selection(handle_selection)

    mouse_tracker = MouseGazeTracker(visualizer, width=width, height=height)
    win_name = "AABAR - Dwell Selection Manual Demo (Mouse Mode)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, width, height)
    cv2.setMouseCallback(win_name, mouse_tracker.on_mouse_event)

    last_time = time.time()

    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now

            gx, gy = mouse_tracker.get_normalized_gaze()
            snapshot = engine.update(gx, gy)

            # Create clean background canvas (deep dark slate)
            canvas = np.full((height, width, 3), (22, 24, 28), dtype=np.uint8)

            # Draw visualizer dwell overlay (cards, progress bars, reticle)
            visualizer._draw_dwell_overlay(canvas, snapshot, engine, width, height)

            # Draw header banner
            cv2.rectangle(canvas, (0, 0), (width, 50), (16, 18, 22), -1)
            cv2.line(canvas, (0, 50), (width, 50), (45, 52, 62), 1)

            title = "AABAR AAC Dwell Selection Demo [Mouse Simulation]"
            cv2.putText(canvas, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)

            engine_st = "ENABLED" if engine.enabled else "DISABLED"
            st_color = (0, 255, 120) if engine.enabled else (0, 60, 255)
            cv2.putText(canvas, f"Engine: {engine_st}", (width - 200, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, st_color, 1, cv2.LINE_AA)

            # Draw bottom telemetry & controls HUD
            cv2.rectangle(canvas, (0, height - 70), (width, height), (16, 18, 22), -1)
            cv2.line(canvas, (0, height - 70), (width, height - 70), (45, 52, 62), 1)

            st_name = snapshot.state.name
            tgt_name = snapshot.target_label or "NONE"
            prog_pct = int(snapshot.progress * 100)
            g_str = f"({gx:.2f}, {gy:.2f})" if gx is not None and gy is not None else "LOST"

            hud_line1 = (
                f"State: {st_name:<11} | Target: {tgt_name:<6} | Dwell: {snapshot.dwell_time_ms:4d}/{dwell_config.dwell_duration_ms}ms ({prog_pct}%) | "
                f"Gaze: {g_str} | Selections: {len(selection_history)}"
            )
            hud_line2 = "Keys: '1'-'4'=Toggle Cards | 'r'=Reset | 't'=Sim Lost Track | 'd'=Toggle Dwell | '+/-'=Dwell Time | 'q'=Quit"

            cv2.putText(canvas, hud_line1, (20, height - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 235, 240), 1, cv2.LINE_AA)
            cv2.putText(canvas, hud_line2, (20, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 165, 180), 1, cv2.LINE_AA)

            # Draw recent selections badge in upper right
            if selection_history:
                last_sel = selection_history[-1]
                badge_txt = f"Last Selected: {last_sel.target_label} ({int((time.time() - last_sel.timestamp))}s ago)"
                cv2.putText(canvas, badge_txt, (width - 450, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 1, cv2.LINE_AA)

            cv2.imshow(win_name, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("r"), ord("R")):
                engine.reset()
                logger.info("Dwell Engine manually reset.")
            elif key in (ord("d"), ord("D")):
                engine.enabled = not engine.enabled
                engine.reset()
                logger.info(f"Dwell Engine toggled: {'ENABLED' if engine.enabled else 'DISABLED'}")
            elif key in (ord("t"), ord("T")):
                mouse_tracker.is_tracking_lost = not mouse_tracker.is_tracking_lost
                logger.info(f"Tracking loss simulation set to: {mouse_tracker.is_tracking_lost}")
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                tid = f"target_{chr(key)}"
                t = target_manager.get_target(tid)
                if t:
                    t.enabled = not t.enabled
                    logger.info(f"Target '{t.label}' ({tid}) toggled enabled: {t.enabled}")
            elif key in (ord("+"), ord("=")):
                dwell_config.dwell_duration_ms = min(3000, dwell_config.dwell_duration_ms + 200)
                logger.info(f"Dwell duration adjusted: {dwell_config.dwell_duration_ms} ms")
            elif key in (ord("-"), ord("_")):
                dwell_config.dwell_duration_ms = max(400, dwell_config.dwell_duration_ms - 200)
                logger.info(f"Dwell duration adjusted: {dwell_config.dwell_duration_ms} ms")

    finally:
        cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print(f" Demo Finished. Total Selections Captured: {len(selection_history)}")
    for i, evt in enumerate(selection_history, 1):
        print(f"  {i}. Target: {evt.target_label:<6} ({evt.target_id}) | Gaze: ({evt.gaze_x:.3f}, {evt.gaze_y:.3f}) | Time: {evt.dwell_duration_ms}ms")
    print("=" * 70 + "\n")
    return 0


def run_camera_demo(dwell_ms: int = 1200, cooldown_ms: int = 800) -> int:
    """Run dwell selection with physical or mock camera via main CV pipeline."""
    from main import main as cv_main
    sys.argv = [
        "main.py",
        "--dwell",
        "--dwell-demo",
        "--dwell-time", str(dwell_ms),
        "--cooldown-time", str(cooldown_ms),
    ]
    return cv_main()


def run_headless_self_test() -> int:
    """Headless self-verification of dwell demo logic."""
    logger.info("Running headless self-verification test of Dwell Demo...")
    tm = TargetManager.create_sample_grid(rows=2, cols=2)
    cfg = DwellConfig(dwell_duration_ms=1000, cooldown_duration_ms=500)
    curr_time = 100.0

    def mock_clock() -> float:
        return curr_time

    engine = DwellEngine(config=cfg, target_manager=tm, clock=mock_clock)
    selections = []
    engine.on_selection(lambda e: selections.append(e))

    # Gaze into target_1 (YES: x=0.08..0.48, y=0.08..0.48)
    curr_time = 100.0
    s1 = engine.update(0.20, 0.20)
    assert s1.state == DwellState.DWELLING
    assert s1.target_id == "target_1"

    curr_time = 101.05
    s2 = engine.update(0.20, 0.20)
    assert s2.state == DwellState.SELECTED
    assert len(selections) == 1
    assert selections[0].target_label == "YES"

    logger.info("Self-verification successful: 1 selection registered cleanly.")
    return 0


def main() -> int:
    """Entry point for manual demo script."""
    parser = argparse.ArgumentParser(
        description="AABAR CV - Phase 4 Dwell Selection Interactive Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        default=True,
        help="Use mouse cursor as interactive gaze simulator",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Use live camera gaze tracking through full CV pipeline",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run headless self-verification check and exit",
    )
    parser.add_argument(
        "--dwell-time",
        type=int,
        default=1200,
        help="Dwell duration in milliseconds before selection triggers",
    )
    parser.add_argument(
        "--cooldown-time",
        type=int,
        default=800,
        help="Post-selection refractory cooldown in milliseconds",
    )
    args = parser.parse_args()

    if args.test:
        return run_headless_self_test()

    if args.camera:
        return run_camera_demo(dwell_ms=args.dwell_time, cooldown_ms=args.cooldown_time)

    return run_mouse_demo(dwell_ms=args.dwell_time, cooldown_ms=args.cooldown_time)


if __name__ == "__main__":
    sys.exit(main())
