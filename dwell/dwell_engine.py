"""
AABAR CV Service - Dwell Selection Engine.

Phase 4: Modular, production-ready dwell selection state machine.
Processes real-time gaze coordinates against registered selectable targets
and emits structured SelectionEvents with deterministic cooldown and noise filtering.
"""

from collections import deque
import math
import time
from typing import Callable, Deque, List, Optional, Tuple

from .dwell_config import DwellConfig
from .dwell_state import DwellSnapshot, DwellState, SelectionEvent
from .target_manager import DwellTarget, TargetManager


class DwellEngine:
    """
    Core Dwell Selection Engine.

    Decoupled from OpenCV rendering and video capture.
    Maintains a deterministic state machine:
        IDLE -> DWELLING -> SELECTED -> COOLDOWN -> IDLE
        (with safe transitions to/from LOST_TRACKING)
    """

    def __init__(
        self,
        config: Optional[DwellConfig] = None,
        target_manager: Optional[TargetManager] = None,
        clock: Optional[Callable[[], float]] = None,
        enabled: bool = True,
    ) -> None:
        self.config = config or DwellConfig()
        self.target_manager = target_manager or TargetManager()
        self._clock: Callable[[], float] = clock or time.time
        self.enabled = bool(enabled)

        # State machine variables
        self.state: DwellState = DwellState.IDLE
        self.current_target: Optional[DwellTarget] = None
        self.accumulated_dwell_ms: float = 0.0
        self.dwell_start_time: Optional[float] = None
        self.last_update_time: Optional[float] = None

        # Tracking loss & saccade grace timing
        self.last_valid_tracking_time: Optional[float] = None
        self.last_inside_target_time: Optional[float] = None

        # Post-selection refractory lockout
        self.selected_state_start_time: Optional[float] = None
        self.cooldown_start_time: Optional[float] = None
        self.selected_target_id: Optional[str] = None
        self.has_left_after_selection: bool = True

        # Gaze stability history
        self.last_gaze_point: Optional[Tuple[float, float]] = None
        self.recent_points: Deque[Tuple[float, float]] = deque(maxlen=5)

        # Event tracking
        self.last_emitted_event: Optional[SelectionEvent] = None
        self.total_selections_count: int = 0
        self._selection_callbacks: List[Callable[[SelectionEvent], None]] = []

    def on_selection(self, callback: Callable[[SelectionEvent], None]) -> None:
        """Register a callback observer to be notified on target selection."""
        self._selection_callbacks.append(callback)

    def reset(self) -> None:
        """Reset the dwell engine to a clean IDLE state."""
        self.state = DwellState.IDLE
        self.current_target = None
        self.accumulated_dwell_ms = 0.0
        self.dwell_start_time = None
        self.last_update_time = None
        self.last_valid_tracking_time = None
        self.last_inside_target_time = None
        self.selected_state_start_time = None
        self.cooldown_start_time = None
        self.selected_target_id = None
        self.has_left_after_selection = True
        self.last_gaze_point = None
        self.recent_points.clear()

    def update(
        self,
        gaze_x: Optional[float],
        gaze_y: Optional[float],
        timestamp: Optional[float] = None,
    ) -> DwellSnapshot:
        """
        Process the current gaze observation and advance the dwell state machine.

        Args:
            gaze_x: Normalized horizontal gaze position [0.0, 1.0], or pixel X.
            gaze_y: Normalized vertical gaze position [0.0, 1.0], or pixel Y.
            timestamp: Optional epoch float timestamp. If None, uses internal clock.

        Returns:
            DwellSnapshot containing current state, target, progress, and any selection event.
        """
        now = float(timestamp) if timestamp is not None else float(self._clock())

        # If engine is disabled, return clean idle snapshot
        if not self.enabled:
            return DwellSnapshot(
                state=DwellState.IDLE,
                current_target_id=None,
                current_target_label=None,
                dwell_time_ms=0.0,
                dwell_progress=0.0,
                remaining_time_ms=0.0,
                is_tracking_valid=False,
                gaze_x=None,
                gaze_y=None,
                selection_event=None,
            )

        # Compute delta time (ms)
        if self.last_update_time is None:
            dt_ms = 0.0
        else:
            dt_ms = max(0.0, (now - self.last_update_time) * 1000.0)
        self.last_update_time = now

        # 1. Validate coordinate inputs
        is_tracking_valid = self._validate_gaze_input(gaze_x, gaze_y)

        # 2. Tracking loss handling
        if is_tracking_valid:
            self.last_valid_tracking_time = now
            self.recent_points.append((float(gaze_x), float(gaze_y)))
            self.last_gaze_point = (float(gaze_x), float(gaze_y))
        else:
            # Handle invalid / missing tracking
            return self._handle_lost_tracking(now, dt_ms)

        assert gaze_x is not None and gaze_y is not None
        gx = float(gaze_x)
        gy = float(gaze_y)

        # 3. Locate target under gaze
        hit_target = self.target_manager.find_target_at(
            gx, gy, policy=self.config.target_selection_policy
        )

        # Track if user looked away from the previously selected target
        if self.selected_target_id is not None:
            if hit_target is None or hit_target.target_id != self.selected_target_id:
                self.has_left_after_selection = True

        selection_event: Optional[SelectionEvent] = None

        # 4. State Machine Transitions
        if self.state == DwellState.COOLDOWN:
            assert self.cooldown_start_time is not None
            cooldown_elapsed = (now - self.cooldown_start_time) * 1000.0
            if cooldown_elapsed >= self.config.cooldown_duration_ms:
                # Cooldown expired: ready for fresh selection
                self.cooldown_start_time = None
                self._transition_after_cooldown(hit_target, now, dt_ms)
            else:
                # Still in cooldown refractory period
                pass

        elif self.state == DwellState.SELECTED:
            assert self.selected_state_start_time is not None
            selected_elapsed = (now - self.selected_state_start_time) * 1000.0
            if selected_elapsed >= self.config.selection_display_duration_ms:
                # Transition from visual selected pulse into cooldown
                self.cooldown_start_time = self.selected_state_start_time + (
                    self.config.selection_display_duration_ms / 1000.0
                )
                cooldown_elapsed = (now - self.cooldown_start_time) * 1000.0
                if cooldown_elapsed >= self.config.cooldown_duration_ms:
                    # Cooldown also expired in this interval
                    self.cooldown_start_time = None
                    self._transition_after_cooldown(hit_target, now, dt_ms)
                else:
                    self.state = DwellState.COOLDOWN
                    self.accumulated_dwell_ms = 0.0

        elif self.state in (DwellState.IDLE, DwellState.LOST_TRACKING):
            if hit_target is not None:
                # Check repeat without leaving policy
                if (
                    hit_target.target_id == self.selected_target_id
                    and not self.has_left_after_selection
                    and not self.config.allow_repeat_without_leaving
                ):
                    # Must look away before re-dwelling on same target
                    self.state = DwellState.IDLE
                    self.current_target = None
                    self.accumulated_dwell_ms = 0.0
                else:
                    # Gaze entered target: start dwelling
                    self.state = DwellState.DWELLING
                    self.current_target = hit_target
                    self.dwell_start_time = now
                    self.accumulated_dwell_ms = 0.0
                    self.last_inside_target_time = now
            else:
                self.state = DwellState.IDLE
                self.current_target = None
                self.accumulated_dwell_ms = 0.0

        elif self.state == DwellState.DWELLING:
            assert self.current_target is not None

            if hit_target is not None and hit_target.target_id == self.current_target.target_id:
                # Gaze remains inside the active target: accumulate dwell
                self.last_inside_target_time = now
                self.accumulated_dwell_ms += dt_ms

                # Check if dwell threshold is reached
                if self.accumulated_dwell_ms >= self.config.dwell_duration_ms:
                    self.state = DwellState.SELECTED
                    self.selected_state_start_time = now
                    self.selected_target_id = hit_target.target_id
                    self.has_left_after_selection = False
                    self.total_selections_count += 1

                    # Emit exactly ONE SelectionEvent
                    selection_event = SelectionEvent(
                        target_id=hit_target.target_id,
                        target_label=hit_target.label,
                        timestamp=now,
                        gaze_x=gx,
                        gaze_y=gy,
                        dwell_duration_ms=self.accumulated_dwell_ms,
                        selection_state="CONFIRMED",
                        metadata={"priority": hit_target.priority, **hit_target.data},
                    )
                    self.last_emitted_event = selection_event

                    for cb in self._selection_callbacks:
                        try:
                            cb(selection_event)
                        except Exception as e:
                            logger.error(f"Error in dwell selection callback: {e}")

            elif hit_target is not None and hit_target.target_id != self.current_target.target_id:
                # Gaze shifted to a DIFFERENT target: clean switch without lingering
                self.current_target = hit_target
                self.dwell_start_time = now
                self.accumulated_dwell_ms = 0.0
                self.last_inside_target_time = now
                self.state = DwellState.DWELLING

            else:
                # Gaze is outside all targets: check leave_target_tolerance grace window
                outside_duration = (
                    (now - self.last_inside_target_time) * 1000.0
                    if self.last_inside_target_time is not None
                    else 999999.0
                )
                if outside_duration > self.config.leave_target_tolerance_ms:
                    # Gaze genuinely exited target: reset
                    self.state = DwellState.IDLE
                    self.current_target = None
                    self.accumulated_dwell_ms = 0.0
                else:
                    # Saccade / micro-jitter grace: do not accumulate dwell time, but hold progress
                    pass

        # 5. Build telemetry snapshot
        return self._build_snapshot(gx, gy, is_tracking_valid=True, selection_event=selection_event, now=now)

    def _validate_gaze_input(self, gaze_x: Optional[float], gaze_y: Optional[float]) -> bool:
        """Verify gaze coordinates are finite numbers within acceptable bounds."""
        if gaze_x is None or gaze_y is None:
            return False
        if not math.isfinite(gaze_x) or not math.isfinite(gaze_y):
            return False
        if self.config.coordinate_space == "normalized":
            # Allow reasonable bounding margin around [0.0, 1.0] for calibration overshoot
            if gaze_x < -0.5 or gaze_x > 1.5 or gaze_y < -0.5 or gaze_y > 1.5:
                return False
        return True

    def _handle_lost_tracking(self, now: float, dt_ms: float) -> DwellSnapshot:
        """Handle lost, invalid, or NaN/Inf gaze coordinates."""
        if self.state == DwellState.DWELLING and self.config.pause_on_lost_tracking:
            lost_duration = (
                (now - self.last_valid_tracking_time) * 1000.0
                if self.last_valid_tracking_time is not None
                else 999999.0
            )
            if lost_duration > self.config.lost_tracking_tolerance_ms:
                # Exceeded tracking loss tolerance: abort dwell
                self.state = DwellState.LOST_TRACKING
                self.current_target = None
                self.accumulated_dwell_ms = 0.0
            else:
                # Within tolerance: hold dwell progress frozen
                pass
        elif self.state == DwellState.COOLDOWN:
            # Cooldown continues running even if tracking is momentarily lost
            assert self.cooldown_start_time is not None
            cooldown_elapsed = (now - self.cooldown_start_time) * 1000.0
            if cooldown_elapsed >= self.config.cooldown_duration_ms:
                self.cooldown_start_time = None
                self.state = DwellState.LOST_TRACKING
                self.current_target = None
                self.accumulated_dwell_ms = 0.0
        elif self.state == DwellState.SELECTED:
            # Transition to cooldown
            self.state = DwellState.COOLDOWN
            self.cooldown_start_time = now
            self.accumulated_dwell_ms = 0.0
        else:
            self.state = DwellState.LOST_TRACKING
            self.current_target = None
            self.accumulated_dwell_ms = 0.0

        return self._build_snapshot(None, None, is_tracking_valid=False, selection_event=None, now=now)

    def _transition_after_cooldown(
        self,
        hit_target: Optional[DwellTarget],
        now: float,
        dt_ms: float,
    ) -> None:
        """Decide next state after cooldown period expires."""
        if hit_target is not None:
            if (
                hit_target.target_id == self.selected_target_id
                and not self.has_left_after_selection
                and not self.config.allow_repeat_without_leaving
            ):
                # Still staring at same selected button: enter IDLE until user looks away
                self.state = DwellState.IDLE
                self.current_target = None
                self.accumulated_dwell_ms = 0.0
            else:
                # Gaze is on a target: start fresh dwell
                self.state = DwellState.DWELLING
                self.current_target = hit_target
                self.dwell_start_time = now
                self.accumulated_dwell_ms = 0.0
                self.last_inside_target_time = now
        else:
            self.state = DwellState.IDLE
            self.current_target = None
            self.accumulated_dwell_ms = 0.0

    def _build_snapshot(
        self,
        gx: Optional[float],
        gy: Optional[float],
        is_tracking_valid: bool,
        selection_event: Optional[SelectionEvent],
        now: float,
    ) -> DwellSnapshot:
        """Construct the live state snapshot."""
        # Progress calculation
        if self.state == DwellState.SELECTED:
            progress = 1.0
            remaining_ms = 0.0
        elif self.state == DwellState.DWELLING and self.config.dwell_duration_ms > 0:
            progress = min(1.0, max(0.0, self.accumulated_dwell_ms / self.config.dwell_duration_ms))
            remaining_ms = max(0.0, self.config.dwell_duration_ms - self.accumulated_dwell_ms)
        else:
            progress = 0.0
            remaining_ms = 0.0

        # Cooldown progress
        if self.state == DwellState.COOLDOWN and self.cooldown_start_time is not None:
            cooldown_elapsed = (now - self.cooldown_start_time) * 1000.0
            cd_progress = min(1.0, max(0.0, cooldown_elapsed / max(1.0, self.config.cooldown_duration_ms)))
        else:
            cd_progress = 0.0

        tid = self.current_target.target_id if self.current_target else None
        label = self.current_target.label if self.current_target else None

        return DwellSnapshot(
            state=self.state,
            current_target_id=tid,
            current_target_label=label,
            dwell_time_ms=self.accumulated_dwell_ms,
            dwell_progress=progress,
            remaining_time_ms=remaining_ms,
            is_tracking_valid=is_tracking_valid,
            gaze_x=gx,
            gaze_y=gy,
            selection_event=selection_event,
            cooldown_progress=cd_progress,
        )
