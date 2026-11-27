"""
AABAR CV Service - Dwell Target Management.

Provides rectangular target definitions, containment testing, enable/disable toggling,
deterministic overlap resolution policies, and sample layout generation.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class DwellTarget:
    """
    Selectable rectangular screen target.

    Supports normalized coordinates (x, y, width, height in [0.0, 1.0])
    or absolute screen pixel coordinates.
    """
    target_id: str
    label: str
    x: float
    y: float
    width: float
    height: float
    enabled: bool = True
    priority: int = 0
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate target boundaries and dimensions."""
        if not self.target_id or not isinstance(self.target_id, str):
            raise ValueError(f"target_id must be a non-empty string, got {self.target_id!r}")
        for attr_name, val in [("x", self.x), ("y", self.y), ("width", self.width), ("height", self.height)]:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"DwellTarget '{self.target_id}' {attr_name} must be a finite float, got {val}")
        if self.width <= 0:
            raise ValueError(f"DwellTarget '{self.target_id}' width must be > 0, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"DwellTarget '{self.target_id}' height must be > 0, got {self.height}")

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def contains(self, px: float, py: float) -> bool:
        """
        Check if point (px, py) lies inside the target's bounding box.
        Boundary edges are inclusive.
        """
        if px is None or py is None or not math.isfinite(px) or not math.isfinite(py):
            return False
        return (self.x <= px <= self.right) and (self.y <= py <= self.bottom)

    def contains_entry(self, px: float, py: float, margin: float = 0.0) -> bool:
        """
        Check if point (px, py) lies inside the entry boundary.
        A positive margin expands the entry region; negative insets it.
        """
        if px is None or py is None or not math.isfinite(px) or not math.isfinite(py):
            return False
        m = float(margin)
        return ((self.x - m) <= px <= (self.right + m)) and (
            (self.y - m) <= py <= (self.bottom + m)
        )

    def contains_exit(self, px: float, py: float, margin: float = 0.04) -> bool:
        """
        Check if point (px, py) lies inside the expanded exit boundary.
        Prevents jitter around target edges from prematurely dropping dwell.
        """
        if px is None or py is None or not math.isfinite(px) or not math.isfinite(py):
            return False
        m = max(0.0, float(margin))
        return ((self.x - m) <= px <= (self.right + m)) and (
            (self.y - m) <= py <= (self.bottom + m)
        )

    def get_exit_rect(self, margin: float = 0.04) -> Tuple[float, float, float, float]:
        """Return (x, y, width, height) of the expanded exit boundary."""
        m = max(0.0, float(margin))
        return (self.x - m, self.y - m, self.width + 2.0 * m, self.height + 2.0 * m)

    def to_rect(self) -> Tuple[float, float, float, float]:
        """Return (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    def to_dict(self) -> Dict[str, Any]:
        """Convert target to dictionary."""
        return {
            "target_id": self.target_id,
            "label": self.label,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
            "enabled": self.enabled,
            "priority": self.priority,
            "data": dict(self.data),
        }


class TargetManager:
    """
    Manages a collection of selectable targets, hit testing, and overlap arbitration.
    """

    def __init__(self, targets: Optional[List[DwellTarget]] = None) -> None:
        self._targets: Dict[str, DwellTarget] = {}
        self._order: List[str] = []
        if targets:
            for t in targets:
                self.add_target(t)

    def add_target(self, target: DwellTarget) -> None:
        """Add or update a target."""
        if not isinstance(target, DwellTarget):
            raise TypeError(f"Expected DwellTarget instance, got {type(target)}")
        tid = target.target_id
        if tid not in self._targets:
            self._order.append(tid)
        self._targets[tid] = target

    def remove_target(self, target_id: str) -> bool:
        """Remove a target by ID. Returns True if removed, False if not found."""
        if target_id in self._targets:
            del self._targets[target_id]
            if target_id in self._order:
                self._order.remove(target_id)
            return True
        return False

    def get_target(self, target_id: str) -> Optional[DwellTarget]:
        """Retrieve target by ID, or None if not found."""
        return self._targets.get(target_id)

    def set_target_enabled(self, target_id: str, enabled: bool) -> bool:
        """Enable or disable target by ID. Returns True if updated."""
        target = self.get_target(target_id)
        if target:
            target.enabled = bool(enabled)
            return True
        return False

    def clear(self) -> None:
        """Remove all targets."""
        self._targets.clear()
        self._order.clear()

    def all_targets(self) -> List[DwellTarget]:
        """Return all registered targets in definition order."""
        return [self._targets[tid] for tid in self._order if tid in self._targets]

    def enabled_targets(self) -> List[DwellTarget]:
        """Return all currently enabled targets."""
        return [t for t in self.all_targets() if t.enabled]

    def find_target_at(
        self,
        x: Optional[float],
        y: Optional[float],
        policy: Literal["highest_priority", "smallest_area", "first_defined"] = "highest_priority",
        current_target_id: Optional[str] = None,
        exit_margin: float = 0.0,
        entry_margin: float = 0.0,
    ) -> Optional[DwellTarget]:
        """
        Locate the target containing coordinate (x, y).

        Supports hysteresis:
            - If (x, y) enters an enabled target's entry boundary, that target is selected.
            - If (x, y) is outside all entry boundaries, but current_target_id is within
              its expanded exit boundary (exit_margin > 0), the current target retains ownership.
            - Resolves overlapping targets using the configured deterministic policy.

        Returns:
            The winning DwellTarget, or None if outside all targets or coordinates are invalid.
        """
        if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
            return None

        # 1. Check entry containment across all enabled targets
        candidates = [t for t in self.enabled_targets() if t.contains_entry(x, y, margin=entry_margin)]

        if not candidates:
            # 2. Hysteresis fallback: Check if current target holds gaze within exit margin
            if current_target_id and exit_margin > 0.0:
                cur = self.get_target(current_target_id)
                if cur and cur.enabled and cur.contains_exit(x, y, margin=exit_margin):
                    return cur
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Multi-target overlap resolution
        if policy == "smallest_area":
            candidates.sort(key=lambda t: (t.area, -t.priority, t.target_id))
        elif policy == "first_defined":
            # Preserve definition order
            pass
        else:  # default "highest_priority"
            candidates.sort(key=lambda t: (-t.priority, t.area, t.target_id))

        return candidates[0]

    @classmethod
    def create_sample_grid(
        cls,
        rows: int = 2,
        cols: int = 2,
        margin: float = 0.08,
        gap: float = 0.04,
        labels: Optional[List[str]] = None,
    ) -> "TargetManager":
        """
        Factory creating a sample normalized grid layout for testing and demo modes.

        Default creates 2x2 AAC quick-response buttons:
            [ YES ]  [ NO ]
            [ HELP ] [ REST ]
        """
        default_labels = ["YES", "NO", "HELP", "REST", "WATER", "MORE"]
        btn_labels = labels or default_labels[: rows * cols]

        usable_w = 1.0 - 2 * margin - (cols - 1) * gap
        usable_h = 1.0 - 2 * margin - (rows - 1) * gap
        w = usable_w / float(cols)
        h = usable_h / float(rows)

        manager = cls()
        idx = 0
        for r in range(rows):
            for c in range(cols):
                lbl = btn_labels[idx] if idx < len(btn_labels) else f"BTN_{idx+1}"
                tx = margin + c * (w + gap)
                ty = margin + r * (h + gap)
                target = DwellTarget(
                    target_id=f"target_{idx+1}",
                    label=lbl,
                    x=tx,
                    y=ty,
                    width=w,
                    height=h,
                    priority=0,
                )
                manager.add_target(target)
                idx += 1
        return manager
