"""
Tests for Phase 4 TargetManager and DwellTarget.
"""

import math
import pytest
from dwell.target_manager import DwellTarget, TargetManager


def test_target_creation_and_bounds() -> None:
    """Verify target boundary calculations and properties."""
    target = DwellTarget(
        target_id="btn_yes",
        label="YES",
        x=0.10,
        y=0.20,
        width=0.30,
        height=0.40,
        priority=2,
    )
    assert target.right == pytest.approx(0.40)
    assert target.bottom == pytest.approx(0.60)
    assert target.area == pytest.approx(0.12)
    assert target.center == pytest.approx((0.25, 0.40))
    assert target.enabled is True
    assert target.priority == 2


def test_target_validation_errors() -> None:
    """Verify target rejects invalid dimensions, non-finite values, and empty IDs."""
    with pytest.raises(ValueError, match="width must be > 0"):
        DwellTarget(target_id="bad", label="B", x=0.1, y=0.1, width=0.0, height=0.2)

    with pytest.raises(ValueError, match="height must be > 0"):
        DwellTarget(target_id="bad", label="B", x=0.1, y=0.1, width=0.2, height=-0.1)

    with pytest.raises(ValueError, match="finite float"):
        DwellTarget(target_id="bad", label="B", x=float("nan"), y=0.1, width=0.2, height=0.2)

    with pytest.raises(ValueError, match="finite float"):
        DwellTarget(target_id="bad", label="B", x=0.1, y=float("inf"), width=0.2, height=0.2)

    with pytest.raises(ValueError, match="non-empty string"):
        DwellTarget(target_id="", label="B", x=0.1, y=0.1, width=0.2, height=0.2)


def test_target_containment_and_edges() -> None:
    """Verify inclusive bounding box containment testing."""
    target = DwellTarget(target_id="t1", label="T1", x=0.2, y=0.3, width=0.4, height=0.3)
    # Center
    assert target.contains(0.4, 0.45) is True
    # Corners (inclusive)
    assert target.contains(0.2, 0.3) is True
    assert target.contains(0.6, 0.6) is True
    # Outside
    assert target.contains(0.19, 0.45) is False
    assert target.contains(0.61, 0.45) is False
    assert target.contains(0.4, 0.29) is False
    assert target.contains(0.4, 0.61) is False
    # Non-finite / None coordinates
    assert target.contains(None, 0.4) is False
    assert target.contains(float("nan"), 0.4) is False
    assert target.contains(0.4, float("inf")) is False


def test_target_manager_crud() -> None:
    """Verify adding, retrieving, removing, and listing targets."""
    manager = TargetManager()
    t1 = DwellTarget("t1", "YES", 0.1, 0.1, 0.3, 0.3)
    t2 = DwellTarget("t2", "NO", 0.5, 0.1, 0.3, 0.3)

    manager.add_target(t1)
    manager.add_target(t2)

    assert len(manager.all_targets()) == 2
    assert manager.get_target("t1") == t1
    assert manager.get_target("nonexistent") is None

    # Enable / disable
    assert manager.set_target_enabled("t1", False) is True
    assert t1.enabled is False
    assert manager.enabled_targets() == [t2]

    # Remove
    assert manager.remove_target("t1") is True
    assert len(manager.all_targets()) == 1
    assert manager.get_target("t1") is None
    assert manager.remove_target("t1") is False

    # Clear
    manager.clear()
    assert len(manager.all_targets()) == 0


def test_target_overlap_policies() -> None:
    """Verify deterministic resolution of overlapping targets."""
    manager = TargetManager()
    # Large outer target with priority 0
    t_large = DwellTarget("outer", "OUTER", 0.1, 0.1, 0.8, 0.8, priority=0)
    # Small inner button with priority 10
    t_inner = DwellTarget("inner", "INNER", 0.3, 0.3, 0.2, 0.2, priority=10)

    manager.add_target(t_large)
    manager.add_target(t_inner)

    # Inside both targets: (0.35, 0.35)
    # 1. highest_priority (inner has priority 10 vs outer 0)
    winner_pri = manager.find_target_at(0.35, 0.35, policy="highest_priority")
    assert winner_pri is not None and winner_pri.target_id == "inner"

    # 2. smallest_area (inner has 0.04 vs outer 0.64)
    winner_area = manager.find_target_at(0.35, 0.35, policy="smallest_area")
    assert winner_area is not None and winner_area.target_id == "inner"

    # 3. first_defined (outer was added first)
    winner_first = manager.find_target_at(0.35, 0.35, policy="first_defined")
    assert winner_first is not None and winner_first.target_id == "outer"


def test_sample_grid_generation() -> None:
    """Verify creation of AAC grid buttons."""
    grid = TargetManager.create_sample_grid(rows=2, cols=2, labels=["YES", "NO", "HELP", "REST"])
    targets = grid.all_targets()
    assert len(targets) == 4
    labels = [t.label for t in targets]
    assert labels == ["YES", "NO", "HELP", "REST"]
    # Check that targets do not overlap
    for i, t_a in enumerate(targets):
        for j, t_b in enumerate(targets):
            if i != j:
                # Centers should not be inside each other
                cx, cy = t_a.center
                assert t_b.contains(cx, cy) is False
