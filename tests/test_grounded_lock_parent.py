"""Regression test for the LOCK / loop-closure tree-parent fixes.

A sub-assembly part that is fixed to the sub-assembly FRAME (grounded) is tied
to the rigid base cluster by a synthetic LOCK edge with no mate geometry.  That
LOCK used to fall to the weak tier, so when the grounded part was ALSO reachable
through a revolute joint -- a mecanum wheel UNIT bolted to the movebase frame
but spun by its motor -- the wheel got attached first and the motor ended up
parented BELOW the wheel (hierarchy inverted).  Treating a LOCK as the rigid tie
it is keeps the grounded motor on the base side and the wheel as its child.
"""

import numpy as np

from sw2robot.exporter.model import Component, _auto_parent_map


def _c(name, xyz=(0, 0, 0)):
    w = np.eye(4)
    w[:3, 3] = xyz
    return Component(name=name, link_name=name, part_path=None,
                     is_subassembly=False, world=w, fixed=False, dof=0)


def test_grounded_motor_parents_the_wheel_not_the_other_way():
    base = _c("base")
    motor = _c("motor", (0.0, 0.0, 0.1))
    wheel = _c("wheel", (0.0, 0.0, 0.2))
    other = _c("other_wheel", (0.2, 0.0, 0.2))   # already on the base cluster

    adjacency = {
        # motor is GROUNDED to the frame -> synthetic LOCK to the base cluster
        frozenset(("base", "motor")):
            {"types": ["LOCK"], "axis": None,
             "mates": [{"type": "LOCK", "etypes": [], "points": [],
                        "dirs": [], "radii": [], "owners": []}]},
        # the wheel spins on its motor (a real revolute)
        frozenset(("motor", "wheel")):
            {"types": ["CONCENTRIC"],
             "axis": (np.zeros(3), np.array([1.0, 0.0, 0.0])), "mates": []},
        # a fixed barrel tie between the two driven wheels (a loop closure)
        frozenset(("wheel", "other_wheel")):
            {"types": ["COINCIDENT"], "axis": None, "mates": []},
        # the other wheel is also driven (so the barrel is "both driven")
        frozenset(("base", "other_wheel")):
            {"types": ["CONCENTRIC"],
             "axis": (np.zeros(3), np.array([1.0, 0.0, 0.0])), "mates": []},
    }
    parent_of, edge_info = _auto_parent_map(
        [base, motor, wheel, other], adjacency, base)

    # the grounded motor hangs off the base cluster, NOT off the wheel
    assert parent_of["motor"] == "base"
    # and the wheel is the motor's child via the revolute joint
    assert parent_of["wheel"] == "motor"
    assert edge_info[("wheel", "motor")]["type"] in (
        "revolute", "continuous")
