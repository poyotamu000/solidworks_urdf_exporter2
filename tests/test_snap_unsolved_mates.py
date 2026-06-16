"""Regression tests for the CONCENTRIC-mate auto-correction guard.

``_snap_unsolved_mates`` repairs a part whose saved pose ignores a suppressed /
erroring SolidWorks mate by sliding it back onto the mate axis (its sibling
instances vote on the correct distance).  But the same part is sometimes used
in DIFFERENT roles -- e.g. one ``finger.SLDPRT`` instance as a proximal phalanx
and another as the middle phalanx.  The proximal one legitimately sits the
inter-joint distance off the shared mate axis, so the outlier test mis-fires and
would snap it right on top of its neighbour (the bug behind the "little finger
overlaps / a link is missing" report).  The guard skips the snap when it would
collapse the part onto a sibling instance.
"""

import numpy as np

from sw2robot.exporter.model import Component, _snap_unsolved_mates


def _part(name, part_path, xyz):
    """A minimal Component at ``xyz`` (only the fields the snap logic reads)."""
    w = np.eye(4)
    w[:3, 3] = xyz
    return Component(name=name, link_name=name, part_path=part_path,
                     is_subassembly=False, world=w, fixed=False, dof=0)


# CONCENTRIC mate on the world X axis through the origin: an instance sitting on
# the axis has offset 0; one sitting at y=+23 mm is 23 mm off it.
_AXIS = {"type": "CONCENTRIC", "points": [[0.0, 0.0, 0.0]],
         "dirs": [[1.0, 0.0, 0.0]]}


def _adjacency(owner_a, owner_b):
    """Two CONCENTRIC mates (same axis) tying two ``finger`` instances to two
    instances of a partner part -- so both land in one sibling-comparison group."""
    return {
        frozenset((owner_a, "partner_a")):
            {"mates": [dict(_AXIS, owners=[owner_a])]},
        frozenset((owner_b, "partner_b")):
            {"mates": [dict(_AXIS, owners=[owner_b])]},
    }


def test_snap_skips_when_it_would_stack_on_a_sibling():
    """The 23 mm-off instance must NOT be snapped when the target is already
    occupied by another instance of the same part (same part, different role)."""
    on_axis = _part("finger_on", "finger.SLDPRT", [0.0, 0.0, 0.0])   # off = 0
    outlier = _part("finger_off", "finger.SLDPRT", [0.10, 0.023, 0.0])  # off = 23mm
    # a third instance occupying exactly where `outlier` would be snapped to
    target = _part("finger_mid", "finger.SLDPRT", [0.10, 0.0, 0.0])
    comps = [on_axis, outlier, target,
             _part("partner_a", "B.SLDPRT", [0.0, 0.0, 0.0]),
             _part("partner_b", "B.SLDPRT", [0.0, 0.0, 0.0])]

    _snap_unsolved_mates(comps, _adjacency("finger_on", "finger_off"))

    # guard fired: the outlier keeps its (correct) exported pose, NOT stacked
    np.testing.assert_allclose(outlier.world[:3, 3], [0.10, 0.023, 0.0])
    # and the part it would have collapsed onto is untouched too
    np.testing.assert_allclose(target.world[:3, 3], [0.10, 0.0, 0.0])


def test_snap_still_repairs_a_genuine_unsolved_mate():
    """With no sibling at the target, a true suppressed-mate outlier is still
    slid back onto its axis -- the guard must not block the legitimate repair."""
    on_axis = _part("finger_on", "finger.SLDPRT", [0.0, 0.0, 0.0])
    outlier = _part("finger_off", "finger.SLDPRT", [0.10, 0.023, 0.0])
    comps = [on_axis, outlier,
             _part("partner_a", "B.SLDPRT", [0.0, 0.0, 0.0]),
             _part("partner_b", "B.SLDPRT", [0.0, 0.0, 0.0])]

    _snap_unsolved_mates(comps, _adjacency("finger_on", "finger_off"))

    # snapped perpendicular onto the X axis: y -> 0, x kept
    np.testing.assert_allclose(outlier.world[:3, 3], [0.10, 0.0, 0.0], atol=1e-9)
