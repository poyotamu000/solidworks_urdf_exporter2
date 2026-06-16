"""Regression test: a configured revolute joint must keep its axis even when the
hinge has NO concentric mate.

MIRRORED SolidWorks parts are often constrained by an ANGLE + coincident-plane
mate set instead of a concentric cylinder.  The extractor only records an
``axis`` from CONCENTRIC mates, so for those edges ``rec["axis"]`` is None and
the config path used to drop the configured ``revolute`` to ``fixed`` ("has no
axis; using fixed") -- the joint then would not move.  ``_config_parent_map``
now falls back to the twist-nullspace geometry (``classify_edge_geo``) to
recover the rotation axis, the same way the auto classifier already does.
"""

import numpy as np

from sw2robot.exporter.model import Component, _config_parent_map

# A real mirror-part hinge from the kxr_humanoid assembly
# (joint_flameA_y_link_4 <-> cross_r_p_link_mirror_8): COINCIDENT + ANGLE +
# COINCIDENT, with NO concentric mate.  Its free DOF is rotation about -Y.
MIRROR_MATES = [
    {"type": "COINCIDENT", "etypes": [1, 1],
     "points": [[0.0785, 0.051, -0.0075], [0.0785, 0.051, -0.0075]],
     "dirs": [[0, -1, 0], [0, 1, 0]], "radii": [0, 0]},
    {"type": "ANGLE", "etypes": [3, 3],
     "points": [[0.0785, 0.051, -0.0075], [0.0785, 0.0685, -0.0075]],
     "dirs": [[-1, 0, 0], [-1, 0, 0]], "radii": [0, 0]},
    {"type": "COINCIDENT", "etypes": [3, 3],
     "points": [[0.0827, 0.051, -0.0033], [0.0818, 0.051, -0.0042]],
     "dirs": [[0, -1, 0], [0, 1, 0]], "radii": [0, 0]},
]


def _comp(name):
    return Component(name=name, link_name=name, part_path=None,
                     is_subassembly=False, world=np.eye(4), fixed=False, dof=0)


def test_config_revolute_recovers_axis_from_angle_mate():
    base, parent, child = _comp("base"), _comp("parent"), _comp("child")
    adjacency = {
        frozenset(("base", "parent")):
            {"types": ["COINCIDENT"], "axis": None, "mates": []},
        # the hinge: no concentric mate, so rec["axis"] is None
        frozenset(("parent", "child")):
            {"types": ["COINCIDENT", "ANGLE", "COINCIDENT"],
             "axis": None, "mates": MIRROR_MATES},
    }
    directed = [
        {"parent": "base", "child": "parent", "type": "fixed"},
        {"parent": "parent", "child": "child", "type": "revolute"},
    ]

    _parent_of, edge_info = _config_parent_map(
        [base, parent, child], adjacency, base, directed)

    ax = edge_info[("child", "parent")]["axis"]
    assert ax is not None, \
        "configured revolute lost its axis (mirror / ANGLE-mate regression)"
    _point, direction = ax
    direction = np.asarray(direction, float)
    assert abs(np.linalg.norm(direction) - 1.0) < 1e-6      # unit axis
    # recovered hinge runs along Y (the assembly's free rotation here)
    assert abs(abs(float(direction[1])) - 1.0) < 1e-6


def test_config_concentric_axis_is_still_preferred():
    """The geometry fallback must not override an explicit concentric axis."""
    base, parent, child = _comp("base"), _comp("parent"), _comp("child")
    concentric_axis = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    adjacency = {
        frozenset(("parent", "child")):
            {"types": ["CONCENTRIC"], "axis": concentric_axis,
             "mates": MIRROR_MATES},     # mates present, but axis already set
    }
    directed = [{"parent": "parent", "child": "child", "type": "revolute"}]

    _parent_of, edge_info = _config_parent_map(
        [base, parent, child], adjacency, base, directed)

    _point, direction = edge_info[("child", "parent")]["axis"]
    np.testing.assert_allclose(direction, [1.0, 0.0, 0.0])   # concentric kept
