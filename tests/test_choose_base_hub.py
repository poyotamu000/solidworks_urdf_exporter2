"""Regression test for hub-based base selection.

The CAD ground set is unreliable: designers often fix a convenient peripheral
part (an arm tip near the origin) rather than the structural frame.  Picking the
grounded part nearest the origin then roots the whole robot at a wrist, giving a
lopsided tree.  ``choose_base`` instead picks the mate-graph HUB -- the part the
most components bolt to -- so the frame becomes the root.
"""

import numpy as np

from sw2robot.exporter.model import Component, choose_base


def _c(name, xyz, fixed=False):
    w = np.eye(4)
    w[:3, 3] = xyz
    return Component(name=name, link_name=name, part_path=None,
                     is_subassembly=False, world=w, fixed=bool(fixed), dof=0)


def test_base_is_the_mate_hub_not_the_grounded_peripheral_part():
    hub = _c("frame", (0.0, 0.0, 0.0))                 # everything bolts here
    arms = [_c(f"arm{i}", (0.5 * i, 0.3, 0.0)) for i in range(4)]
    # a tiny part the CAD happens to ground, sitting nearest the origin
    tip = _c("wrist_tip", (0.01, 0.0, 0.0), fixed=True)

    comps = [hub, *arms, tip]
    adjacency = {frozenset(("frame", a.name)): {"types": ["COINCIDENT"]}
                 for a in arms}
    adjacency[frozenset(("arm0", "wrist_tip"))] = {"types": ["COINCIDENT"]}
    ground = {"wrist_tip"}

    base = choose_base(comps, ground, base_hint=None, adjacency=adjacency)
    assert base.name == "frame", \
        f"base should be the hub 'frame', got {base.name!r}"


def test_base_hint_still_wins():
    hub = _c("frame", (0, 0, 0))
    arm = _c("arm", (1, 0, 0))
    adjacency = {frozenset(("frame", "arm")): {"types": ["COINCIDENT"]}}
    base = choose_base([hub, arm], set(), base_hint="arm", adjacency=adjacency)
    assert base.name == "arm"
