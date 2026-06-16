"""Regression test for the self-collision baseline broadphase-sync fix.

trimesh 4.x's ``CollisionManager.add_object(name, mesh, transform=T)`` sets the
fcl object's pose but does NOT refresh the broadphase AABB-tree node for that
object, so for some geometry+rotation combinations ``in_collision_internal``
fails to surface a pair that genuinely overlaps once the parts are moved to
their world poses.  ``SelfCollision`` built its rest-pose baseline that way, so
those overlapping rest contacts escaped the baseline -- and then lit up red as
false "new" collisions the instant a query (which DOES refresh the tree via
``set_transform``) ran at the very same home pose, even on parts that never
move (base/body panels).

The two hulls + transforms in ``tests/fixtures`` are the exact pair from the
kxr_humanoid_movebase model that exhibited the miss (a 90-degree-rotated servo
horn overlapping a yaw link at the home pose); a box stand-in or a
rotation-stripped transform does NOT reproduce it, so the real geometry is kept
as a small fixture.
"""

import os

import numpy as np
import trimesh

from sw2robot.editor.autoinit import SelfCollision

_FIX = os.path.join(os.path.dirname(__file__), "fixtures")
_PAIR = ("servo_hone_38mm_1", "single_yaw_link_1")


class _Coords:
    def __init__(self, T):
        self._T = np.asarray(T, float)

    def T(self):
        return self._T


class _Link:
    def __init__(self, name, T):
        self.name = name
        self._c = _Coords(T)

    def worldcoords(self):
        return self._c


class _Robot:
    def __init__(self, links):
        self.link_list = links
        self.joint_list = []          # no joints -> no adjacency in the baseline


def test_rotated_world_overlap_is_baselined_not_flagged_new():
    meshes, links = {}, []
    for n in _PAIR:
        meshes[n] = trimesh.load(os.path.join(_FIX, n + ".hull.ply"),
                                 process=False)
        links.append(_Link(n, np.load(os.path.join(_FIX, n + ".T.npy"))))
    robot = _Robot(links)

    sc = SelfCollision(robot, meshes, confirm=True)

    # the pair deeply overlaps at the home pose, so it MUST be in the baseline
    assert frozenset(_PAIR) in sc.baseline, \
        "rest-pose overlap escaped the baseline (broadphase not synced)"
    # ... and therefore reports as NO new collision at that same home pose
    new, offenders = sc.offenders()
    assert new == set(), f"rest overlap leaked as a false new collision: {new}"
    assert offenders == set()
