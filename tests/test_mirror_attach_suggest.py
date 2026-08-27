"""Filling ``attach_to:`` in for the user.

A robot you would mirror is symmetric, so the mount the copy belongs on is
usually already in the model.  Where it is, is not a guess: reflect the source
limb's own parent and look for a link sitting there.  Getting this wrong in
either direction is worse than not offering it -- a wrong mount silently moves
the limb, so the rule is "one clear match or nothing".
"""
import numpy as np
import pytest

from sw2robot.exporter.limb_mirror import suggest_attach


def _pose(x, y, z):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


@pytest.fixture
def quad():
    """A torso with both hips modelled and only the left leg built."""
    poses = {
        "torso": _pose(0.0, 0.0, 0.0),
        "hip_l": _pose(0.09, 0.05, 0.0),
        "hip_r": _pose(0.09, -0.05, 0.0),        # the reflection of hip_l
        "leg_left_0": _pose(0.09, 0.07, 0.0),
        "leg_left_1": _pose(0.09, 0.07, -0.1),
    }
    parent_of = {"hip_l": "torso", "hip_r": "torso",
                 "leg_left_0": "hip_l", "leg_left_1": "leg_left_0"}
    return poses, parent_of


def test_it_finds_the_mount_sitting_at_the_reflected_point(quad):
    poses, parent_of = quad
    assert suggest_attach(poses, parent_of, "leg_left_0", "xz") == "hip_r"


def test_the_plane_decides_it(quad):
    """The same limb reflected through a different plane lands somewhere with
    nothing on it, and there is then nothing honest to offer."""
    poses, parent_of = quad
    assert suggest_attach(poses, parent_of, "leg_left_0", "yz") is None
    assert suggest_attach(poses, parent_of, "leg_left_0", "xy") is None


def test_no_counterpart_means_no_suggestion(quad):
    """The humanoid case: only one shoulder is modelled, so the copy really does
    belong on the source's own parent and the box stays blank."""
    poses, parent_of = quad
    del poses["hip_r"]
    del parent_of["hip_r"]
    assert suggest_attach(poses, parent_of, "leg_left_0", "xz") is None


def test_two_candidates_are_ambiguous_so_it_declines(quad):
    poses, parent_of = quad
    poses["a_washer"] = _pose(0.09, -0.05, 0.0)      # same spot as hip_r
    parent_of["a_washer"] = "torso"
    assert suggest_attach(poses, parent_of, "leg_left_0", "xz") is None


def test_a_parent_on_the_plane_is_its_own_reflection():
    """A limb hanging off a link the plane passes through reflects onto that
    same link -- the default attachment is already right, so offer nothing."""
    poses = {"torso": _pose(0.0, 0.0, 0.0), "arm_0": _pose(0.0, 0.2, 0.0)}
    parent_of = {"arm_0": "torso"}
    assert suggest_attach(poses, parent_of, "arm_0", "xz") is None


def test_it_never_points_into_the_limb_being_copied():
    """A link of the limb itself can land on the reflected point (a symmetric
    limb straddling the plane); attaching the copy there would be a cycle."""
    poses = {"torso": _pose(0.0, 0.0, 0.0),
             "wrist": _pose(0.0, 0.02, 0.0),
             "jaw": _pose(0.0, 0.01, 0.0),
             "jaw_tip": _pose(0.0, -0.02, 0.0)}     # reflection of `wrist`
    parent_of = {"wrist": "torso", "jaw": "wrist", "jaw_tip": "jaw"}
    assert suggest_attach(poses, parent_of, "jaw", "xz") is None


def test_a_limb_root_with_no_parent_is_not_mirrorable(quad):
    poses, parent_of = quad
    assert suggest_attach(poses, parent_of, "torso", "xz") is None


def test_an_unknown_plane_is_declined(quad):
    poses, parent_of = quad
    assert suggest_attach(poses, parent_of, "leg_left_0", "sideways") is None


def test_the_tolerance_is_a_real_match_not_the_nearest_thing(quad):
    """`hip_r` 8 mm off the reflected point is a different mount, not this one;
    silently snapping to it would move the limb."""
    poses, parent_of = quad
    poses["hip_r"] = _pose(0.09, -0.058, 0.0)
    assert suggest_attach(poses, parent_of, "leg_left_0", "xz") is None
    assert suggest_attach(poses, parent_of, "leg_left_0", "xz",
                          tol=0.01) == "hip_r"
