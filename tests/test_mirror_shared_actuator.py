"""A mirrored limb that rides the original's actuator.

``mirror_limbs`` hangs the copy off the limb root's own parent.  When a joint
drives that parent, the one actuator swings both limbs -- which is a bug on a
quadruped (each leg has its own hip; the copy lost it) and correct on a
humanoid (one chest joint carries both arms, one wrist carries both jaws).

Nothing in the tree tells those apart.  What does is whether the model already
holds the mount the copy should have used: a quadruped's torso carries all four
hip servos, so the spare one is sitting right there, while no robot has a second
chest.  So the warning fires only when there is a free counterpart mount to
point at -- otherwise it would fire on every humanoid arm anyone ever mirrors,
which is the main thing the feature exists for.
"""
import numpy as np
import pytest

from sw2robot.exporter.limb_mirror import mirror_limbs, shared_actuator
from sw2robot.exporter.model import Component, Joint, RobotModel


def _comp(name):
    return Component(name=name + "-1", link_name=name, part_path="p.SLDPRT",
                     is_subassembly=False, world=np.eye(4), fixed=False,
                     dof=None, sw_mass=1.0, sw_com=[0.05, 0.0, 0.0],
                     sw_inertia=[0.1, 0.01, 0.02, 0.2, 0.03, 0.3])


def _joint(parent, child, jtype="fixed", xyz=None):
    return Joint(name=f"{parent}__{child}", parent=parent, child=child,
                 jtype=jtype, xyz=xyz or [0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0],
                 axis=[0.0, 0.0, 1.0] if jtype != "fixed" else None)


@pytest.fixture
def quad():
    """A torso carrying BOTH hips, with only one leg built::

        torso -(rev)-> hip_l -> leg_right_0 -(rev)-> leg_right_1
              -(rev)-> hip_r        (free: the mount the copy belongs on)
    """
    names = ["torso", "hip_l", "hip_r", "leg_right_0", "leg_right_1"]
    return RobotModel(
        name="quad", base_link="torso",
        components=[_comp(n) for n in names],
        joints=[_joint("torso", "hip_l", "revolute", [0.0, 0.05, 0.0]),
                _joint("torso", "hip_r", "revolute", [0.0, -0.05, 0.0]),
                _joint("hip_l", "leg_right_0", xyz=[0.0, 0.02, 0.0]),
                _joint("leg_right_0", "leg_right_1", "revolute",
                       [0.0, 0.0, -0.1])])


@pytest.fixture
def humanoid():
    """One chest joint, both arms hanging off it -- sharing by design::

        base -(rev)-> chest -> arm_l_0 -(rev)-> arm_l_1
    """
    names = ["base", "chest", "arm_l_0", "arm_l_1"]
    return RobotModel(
        name="h", base_link="base",
        components=[_comp(n) for n in names],
        joints=[_joint("base", "chest", "revolute", [0.0, 0.0, 0.3]),
                _joint("chest", "arm_l_0", xyz=[0.0, 0.15, 0.0]),
                _joint("arm_l_0", "arm_l_1", "revolute", [0.0, 0.0, -0.2])])


# ------------------------------------------------------- the predicate itself

def test_it_names_the_joint_that_would_drive_both_limbs(quad):
    assert shared_actuator(quad.joints, "leg_right_0")["joint"] == "torso__hip_l"


def test_using_the_far_mount_is_not_sharing(quad):
    assert shared_actuator(quad.joints, "leg_right_0", "hip_r") is None


def test_a_rigidly_grounded_parent_is_not_sharing(quad):
    """Rooting at the hip itself is the other way out: its parent is the torso,
    which nothing drives."""
    assert shared_actuator(quad.joints, "hip_l") is None


def test_the_robot_root_has_no_attachment_joint(quad):
    assert shared_actuator(quad.joints, "torso") is None


# ------------------------------------------------------- what gets reported

def test_the_report_points_at_the_mount_that_was_missed(quad):
    reports = mirror_limbs(
        quad, [{"root": "leg_right_0", "plane": "xz", "prefix": "L_"}])
    assert not reports[0].get("skip")
    assert reports[0]["shared"] == {"joint": "torso__hip_l",
                                    "attach_to": "hip_r"}
    # and the limb really did share -- both roots on the same parent
    by_child = {j.child: j.parent for j in quad.joints}
    assert by_child["L_leg_right_0"] == by_child["leg_right_0"] == "hip_l"


def test_taking_the_advice_silences_it(quad):
    reports = mirror_limbs(
        quad, [{"root": "leg_right_0", "plane": "xz", "prefix": "L_",
                "attach_to": "hip_r"}])
    assert reports[0]["shared"] is None


def test_two_arms_on_one_chest_joint_are_not_warned_about(humanoid):
    """The regression this rule exists for: there is no second chest, so the
    sharing is the design and saying otherwise on every humanoid arm would make
    the warning worthless."""
    reports = mirror_limbs(
        humanoid, [{"root": "arm_l_0", "plane": "xz", "rename": {"_l_": "_r_"}}])
    assert not reports[0].get("skip")
    assert reports[0]["shared"] is None
    # the chest joint does drive both -- it is simply the right answer
    assert shared_actuator(humanoid.joints, "arm_l_0")["joint"] == "base__chest"


def test_it_is_a_warning_and_not_a_refusal(quad):
    reports = mirror_limbs(
        quad, [{"root": "leg_right_0", "plane": "xz", "prefix": "L_"}])
    assert reports[0]["links"] == 2
    assert {c.link_name for c in quad.components} >= {"L_leg_right_0",
                                                      "L_leg_right_1"}
