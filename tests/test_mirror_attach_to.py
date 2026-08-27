"""``attach_to:`` -- hanging a mirrored limb on a mount that already exists.

By default the copy hangs off the limb root's own parent, which assumes the far
side has nothing there yet.  That assumption breaks on a robot whose TORSO
carries both sides' mounts and is only missing the limbs: a quadruped models all
four hip servos in the body, so mirroring a leg onto its own parent either puts
both legs on one horn (one joint, two limbs) or -- if you root higher to avoid
that -- clones a servo that is already in the model, double-counting its mass.

Naming the far side's existing mount fixes both: the copy hangs where the CAD
has it, driven by that side's own joint, with nothing duplicated.
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
    """A torso carrying BOTH hips, with only the left leg modelled::

        torso -(rev)-> hip_l -> leg_left_0 -(rev)-> leg_left_1
              -(rev)-> hip_r          (bare: the right leg is missing)
    """
    names = ["torso", "hip_l", "hip_r", "leg_left_0", "leg_left_1"]
    return RobotModel(
        name="quad", base_link="torso",
        components=[_comp(n) for n in names],
        joints=[_joint("torso", "hip_l", "revolute", [0.0, 0.05, 0.0]),
                _joint("torso", "hip_r", "revolute", [0.0, -0.05, 0.0]),
                _joint("hip_l", "leg_left_0", xyz=[0.0, 0.02, 0.0]),
                _joint("leg_left_0", "leg_left_1", "revolute",
                       [0.0, 0.0, -0.1])])


def _spec(**kw):
    base = {"root": "leg_left_0", "plane": "xz", "prefix": "R_"}
    base.update(kw)
    return [base]


def test_the_copy_hangs_on_the_mount_that_was_named(quad):
    reports = mirror_limbs(quad, _spec(attach_to="hip_r"))
    assert not reports[0].get("skip")
    attach = next(j for j in quad.joints if j.child == "R_leg_left_0")
    assert attach.parent == "hip_r"


def test_nothing_is_cloned(quad):
    """The point of the option: the far mount is reused, not duplicated."""
    mirror_limbs(quad, _spec(attach_to="hip_r"))
    links = [c.link_name for c in quad.components]
    assert "R_hip_l" not in links and "R_hip_r" not in links
    assert len(links) == len(set(links))
    assert sorted(ln for ln in links if ln.startswith("R_")) == [
        "R_leg_left_0", "R_leg_left_1"]


def test_the_copy_lands_at_the_reflected_pose_relative_to_its_new_parent(quad):
    """The offset is recomputed against the new parent, so the limb still ends
    up where the reflection puts it -- not where the source's offset would."""
    mirror_limbs(quad, _spec(attach_to="hip_r"))
    attach = next(j for j in quad.joints if j.child == "R_leg_left_0")
    # source: torso -> hip_l (+0.05) -> leg (+0.02) = +0.07; reflected = -0.07,
    # and hip_r is already at -0.05, so the remaining offset is -0.02
    assert np.allclose(attach.xyz, [0.0, -0.02, 0.0])


def test_without_it_the_copy_shares_the_original_hip(quad):
    reports = mirror_limbs(quad, _spec())
    attach = next(j for j in quad.joints if j.child == "R_leg_left_0")
    assert attach.parent == "hip_l"                     # same horn as the twin
    assert reports[0]["shared"]["joint"] == "torso__hip_l"


def test_naming_the_far_mount_clears_the_shared_actuator_warning(quad):
    """The copy is now driven by hip_r, which drives nothing else."""
    reports = mirror_limbs(quad, _spec(attach_to="hip_r"))
    assert reports[0]["shared"] is None
    assert shared_actuator(quad.joints, "leg_left_0", "hip_r") is None
    assert shared_actuator(quad.joints, "leg_left_0")["joint"] == "torso__hip_l"


def test_a_mount_that_does_not_exist_is_refused(quad):
    reports = mirror_limbs(quad, _spec(attach_to="hip_nowhere"))
    assert "not a link" in reports[0]["skip"]
    assert not any(c.link_name.startswith("R_") for c in quad.components)


def test_a_mount_inside_the_limb_is_refused(quad):
    """It would be its own ancestor -- a cycle, and the generated tree would
    never close."""
    reports = mirror_limbs(quad, _spec(attach_to="leg_left_1"))
    assert "inside the limb" in reports[0]["skip"]
    assert not any(c.link_name.startswith("R_") for c in quad.components)


def test_the_limb_below_the_attachment_is_still_a_1_to_1_copy(quad):
    mirror_limbs(quad, _spec(attach_to="hip_r"))
    by_child = {j.child: j for j in quad.joints}
    # the knee keeps its type, and its parent is the copied link, not the source
    assert by_child["R_leg_left_1"].jtype == "revolute"
    assert by_child["R_leg_left_1"].parent == "R_leg_left_0"
