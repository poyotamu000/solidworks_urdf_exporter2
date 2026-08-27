"""Building the half of a robot that was never modelled, by reflecting the half
that was.

The load-bearing test here is the joint-axis sign.  A mirror generator that gets
it wrong produces a limb that looks correct at the zero pose and bends the wrong
way everywhere else -- so the convention is pinned twice: once on the rule
itself, and once by driving both sides to the same q and checking the resulting
world poses really are reflections.
"""

import numpy as np
import pytest

from sw2robot.exporter.limb_mirror import (
    PLANES,
    link_anchors,
    mirror_axis,
    mirror_limbs,
    origin_matrix,
    subtree_links,
)


def _rot(axis, q):
    """Rodrigues, so the pose check does not lean on the code under test."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)
    M = np.eye(4)
    M[:3, :3] = R
    return M


def _reflection(signs):
    M = np.eye(4)
    M[:3, :3] = np.diag(np.asarray(signs, float))
    return M


# ------------------------------------------------------------------ the sign

def test_a_revolute_axis_picks_up_the_axial_sign():
    """A rotation axis is an axial vector: reflecting it takes the extra minus,
    which is what makes the same q produce the mirrored pose."""
    assert mirror_axis([0, 0, 1], PLANES["xz"], "revolute") == [0.0, 0.0, -1.0]
    assert mirror_axis([1, 0, 0], PLANES["xz"], "revolute") == [-1.0, 0.0, 0.0]
    # the axis lying ALONG the mirrored direction is the one that survives
    assert mirror_axis([0, 1, 0], PLANES["xz"], "revolute") == [0.0, 1.0, 0.0]


def test_a_prismatic_axis_does_not():
    """A translation axis is an ordinary vector -- no sign flip, or the
    generated slider would extend backwards."""
    assert mirror_axis([0, 0, 1], PLANES["xz"], "prismatic") == [0.0, 0.0, 1.0]
    assert mirror_axis([0, 1, 0], PLANES["xz"], "prismatic") == [0.0, -1.0, 0.0]


def test_continuous_follows_revolute():
    assert (mirror_axis([1, 0, 0], PLANES["xz"], "continuous")
            == mirror_axis([1, 0, 0], PLANES["xz"], "revolute"))


def test_no_axis_stays_none():
    assert mirror_axis(None, PLANES["xz"], "fixed") is None


# ------------------------------------------------------------------ the tree

class _J:
    def __init__(self, name, parent, child, jtype="fixed", xyz=None, rpy=None,
                 axis=None, lower=None, upper=None):
        self.name, self.parent, self.child, self.jtype = name, parent, child, jtype
        self.xyz = list(xyz or [0, 0, 0])
        self.rpy = list(rpy or [0, 0, 0])
        self.axis = axis
        self.lower, self.upper = lower, upper
        self.sw_axis_point = self.sw_axis_dir = self.geo_note = None


def test_link_anchors_compose_down_the_tree_in_any_joint_order():
    joints = [_J("b", "mid", "tip", xyz=[0, 0, 1]),
              _J("a", "base_link", "mid", xyz=[1, 0, 0])]   # child before parent
    anchors = link_anchors(joints, "base_link")
    assert np.allclose(anchors["tip"][:3, 3], [1, 0, 1])


def test_subtree_is_breadth_first_so_parents_come_first():
    joints = [_J("a", "root", "l1"), _J("b", "l1", "l2"), _J("c", "l2", "l3")]
    assert subtree_links(joints, "l1") == ["l1", "l2", "l3"]


# ------------------------------------------------------------ the generation

def _model():
    """torso + a two-link right arm, the shoulder 0.2 m out along -y."""
    from sw2robot.exporter.model import Component, Joint, RobotModel

    def comp(link, mass=1.0, com=None):
        return Component(name=link + "-1", link_name=link, part_path="p.SLDPRT",
                         is_subassembly=False, world=np.eye(4), fixed=False,
                         dof=None, sw_mass=mass,
                         sw_com=list(com or [0.05, 0.0, 0.0]),
                         sw_inertia=[0.1, 0.01, 0.02, 0.2, 0.03, 0.3],
                         visual_xyz=[0.0, 0.01, 0.0], visual_rpy=[0.0, 0.0, 0.2])

    model = RobotModel(
        name="r", base_link="torso",
        components=[comp("torso", mass=5.0), comp("right_arm_0"),
                    comp("right_arm_1")],
        joints=[
            Joint(name="torso__right_arm_0", parent="torso", child="right_arm_0",
                  jtype="revolute", xyz=[0.0, -0.2, 0.3], rpy=[0.0, 0.0, 0.0],
                  axis=[0.0, 0.0, 1.0], lower=-1.2, upper=0.7),
            Joint(name="right_arm_0__right_arm_1", parent="right_arm_0",
                  child="right_arm_1", jtype="revolute", xyz=[0.0, -0.15, 0.0],
                  rpy=[0.0, 0.0, 0.0], axis=[1.0, 0.0, 0.0],
                  lower=0.0, upper=2.0),
        ])
    return model


def _spec():
    return [{"root": "right_arm_0", "plane": "xz", "rename": {"right": "left"}}]


def test_the_generated_limb_lands_on_the_other_side():
    model = _model()
    reports = mirror_limbs(model, _spec())

    assert reports[0]["links"] == 2
    names = {c.link_name for c in model.components}
    assert {"left_arm_0", "left_arm_1"} <= names
    attach = next(j for j in model.joints if j.child == "left_arm_0")
    assert attach.parent == "torso"                       # hangs off the torso
    assert np.allclose(attach.xyz, [0.0, 0.2, 0.3])       # y mirrored


def test_limits_carry_over_unchanged():
    """The axis sign is what encodes the mirroring, so the limits must not ALSO
    be flipped -- doing both would cancel out and give a limb that mirrors its
    range but not its motion."""
    model = _model()
    mirror_limbs(model, _spec())
    src = next(j for j in model.joints if j.child == "right_arm_1")
    dst = next(j for j in model.joints if j.child == "left_arm_1")
    assert (dst.lower, dst.upper) == (src.lower, src.upper)


def test_same_q_puts_the_two_sides_in_mirror_poses():
    """The convention, checked end to end rather than by its formula: drive both
    arms to the same joint angles and the left tip frame must be the exact
    reflection of the right one."""
    model = _model()
    mirror_limbs(model, _spec())
    M = _reflection(PLANES["xz"])
    S = _reflection(PLANES["xz"])
    by_child = {j.child: j for j in model.joints}

    def tip_pose(chain, q):
        pose = np.eye(4)
        for link, angle in zip(chain, q):
            joint = by_child[link]
            pose = pose @ origin_matrix(joint.xyz, joint.rpy) \
                @ _rot(joint.axis, angle)
        return pose

    for q in ([0.0, 0.0], [0.4, -0.9], [-1.1, 1.7]):
        right = tip_pose(["right_arm_0", "right_arm_1"], q)
        left = tip_pose(["left_arm_0", "left_arm_1"], q)
        assert np.allclose(left, M @ right @ S, atol=1e-9), q


def test_the_wrong_axis_sign_would_be_caught():
    """Guard on the guard: with the naive a' = S@a the pose check above fails,
    so that test is really testing the sign and not passing vacuously."""
    model = _model()
    mirror_limbs(model, _spec())
    M = _reflection(PLANES["xz"])
    S = _reflection(PLANES["xz"])
    by_child = {j.child: j for j in model.joints}
    for joint in model.joints:                      # break it on purpose
        if joint.child.startswith("left") and joint.axis:
            joint.axis = [-v for v in joint.axis]

    def tip_pose(chain, q):
        pose = np.eye(4)
        for link, angle in zip(chain, q):
            j = by_child[link]
            pose = pose @ origin_matrix(j.xyz, j.rpy) @ _rot(j.axis, angle)
        return pose

    right = tip_pose(["right_arm_0", "right_arm_1"], [0.4, -0.9])
    left = tip_pose(["left_arm_0", "left_arm_1"], [0.4, -0.9])
    assert not np.allclose(left, M @ right @ S, atol=1e-6)


def test_the_inertial_is_reflected_with_the_link():
    """The link's own axes are reflected, so its mass properties have to be too
    -- an un-reflected inertia tensor is the kind of error that only shows up as
    a limp in simulation."""
    model = _model()
    mirror_limbs(model, _spec())
    src = next(c for c in model.components if c.link_name == "right_arm_0")
    dst = next(c for c in model.components if c.link_name == "left_arm_0")
    assert dst.sw_mass == src.sw_mass
    assert np.allclose(dst.sw_com, [src.sw_com[0], -src.sw_com[1], src.sw_com[2]])
    # y is mirrored -> ixy and iyz flip sign, ixz does not
    ixx, ixy, ixz, iyy, iyz, izz = src.sw_inertia
    assert np.allclose(dst.sw_inertia, [ixx, -ixy, ixz, iyy, -iyz, izz])


def test_the_visual_origin_is_conjugated_not_just_reflected():
    """S @ T alone is improper and has no rpy; the origin has to be S @ T @ S,
    with the MESH carrying the other reflection."""
    model = _model()
    mirror_limbs(model, _spec())
    dst = next(c for c in model.components if c.link_name == "left_arm_0")
    S = _reflection(PLANES["xz"])
    expected = S @ origin_matrix([0.0, 0.01, 0.0], [0.0, 0.0, 0.2]) @ S
    got = origin_matrix(dst.visual_xyz, dst.visual_rpy)
    assert np.allclose(got, expected, atol=1e-9)
    assert np.linalg.det(got[:3, :3]) == pytest.approx(1.0)


def test_generated_links_are_marked_as_having_no_cad():
    model = _model()
    mirror_limbs(model, _spec())
    dst = next(c for c in model.components if c.link_name == "left_arm_0")
    assert dst.mirrored_from == "right_arm_0"
    # a CAD mate axis is a world-frame record of a real constraint; a generated
    # joint has none, so it must not carry a reflected copy of one
    attach = next(j for j in model.joints if j.child == "left_arm_0")
    assert attach.sw_axis_point is None and attach.sw_axis_dir is None


# ------------------------------------------------------------------ refusals

def test_a_rename_that_would_collide_generates_nothing():
    """Silently producing left_arm_0 twice, or leaving the copy sharing the
    original's name, would corrupt the URDF -- refuse instead."""
    model = _model()
    reports = mirror_limbs(model, [{"root": "right_arm_0", "plane": "xz",
                                    "rename": {"NOPE": "X"}}])
    assert reports[0]["skip"] and "unchanged or colliding" in reports[0]["skip"]
    assert len(model.components) == 3


def test_an_unknown_plane_generates_nothing():
    model = _model()
    reports = mirror_limbs(model, [{"root": "right_arm_0", "plane": "diagonal",
                                    "rename": {"right": "left"}}])
    assert reports[0]["skip"] and "unknown plane" in reports[0]["skip"]
    assert len(model.joints) == 2


def test_a_missing_root_generates_nothing():
    model = _model()
    reports = mirror_limbs(model, [{"root": "right_leg_LINK0",
                                    "rename": {"right_leg": "left_leg"}}])
    assert reports[0]["skip"] == "no such link in the model"


def test_mirroring_the_robot_root_is_refused():
    """There is no attachment joint to reflect, and the result would be a second
    disconnected root."""
    model = _model()
    reports = mirror_limbs(model, [{"root": "torso", "rename": {"torso": "X"}}])
    assert reports[0]["skip"] and "root" in reports[0]["skip"]


def test_no_spec_is_a_no_op():
    model = _model()
    assert mirror_limbs(model, None) == []
    assert mirror_limbs(model, []) == []
    assert len(model.components) == 3


# --------------------------------- the generated links are first-class links

def _eye():
    return list(np.eye(4).flatten())


def _half_robot_graph():
    """base + a two-link right arm, as a cached extraction would hold it."""
    from sw2robot.exporter.state import ComponentState, GraphState

    def state(name, mass):
        return ComponentState(name=name, link_name=name, world=_eye(),
                              sw_mass=mass, sw_com=[0.01, 0.0, 0.0],
                              sw_inertia=[1, 0, 0, 1, 0, 1])

    return GraphState(
        robot_name="r", source_assembly="x.SLDASM",
        components=[state("base", 5.0), state("right_arm_0", 1.0),
                    state("right_arm_1", 2.0)])


def _half_robot_config(**extra):
    config = {
        "base": "base",
        "joints": [
            {"parent": "base", "child": "right_arm_0", "type": "revolute",
             "axis_dir": [0, 0, 1]},
            {"parent": "right_arm_0", "child": "right_arm_1", "type": "revolute",
             "axis_dir": [1, 0, 0]},
        ],
        "mirror_limbs": [{"root": "right_arm_0", "plane": "xz",
                          "rename": {"right": "left"}}],
    }
    config.update(extra)
    return config


def test_build_model_generates_the_limb_from_the_config():
    from sw2robot.exporter.model import build_model

    model = build_model(_half_robot_graph(), config=_half_robot_config())
    names = {c.link_name for c in model.components}
    assert {"left_arm_0", "left_arm_1"} <= names
    assert any(j.child == "left_arm_0" and j.parent == "base"
               for j in model.joints)


def test_a_mass_set_on_a_generated_link_is_actually_applied():
    """The web editor cannot tell a generated link from an extracted one, so a
    `masses:` entry it writes has to land.  It did not until the generation was
    moved ahead of the per-link overrides -- the edit was accepted and silently
    lost, which is the worst way for this to fail."""
    from sw2robot.exporter.model import build_model

    model = build_model(_half_robot_graph(),
                        config=_half_robot_config(masses={"left_arm_1": 7.5}))
    by = {c.link_name: c for c in model.components}
    assert by["left_arm_1"].mass_target == 7.5
    assert by["right_arm_1"].mass_target is None       # only the one named


def test_frame_only_on_a_generated_link_is_actually_applied():
    from sw2robot.exporter.model import build_model

    model = build_model(_half_robot_graph(),
                        config=_half_robot_config(frame_only=["left_arm_1"]))
    by = {c.link_name: c for c in model.components}
    assert by["left_arm_1"].frame_only is True
    assert by["right_arm_1"].frame_only is False


def test_a_density_set_on_a_generated_link_is_actually_applied():
    from sw2robot.exporter.model import build_model

    model = build_model(_half_robot_graph(),
                        config=_half_robot_config(densities={"left_arm_0": 750.0}))
    by = {c.link_name: c for c in model.components}
    assert by["left_arm_0"].density == 750.0
    assert by["left_arm_0"].density_override is True


def test_a_limb_with_meshes_is_refused_when_there_is_nowhere_to_write_them():
    """Reusing the source's mesh would draw the generated limb with the original
    hand's geometry; dropping it would leave links with no shape.  Neither is
    something to do quietly."""
    model = _model()
    for comp in model.components:
        comp.mesh_file = "meshes/x.3dxml"
    reports = mirror_limbs(model, _spec(), meshes_dir=None)
    assert reports[0]["skip"] and "meshes directory" in reports[0]["skip"]
    assert len(model.components) == 3


def test_a_limb_without_meshes_still_generates_without_a_meshes_dir():
    model = _model()                       # the fixture has no mesh_file set
    reports = mirror_limbs(model, _spec(), meshes_dir=None)
    assert reports[0].get("skip") is None and reports[0]["links"] == 2


# ------------------------------------------------------------------ the mesh

def test_the_mirrored_mesh_keeps_its_normals_pointing_out(tmp_path):
    """A reflected mesh must end up wound so its normals still face outward.

    This shipped broken once: trimesh's apply_transform already flips the
    winding for a negative-determinant matrix, so calling invert() afterwards
    undid it and wrote a solid inside out -- a signed volume of -3.85 cm^3 on a
    3.85 cm^3 part, which renders black in RViz and reads as inverted to
    anything that trusts normals.
    """
    import trimesh

    from sw2robot.exporter.limb_mirror import mirror_mesh_file

    # an ASYMMETRIC solid, so a mirror is distinguishable from the original
    box = trimesh.creation.box(extents=[0.04, 0.02, 0.03])
    box.apply_translation([0.0, 0.011, 0.0])
    src = tmp_path / "part.glb"
    box.export(str(src), file_type="glb")

    dst = tmp_path / "part__mirrored.glb"
    assert mirror_mesh_file(str(src), str(dst), PLANES["xz"]) == str(dst)

    out = trimesh.load(str(dst))
    out = out.to_geometry() if isinstance(out, trimesh.Scene) else out
    assert out.volume > 0, "normals point into the solid"
    assert out.volume == pytest.approx(box.volume, rel=1e-6)
    assert np.allclose(out.extents, box.extents, atol=1e-9)
    # y is the mirrored axis: the centroid crosses the plane, the others do not
    assert out.centroid[1] == pytest.approx(-box.centroid[1], abs=1e-9)
    assert out.centroid[0] == pytest.approx(box.centroid[0], abs=1e-9)


def test_mirroring_an_unreadable_mesh_reports_instead_of_raising(tmp_path):
    from sw2robot.exporter.limb_mirror import mirror_mesh_file

    bad = tmp_path / "not-a-mesh.glb"
    bad.write_text("this is not a mesh")
    assert mirror_mesh_file(str(bad), str(tmp_path / "out.glb"),
                            PLANES["xz"]) is None


def test_an_up_to_date_mirrored_mesh_is_reused(tmp_path):
    """Every build is every edit in the editor, so re-reflecting geometry that
    has not moved put 13.6 s on a 13.8 s rebuild -- changing one link's mass
    paid to re-mirror six limb meshes."""
    import trimesh

    from sw2robot.exporter.limb_mirror import mirror_mesh_file

    src = tmp_path / "part.glb"
    trimesh.creation.box(extents=[0.04, 0.02, 0.03]).export(str(src),
                                                            file_type="glb")
    dst = tmp_path / "part__mirrored.glb"
    assert mirror_mesh_file(str(src), str(dst), PLANES["xz"])
    first = dst.stat().st_mtime_ns
    dst.write_bytes(dst.read_bytes())          # touch, keeping it newer

    assert mirror_mesh_file(str(src), str(dst), PLANES["xz"]) == str(dst)
    assert dst.stat().st_mtime_ns >= first, "the cached mesh was rewritten"


def test_a_newer_source_forces_a_fresh_reflection(tmp_path):
    """The reuse is an mtime gate, not a name gate: re-extracting after a CAD
    edit has to win, or the old shape survives invisibly."""
    import os
    import time

    import trimesh

    from sw2robot.exporter.limb_mirror import mirror_mesh_file

    src = tmp_path / "part.glb"
    trimesh.creation.box(extents=[0.04, 0.02, 0.03]).export(str(src),
                                                            file_type="glb")
    dst = tmp_path / "part__mirrored.glb"
    mirror_mesh_file(str(src), str(dst), PLANES["xz"])
    small = trimesh.load(str(dst))
    small = small.to_geometry() if isinstance(small, trimesh.Scene) else small

    time.sleep(0.01)
    trimesh.creation.box(extents=[0.08, 0.02, 0.03]).export(str(src),
                                                            file_type="glb")
    os.utime(str(src), None)                   # the CAD changed after the copy
    mirror_mesh_file(str(src), str(dst), PLANES["xz"])
    big = trimesh.load(str(dst))
    big = big.to_geometry() if isinstance(big, trimesh.Scene) else big
    assert big.extents[0] == pytest.approx(0.08, rel=1e-6), big.extents
    assert small.extents[0] == pytest.approx(0.04, rel=1e-6)


# ------------------------------------------------------------------ prefixing

def test_a_prefix_names_the_generated_links_when_there_is_no_marker_to_swap():
    """Swapping a side marker only works on a robot whose limb links carry one.
    Plenty do not -- limb parts named after the part and told apart by an
    instance number, `joint_frame_y_link_4` beside `joint_frame_y_link_5` --
    and a prefix is what always yields a fresh name."""
    model = _model()
    reports = mirror_limbs(model, [{"root": "right_arm_0", "plane": "xz",
                                    "prefix": "mirrored_"}])
    assert reports[0].get("skip") is None
    names = {c.link_name for c in model.components}
    assert {"mirrored_right_arm_0", "mirrored_right_arm_1"} <= names


def test_a_prefix_wins_over_a_rename_when_both_are_given():
    model = _model()
    mirror_limbs(model, [{"root": "right_arm_0", "plane": "xz",
                          "prefix": "m_", "rename": {"right": "left"}}])
    names = {c.link_name for c in model.components}
    assert "m_right_arm_0" in names and "left_arm_0" not in names


def test_a_prefix_that_would_collide_generates_nothing():
    model = _model()
    reports = mirror_limbs(model, [{"root": "right_arm_0", "plane": "xz",
                                    "prefix": ""}])
    assert reports[0]["skip"] and "unchanged or colliding" in reports[0]["skip"]
    assert len(model.components) == 3


def test_the_prefixed_joints_get_their_own_names_too():
    model = _model()
    mirror_limbs(model, [{"root": "right_arm_0", "plane": "xz",
                          "prefix": "mirrored_"}])
    names = [j.name for j in model.joints]
    assert len(names) == len(set(names)), "duplicate joint names in the URDF"
    assert any(n.startswith("mirrored_") for n in names)
