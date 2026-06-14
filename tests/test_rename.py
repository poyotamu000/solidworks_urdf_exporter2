"""Rename overlay: write_urdf applies link/joint name overrides (refs follow,
names sanitised) and the webserver's joints.yaml helpers round-trip."""
import os
import xml.etree.ElementTree as ET

import numpy as np


def _model():
    from sw2robot.exporter.model import Component, Joint, RobotModel
    comps = [
        Component(name="Base-1", link_name="base_link", part_path=None,
                  is_subassembly=False, world=np.eye(4), fixed=True, dof=0),
        Component(name="Arm-1", link_name="arm_link", part_path=None,
                  is_subassembly=False, world=np.eye(4), fixed=False, dof=1),
    ]
    joints = [Joint(name="base_link__arm_link", parent="base_link",
                    child="arm_link", jtype="revolute", axis=[0, 0, 1],
                    lower=-1, upper=1)]
    return RobotModel(name="demo", components=comps, joints=joints,
                      base_link="base_link")


def test_link_and_joint_overrides_applied_with_refs(tmp_path):
    from sw2robot.exporter import urdf_writer
    out = tmp_path / "urdf" / "demo.urdf"
    urdf_writer.write_urdf(
        _model(), str(out),
        link_overrides={"arm_link": "distal", "base_link": "ignored_root"},
        joint_overrides={"base_link__arm_link": "hinge-1"})   # dirty on purpose
    root = ET.parse(out).getroot()

    links = {l.get("name") for l in root.findall("link")}
    assert links == {"base_link", "distal"}     # root_link_name wins
    j = root.find("joint")
    assert j.get("name") == "hinge_1"           # sanitised (hyphen -> _)
    assert j.find("parent").get("link") == "base_link"  # refs follow
    assert j.find("child").get("link") == "distal"


def test_no_overrides_is_unchanged(tmp_path):
    from sw2robot.exporter import urdf_writer
    a = tmp_path / "a" / "demo.urdf"
    b = tmp_path / "b" / "demo.urdf"
    urdf_writer.write_urdf(_model(), str(a))
    urdf_writer.write_urdf(_model(), str(b), link_overrides={}, joint_overrides={})
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
    names = {l.get("name") for l in ET.parse(a).getroot().findall("link")}
    assert names == {"base_link", "arm_link"}


def test_webserver_yaml_helpers():
    from sw2robot.editor import webserver as w
    txt = ("base: foo\njoints:\n  - parent: a\n    child:  b\n"
           "    type:   fixed\n")
    t2 = w._upsert_yaml_map(txt, "link_names", "b", "distal")
    assert w._names_inverse(t2, "link_names") == {"distal": "b"}
    # updating the same key replaces, does not duplicate
    t3 = w._upsert_yaml_map(t2, "link_names", "b", "distal2")
    assert w._names_inverse(t3, "link_names") == {"distal2": "b"}
    # the root's display name (root_link_name) reverse-maps to the base component
    root_yaml = ("base: rootcomp\nroot_link_name: root_disp\n"
                 "link_names:\n  b: distal2\n"
                 "joints:\n  - parent: a\n    child:  b\n    type: fixed\n")
    inv = w._link_names_inverse(root_yaml)
    assert inv["root_disp"] == "rootcomp"
    assert inv["distal2"] == "b"
    assert t3.count("b:") == 1 + txt.count("b:")  # one map entry (+ child: b line)
    # joints list survives the prepended map
    assert "child:  b" in t3 and "base: foo" in t3
    assert "root_link_name: r" in w._set_root_link_name(txt, "r")
