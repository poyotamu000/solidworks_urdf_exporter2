"""The editor side of ``mirror_limbs:``: generating the other half of a robot
from a link panel, and showing a generated link for what it is.

Drives a real in-process HTTP server against a synthetic half-robot package (a
torso plus one arm), because the parts worth testing are the wiring: does the
config the endpoint writes actually produce links, does a refusal roll back
instead of reporting a success that changed nothing, and does a link with no
graph.json component behind it still come back describable.
"""
import json
import threading
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml


def _eye():
    return list(np.eye(4).flatten())


def _free_port():
    import socket
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _get_json(base, path):
    with urllib.request.urlopen(base + path) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(base, path, body):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _urdf_links(pkg):
    root = ET.parse(str(pkg / "urdf" / "demo.urdf")).getroot()
    return {ln.get("name") for ln in root.findall("link")}


@pytest.fixture
def server(tmp_path):
    from sw2robot.editor import webserver
    from sw2robot.exporter.export import build
    from sw2robot.exporter.state import ComponentState, GraphState

    pkg = tmp_path / "pkg"
    pkg.mkdir()

    def state(name, mass):
        return ComponentState(name=name, link_name=name, world=_eye(),
                              material="ABS", density=1040.0, sw_mass=mass,
                              sw_com=[0.01, 0.0, 0.0],
                              sw_inertia=[1, 0, 0, 1, 0, 1])

    GraphState(
        robot_name="demo", source_assembly="x.SLDASM",
        components=[state("base", 5.0), state("right_arm_0", 1.0),
                    state("right_arm_1", 2.0)]).save(str(pkg / "graph.json"))
    cfg = pkg / "demo.joints.yaml"
    cfg.write_text(yaml.safe_dump({
        "base": "base",
        "joints": [
            {"parent": "base", "child": "right_arm_0", "type": "revolute",
             "axis_dir": [0, 0, 1]},
            {"parent": "right_arm_0", "child": "right_arm_1", "type": "revolute",
             "axis_dir": [1, 0, 0]},
        ]}))
    build(str(pkg), config_path=str(cfg))

    httpd, port = webserver._bind_free_port(webserver._Handler, _free_port())
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        assert _get_json(base, f"/api/open?path={pkg}")["mode"] == "cad"
        yield base, pkg
    finally:
        httpd.shutdown()
        httpd.server_close()
        webserver._um["state"] = None


def _cfg(pkg):
    return yaml.safe_load((pkg / "demo.joints.yaml").read_text()) or {}


# ------------------------------------------------------------------ generating

def test_generating_a_limb_writes_the_config_and_builds_the_links(server):
    base, pkg = server
    assert "left_arm_0" not in _urdf_links(pkg)

    code, r = _post(base, "/api/set_mirror_limb",
                    {"link": "right_arm_0", "on": True, "plane": "xz",
                     "rename_from": "right", "rename_to": "left"})
    assert code == 200 and r["ok"] is True

    assert _cfg(pkg)["mirror_limbs"] == [
        {"root": "right_arm_0", "plane": "xz", "rename": {"right": "left"}}]
    assert {"left_arm_0", "left_arm_1"} <= _urdf_links(pkg)


def test_the_generated_links_report_where_they_came_from(server):
    """A generated link has no graph.json component, so every per-link lookup in
    the editor comes back empty and it reads as a broken extraction.  Its
    provenance has to survive the build -- it does, through the URDF comment."""
    base, _pkg = server
    _post(base, "/api/set_mirror_limb",
          {"link": "right_arm_0", "on": True, "plane": "xz",
           "rename_from": "right", "rename_to": "left"})
    links = _get_json(base, "/api/components")["links"]
    assert links["left_arm_1"]["mirrored_from"] == "right_arm_1"
    assert links["right_arm_1"]["mirrored_from"] is None
    # and it is NOT reported as an unreviewed default mass: its weight is the
    # twin's real one, so the export gate must not stop on it
    assert "left_arm_1" not in _get_json(
        base, "/api/components")["default_mass_links"]


def test_the_configured_spec_comes_back_so_the_panel_can_offer_undo(server):
    base, _pkg = server
    _post(base, "/api/set_mirror_limb",
          {"link": "right_arm_0", "on": True, "plane": "xz",
           "rename_from": "right", "rename_to": "left"})
    specs = _get_json(base, "/api/components")["mirror_limbs"]
    assert [s["root"] for s in specs] == ["right_arm_0"]


def test_removing_it_takes_the_links_back_out(server):
    base, pkg = server
    _post(base, "/api/set_mirror_limb",
          {"link": "right_arm_0", "on": True, "plane": "xz",
           "rename_from": "right", "rename_to": "left"})
    code, r = _post(base, "/api/set_mirror_limb",
                    {"link": "right_arm_0", "on": False})
    assert code == 200 and r["ok"] is True
    assert "mirror_limbs" not in _cfg(pkg)
    assert "left_arm_0" not in _urdf_links(pkg)


# -------------------------------------------------------------------- refusals

def test_a_rename_that_generates_nothing_is_an_error_not_a_green_toast(server):
    """The build is the only thing that knows a rename collides.  If it refuses,
    the endpoint has to roll the config back and say why -- reporting success
    over a robot that did not change is the failure this guards."""
    base, pkg = server
    code, r = _post(base, "/api/set_mirror_limb",
                    {"link": "right_arm_0", "on": True, "plane": "xz",
                     "rename_from": "right", "rename_to": "right"})
    assert code == 400 and "error" in r
    assert "mirror_limbs" not in _cfg(pkg)          # rolled back

    # one that passes the endpoint's own checks but the BUILD refuses
    code, r = _post(base, "/api/set_mirror_limb",
                    {"link": "right_arm_0", "on": True, "plane": "xz",
                     "rename_from": "right_arm_0", "rename_to": "base"})
    assert code == 400
    assert "colliding" in r["error"] or "unchanged" in r["error"]
    assert "mirror_limbs" not in _cfg(pkg)
    assert _urdf_links(pkg) == {"base_link", "right_arm_0", "right_arm_1"}


def test_an_unknown_plane_is_refused_and_rolled_back(server):
    base, pkg = server
    code, r = _post(base, "/api/set_mirror_limb",
                    {"link": "right_arm_0", "on": True, "plane": "sideways",
                     "rename_from": "right", "rename_to": "left"})
    assert code == 400 and "plane" in r["error"]
    assert "mirror_limbs" not in _cfg(pkg)


def test_mirroring_the_same_limb_twice_is_refused(server):
    base, _pkg = server
    _post(base, "/api/set_mirror_limb",
          {"link": "right_arm_0", "on": True, "plane": "xz",
           "rename_from": "right", "rename_to": "left"})
    code, r = _post(base, "/api/set_mirror_limb",
                    {"link": "right_arm_0", "on": True, "plane": "xz",
                     "rename_from": "right", "rename_to": "other"})
    assert code == 400 and "already" in r["error"]


def test_removing_one_that_was_never_configured_is_refused(server):
    base, _pkg = server
    code, r = _post(base, "/api/set_mirror_limb",
                    {"link": "right_arm_0", "on": False})
    assert code == 400 and "does not mirror" in r["error"]


def test_a_rename_that_does_not_match_the_link_is_refused(server):
    """Catching it here rather than at build time means the message can name the
    link, instead of "3 names unchanged"."""
    base, _pkg = server
    code, r = _post(base, "/api/set_mirror_limb",
                    {"link": "right_arm_0", "on": True, "plane": "xz",
                     "rename_from": "left_leg", "rename_to": "right_leg"})
    assert code == 400 and "does not appear" in r["error"]


def test_a_prefix_is_accepted_when_there_is_no_side_marker(server):
    """The panel offers a prefix box for a link with no leading R/L, so the
    endpoint has to take one -- otherwise those robots cannot use the feature
    at all."""
    base, pkg = server
    code, r = _post(base, "/api/set_mirror_limb",
                    {"link": "right_arm_0", "on": True, "plane": "yz",
                     "prefix": "mirrored_"})
    assert code == 200 and r["prefix"] == "mirrored_"
    assert _cfg(pkg)["mirror_limbs"] == [
        {"root": "right_arm_0", "plane": "yz", "prefix": "mirrored_"}]
    assert {"mirrored_right_arm_0", "mirrored_right_arm_1"} <= _urdf_links(pkg)


def test_neither_a_prefix_nor_a_rename_is_refused(server):
    base, _pkg = server
    code, r = _post(base, "/api/set_mirror_limb",
                    {"link": "right_arm_0", "on": True, "plane": "xz"})
    assert code == 400 and "prefix" in r["error"]
