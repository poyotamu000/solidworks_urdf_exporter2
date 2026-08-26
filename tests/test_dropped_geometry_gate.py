"""Where the dropped-geometry check runs.

It re-decodes every mesh in the package, which on a humanoid is essentially the
whole build -- 46.7 s of a 48.1 s rebuild, 197 meshes -- while answering a
question only the extraction and the `expand:` config can change.  So the
interactive editor turns it off and the export path runs it instead.

The risk in that trade is losing the warning entirely, so these pin both halves:
the flag really skips (and says so), and the export path really runs it.
"""
import numpy as np
import yaml


def _eye():
    return list(np.eye(4).flatten())


def _pkg(tmp_path):
    from sw2robot.exporter.export import build
    from sw2robot.exporter.state import ComponentState, GraphState

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    GraphState(
        robot_name="demo", source_assembly="x.SLDASM",
        components=[
            ComponentState(name="base", link_name="base", world=_eye(),
                           fixed=True, material="ABS", density=1040.0,
                           sw_mass=1.0, sw_com=[0, 0, 0],
                           sw_inertia=[1, 0, 0, 1, 0, 1]),
            ComponentState(name="tip", link_name="tip", world=_eye(),
                           material="ABS", density=1040.0, sw_mass=0.2,
                           sw_com=[0, 0, 0], sw_inertia=[2, 0, 0, 2, 0, 2]),
        ]).save(str(pkg / "graph.json"))
    cfg = pkg / "demo.joints.yaml"
    cfg.write_text(yaml.safe_dump({
        "base": "base",
        "joints": [{"parent": "base", "child": "tip", "type": "fixed"}]}))
    build(str(pkg), config_path=str(cfg))
    return pkg, cfg


def test_the_flag_skips_the_check_and_says_so(tmp_path, capsys):
    from sw2robot.exporter.export import build

    pkg, cfg = _pkg(tmp_path)
    capsys.readouterr()
    build(str(pkg), config_path=str(cfg), check_geometry=False)
    out = capsys.readouterr().out
    assert "dropped-geometry check skipped" in out
    assert "runs on export" in out, "a silent skip is the thing to avoid"


def test_it_runs_by_default(tmp_path, capsys):
    """The CLI build keeps the check: `sw2urdf-build <pkg>` is where someone
    goes to find out whether the package is whole."""
    from sw2robot.exporter.export import build

    pkg, cfg = _pkg(tmp_path)
    capsys.readouterr()
    build(str(pkg), config_path=str(cfg))
    assert "runs on export" not in capsys.readouterr().out


def test_every_editor_rebuild_passes_the_flag():
    """The editor has seventeen rebuild call sites; one left on the default
    would put the 46-second check back on that one edit and nobody would know
    which."""
    import re
    from pathlib import Path

    src = Path("sw2robot/editor/webserver.py").read_text(encoding="utf-8")
    calls = re.findall(r"build\(cls\.pkg_dir, config_path=yml[^)]*\)", src)
    assert calls, "the rebuild call shape changed -- update this test"
    missing = [c for c in calls if "check_geometry=False" not in c]
    assert not missing, missing


def test_the_export_path_runs_it(tmp_path, monkeypatch):
    """Skipping it during editing is only acceptable because shipping a package
    still checks."""
    from sw2robot.editor import webserver

    pkg, _cfg = _pkg(tmp_path)
    called = {}
    monkeypatch.setattr(webserver, "_warn_dropped_geometry_on_export",
                        lambda p: called.setdefault("pkg", p))
    try:
        webserver._export_zip(str(pkg), "demo")
    except Exception:
        pass                     # the zip itself may need optional deps
    assert called.get("pkg") == str(pkg)


def test_the_export_check_survives_a_broken_package(tmp_path, capsys):
    """Advisory means advisory: an export must not fail because the check
    could not run."""
    from sw2robot.editor.webserver import _warn_dropped_geometry_on_export

    empty = tmp_path / "empty"
    empty.mkdir()
    _warn_dropped_geometry_on_export(str(empty))     # no graph.json
    assert "Traceback" not in capsys.readouterr().out
