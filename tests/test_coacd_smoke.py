"""Real-CoACD smoke test: actually run the ``coacd`` wheel (no stubbing) on a
tiny concave mesh.  Unlike the other CoACD tests (which stub the slow C call),
this proves the optional ``coacd`` package installs AND runs on the host OS --
the CI signal that catches a missing/broken wheel on Linux/macOS.

Skipped where ``coacd`` is not installed, so it is a no-op for the default
(coacd-less) environment and only bites in the CI job that installs ``[coacd]``.
"""
import pytest

from sw2robot.exporter.ros_export import _run_coacd, coacd_available

if not coacd_available():
    pytest.skip("coacd not installed", allow_module_level=True)


def _l_shape():
    """A small NON-convex L-prism (two overlapping boxes) -- something CoACD has
    a reason to split, kept tiny so the real run is a couple of seconds."""
    import trimesh

    a = trimesh.creation.box(extents=(2, 1, 1))
    b = trimesh.creation.box(extents=(1, 2, 1))
    b.apply_translation((0.5, 0.5, 0))
    return trimesh.util.concatenate([a, b])


def test_real_coacd_runs_and_decomposes():
    import numpy as np

    mesh = _l_shape()
    # coarse + cheap params: this is an "it runs on this OS" check, not a
    # quality check, so keep the MCTS search small
    parts = _run_coacd(
        np.asarray(mesh.vertices), np.asarray(mesh.faces),
        {"threshold": 0.2, "max_convex_hull": 4,
         "preprocess_resolution": 20, "mcts_iterations": 20})

    assert len(parts) >= 1                       # produced at least one hull
    for verts, faces in parts:
        v, f = np.asarray(verts), np.asarray(faces)
        assert v.ndim == 2 and v.shape[1] == 3 and len(v) >= 4
        assert f.ndim == 2 and f.shape[1] == 3 and len(f) >= 4
