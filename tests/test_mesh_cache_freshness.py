"""Per-part mesh cache invalidation (_cache_is_fresh).

export_meshes reuses meshes/<part>.3dxml verbatim across re-extractions to keep
re-runs fast.  Keyed on the part NAME alone, that reuse used to survive a CAD
edit -- the edited part re-exports to the same filename the stale mesh already
occupies, so the change was masked until the user wiped %TEMP%\\sw2robot\\output
by hand (the reported bug).  _cache_is_fresh gates the reuse on mtime so an edit
since the last extract forces a fresh export.
"""
import os

from sw2robot.exporter.mesh import _MIN_MESH_BYTES, _cache_is_fresh


def _write(path, size=_MIN_MESH_BYTES, mtime=None):
    path.write_bytes(b"x" * size)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_cache_reused_when_newer_than_source(tmp_path):
    src = _write(tmp_path / "part.SLDPRT", mtime=1000)
    cache = _write(tmp_path / "part.3dxml", mtime=2000)
    assert _cache_is_fresh(str(cache), str(src)) is True


def test_stale_cache_forces_reexport(tmp_path):
    # the bug: part edited AFTER the mesh was cached -> must not be reused
    cache = _write(tmp_path / "part.3dxml", mtime=1000)
    src = _write(tmp_path / "part.SLDPRT", mtime=2000)
    assert _cache_is_fresh(str(cache), str(src)) is False


def test_equal_mtime_is_reusable(tmp_path):
    src = _write(tmp_path / "part.SLDPRT", mtime=1500)
    cache = _write(tmp_path / "part.3dxml", mtime=1500)
    assert _cache_is_fresh(str(cache), str(src)) is True


def test_envelope_sized_cache_never_reused(tmp_path):
    # a sub-_MIN_MESH_BYTES file is just the empty-document envelope
    src = _write(tmp_path / "part.SLDPRT", mtime=1000)
    cache = _write(tmp_path / "part.3dxml", size=_MIN_MESH_BYTES - 1, mtime=2000)
    assert _cache_is_fresh(str(cache), str(src)) is False


def test_missing_cache_not_reused(tmp_path):
    src = _write(tmp_path / "part.SLDPRT", mtime=1000)
    assert _cache_is_fresh(str(tmp_path / "absent.3dxml"), str(src)) is False


def test_unreadable_source_falls_back_to_reuse(tmp_path):
    # part moved/renamed away since extraction: we cannot re-export it, so the
    # cache (all we have) is reused rather than dropped
    cache = _write(tmp_path / "part.3dxml", mtime=1000)
    assert _cache_is_fresh(str(cache), str(tmp_path / "gone.SLDPRT")) is True
