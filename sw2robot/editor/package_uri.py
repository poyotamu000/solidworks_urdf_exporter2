"""Resolve and serve ROS ``package://`` mesh references for the web editor.

Package *resolution* is delegated to scikit-robot (>= 0.3.16), whose
``get_path_with_cache`` resolves from ``ament_index`` / ``rospkg`` AND the
sourced ROS search-path environment variables -- so it also works in the
standalone binary, which bundles neither ROS Python package.

What stays here is the editor's own concern: mapping a ``package://`` URI to a
file so the server can hand the mesh bytes to the browser (which cannot read the
filesystem), preferring the opened package's own folder first.
"""

from __future__ import annotations

import os
import re
import urllib.parse

_PACKAGE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def find_package(package):
    """Return the filesystem root of a sourced ROS package, or ``None``.

    Delegates to scikit-robot, which resolves via ``ament_index`` / ``rospkg``
    and the ROS search-path environment variables (``ROS_PACKAGE_PATH`` /
    ``AMENT_PREFIX_PATH`` / ``COLCON_PREFIX_PATH`` / ``CMAKE_PREFIX_PATH``) and
    caches the result.
    """
    if not _PACKAGE_NAME.fullmatch(package or ""):
        return None
    try:
        from skrobot.utils.urdf import get_path_with_cache
        return get_path_with_cache(package)
    except Exception:
        # get_path_with_cache raises ImportError (no resolver installed) or
        # LookupError (package not found); either way we have no path.
        return None


def _inside(root, relative):
    root = os.path.normpath(root)
    candidate = os.path.normpath(os.path.join(root, *relative.split("/")))
    try:
        if os.path.commonpath([root, candidate]) != root:
            return None
    except ValueError:
        return None
    return candidate


def split_package_uri(uri):
    """Return ``(package, relative_path)`` for a safe package URI."""
    parsed = urllib.parse.urlsplit(uri)
    package = parsed.netloc
    relative = urllib.parse.unquote(parsed.path).lstrip("/").replace("\\", "/")
    if parsed.scheme != "package" or not _PACKAGE_NAME.fullmatch(package or ""):
        return None
    if not relative or relative.startswith("../") or "/../" in relative:
        return None
    return package, relative


def resolve_package_uri(uri, current_package=None):
    """Resolve a package URI to an existing file.

    ``current_package/<uri path>`` is intentionally checked first.  That is the
    editor's historical behavior and supports exported packages whose folder
    name differs from the package name embedded in the URDF.  The ROS lookup
    (via scikit-robot) is an additive fallback only.
    """
    parts = split_package_uri(uri)
    if parts is None:
        return None
    package, relative = parts
    if current_package:
        local = _inside(current_package, relative)
        if local and os.path.isfile(local):
            return local
    root = find_package(package)
    external = _inside(root, relative) if root else None
    return external if external and os.path.isfile(external) else None
