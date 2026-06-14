"""cad2rc -- the headless core bridging sw2urdf (CAD->module) and the ROS config
export.

Design (a standing directive):
- **State**: ``RobotCompilerState`` (Pydantic) is the single source of truth for
  a CAD-derived module being configured.  It serializes to JSON, so it moves to
  a REST/WebSocket payload unchanged.
- **Core API** (``core``): pure functions that mutate State / produce artifacts.
  A viser callback or a FastAPI endpoint is a *thin* wrapper over these.
- **Headless**: the whole pipeline (CAD build -> edit -> ROS export) runs from
  the CLI (``python -m cad2rc``) with no GUI and no SolidWorks.

It uses the *pure* halves: ``sw2urdf.export.build`` (no SolidWorks) and the
vendored ``cad2rc._vendor.rc_config`` (ROS/MoveIt/Gazebo config generators).
"""

from .state import JointEdit, RobotCompilerState
from . import core

__all__ = ["RobotCompilerState", "JointEdit", "core"]
