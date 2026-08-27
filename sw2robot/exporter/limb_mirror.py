"""Generate the other half of a symmetric robot from the half that exists in CAD.

Modelling both arms and both legs is duplicated work, and keeping the two copies
in step through a redesign is worse.  Given an assembly that holds the torso plus
ONE arm and ONE leg, this builds the missing side by reflecting each limb through
the robot's sagittal plane -- links, joints, inertials and meshes -- so the URDF
comes out whole while the CAD stays half the size.

Nothing here touches SolidWorks: it runs at BUILD time on the finished
``RobotModel``, which means it also works on a package rebuilt from a cached
``graph.json``, and a mirrored limb costs nothing to regenerate after the real
side changes.

The frames
----------
Reflecting a link's world pose alone does not give a usable frame: ``M @ W`` has
determinant -1, i.e. it is left-handed, and no ``rpy`` describes it.  Every link
therefore gets a LOCAL reflection ``S`` as well::

    anchor' = M @ anchor @ S

which is proper again (``det = (-1)(+1)(-1) = +1``).  ``S`` is a gauge choice --
any improper orthogonal matrix restores handedness -- and taking the same
diagonal as ``M`` keeps each mirrored frame pointing where a reader expects.
Because ``S`` reflects the link's OWN axes, everything expressed in that frame
has to be reflected with it: the inertial, the visual origin, and the mesh.

The joint-axis sign
-------------------
This is the one place a mirror generator usually goes wrong.  For an improper
``Q``, ``Q R(a, θ) Qᵀ = R(-Q a, θ)`` -- the extra minus is what makes a rotation
axis an *axial* vector.  Working the frames through gives, for the axis as URDF
stores it (in the child frame):

* **revolute / continuous**: ``a' = -S @ a``
* **prismatic**: ``a' = S @ a`` (a translation axis is an ordinary vector, no
  sign flip)

With those, joint limits carry over UNCHANGED and the two sides share one
convention: driving the left and right joint to the same ``q`` puts the robot in
a mirror-symmetric pose.  Get the revolute sign wrong instead and the generated
limb still looks right at zero and bends backwards everywhere else -- which is
why :func:`mirror_axis` is pinned by its own tests.

What this cannot do
-------------------
The generated side is a perfect mirror by construction, so it is only as true as
the robot's symmetry.  A cable tray on one side, a gripper that exists only on
the right, a connector facing one way -- none of that is knowable from the half
that was modelled, and it is not invented here.  Every generated link is named in
the build log and carries ``mirrored_from``, so nothing looks like it came from
CAD when it did not.
"""

from __future__ import annotations

import os

import numpy as np
from skrobot.coordinates.math import rpy2homogeneous

from .geometry import matrix_to_xyz_rpy
from .mirror import reflect_inertial

# Which base_link plane the limbs are reflected through.  A humanoid's sagittal
# plane is the XZ one (y -> -y) for the usual x-forward / z-up convention, so
# that is the default; the others are here because "the robot's own convention"
# is not something to assume.
PLANES = {
    "xz": (1.0, -1.0, 1.0),
    "yz": (-1.0, 1.0, 1.0),
    "xy": (1.0, 1.0, -1.0),
}
DEFAULT_PLANE = "xz"

# Joint types whose axis is an AXIAL vector (see the module docstring).
_AXIAL_JOINTS = ("revolute", "continuous")


def _reflection(signs):
    """4x4 homogeneous reflection from a per-axis sign triple."""
    M = np.eye(4)
    M[:3, :3] = np.diag(np.asarray(signs, float))
    return M


def origin_matrix(xyz, rpy):
    """A URDF ``<origin>`` as a 4x4."""
    M = np.asarray(rpy2homogeneous(*[float(v) for v in (rpy or [0, 0, 0])]),
                   float).copy()
    M[:3, 3] = [float(v) for v in (xyz or [0, 0, 0])]
    return M


def link_anchors(joints, root):
    """``{link name -> 4x4 in the root's frame}`` from the joint tree.

    The model stores each joint's origin relative to its parent, so the absolute
    frames are recovered by composing down the tree.  A link whose parent has not
    been placed yet is deferred rather than skipped, so joint ORDER in the model
    does not matter.  Links unreachable from ``root`` are simply absent.
    """
    anchors = {root: np.eye(4)}
    pending = list(joints)
    while pending:
        progressed = False
        still = []
        for joint in pending:
            parent = anchors.get(joint.parent)
            if parent is None:
                still.append(joint)
                continue
            anchors[joint.child] = parent @ origin_matrix(joint.xyz, joint.rpy)
            progressed = True
        pending = still
        if not progressed:
            break
    return anchors


def subtree_links(joints, root):
    """``root`` plus every link below it, in breadth-first order.

    Breadth-first matters downstream: a parent is always emitted before its
    children, so the generated joints can be appended in a valid order.
    """
    children = {}
    for joint in joints:
        children.setdefault(joint.parent, []).append(joint.child)
    out, queue, seen = [], [root], {root}
    while queue:
        link = queue.pop(0)
        out.append(link)
        for child in children.get(link, ()):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return out


def _driver(parent_of, link):
    """The nearest movable joint at or above ``link``, or None."""
    while link in parent_of:
        up = parent_of[link]
        if up.jtype != "fixed":
            return up
        link = up.parent
    return None


def shared_actuator(joints, limb_root, attach_to=None):
    """The joint that would drive BOTH the limb and its mirrored copy, or None.

    ``mirror_limbs`` hangs the copy off the limb root's own parent, which is
    right when that parent is rigidly attached to the base -- and a silent trap
    when it is not.  A quadruped keeps its hip servos in the torso, so rooting
    the mirror at the leg gives the copy no hip of its own: it bolts onto the
    same horn as the original and the one hip joint swings both legs.  (The
    same sharing is CORRECT for a pair of gripper jaws on one wrist, so this is
    reported, not refused.)

    ``attach_to`` is the spec's own attachment override.  Hanging the copy on
    the OTHER side's existing mount is the way out of the trap without cloning
    that mount: the copy is then driven by the far side's own joint, which is
    not shared with anything, so nothing is reported.

    Returns ``{"joint", "suggest"}`` -- the shared joint, and the link to root
    the mirror at instead so the copy brings its own actuator -- or None.
    """
    parent_of = {j.child: j for j in joints}
    joint = parent_of.get(limb_root)
    if joint is None:
        return None
    mine = _driver(parent_of, attach_to or joint.parent)
    if mine is None:
        return None                      # rigidly grounded: nothing is shared
    # shared only when that SAME joint also drives the limb being copied
    if _driver(parent_of, joint.parent) is not mine:
        return None
    return {"joint": mine.name, "suggest": mine.child}


def suggest_attach(poses, parent_of, limb_root, plane, tol=0.002):
    """The far side's EXISTING mount for ``limb_root``, or None.

    Robots that get mirrored are symmetric, so the mount the copy belongs on is
    usually already modelled -- a quadruped's torso carries all four hip servos
    even when two legs are missing.  Where that mount is, is not a guess: it is
    wherever the source's own parent lands when reflected.  Look for a link
    already sitting there, and it is the ``attach_to:`` the user would have
    typed.

    ``poses`` maps link name -> 4x4 world pose, ``parent_of`` child -> parent.
    Position only: the counterpart's ORIENTATION follows whatever convention the
    CAD used to place that side (this quad's legs are rotated copies, not
    reflections), so requiring the rotation to match too would reject the very
    mounts this is meant to find.

    Returns None rather than a doubtful answer -- when nothing sits at the
    reflected point, when more than one link does, or when the source's own
    parent is the match (a parent ON the plane is its own reflection, and the
    default attachment is already right).
    """
    signs = PLANES.get(str(plane or "").lower())
    parent = parent_of.get(limb_root)
    if signs is None or parent is None or parent not in poses:
        return None
    kids = {}
    for child, par in parent_of.items():
        kids.setdefault(par, []).append(child)
    subtree, queue = set(), [limb_root]
    while queue:
        n = queue.pop()
        if n in subtree:
            continue
        subtree.add(n)
        queue.extend(kids.get(n, ()))

    target = np.asarray(poses[parent], dtype=float)[:3, 3] * np.asarray(signs)
    near = [n for n, T in poses.items()
            if n not in subtree
            and float(np.linalg.norm(np.asarray(T, dtype=float)[:3, 3]
                                     - target)) <= tol]
    if len(near) != 1:
        return None                  # nothing there, or ambiguous -- don't guess
    return None if near[0] == parent else near[0]


def mirror_axis(axis, signs, jtype):
    """A joint axis, reflected, in the mirrored child's frame.

    A rotation axis picks up the extra minus sign that makes it an axial vector;
    a prismatic axis does not.  See the module docstring -- this is the sign that
    decides whether the generated limb bends with its twin or against it.
    """
    if axis is None:
        return None
    reflected = np.asarray(signs, float) * np.asarray(axis, float)
    if jtype in _AXIAL_JOINTS:
        reflected = -reflected
    return [round(float(v), 8) for v in reflected]


def mirror_mesh_file(src_path, dst_path, signs):
    """Write ``src_path`` reflected through the plane ``signs`` as a ``.glb``.

    A reflection turns every triangle inside out, so the winding has to end up
    flipped -- a mesh whose normals point into the solid renders black in RViz
    and reads as inside-out to anything that trusts them.

    Do NOT simply call ``invert()`` here: trimesh's ``apply_transform`` already
    flips the winding itself when the matrix determinant is negative, so an
    unconditional inversion undoes its work and ships exactly the broken mesh it
    was meant to prevent.  (That is not a hypothetical -- it shipped, and showed
    up as a signed volume of -3.85 cm^3 on a 3.85 cm^3 part.)  Check the signed
    volume instead and only correct it when it really is negative, which holds
    whichever way a future trimesh decides to behave.

    The output is glb rather than the source format because trimesh cannot write
    the 3DXML the SolidWorks step produces, and because glb is what the rest of
    this package already treats as "geometry, in metres".  Referencing the
    original file with a negative ``<mesh scale>`` instead was rejected: it is
    legal XML that several consumers silently mis-handle, which is exactly the
    kind of half-working that is worse than an extra file.

    An existing output at least as new as its source is REUSED.  Without that
    the reflection re-runs on every build, and every build is every edit in the
    editor: six limb meshes cost 13.6 s of a 13.8 s rebuild, so changing one
    link's mass paid for re-mirroring geometry that had not moved.  The mtime
    gate is the same one the per-part mesh cache uses (mesh._cache_is_fresh), so
    editing the CAD and re-extracting still forces a fresh reflection.

    Returns the path written or reused, or None when the source could not be
    loaded.
    """
    from .mesh import _cache_is_fresh
    from .ros_export import _load_mesh_metres

    if _cache_is_fresh(dst_path, src_path):
        return dst_path
    try:
        mesh = _load_mesh_metres(src_path)
    except Exception as e:
        print(f"      WARN: cannot mirror mesh {os.path.basename(src_path)}: "
              f"{e!r}")
        return None
    mesh.apply_transform(_reflection(signs))
    try:
        if mesh.is_volume and mesh.volume < 0:
            mesh.invert()
    except Exception:                  # open / non-watertight: nothing to judge
        pass
    mesh.units = "meter"
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    mesh.export(dst_path, file_type="glb")
    return dst_path


# Reflecting a limb's meshes is the only expensive part of this module, and the
# one part that never touches shared state.  Two or more of them go to a process
# POOL for the same reason mesh._verify_parallel does: the 3DXML loader is pure
# Python, so threads only fight over the GIL.  Six leg meshes took 64.7 s the
# first time a limb was generated; after that the mtime cache makes it free.
# Gate the pool on BYTES, not on how many files there are: the work is
# proportional to the geometry decoded, and spawning twelve interpreters to
# reflect an arm's three small meshes (1.5 MB total) turned 1.6 s into 2.8 s,
# while the leg's six (18.9 MB) went 64.7 s -> 8.6 s.  4 MB sits between the two
# with room on either side.
_MIRROR_POOL_MIN_BYTES = 4 * 1024 * 1024
_MIRROR_POOL_MAX_WORKERS = 8


def _mirror_one(job):
    """``((src, dst, signs)) -> (dst, written_or_None)``; a module-level
    function so a process pool can pickle it."""
    src, dst, signs = job
    return dst, mirror_mesh_file(src, dst, signs)


def _mirror_meshes(jobs):
    """Reflect several meshes, in parallel when that is possible and worth it.

    Returns ``{dst: written path or None}``.  Falls back to serial on any pool
    failure rather than raising: a machine that cannot spawn -- a sandbox, a
    frozen build without ``freeze_support()`` -- must still be able to build.
    """
    jobs = list(jobs)
    if not jobs:
        return {}
    workers = min(os.cpu_count() or 1, _MIRROR_POOL_MAX_WORKERS, len(jobs))
    try:
        sizes = {j[0]: os.path.getsize(j[0]) for j in jobs}
    except OSError:
        sizes = {}
    if (len(jobs) >= 2 and workers >= 2
            and sum(sizes.values()) >= _MIRROR_POOL_MIN_BYTES):
        # biggest first: one 10 s mesh among six decides whether the run ends
        # at 10 s or at 10 s plus everything else
        ordered = sorted(jobs, key=lambda j: -sizes.get(j[0], 0))
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as pool:
                return dict(pool.map(_mirror_one, ordered))
        except Exception as e:
            print(f"      note: mirroring meshes fell back to one core ({e!r})")
    return dict(_mirror_one(j) for j in jobs)


def _rename(name, rules, prefix=None):
    """The generated link's name: a prefix, or a substring substitution.

    Substituting a side marker (``right`` -> ``left``) reads best, but it only
    works when the limb's links actually carry one.  On a robot whose limb parts
    are named after the part and told apart by an instance number --
    ``joint_frame_y_link_4`` next to ``joint_frame_y_link_5`` -- there is no
    shared substring to swap, and every such robot would be locked out.  A
    prefix always yields a fresh name, so it is the fallback that always works.
    """
    if prefix:
        return f"{prefix}{name}"
    for src, dst in (rules or {}).items():
        if src in name:
            return name.replace(src, dst, 1)
    return name


def _mirrored_component(comp, source_link, new_name, signs, mesh_file):
    """A copy of ``comp`` with everything in its own frame reflected."""
    import copy as _copy

    out = _copy.copy(comp)
    out.name = f"{comp.name}__mirrored"
    out.link_name = new_name
    out.mesh_file = mesh_file
    out.mirrored_from = source_link
    S = _reflection(signs)
    visual = S @ origin_matrix(comp.visual_xyz, comp.visual_rpy) @ S
    out.visual_xyz, out.visual_rpy = matrix_to_xyz_rpy(visual)
    if comp.sw_com is not None and comp.sw_inertia is not None:
        _mass, com, inertia = reflect_inertial(
            comp.sw_mass or 0.0, comp.sw_com, comp.sw_inertia, signs)
        out.sw_com, out.sw_inertia = com, list(inertia)
    if isinstance(getattr(comp, "inertial_override", None), dict):
        ov = comp.inertial_override
        _mass, com, inertia = reflect_inertial(
            ov["mass"], ov["com"], ov["inertia"], signs)
        out.inertial_override = {"mass": ov["mass"], "com": com,
                                 "inertia": inertia}
    return out


def _sharing_to_report(joints, anchors, limb_root, plane, attach_to):
    """The shared-actuator warning, or None when there is nothing to say.

    Sharing the parent's joint is only a MISTAKE when the model already holds
    the mount the copy should have used -- a quadruped's spare hip servo.  Two
    arms on one chest joint, or two jaws on one wrist, share by design and there
    is no second mount anywhere: warning about those would fire on every
    humanoid arm ever mirrored, which is the main thing this feature is for.

    So: report only when the limb is driven by its twin's joint AND a free
    counterpart mount exists.  Then the advice is concrete -- use that mount.
    """
    sh = shared_actuator(joints, limb_root, attach_to)
    if sh is None:
        return None
    parent_of = {j.child: j.parent for j in joints}
    mount = suggest_attach(anchors, parent_of, limb_root, plane)
    if mount is None:
        return None                  # nowhere better to put it: stay quiet
    return {"joint": sh["joint"], "attach_to": mount}


def mirror_limbs(model, specs, meshes_dir=None):
    """Add a mirrored copy of each limb named in ``specs`` to ``model``.

    ``specs`` is the ``mirror_limbs:`` block of joints.yaml::

        mirror_limbs:
          - root: right_arm_0        # the link the limb hangs from
            plane: xz                # base_link plane to reflect through
            rename: {right: left}    # substring substitution for the new names
          # ...or, when the limb's links carry no side marker to swap:
          - root: shoulder_link_1
            plane: yz
            prefix: mirrored_        # prepended to every generated name
            attach_to: right_hip_out # OPTIONAL: hang the copy here instead of
                                     # on the source's own parent

    The limb's own attachment joint is regenerated too, so the mirrored limb
    hangs off the same parent at the reflected pose -- or off ``attach_to`` when
    the spec names one.  That matters when the other side's MOUNT is already
    modelled and only the limb itself is missing: a quadruped's torso carries
    all four hip servos, so mirroring a leg onto its own parent either clones a
    servo that already exists or leaves both legs on one horn.  Naming the far
    side's horn puts the copy where the CAD has it, with no duplicate.

    ``model`` is mutated; the return value is a list of report records for
    :func:`print_limb_report`.
    """
    if not specs:
        return []
    by_link = {c.link_name: c for c in model.components}
    parent_of = {j.child: j for j in model.joints}
    root = next((c.link_name for c in model.components
                 if c.link_name not in parent_of), model.base_link)
    anchors = link_anchors(model.joints, root)

    reports = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        limb_root = spec.get("root")
        plane = str(spec.get("plane") or DEFAULT_PLANE).lower()
        rules = spec.get("rename") or {}
        prefix = str(spec.get("prefix") or "")
        attach_to = spec.get("attach_to") or None
        if plane not in PLANES:
            reports.append({"root": limb_root,
                            "skip": f"unknown plane {plane!r}; expected one of "
                                    + ", ".join(sorted(PLANES))})
            continue
        if limb_root not in by_link:
            reports.append({"root": limb_root,
                            "skip": "no such link in the model"})
            continue
        if limb_root not in parent_of:
            reports.append({"root": limb_root,
                            "skip": "it is the robot's root, so there is no "
                                    "attachment joint to mirror"})
            continue
        if attach_to is not None and attach_to not in by_link:
            reports.append({"root": limb_root,
                            "skip": f"attach_to: {attach_to!r} is not a link "
                                    f"in the model"})
            continue
        signs = PLANES[plane]
        S = _reflection(signs)
        M = _reflection(signs)

        links = [ln for ln in subtree_links(model.joints, limb_root)
                 if ln in anchors]
        if attach_to in links:
            # it would hang off a link inside its own copy -- a cycle, or a
            # parent that does not exist until the copy is made
            reports.append({"root": limb_root,
                            "skip": f"attach_to: {attach_to!r} is inside the "
                                    f"limb being mirrored"})
            continue
        names = {ln: _rename(ln, rules, prefix) for ln in links}
        clashes = [ln for ln, new in names.items()
                   if new == ln or new in by_link]
        if clashes:
            reports.append({
                "root": limb_root,
                "skip": f"`{'prefix' if prefix else 'rename'}` leaves "
                        f"{len(clashes)} name(s) unchanged or colliding (e.g. "
                        f"{names[clashes[0]]!r}); a mirrored link cannot share "
                        f"a name with a real one"})
            continue

        # Without somewhere to write the reflected meshes there is no honest
        # answer: reusing the source's mesh file would draw the generated limb
        # with the ORIGINAL hand's geometry, and dropping it would produce links
        # that silently have no shape.  Refuse the limb and say why.
        if any(by_link[ln].mesh_file for ln in links) and not meshes_dir:
            reports.append({
                "root": limb_root,
                "skip": "no meshes directory to write the reflected geometry "
                        "into (build_model needs meshes_dir=)"})
            continue

        mirrored_anchors = {ln: M @ anchors[ln] @ S for ln in links}
        # collect the distinct meshes FIRST so they can be reflected together;
        # several links of one limb often share a mesh file
        jobs, dst_of = {}, {}
        if meshes_dir:
            for link in links:
                mesh_file = by_link[link].mesh_file
                if not mesh_file or mesh_file in dst_of:
                    continue
                src = os.path.join(os.path.dirname(meshes_dir), mesh_file)
                stem = os.path.splitext(os.path.basename(mesh_file))[0]
                dst = os.path.join(meshes_dir, f"{stem}__mirrored.glb")
                dst_of[mesh_file] = dst
                jobs[dst] = (src, dst, signs)
        written = _mirror_meshes(jobs.values())
        mesh_cache = {
            mf: (os.path.join("meshes", os.path.basename(dst))
                 if written.get(dst) else None)
            for mf, dst in dst_of.items()}

        added_mass = 0.0
        for link in links:
            comp = by_link[link]
            mesh_file = comp.mesh_file
            if mesh_file and meshes_dir:
                mesh_file = mesh_cache.get(mesh_file)
            new = _mirrored_component(comp, link, names[link], signs, mesh_file)
            model.components.append(new)
            by_link[new.link_name] = new
            added_mass += float(comp.sw_mass or 0.0)

        for link in links:
            joint = parent_of[link]
            parent_anchor = mirrored_anchors.get(joint.parent)
            if parent_anchor is None:            # the attachment joint
                new_parent = attach_to or joint.parent
                parent_anchor = anchors[new_parent]
            else:
                new_parent = names[joint.parent]
            rel = np.linalg.inv(parent_anchor) @ mirrored_anchors[link]
            xyz, rpy = matrix_to_xyz_rpy(rel)
            import copy as _copy
            new_joint = _copy.copy(joint)
            new_joint.name = _rename(joint.name, rules, prefix) or joint.name
            if new_joint.name == joint.name:
                new_joint.name = f"{joint.name}__mirrored"
            new_joint.parent = new_parent
            new_joint.child = names[link]
            new_joint.xyz, new_joint.rpy = xyz, rpy
            new_joint.axis = mirror_axis(joint.axis, signs, joint.jtype)
            # the CAD axis line is a WORLD-frame debug record of a real mate;
            # this joint has no mate behind it, so leave it empty rather than
            # reflecting a number that points at nothing
            new_joint.sw_axis_point = None
            new_joint.sw_axis_dir = None
            new_joint.geo_note = f"mirrored from {joint.name}"
            model.joints.append(new_joint)

        reports.append({"root": limb_root, "plane": plane, "links": len(links),
                        "mass": added_mass,
                        "names": [names[ln] for ln in links],
                        "attach_to": attach_to,
                        "shared": _sharing_to_report(
                            model.joints, anchors, limb_root, plane,
                            attach_to)})
    return reports


def print_limb_report(reports):
    """Print what :func:`mirror_limbs` generated, or say nothing."""
    for r in reports:
        if r.get("skip"):
            print(f"      mirror_limbs '{r['root']}': NOT generated -- "
                  f"{r['skip']}")
            continue
        print(f"      mirror_limbs: generated {r['links']} link(s) from "
              f"'{r['root']}' through the {r['plane']} plane "
              f"(+{r['mass']:.3f} kg): {', '.join(r['names'])}")
        sh = r.get("shared")
        if sh:
            print(f"        WARN: the copy hangs off the same parent as "
                  f"'{r['root']}', which joint '{sh['joint']}' drives -- so "
                  f"that one joint now moves BOTH limbs.  The mount the copy "
                  f"belongs on is already in the model: set "
                  f"attach_to: {sh['attach_to']}")
    if any(not r.get("skip") for r in reports):
        print("        these links have NO CAD behind them -- they are exact "
              "reflections, so anything genuinely asymmetric is not modelled")


__all__ = [
    "DEFAULT_PLANE",
    "PLANES",
    "link_anchors",
    "mirror_axis",
    "mirror_limbs",
    "mirror_mesh_file",
    "origin_matrix",
    "print_limb_report",
    "shared_actuator",
    "subtree_links",
    "suggest_attach",
]
