// The plane a limb is reflected through, drawn in the viewer.
//
// `mirror_limbs:` names its plane as a two-letter string, which tells you
// nothing about where it actually cuts this robot -- and picking the wrong one
// produces a limb that is confidently in the wrong place.  Showing the plane
// turns that into something you can see before pressing the button, and after
// the fact it answers "which side did this link come from" for a link that has
// no CAD behind it.
//
// The quad is attached to the ROBOT ROOT, not to the scene: the plane is defined
// in base_link's frame, so parenting it there means it follows the root pose
// (and the viewer's Z-up world rotation) without any conversion of its own.
import { markHelper } from './bootstrap.js';
import { robotBox } from './camera-reroot.js';
import { viewer } from './dom.js';
import { THREE } from './three-setup.js';

// Which axis each plane's normal runs along, and the rotation that takes
// PlaneGeometry (which lies in local XY, normal +Z) onto it.
const PLANES = {
  xz: { normal: 1, rot: [-Math.PI / 2, 0, 0] },   // y = 0, the sagittal plane
  yz: { normal: 0, rot: [0, Math.PI / 2, 0] },    // x = 0
  xy: { normal: 2, rot: [0, 0, 0] },              // z = 0
};

// the sagittal plane of the usual x-forward / z-up robot; the panel's
// dropdown starts here and the preview matches it
export const DEFAULT_MIRROR_PLANE = 'xz';

let current = null;

export function hideMirrorPlane() {
  if (!current) { return; }
  current.removeFromParent();
  current.traverse(o => {
    o.geometry?.dispose();
    o.material?.dispose();
  });
  current = null;
  viewer.redraw?.();
}

/** Draw the ``plane`` ('xz' | 'yz' | 'xy') of the robot root, or hide it for
 *  anything falsy or unknown.  Idempotent: calling it repeatedly with the same
 *  plane rebuilds one quad, never stacks them. */
export function showMirrorPlane(plane) {
  hideMirrorPlane();
  const spec = PLANES[String(plane ?? '').toLowerCase()];
  const robot = viewer.robot;
  if (!spec || !robot) { return; }

  // size it from the MODEL, not the helpers -- robotBox already excludes the
  // markers, which stick far outside the geometry
  const box = robotBox();
  if (!box) { return; }
  const size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z, 0.05) * 1.15;
  // the plane passes through the root origin along its normal; centre it on the
  // robot in the other two axes so it frames the parts it will reflect
  const centre = robot.worldToLocal(box.getCenter(new THREE.Vector3()));
  centre.setComponent(spec.normal, 0);

  const group = new THREE.Group();
  const quad = new THREE.Mesh(
    new THREE.PlaneGeometry(span, span),
    new THREE.MeshBasicMaterial({
      color: 0x4da3ff, transparent: true, opacity: 0.10,
      side: THREE.DoubleSide,
      // a filled plane through the middle of a robot would otherwise z-fight
      // its way over half the geometry
      depthWrite: false,
    }));
  const edge = new THREE.LineSegments(
    new THREE.EdgesGeometry(quad.geometry),
    new THREE.LineBasicMaterial({ color: 0x4da3ff, transparent: true,
                                  opacity: 0.55, depthWrite: false }));
  group.add(quad, edge);
  group.rotation.set(...spec.rot);
  group.position.copy(centre);
  // under the selection triad and the axis glyphs, over the meshes
  quad.renderOrder = 900;
  edge.renderOrder = 901;

  markHelper(group);
  robot.add(group);
  current = group;
  viewer.redraw?.();
}

/** The link name ``rename`` would give ``name``, matching the server's rule:
 *  the FIRST occurrence of each key, one substitution per name. */
export function applyRename(name, rules, prefix) {
  if (prefix) { return prefix + name; }
  for (const [from, to] of Object.entries(rules ?? {})) {
    if (from && name.includes(from)) { return name.replace(from, to); }
  }
  return name;
}

/** The ``mirror_limbs:`` spec that produced ``link`` from ``source``, or null.
 *  Matching on the rename (rather than walking the tree) is exact and needs
 *  nothing but what the panel already has. */
export function specThatGenerated(specs, link, source) {
  if (!source) { return null; }
  return (specs ?? []).find(
    s => applyRename(source, s.rename, s.prefix) === link) ?? null;
}
