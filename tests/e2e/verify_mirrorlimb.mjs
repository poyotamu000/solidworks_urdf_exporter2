// Browser E2E for `mirror_limbs:` in the link panel: generate the limb on the
// other side of the robot from the selected link, and drop it again.
//
// The endpoint has unit coverage (tests/test_mirror_limb_webserver.py); what
// only a browser can show is that the panel actually RENDERS its controls and
// that clicking through them changes the loaded robot.  A missing import or a
// typo'd i18n key leaves the panel silently short of a row, which no Python
// test would ever notice.
//
// Prereqs (needs a CAD package, i.e. one with a graph.json):
//   1. build once:  cp -r examples/fingertip <pkg> && \
//                   uv run python -m sw2robot.exporter.build <pkg>
//   2. server:      uv run python -m sw2robot.editor.webserver <pkg> --port 8092
//   3. once:        cd tests/e2e && npm i
// Run:
//   node tests/e2e/verify_mirrorlimb.mjs [url]
//
// Removes the limb it generates before exiting, so the served package is left
// exactly as it was found.
import puppeteer from 'puppeteer-core';

const BASE = process.argv[2] ?? 'http://127.0.0.1:8092';
// force Japanese: the panel assertions below check localized strings
const URL = BASE + (BASE.includes('?') ? '&' : '?') + 'lang=ja';
const CHROME = process.env.CHROME_PATH
  ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe';
let fails = 0;
const check = (n, ok, d = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${n}${d ? '  -- ' + d : ''}`);
  if (!ok) { fails += 1; }
};

const EXTRA_ARGS = (process.env.CHROME_ARGS ?? '')
  .split(',').map(s => s.trim()).filter(Boolean);
const browser = await puppeteer.launch({
  executablePath: CHROME, headless: 'new',
  args: ['--no-sandbox', ...EXTRA_ARGS] });
const page = await browser.newPage();
await page.setViewport({ width: 1500, height: 900 });
page.on('pageerror', e => {
  console.log('PAGEERROR', e.stack ?? e.message);
  fails += 1;
});
await page.goto(URL, { waitUntil: 'networkidle2', timeout: 60000 });
const links = () => page.evaluate(
  () => Object.keys(window.viewer?.robot?.links ?? {}));
await page.waitForFunction(
  () => Object.keys(window.viewer?.robot?.links ?? {}).length > 0,
  { timeout: 60000, polling: 500 });

// any link hanging off a joint is a limb root: it plus everything below it is
// what gets reflected.  The root link itself has no attachment joint to mirror,
// so pick a child.
const before = await links();
const target = await page.evaluate(() => {
  const r = window.viewer.robot;
  return Object.keys(r.links).find(
    n => r.links[n]?.parent?.isURDFJoint) ?? null;
});
check('found a link with a parent joint', !!target, String(target));
if (!target) { await browser.close(); process.exit(1); }

const PREFIX = 'mirchk';
const panel = (which = target) => page.evaluate(async (name) => {
  const mod = await import('/link-info.js');
  mod.fillLinkInfo(name);
  const el = document.getElementById('linkinfo');
  return {
    visible: el?.style.display !== 'none',
    from: el?.querySelector('#li_mirf') != null,
    to: el?.querySelector('#li_mirt') != null,
    plane: [...(el?.querySelectorAll('#li_mirp option') ?? [])]
      .map(o => o.value),
    go: el?.querySelector('#li_mirgo') != null,
    undo: el?.querySelector('#li_unmirror') != null,
    label: [...(el?.querySelectorAll('td') ?? [])]
      .map(td => td.textContent.trim()).find(x => x.includes('ミラー')) ?? null,
  };
}, which);

const p0 = await panel();
check('panel renders the mirror row', p0.visible && !!p0.label, p0.label ?? '');
check('it offers from / to / generate', p0.from && p0.to && p0.go);
check('it offers all three planes', p0.plane.join(',') === 'xz,yz,xy',
      p0.plane.join(','));
check('no undo button before anything is generated', !p0.undo);

// the plane is drawn in the viewer while the panel offers it: a two-letter
// name says nothing about where it cuts THIS robot
const planes = () => page.evaluate(() => {
  let n = 0;
  window.viewer.robot.traverse(
    o => { if (o.isMesh && o.geometry?.type === 'PlaneGeometry') { n += 1; } });
  return n;
});
check('the mirror plane is drawn for a mirrorable link', await planes() === 1,
      String(await planes()));
await page.select('#li_mirp', 'yz');
await new Promise(r => setTimeout(r, 400));
check('switching the plane replaces it rather than stacking',
      await planes() === 1, String(await planes()));
await page.select('#li_mirp', 'xz');
await new Promise(r => setTimeout(r, 400));
// reopening must not leak a quad per open
await panel(); await panel();
check('reopening the panel does not stack planes', await planes() === 1,
      String(await planes()));

// generate: fill the rename boxes so the copies get names of their own
await page.evaluate((name, prefix) => {
  const el = document.getElementById('linkinfo');
  el.querySelector('#li_mirf').value = name;
  el.querySelector('#li_mirt').value = prefix + name;
  el.querySelector('#li_mirgo').click();
}, target, PREFIX);
let generated = false;
try {
  await page.waitForFunction(
    (p) => Object.keys(window.viewer?.robot?.links ?? {})
      .some(l => l.startsWith(p)),
    { timeout: 120000, polling: 500 }, PREFIX);
  generated = true;
} catch { /* reported below */ }
const after = await links();
check('clicking generate adds the mirrored links', generated,
      `${before.length} -> ${after.length}`);
check('the generated links are named by the rename rule',
      after.some(l => l === PREFIX + target),
      after.filter(l => l.startsWith(PREFIX)).join(','));

// the panel now offers to undo it instead of generating again
const p1 = await panel();
check('the panel switches to an undo control', p1.undo && !p1.go);

// and the generated link says where it came from rather than looking unresolved
// packageState is a module export, not a global -- import it the same way the
// front end does rather than reaching for a window property that is not there
const meta = await page.evaluate(async (l) => {
  const st = await import('/state.js');
  return st.packageState?.compMeta?.[l]?.mirrored_from ?? null;
}, PREFIX + target);
check('the generated link reports its origin', meta === target, String(meta));

// a generated link says which plane it came from; a link the feature does not
// apply to shows none
await panel(PREFIX + target);
await new Promise(r => setTimeout(r, 400));
check('a generated link still shows its plane', await planes() === 1,
      String(await planes()));
const root = await page.evaluate(() => {
  const r = window.viewer.robot;
  return Object.keys(r.links).find(n => !r.links[n]?.parent?.isURDFJoint);
});
if (root) {
  await panel(root);
  await new Promise(r => setTimeout(r, 400));
  check('the root link shows no plane', await planes() === 0,
        `${root}: ${await planes()}`);
}
await panel();

// restore: drop it again
await page.evaluate(() => document.getElementById('li_unmirror')?.click());
let removed = false;
try {
  await page.waitForFunction(
    (p) => !Object.keys(window.viewer?.robot?.links ?? {})
      .some(l => l.startsWith(p)),
    { timeout: 120000, polling: 500 }, PREFIX);
  removed = true;
} catch { /* reported below */ }
check('undo takes the generated links back out', removed);

await browser.close();
console.log(fails ? `${fails} FAILED` : 'all passed');
process.exit(fails ? 1 : 0);
