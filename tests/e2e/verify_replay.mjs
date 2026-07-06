// Step-2 check: replay re-performs a recorded op() stream on the viewer.
//  - select + setJoint + resetPose dispatch and actually change viewer state
//  - replay does NOT append the re-performed actions back into oplog
//  - an unknown/destructive op is skipped, not fatal
//  - no page errors
import puppeteer from 'puppeteer-core';

const BASE = process.argv[2] ?? 'http://localhost:8090';
const URL = BASE + (BASE.includes('?') ? '&' : '?') + 'lang=ja';
const CHROME = process.env.CHROME_PATH
  ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const EXTRA = (process.env.CHROME_ARGS ?? '').split(',').map(s => s.trim()).filter(Boolean);
const sleep = ms => new Promise(r => setTimeout(r, ms));
let failures = 0;
const check = (n, ok, d = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${n}${d ? '  -- ' + d : ''}`);
  if (!ok) failures++;
};

const browser = await puppeteer.launch({
  executablePath: CHROME, headless: 'new', args: ['--disable-gpu', ...EXTRA] });
const page = await browser.newPage();
const errs = [];
page.on('pageerror', e => errs.push(e.message.split('\n')[0]));

await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.waitForFunction(() => window.sw2robot, { timeout: 60000 });
await sleep(1500);

// pick a link (not the current selection) and a movable joint + target angle
const plan = await page.evaluate(() => {
  const links = Object.keys(window.viewer.robot.links);
  const j = Object.values(window.viewer.robot.joints)
    .find(x => x.jointType !== 'fixed' && !x.mimicJoint);
  return { link: links[links.length - 1], joint: j?.name ?? null };
});
check('setup: have a link and a movable joint', !!plan.link && !!plan.joint,
      JSON.stringify(plan));

// craft a recorded stream with realistic timestamps + one unknown op
const entries = [
  { t: 0,   op: 'select',    link: plan.link },
  { t: 120, op: 'setJoint',  joint: plan.joint, value: 0.3 },
  { t: 240, op: 'rename',    kind: 'link', old: 'x', new: 'y' },  // destructive -> skip
  { t: 360, op: 'resetPose', n: 1 },
];

// snapshot the log length; replay must not grow it
const before = await page.evaluate(() => window.sw2robot.oplog().length);

// replay at real speed; sample between setJoint (120ms) and resetPose (360ms)
// NB: block body so evaluate does not auto-await the replay promise
await page.evaluate(e => { window.__rp = window.sw2robot.replay(e, { speed: 1 }); }, entries);
await sleep(220);
const mid = await page.evaluate(j => ({
  selected: window.sw2robot.dump().selected,
  angle: Number(window.viewer.robot.joints[j].angle),
}), plan.joint);
check('replay: select was dispatched', mid.selected === plan.link,
      `selected=${mid.selected}`);
check('replay: setJoint drove the joint to 0.3', Math.abs(mid.angle - 0.3) < 1e-6,
      `angle=${mid.angle}`);

await page.evaluate(() => window.__rp);   // wait for the run to finish
await sleep(200);
const after = await page.evaluate(() => ({
  len: window.sw2robot.oplog().length,
  angle: Number(window.viewer.robot.joints[
    Object.values(window.viewer.robot.joints)
      .find(x => x.jointType !== 'fixed' && !x.mimicJoint).name].angle),
}));
check('replay: did NOT append to oplog', after.len === before,
      `before=${before} after=${after.len}`);
check('replay: resetPose ran last (joint back to 0)', Math.abs(after.angle) < 1e-6,
      `angle=${after.angle}`);

check('no page errors', errs.length === 0, errs.join(' | '));

await browser.close();
console.log(failures ? `\n${failures} FAILURE(S)` : '\nALL PASS');
process.exit(failures ? 1 : 0);
