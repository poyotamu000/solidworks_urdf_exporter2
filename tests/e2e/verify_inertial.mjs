// Browser E2E for URDF-input mode inertial editing, against a box-geometry URDF
// (no meshes, so no glb-conversion deps -- the viewer renders the boxes itself).
//
// Prereqs:
//   1. server:  uv run python -m sw2robot.editor.webserver \
//                 tests/e2e/fixtures/boxbot --port 8099
//   2. once:    cd tests/e2e && npm i
// Run:
//   node tests/e2e/verify_inertial.mjs [url]
import puppeteer from 'puppeteer-core';

const BASE = process.argv[2] ?? 'http://127.0.0.1:8099';
const CHROME = process.env.CHROME_PATH
  ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const sleep = ms => new Promise(r => setTimeout(r, ms));
let fails = 0;
const check = (n, ok, d = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${n}${d ? '  -- ' + d : ''}`);
  if (!ok) { fails += 1; }
};

// CI (headless Linux) passes --no-sandbox etc. via CHROME_ARGS; local runs leave
// it unset and keep the default flags.
const EXTRA_ARGS = (process.env.CHROME_ARGS ?? '')
  .split(',').map(s => s.trim()).filter(Boolean);
const browser = await puppeteer.launch({
  executablePath: CHROME, headless: 'new',
  args: ['--disable-gpu', ...EXTRA_ARGS] });
const page = await browser.newPage();
const errs = [];
page.on('pageerror', e => errs.push(e.message.split('\n')[0]));

await page.goto(BASE + '?lang=en', { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.waitForFunction(
  () => document.getElementById('viewer')?.robot?.links?.tip,
  { timeout: 60000 }).catch(() => {});
await sleep(1000);
check('no page errors', errs.length === 0, errs[0] ?? '');

const mode = await page.evaluate(async () =>
  (await (await fetch('/api/info')).json()).mode);
check('info.mode == urdf', mode === 'urdf', `mode=${mode}`);

// select link 'tip' by clicking joint j's row (selects the child link)
await page.evaluate(() => {
  [...document.querySelectorAll('.joint:not(.root)')]
    .find(r => r.querySelector('.jname')?.textContent === 'tip')
    .querySelector('.jname').click();
});
await sleep(700);

const ui = await page.evaluate(() => ({
  mass: document.getElementById('li_mass')?.value,
  hasApply: !!document.getElementById('li_inertial_apply'),
  hasIxx: !!document.getElementById('li_ixx'),
  cz: document.getElementById('li_cz')?.value,
}));
check('inertial editor rendered', ui.hasApply && ui.hasIxx, JSON.stringify(ui));
check('mass prefilled from URDF', ui.mass === '0.3', `mass=${ui.mass}`);

// edit mass + CoM, then apply
await page.evaluate(() => {
  document.getElementById('li_mass').value = '0.75';
  document.getElementById('li_cz').value = '0.05';
});
await page.click('#li_inertial_apply');
await sleep(2500);                       // loadRobot rebuild + reselect

const urdf = await page.evaluate(async () => {
  const i = await (await fetch('/api/info')).json();
  return await (await fetch(i.urdf)).text();
});
check('served URDF carries the new mass', /value="0\.75"/.test(urdf),
      (urdf.match(/<mass[^>]*>/g) ?? []).join(' '));
check('served URDF carries the new CoM',
      /xyz="0 0 0\.05"/.test(urdf),
      (urdf.match(/<origin[^>]*xyz="[^"]*"[^>]*\/>/g) ?? []).slice(-1)[0] ?? '');

// empty-field guard: clearing a field must not 500
await page.evaluate(() => {
  [...document.querySelectorAll('.joint:not(.root)')]
    .find(r => r.querySelector('.jname')?.textContent === 'tip')
    .querySelector('.jname').click();
});
await sleep(600);
await page.evaluate(() => { document.getElementById('li_ixx').value = ''; });
await page.click('#li_inertial_apply');
await sleep(800);
const logTxt = await page.evaluate(() =>
  [...document.querySelectorAll('#log div')].map(d => d.textContent).join('\n'));
check('empty field caught client-side', /must be a number/.test(logTxt),
      logTxt.split('\n').slice(-2).join(' | '));

// restore the fixture's pristine inertial (like run.mjs, leave no edits behind)
// so a re-run against the same checkout still starts from mass 0.3
await page.evaluate(() => {
  [...document.querySelectorAll('.joint:not(.root)')]
    .find(r => r.querySelector('.jname')?.textContent === 'tip')
    .querySelector('.jname').click();
});
await sleep(500);
await page.evaluate(() => {
  const set = (id, v) => { document.getElementById(id).value = v; };
  set('li_mass', '0.3'); set('li_cx', '0'); set('li_cy', '0'); set('li_cz', '0');
  set('li_ixx', '0.001'); set('li_ixy', '0'); set('li_ixz', '0');
  set('li_iyy', '0.001'); set('li_iyz', '0'); set('li_izz', '0.001');
});
await page.click('#li_inertial_apply');
await sleep(1500);

await browser.close();
console.log(fails ? `\n${fails} CHECK(S) FAILED` : '\nALL CHECKS PASS');
process.exit(fails ? 1 : 0);
