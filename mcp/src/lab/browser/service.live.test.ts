/**
 * The service against a REAL Playwright server. Skipped unless BROWSER_WS_URL
 * is set:
 *
 *   docker run --rm -p 3300:3000 -e BROWSER_WS_PATH=devpath ghcr.io/stakwork/strut-browser:1.63
 *   BROWSER_WS_URL=ws://127.0.0.1:3300 BROWSER_WS_PATH=devpath \
 *     npx tsx --test src/lab/browser/service.live.test.ts
 *
 * The pages are `data:` URLs the test writes, so what is asserted is ours.
 * Three cases reach example.com, for a real navigation, a real 404 and real
 * cookies, and assert only on URLs, statuses and what the test itself set.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileArtifactsCapability } from "strut";
import { BrowserService } from "./service.js";
import { sleep } from "./settle.js";

const WS = process.env["BROWSER_WS_URL"];
const opts = { skip: WS ? false : "BROWSER_WS_URL is unset" };
const env = { ...process.env, BROWSER_WS_URL: WS, BROWSER_CDP_URL: undefined };

const page = (html: string) => `data:text/html;charset=utf-8,${encodeURIComponent(html)}`;
const HOME = page(`<!doctype html><title>Home</title>
<h1>Prices</h1>
<p id="plan">Monthly</p>
<p>Two paragraphs.</p>
<button onclick="document.getElementById('plan').textContent = 'Annual'">Switch</button>
<a href="https://example.com/">Elsewhere</a>
<div style="height: 2000px"></div>`);

test("open → snapshot (refs) → act by ref → navigate → back → screenshot (a real 1280×800 PNG)", opts, async () => {
  const dir = await mkdtemp(join(tmpdir(), "lab-browser-live-"));
  const artifacts = fileArtifactsCapability(dir);
  const browser = BrowserService.fromEnv(env);
  const runId = "live1";
  const sink = { artifacts, runId };
  try {
    const opened = await browser.open(runId, { url: HOME });
    assert.equal(opened.title, "Home");
    assert.equal((await browser.open(runId)).title, "Home", "idempotent: the run's page, not a new one");

    const snap = await browser.snapshot(runId);
    assert.match(snap.tree, /heading "Prices"/);
    const button = /button "Switch" \[ref=(e\d+)\]/.exec(snap.tree)?.[1];
    const link = /link "Elsewhere" \[ref=(e\d+)\]/.exec(snap.tree)?.[1];
    assert.ok(button && link, `the button and the link have refs:\n${snap.tree}`);

    assert.throws(() => browser.locate(runId, { ref: "e999" }), /unknown ref "e999"/);
    assert.throws(() => browser.locate(runId, {}), /exactly one of ref or selector/);

    // an action that does not navigate: the page changed, the refs still hold
    await browser.click(runId, { ref: button });
    assert.equal((await browser.text(runId, { selector: "#plan" })).text, "Annual");
    assert.deepEqual(await browser.wait(runId, { text: "Annual" }).then((r) => r.ok), true);
    assert.deepEqual(await browser.evaluate(runId, { expression: "document.querySelectorAll('p').length" }), { result: 2 });
    await browser.scroll(runId, { y: 100 });
    await browser.hover(runId, { selector: "a" });

    // a same-document URL change (pushState, a hash) is not a navigation: the refs hold
    await browser.evaluate(runId, { expression: "history.pushState(null, '', '#pricing'); location.hash = 'annual'; 1" });
    await browser.click(runId, { ref: button });
    // a selector matching several elements says how to get out of it
    await assert.rejects(browser.click(runId, { selector: "p" }), (e: Error) => /strict mode violation/.test(e.message) && /act by ref instead/.test(e.message) && !/Call log/.test(e.message));

    const { small, ...shot } = await browser.screenshot(runId, {}, sink);
    assert.deepEqual(shot, { path: "shots/001.png", dir: join(dir, runId), url: `/artifacts/${runId}/shots/001.png`, width: 1280, height: 800, bytes: shot.bytes });
    assert.ok(shot.bytes > 1024);
    const png = await readFile(join(dir, runId, "shots", "001.png"));
    assert.equal(png.length, shot.bytes);
    assert.equal(png.subarray(1, 4).toString(), "PNG");
    assert.ok(small.length > 0 && small.length < png.length, "the model's copy is smaller");
    const second = await browser.screenshot(runId, { selector: "h1", name: "Hero shot" }, sink);
    assert.equal(second.path, "shots/Hero_shot.png");
    assert.ok(second.height < 800);
    const whole = await browser.screenshot(runId, { fullPage: true }, sink);
    assert.ok(whole.height > 2000, "fullPage is the scroll height");

    const state = await browser.state(runId, artifacts);
    assert.equal(state.path, "state.json");
    assert.ok(JSON.parse(await readFile(join(dir, runId, "state.json"), "utf-8")).cookies);

    // a real navigation resets the refs; back returns to where we were
    const clicked = await browser.click(runId, { ref: link });
    assert.match(clicked.url, /^https:\/\/example\.com\//);
    assert.throws(() => browser.locate(runId, { ref: link }), /unknown ref/, "navigation reset the refs");
    const back = await browser.back(runId);
    assert.equal(back.title, "Home");
  } finally {
    await browser.close();
    await rm(dir, { recursive: true, force: true });
  }
});

test("observe drains what the page complained about: console errors, failed documents", opts, async () => {
  const browser = BrowserService.fromEnv(env);
  const runId = "obs";
  try {
    await browser.open(runId, { url: HOME });
    assert.deepEqual(browser.observe(runId), { console: [], pageErrors: [], failedRequests: [], httpErrors: [] });
    await browser.evaluate(runId, { expression: "console.error('boom')" });
    assert.deepEqual(browser.observe(runId).console, ["boom"]);
    assert.deepEqual(browser.observe(runId).console, [], "drained");

    const ok = await browser.goto(runId, { url: "https://example.com/" });
    assert.equal(ok.status, 200);
    browser.observe(runId);
    const missing = await browser.goto(runId, { url: "https://example.com/definitely-not-here" });
    assert.equal(missing.status, 404);
    assert.match(browser.observe(runId).httpErrors.join("\n"), /404 GET https:\/\/example\.com\/definitely-not-here/);
  } finally {
    await browser.close();
  }
});

test("two runs at once have a page and a cookie jar each", opts, async () => {
  const browser = BrowserService.fromEnv(env);
  try {
    const visit = async (runId: string) => {
      await browser.open(runId, { url: "https://example.com/" });
      await browser.evaluate(runId, { expression: `document.cookie = "run=${runId}; path=/"` });
      await browser.goto(runId, { url: page(`<title>${runId}</title><h1>${runId}</h1>`) });
      const title = (await browser.snapshot(runId)).title;
      await browser.goto(runId, { url: "https://example.com/" });
      return { title, cookie: (await browser.evaluate(runId, { expression: "document.cookie" })).result };
    };
    const [a, b] = await Promise.all([visit("a"), visit("b")]);
    assert.deepEqual(a, { title: "a", cookie: "run=a" });
    assert.deepEqual(b, { title: "b", cookie: "run=b" });

    await browser.dispose("a");
    await assert.rejects(browser.snapshot("a"), /call browser\/open/);
    assert.equal((await browser.snapshot("b")).url, "https://example.com/", "the other run is untouched");
  } finally {
    await browser.close();
  }
});

test("the denylist refuses file: and a metadata address before anything is fetched", opts, async () => {
  const browser = BrowserService.fromEnv(env);
  try {
    await assert.rejects(browser.open("deny", { url: "file:///etc/passwd" }), /refused file:\/\/\/etc\/passwd: scheme file:/);
    await browser.open("deny");
    await assert.rejects(browser.goto("deny", { url: "http://169.254.169.254/latest/meta-data/" }), /denied range/);
    assert.equal((await browser.open("deny")).url, "about:blank", "nothing was navigated");
  } finally {
    await browser.close();
  }
});

test("capture: declared steps run in order, a step that matches nothing names itself, unknown verbs are refused", opts, async () => {
  const dir = await mkdtemp(join(tmpdir(), "lab-browser-live-"));
  const artifacts = fileArtifactsCapability(dir);
  const browser = BrowserService.fromEnv(env);
  try {
    const out = await browser.capture(
      "cap",
      { url: HOME, steps: [{ do: "wait", text: "Monthly" }, { do: "click", selector: "text=Switch" }, { do: "wait", text: "Annual" }], fullPage: true },
      { artifacts, runId: "cap" },
    );
    assert.equal(out.path, "shots/001.png");
    assert.ok(out.height > 2000);
    assert.equal(out.title, "Home");
    assert.equal((await browser.open("cap")).url, out.pageUrl);
    assert.deepEqual(out.errors, []);
    assert.equal((await browser.text("cap", { selector: "#plan" })).text, "Annual", "the steps ran before the shot");

    await assert.rejects(
      browser.capture("cap2", { url: HOME, steps: [{ do: "click", selector: "text=No such thing", timeoutMs: 1000 }] }, { artifacts, runId: "cap2" }),
      /capture: step 1 \(click .*No such thing.*\) failed: .*Timeout 1000ms/,
    );
    await assert.rejects(browser.capture("cap3", { url: HOME, steps: [{ do: "explode" }] }, { artifacts, runId: "cap3" }), /unknown verb "explode"/);
  } finally {
    await browser.close();
    await rm(dir, { recursive: true, force: true });
  }
});

test("losing the per-call race closes the context; a later open starts a fresh one", opts, async () => {
  const browser = BrowserService.fromEnv(env);
  try {
    const p = await browser.page("hang");
    await browser.open("hang", { url: HOME });
    // Playwright puts no timeout on evaluate: only the hard race can end this.
    await assert.rejects(browser.evaluate("hang", { expression: "new Promise(() => {})", timeoutMs: 300 }), /browser\/evaluate: timed out after 300 ms/);
    assert.ok(p.isClosed(), "the context was closed");
    await assert.rejects(browser.snapshot("hang"), /call browser\/open/);
    assert.equal((await browser.open("hang", { url: HOME })).title, "Home");
  } finally {
    await browser.close();
  }
});

test("budget expiry closes the context and refuses the run; dispose is idempotent", opts, async () => {
  const browser = new BrowserService({ wsUrl: WS, wsPath: process.env["BROWSER_WS_PATH"], runBudgetMs: 1500 });
  try {
    const p = await browser.page("budget");
    assert.ok(browser.remaining("budget") <= 1500);
    await sleep(2000);
    assert.ok(p.isClosed(), "the budget timer closed the context");
    assert.equal(browser.remaining("budget"), 0);
    await assert.rejects(browser.open("budget", { url: HOME }), /browser budget \(1500 ms\) is spent/);
    await browser.dispose("budget");
    await browser.dispose("budget");
    assert.deepEqual(browser.observe("budget"), { console: [], pageErrors: [], failedRequests: [], httpErrors: [] });
    assert.equal(browser.remaining("budget"), 1500, "a disposed run is forgotten");
  } finally {
    await browser.close();
  }
});
