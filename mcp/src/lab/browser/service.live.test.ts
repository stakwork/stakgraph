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

test("verify: a page that holds supports; one that does not is a result with the shot where it got to; only the instrument throws", opts, async () => {
  const dir = await mkdtemp(join(tmpdir(), "lab-browser-live-"));
  const artifacts = fileArtifactsCapability(dir);
  const browser = BrowserService.fromEnv(env);
  const sink = (runId: string) => ({ artifacts, runId });
  try {
    const ok = await browser.verify(
      "ver1",
      {
        url: HOME,
        steps: [{ do: "click", selector: "text=Switch" }, { do: "wait", text: "Annual" }],
        expect: { text: ["Annual", "two paragraphs"], selector: ["#plan", "button"], url: "data:**", noErrors: true, evaluate: "document.querySelectorAll('p').length === 2" },
      },
      sink("ver1"),
    );
    assert.equal(ok.supports, true, ok.content);
    assert.deepEqual(ok.failed, []);
    assert.equal(ok.content, "the page reached; all 7 expectations hold");
    assert.equal(ok.path, "shots/001.png");
    assert.ok((await readFile(join(dir, "ver1", "shots", "001.png"))).subarray(1, 4).toString() === "PNG", "a real PNG under the run's artifacts");

    // The page failing: a step that matches nothing stops the list, the
    // expectations are still tested, the shot is still taken.
    const bad = await browser.verify(
      "ver2",
      {
        url: HOME,
        steps: [{ do: "click", selector: "text=No such thing", timeoutMs: 1000 }, { do: "click", selector: "text=Switch" }],
        expect: { text: ["Annual"], selector: ["form"], noErrors: true, evaluate: "document.title === 'Elsewhere'" },
      },
      sink("ver2"),
    );
    assert.equal(bad.supports, false);
    assert.match(bad.failed[0]!, /^step 1 \(click text=No such thing\) failed: .*Timeout 1000ms/);
    assert.deepEqual(bad.failed.slice(1), ['text "Annual": not visible', "selector form: matches nothing", "evaluate: false is not truthy"], "the second step never ran");
    assert.match(bad.content, /^stopped at step 1 \(click text=No such thing\): .*; 3 of 4 expectations failed: text "Annual" not visible; selector form matches nothing; evaluate false is not truthy$/);
    assert.equal(bad.path, "shots/001.png", "the shot is taken where the page got to");

    // An error on the page refutes noErrors.
    const noisy = await browser.verify("ver3", { url: page(`<title>Noisy</title><script>console.error("boom")</script><p>hi</p>`), expect: { text: ["hi"], noErrors: true } }, sink("ver3"));
    assert.equal(noisy.supports, false);
    assert.deepEqual(noisy.errors, ["console: boom"]);
    assert.deepEqual(noisy.failed, ["no errors: 1 on the page, first console: boom"]);

    // The instrument failing throws: a URL that does not open, a selector that does not parse.
    await assert.rejects(browser.verify("ver4", { url: "http://127.0.0.1:9/", timeoutMs: 5000 }, sink("ver4")), /ERR_CONNECTION_REFUSED|ECONNREFUSED|ERR_/);
    await assert.rejects(browser.verify("ver5", { url: HOME, expect: { selector: ["div[[["] } }, sink("ver5")), /browser\/verify: selector "div\[\[\[" cannot be evaluated/);
    await assert.rejects(browser.verify("ver6", { url: HOME, expect: { evaluate: "return )(" } }, sink("ver6")), /evaluate does not parse/);
    await assert.rejects(browser.verify("ver7", { url: "file:///etc/passwd" }, sink("ver7")), /refused file:/, "the denylist applies");
  } finally {
    await browser.close();
    await rm(dir, { recursive: true, force: true });
  }
});

test("walkthrough: a frame per captioned step and the manifest naming them, told to the job's host; a failing step stops on a marked frame", opts, async () => {
  const dir = await mkdtemp(join(tmpdir(), "lab-browser-walk-"));
  const browser = BrowserService.fromEnv(env);
  const told: Array<Record<string, unknown>> = [];
  const sink = (runId: string) => ({ artifacts: fileArtifactsCapability(join(dir, "artifacts")), runId, job: { name: "j-walk", root: join(dir, "jobs", runId) }, notify: (u: Record<string, unknown>) => void told.push(u) });
  try {
    const ok = await browser.walkthrough(
      "walk1",
      {
        url: HOME,
        title: "Switch to annual",
        caption: "The prices page",
        steps: [{ do: "click", selector: "text=Switch", caption: "Switched" }, { do: "wait", text: "Annual" }],
        expect: { text: ["Annual"] },
      },
      sink("walk1"),
    );
    assert.equal(ok.supports, true, ok.content);
    assert.deepEqual(ok.frames.map((f) => [f.file, f.caption]), [["01.png", "The prices page"], ["02.png", "Switched"], ["03.png", "Home"]], "the wait has no caption: the end gets a frame of its own");
    assert.equal(ok.path, "walkthroughs/001/walkthrough.json");
    const m = JSON.parse(await readFile(join(dir, "jobs", "walk1", ok.path), "utf8"));
    assert.equal(m.status, "done");
    assert.equal(m.frames[1].step.selector, "text=Switch");
    assert.ok(m.frames[2].ms >= m.frames[1].ms);
    const png = await readFile(join(dir, "jobs", "walk1", "walkthroughs", "001", "02.png"));
    assert.equal(png.subarray(1, 4).toString(), "PNG");
    assert.deepEqual(told.map((u) => [u.kind, u.state, (u.detail as { frames: number }).frames]), [["walkthrough", "recording", 1], ["walkthrough", "recording", 2], ["walkthrough", "recording", 3], ["walkthrough", "done", 3]]);
    assert.ok(ok.small.length > 0);

    const bad = await browser.walkthrough(
      "walk2",
      { url: HOME, steps: [{ do: "click", selector: "text=Switch", caption: "Switched" }, { do: "click", selector: "text=Nope", timeoutMs: 1500, caption: "Never" }] },
      sink("walk2"),
    );
    assert.equal(bad.supports, false);
    assert.match(bad.failed.join(" "), /step 2 \(click text=Nope\)/);
    assert.deepEqual(bad.frames.map((f) => f.caption), ["Home", "Switched", "Stopped at step 2"]);
    const m2 = JSON.parse(await readFile(join(dir, "jobs", "walk2", bad.path), "utf8"));
    assert.equal(m2.status, "stopped");
    assert.match(m2.frames[2].failed, /step 2/);

    await assert.rejects(browser.walkthrough("walk3", { url: HOME, steps: [{ do: "close" }] }, sink("walk3")), /unknown verb "close"/);
  } finally {
    for (const r of ["walk1", "walk2", "walk3"]) await browser.dispose(r).catch(() => {});
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
