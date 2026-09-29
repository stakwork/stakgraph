/**
 * Every browser/* step over a FAKE BrowserService (offline). A step is a thin
 * wrapper: validate its input (as the runner does), forward to one verb with
 * the run id, return the verb's output. So the tests check that mapping, the
 * one-of refinements, the two steps that touch secrets/artifacts, and that
 * the seeded files really load from a workspace through strut's registry.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readdir, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { WorkspaceManager, buildRegistry, mediaOf, type AnyStepDef } from "strut";
import { CAPTURE_VERBS as SERVICE_VERBS } from "./service.js";
import { CAPTURE_VERBS, type Ctx } from "./steps/_shared.js";
import { seedBrowserSteps, SEED_STEPS, STEPS_DIR } from "./seed.js";

import open from "./steps/open.js";
import goto from "./steps/goto.js";
import snapshot from "./steps/snapshot.js";
import click from "./steps/click.js";
import fill from "./steps/fill.js";
import type_ from "./steps/type.js";
import press from "./steps/press.js";
import select from "./steps/select.js";
import hover from "./steps/hover.js";
import scroll from "./steps/scroll.js";
import wait from "./steps/wait.js";
import text from "./steps/text.js";
import screenshot from "./steps/screenshot.js";
import evaluate from "./steps/evaluate.js";
import observe from "./steps/observe.js";
import back from "./steps/back.js";
import state from "./steps/state.js";
import close from "./steps/close.js";
import capture from "./steps/capture.js";

const STEPS = { open, goto, snapshot, click, fill, type: type_, press, select, hover, scroll, wait, text, screenshot, evaluate, observe, back, state, close, capture };

const NAV = { url: "https://example.com/", title: "Example Domain", status: 200 };
const SHOT = { path: "shots/001.png", url: "/artifacts/r1/shots/001.png", width: 1280, height: 800, bytes: 4321, small: Buffer.from("png") };
const OBS = { console: ["boom"], pageErrors: [], failedRequests: [], httpErrors: ["500 GET /api"] };

/** Records every verb call; each verb returns a canned output. */
class FakeBrowser {
  calls: Array<{ verb: string; runId: string; args: unknown[] }> = [];
  fail?: string;
  private rec<T>(verb: string, runId: string, args: unknown[], out: T): T {
    this.calls.push({ verb, runId, args });
    if (this.fail === verb) throw new Error(`browser/${verb}: nope`);
    return out;
  }
  async open(runId: string, input: unknown) { return this.rec("open", runId, [input], NAV); }
  async goto(runId: string, input: unknown) { return this.rec("goto", runId, [input], NAV); }
  async snapshot(runId: string) { return this.rec("snapshot", runId, [], { ...NAV, tree: '- link "x" [ref=e1]' }); }
  async click(runId: string, input: unknown) { return this.rec("click", runId, [input], { url: NAV.url }); }
  async fill(runId: string, input: unknown) { return this.rec("fill", runId, [input], { ok: true as const }); }
  async type(runId: string, input: unknown) { return this.rec("type", runId, [input], { ok: true as const }); }
  async press(runId: string, input: unknown) { return this.rec("press", runId, [input], { url: NAV.url }); }
  async select(runId: string, input: unknown) { return this.rec("select", runId, [input], { ok: true as const }); }
  async hover(runId: string, input: unknown) { return this.rec("hover", runId, [input], { ok: true as const }); }
  async scroll(runId: string, input: unknown) { return this.rec("scroll", runId, [input], { ok: true as const }); }
  async wait(runId: string, input: unknown) { return this.rec("wait", runId, [input], { ok: true as const, waitedMs: 12 }); }
  async text(runId: string, input: unknown) { return this.rec("text", runId, [input], { text: "hello" }); }
  async screenshot(runId: string, input: unknown, artifacts: unknown) { return this.rec("screenshot", runId, [input, artifacts], SHOT); }
  async evaluate(runId: string, input: unknown) { return this.rec("evaluate", runId, [input], { result: 3 }); }
  observe(runId: string) { return this.rec("observe", runId, [], OBS); }
  async back(runId: string, input: unknown) { return this.rec("back", runId, [input], { url: NAV.url, title: NAV.title }); }
  async state(runId: string, artifacts: unknown) { return this.rec("state", runId, [artifacts], { path: "state.json", url: "/artifacts/r1/state.json" }); }
  async closeRun(runId: string) { return this.rec("close", runId, [], { ok: true as const }); }
  async capture(runId: string, input: unknown, artifacts: unknown) { return this.rec("capture", runId, [input, artifacts], { ...SHOT, errors: ["http: 500 GET /api"], pageUrl: "https://example.com/after", title: "After" }); }
}

const artifacts = { tag: "artifacts" };

function ctxFor(browser: FakeBrowser, secrets: Record<string, string> = {}): Ctx {
  return {
    runId: "r1",
    path: "wf/step",
    scope: {},
    input: {},
    emit: async () => {},
    services: {
      browser: browser as never,
      artifacts: artifacts as never,
      http: {} as never,
      secrets: { get: async (name: string) => secrets[name] },
    },
  };
}

/** Run a step the way the runner does: validate, run, validate the output. */
async function run(def: AnyStepDef, input: unknown, browser = new FakeBrowser(), secrets?: Record<string, string>) {
  const cfg = def.input.parse(input);
  const out = await def.run(cfg, ctxFor(browser, secrets));
  return def.output.parse(out);
}

test("every step is browser/<file>, has a description written for a model, and the seed lists every file", async () => {
  const files = (await readdir(STEPS_DIR)).filter((f) => f.endsWith(".ts")).map((f) => f.slice(0, -3)).sort();
  assert.deepEqual([...SEED_STEPS].sort(), files, "a step file the seed does not list is never published");
  assert.deepEqual(files.filter((f) => !f.startsWith("_")), Object.keys(STEPS).sort());
  for (const [name, def] of Object.entries(STEPS)) {
    assert.equal(def.type, `browser/${name}`);
    assert.ok((def.description ?? "").length > 40, `${def.type} needs a real description`);
  }
  assert.deepEqual([...CAPTURE_VERBS], [...SERVICE_VERBS], "the step's verb list must match the service's");
});

test("each verb step forwards its validated input to the service under the run id", async () => {
  const cases: Array<[AnyStepDef, unknown, string, unknown]> = [
    [goto, { url: "https://example.com", wait: 100 }, "goto", { url: "https://example.com", wait: 100 }],
    [click, { ref: "e3", double: true }, "click", { ref: "e3", double: true }],
    [click, { selector: "text=Go", button: "right" }, "click", { selector: "text=Go", button: "right" }],
    [fill, { ref: "e1", value: "hi" }, "fill", { ref: "e1", value: "hi" }],
    [type_, { text: "abc", delay: 5 }, "type", { text: "abc", delay: 5 }],
    [press, { key: "Enter" }, "press", { key: "Enter" }],
    [select, { selector: "#s", value: "blue" }, "select", { selector: "#s", value: "blue" }],
    [hover, { ref: "e9" }, "hover", { ref: "e9" }],
    [scroll, { y: -300 }, "scroll", { y: -300 }],
    [scroll, { selector: "footer" }, "scroll", { selector: "footer" }],
    [wait, { text: "Total" }, "wait", { text: "Total" }],
    [wait, { ms: 250 }, "wait", { ms: 250 }],
    [wait, { url: "**/done" }, "wait", { url: "**/done" }],
    [text, {}, "text", {}],
    [text, { ref: "e2" }, "text", { ref: "e2" }],
    [evaluate, { expression: "1+2" }, "evaluate", { expression: "1+2" }],
    [back, { timeoutMs: 500 }, "back", { timeoutMs: 500 }],
  ];
  for (const [def, input, verb, args] of cases) {
    const browser = new FakeBrowser();
    await run(def, input, browser);
    assert.equal(browser.calls.length, 1, `${def.type}: one verb call`);
    assert.equal(browser.calls[0]!.verb, verb, def.type);
    assert.equal(browser.calls[0]!.runId, "r1");
    assert.deepEqual(browser.calls[0]!.args[0], args, def.type);
  }
});

test("snapshot, observe and close take no input and return the verb's result", async () => {
  const browser = new FakeBrowser();
  const snap = await run(snapshot, {}, browser);
  assert.match(snap.tree, /\[ref=e1\]/);
  assert.deepEqual(await run(observe, {}, browser), OBS);
  assert.deepEqual(await run(close, {}, browser), { ok: true });
  assert.deepEqual(browser.calls.map((c) => c.verb), ["snapshot", "observe", "close"]);
});

test("element steps take exactly one of ref / selector", () => {
  for (const def of [click, fill, select, hover]) {
    const extra = def === fill || def === select ? { value: "v" } : {};
    assert.equal(def.input.safeParse({ ...extra }).success, false, `${def.type}: neither`);
    assert.equal(def.input.safeParse({ ref: "e1", selector: "a", ...extra }).success, false, `${def.type}: both`);
    assert.equal(def.input.safeParse({ ref: "e1", ...extra }).success, true, `${def.type}: ref`);
    assert.equal(def.input.safeParse({ selector: "a", ...extra }).success, true, `${def.type}: selector`);
  }
  assert.equal(scroll.input.safeParse({}).success, false);
  assert.equal(scroll.input.safeParse({ ref: "e1", y: 10 }).success, false);
  assert.equal(wait.input.safeParse({}).success, false);
  assert.equal(wait.input.safeParse({ text: "a", ms: 1 }).success, false);
  assert.equal(text.input.safeParse({ ref: "e1", selector: "a" }).success, false);
  assert.equal(screenshot.input.safeParse({ ref: "e1", selector: "a" }).success, false);
  assert.equal(screenshot.input.safeParse({}).success, true);
});

test("open passes the viewport through and loads a storage state from the named secret, never the value into the output", async () => {
  const plain = new FakeBrowser();
  await run(open, { url: "https://example.com", viewport: { width: 800, height: 600 } }, plain);
  assert.deepEqual(plain.calls[0]!.args[0], { url: "https://example.com", viewport: { width: 800, height: 600 }, storageState: undefined, timeoutMs: undefined });

  const withState = new FakeBrowser();
  const stateJson = JSON.stringify({ cookies: [{ name: "sid", value: "s3cret" }], origins: [] });
  const out = await run(open, { storageStateSecret: "GITHUB_SESSION" }, withState, { GITHUB_SESSION: stateJson });
  assert.deepEqual((withState.calls[0]!.args[0] as { storageState: unknown }).storageState, JSON.parse(stateJson));
  assert.ok(!JSON.stringify(out).includes("s3cret"));

  await assert.rejects(run(open, { storageStateSecret: "MISSING" }), /secret MISSING is not set/);
  await assert.rejects(run(open, { storageStateSecret: "BAD" }, new FakeBrowser(), { BAD: "not json" }), /not JSON/);
});

test("screenshot and state hand the artifacts capability to the service; the model's copy stays out of the output", async () => {
  const browser = new FakeBrowser();
  const shot = await run(screenshot, { fullPage: true, name: "hero" }, browser);
  assert.deepEqual(browser.calls[0]!.args, [{ fullPage: true, name: "hero" }, artifacts]);
  assert.deepEqual(shot, { path: "shots/001.png", url: "/artifacts/r1/shots/001.png", width: 1280, height: 800, bytes: 4321 });
  assert.ok(!("small" in shot));

  const st = await run(state, {}, browser);
  assert.deepEqual(browser.calls[1]!.args, [artifacts]);
  assert.deepEqual(st, { path: "state.json", url: "/artifacts/r1/state.json" });
});

test("screenshot and capture mark their output with the model's copy (withMedia): one image/png file part, no bytes in the JSON", async () => {
  const cases: Array<[AnyStepDef, unknown, string[]]> = [
    [screenshot, {}, ["bytes", "height", "path", "url", "width"]],
    [capture, { url: "https://example.com" }, ["bytes", "errors", "height", "pageUrl", "path", "title", "url", "width"]],
  ];
  for (const [def, input, keys] of cases) {
    // The raw run() result, as the agent's tool executor gets it (the runner
    // does not re-parse outputs; zod's parse would drop the marker).
    const out = await def.run(def.input.parse(input), ctxFor(new FakeBrowser()));
    const media = mediaOf(out);
    assert.equal(media?.length, 1, `${def.type}: one media part`);
    assert.deepEqual({ ...media![0], data: undefined }, { mediaType: "image/png", filename: "shots/001.png", data: undefined });
    assert.ok(media![0]!.data instanceof Uint8Array && media![0]!.data.length > 0, `${def.type}: raw bytes, not base64`);
    const json = JSON.stringify(out);
    assert.ok(!json.includes("_media") && !json.includes("cG5n") && !json.includes("small"), `${def.type}: nothing of the image in the JSON: ${json}`); // cG5n = base64("png")
    assert.deepEqual(Object.keys(out).sort(), keys, `${def.type}: the plain output shape`);
    assert.doesNotThrow(() => def.output.parse(out));
  }
});

test("capture validates the declared steps and returns the shot with the page's errors", async () => {
  const browser = new FakeBrowser();
  const steps = [
    { do: "wait", text: "Learn more" },
    { do: "click", selector: "text=Learn more" },
  ];
  const out = await run(capture, { url: "https://example.com", steps, fullPage: false, wait: 50 }, browser);
  assert.equal(browser.calls[0]!.verb, "capture");
  assert.deepEqual(browser.calls[0]!.args, [{ url: "https://example.com", steps, fullPage: false, wait: 50 }, artifacts]);
  assert.deepEqual(out.errors, ["http: 500 GET /api"]);
  assert.equal(out.pageUrl, "https://example.com/after");
  assert.ok(!("small" in out));
  assert.equal(capture.input.safeParse({ url: "https://x", steps: [{ do: "close" }] }).success, false, "not a capture verb");
  assert.equal(capture.input.safeParse({ url: "https://x", steps: [{ do: "screenshot" }] }).success, false, "the compound shoots itself");
});

test("a failing verb throws through the step (a declared workflow must fail loudly)", async () => {
  const browser = new FakeBrowser();
  browser.fail = "click";
  await assert.rejects(run(click, { ref: "e1" }, browser), /browser\/click: nope/);
});

test("a step on a host with no browser service throws a clear error", async () => {
  const ctx = ctxFor(new FakeBrowser());
  delete (ctx.services as { browser?: unknown }).browser;
  await assert.rejects(snapshot.run({}, ctx), /no `browser` service/);
});

test("the seeded files load from a workspace through strut's registry (import 'strut' + ./_shared.js resolve)", async () => {
  // Under the mcp dir, like the real lab-workspace: a seeded step's `strut`
  // import resolves via mcp/node_modules from there.
  const mcpDir = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
  const dir = await mkdtemp(join(mcpDir, ".lab-browser-steps-"));
  try {
    const workspace = new WorkspaceManager(dir);
    await seedBrowserSteps(workspace);
    await seedBrowserSteps(workspace); // content-hash keyed: a reseed is a no-op
    const listed = (await workspace.listSteps()).map((s) => s.type).sort();
    assert.deepEqual(listed, Object.keys(STEPS).map((n) => `browser/${n}`).sort(), "helpers are hidden, every step is listed");
    const { registry, sources } = await buildRegistry(await workspace.materializeCustomSteps());
    for (const name of Object.keys(STEPS)) {
      const type = `browser/${name}`;
      assert.ok(registry[type], `${type} loaded`);
      assert.equal(sources[type], "custom");
    }
    assert.ok(!registry["browser/_shared"]);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
