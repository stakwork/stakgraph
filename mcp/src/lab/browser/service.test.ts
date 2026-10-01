/**
 * The service's offline surface: what boot accepts, and the session
 * lifecycle over a FAKE Playwright (`launch()` overridden) — no browser is
 * touched. What a real server does with a page is service.live.test.ts.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import type { Browser } from "playwright-core";
import { BrowserService, strictHint } from "./service.js";
import { sleep } from "./settle.js";

test("BROWSER_WS_URL and BROWSER_CDP_URL together are refused at boot; either alone is a backend", () => {
  assert.throws(() => BrowserService.fromEnv({ BROWSER_WS_URL: "ws://browser:3000", BROWSER_CDP_URL: "http://127.0.0.1:9222" }), /not both/);
  assert.equal(BrowserService.fromEnv({ BROWSER_WS_URL: "ws://browser:3000" }).cdp, false);
  assert.equal(BrowserService.fromEnv({ BROWSER_CDP_URL: "http://127.0.0.1:9222" }).cdp, true);
  assert.equal(BrowserService.fromEnv({}).cdp, false, "unset: a local launch");
  assert.throws(() => BrowserService.fromEnv({ BROWSER_VIEWPORT: "wide" }), /WIDTHxHEIGHT/);
});

test("a strict-mode violation keeps the matches, drops the call log and names the way out", () => {
  const pw = "locator.click: Error: strict mode violation: locator('text=algorithms') resolved to 2 elements:\n    1) <p>…</p>\n    2) <strong>algorithms</strong>\n\nCall log:\n  - waiting for locator('text=algorithms')\n";
  const m = strictHint(pw);
  assert.match(m, /resolved to 2 elements:\n    1\) <p>…<\/p>\n    2\) <strong>algorithms<\/strong>\n→ act by ref instead/);
  assert.doesNotMatch(m, /Call log/);
  assert.equal(strictHint("locator.click: Timeout 5000ms exceeded."), "locator.click: Timeout 5000ms exceeded.", "anything else is untouched");
});

// ── The session lifecycle over a fake Playwright ─────────────────────────────
// Every context the service makes is recorded, with whether it was closed:
// a context the service made and no longer tracks must be closed, or it
// would live in the browser container until the host restarts.

class FakePage {
  closed = false;
  isClosed() { return this.closed; }
  setDefaultTimeout() {}
  on() {}
  async close() { this.closed = true; }
}

class FakeContext {
  closed = false;
  constructor(private readonly pw: FakePlaywright) {}
  async newPage() {
    if (this.pw.newPageFails) throw new Error("Target page, context or browser has been closed");
    return new FakePage();
  }
  async close() { this.closed = true; }
}

class FakePlaywright {
  contexts: FakeContext[] = [];
  /** `newContext` waits for this before it makes one. */
  gate: Promise<void> = Promise.resolve();
  newPageFails = false;
  readonly browser = {
    isConnected: () => true,
    on: () => {},
    contexts: () => [],
    close: async () => {},
    newContext: async () => {
      await this.gate;
      const c = new FakeContext(this);
      this.contexts.push(c);
      return c;
    },
  } as unknown as Browser;
}

class Service extends BrowserService {
  constructor(readonly pw: FakePlaywright, runBudgetMs?: number) {
    super({ wsUrl: "ws://fake:3000", ...(runBudgetMs ? { runBudgetMs } : {}) });
  }
  protected override async launch(): Promise<Browser> {
    return this.pw.browser;
  }
}

const open = (count: number) => (c: FakeContext) => Number(c.closed === false) === count;
const leaked = (pw: FakePlaywright) => pw.contexts.filter(open(1)).length;

test("two page() calls for one run in flight together share ONE context; dispose closes it", async () => {
  const pw = new FakePlaywright();
  const svc = new Service(pw);
  const [a, b] = await Promise.all([svc.page("r1"), svc.page("r1")]);
  assert.equal(a, b, "the second call joined the first open");
  assert.equal(pw.contexts.length, 1);
  assert.equal(await svc.page("r1"), a, "a later call returns the live page");
  assert.equal(pw.contexts.length, 1);
  await svc.dispose("r1");
  assert.equal(leaked(pw), 0);
  assert.equal(svc.remaining("r1"), svc.runBudgetMs, "the run is forgotten");
});

test("dispose() while the open is in flight: the open closes the context it made and fails", async () => {
  const pw = new FakePlaywright();
  let release!: () => void;
  pw.gate = new Promise<void>((r) => (release = r));
  const svc = new Service(pw);
  const opening = svc.page("r1");
  await sleep(5); // past connect(), parked in newContext
  await svc.dispose("r1");
  release();
  await assert.rejects(opening, /run ended while its page was opening/);
  assert.equal(pw.contexts.length, 1, "the context was made…");
  assert.equal(leaked(pw), 0, "…and closed by the open itself");
  // The run is gone, so the next call starts a fresh one.
  pw.gate = Promise.resolve();
  await svc.page("r1");
  assert.equal(pw.contexts.length, 2);
  await svc.dispose("r1");
  assert.equal(leaked(pw), 0);
});

test("the budget running out while the open is in flight closes the context too", async () => {
  const pw = new FakePlaywright();
  let release!: () => void;
  pw.gate = new Promise<void>((r) => (release = r));
  const svc = new Service(pw, 10);
  const opening = svc.page("r1");
  await sleep(30); // the budget timer fired with no session to close
  release();
  await assert.rejects(opening, /budget \(10 ms\) is spent/);
  assert.equal(leaked(pw), 0);
  await assert.rejects(svc.page("r1"), /is spent/, "the run stays spent until dispose");
  await svc.dispose("r1");
});

test("newPage() failing closes the context it would have lived in; the next call opens again", async () => {
  const pw = new FakePlaywright();
  const svc = new Service(pw);
  pw.newPageFails = true;
  await assert.rejects(svc.page("r1"), /has been closed/);
  assert.equal(pw.contexts.length, 1);
  assert.equal(leaked(pw), 0);
  pw.newPageFails = false;
  const p = await svc.page("r1");
  assert.equal(pw.contexts.length, 2, "the failed open did not pin the run");
  assert.equal(p.isClosed(), false);
  await svc.dispose("r1");
  assert.equal(leaked(pw), 0);
});
