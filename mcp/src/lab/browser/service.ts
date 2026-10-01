/**
 * BrowserService: the lab's side of the browser that the seeded `browser/*`
 * steps drive (`ctx.services.browser`). ONE lazy Playwright connection, ONE
 * `BrowserContext` + page per run keyed by `runId`, observations accumulated
 * from the moment the page exists, a ref table from the last accessibility
 * snapshot, settle after every action, a per-call timeout and a per-run
 * budget (losing either closes the run's pages), and `dispose(runId)` wired
 * into `services.onRunEnd`.
 *
 * Three backends, one service; the environment picks.
 *   - `BROWSER_WS_URL` (+ `BROWSER_WS_PATH`): a Playwright SERVER, what a
 *     swarm runs beside this host (the `browser` container). Every run gets a
 *     context of its own, cookies, storage and page, closed when the run
 *     ends. The server refuses a client on another Playwright major.minor:
 *     `playwright-core` in package.json and the server image move together.
 *   - `BROWSER_CDP_URL`: a real Chrome someone started with
 *     `--remote-debugging-port` on a dedicated profile and logged into. The
 *     service connects over CDP and opens ONE PAGE PER RUN in the browser's
 *     DEFAULT context: runs share the profile's cookies and are isolated only
 *     by page, and `dispose` closes the run's pages, never the context. So
 *     over CDP `storageStateSecret` is refused (the profile's state is the
 *     session), `viewport` is refused (the window is the human's) and
 *     `browser/state` is refused (it would export the whole profile).
 *   - Neither: a local headless Chrome (`BROWSER_EXECUTABLE`, else the
 *     `BROWSER_CHANNEL` channel, default `chrome`). For development.
 * Setting both URLs is refused.
 *
 * Every `browser/*` step is a thin wrapper over one verb here, because a
 * seeded step is self-contained source that can import only `strut`: the
 * Playwright calls, the artifact write and the downscale live in-code, once.
 * `capture` runs its declared list through the same verbs.
 *
 * `playwright-core` is imported lazily, on the first connection, so a lab
 * that never opens a browser never loads it.
 */
import type { Browser, BrowserContext, BrowserContextOptions, Page, Locator } from "playwright-core";
import type { ArtifactsCapability } from "strut";
import { settle as settlePage, sleep } from "./settle.js";
import { saveShot, type Shot } from "./shot.js";
import { parseDenyList, denyReason, type DenyRule } from "./deny.js";

export interface Observations {
  console: string[];
  pageErrors: string[];
  failedRequests: string[];
  httpErrors: string[];
}

export interface Viewport {
  width: number;
  height: number;
}

/** Exactly one of `ref` (from the last `snapshot`) or a Playwright selector. */
export interface Target {
  ref?: string;
  selector?: string;
}

export interface BrowserServiceOptions {
  /** `ws://host:port` of a Playwright server (`playwright run-server`); the
   *  secret endpoint path goes in `wsPath`. Unset → launch a local browser. */
  wsUrl?: string;
  wsPath?: string;
  /** A real Chrome over CDP instead of a Playwright server. */
  cdpUrl?: string;
  /** Local launch only: a browser binary, else a branded `channel`
   *  (default `chrome` — the Google Chrome on the machine, no download). */
  executablePath?: string;
  channel?: string;
  /** Default viewport for `open` (1280×800). */
  viewport?: Viewport;
  /** Per-call cap, default 10 s. */
  stepTimeoutMs?: number;
  /** Per-run browser budget, default 10 min. */
  runBudgetMs?: number;
  /** `BROWSER_URL_DENY` (see deny.ts). */
  deny?: string;
}

export interface Navigated {
  url: string;
  title: string;
  /** HTTP status of the main document; 0 when nothing was fetched. */
  status: number;
}

export type CaptureStep = { do: string } & Record<string, unknown>;

/** The verbs a `capture` list may name — everything that drives the page
 *  the run already has; `open`/`close`/`screenshot` are the compound's own. */
export const CAPTURE_VERBS = ["goto", "click", "fill", "type", "press", "select", "hover", "scroll", "wait", "text", "evaluate", "back"] as const;

export const TEXT_CAP = 50_000;
export const TREE_CAP = 100_000;
const OBS_CAP = 200;
/** Slack over the per-call cap before the hard race closes the context:
 *  settle can add up to ~7.5 s on its own. */
const SETTLE_SLACK_MS = 8_500;

interface Session {
  context: BrowserContext;
  /** The run's own context (a Playwright server) — closed with the run. Over
   *  CDP it is the profile's default context, shared with the human and every
   *  other run, and only `pages` are closed. */
  own: boolean;
  page: Page;
  /** Every page the run opened: the first plus any popup that replaced it. */
  pages: Page[];
  refs: Set<string>;
  /** A popup / new tab became the page since the last snapshot. */
  replaced: boolean;
  obs: Observations;
}

interface Run {
  startedAt: number;
  timer?: NodeJS.Timeout;
  exhausted: boolean;
  session?: Session;
}

export class BrowserService {
  readonly viewport: Viewport;
  readonly stepTimeoutMs: number;
  readonly runBudgetMs: number;
  private readonly rules: DenyRule[];
  private readonly opts: BrowserServiceOptions;
  private browser?: Browser;
  private connecting?: Promise<Browser>;
  private readonly runs = new Map<string, Run>();

  constructor(opts: BrowserServiceOptions = {}) {
    if (opts.wsUrl && opts.cdpUrl) throw new Error("browser: set BROWSER_WS_URL (a Playwright server) or BROWSER_CDP_URL (a real Chrome), not both");
    this.opts = opts;
    this.viewport = opts.viewport ?? { width: 1280, height: 800 };
    this.stepTimeoutMs = opts.stepTimeoutMs ?? 10_000;
    this.runBudgetMs = opts.runBudgetMs ?? 600_000;
    this.rules = parseDenyList(opts.deny);
  }

  /** The service the environment describes (see the file comment). */
  static fromEnv(env: Record<string, string | undefined> = process.env): BrowserService {
    const [width, height] = (env["BROWSER_VIEWPORT"] ?? "1280x800").split("x").map(Number);
    if (!width || !height) throw new Error(`BROWSER_VIEWPORT must be WIDTHxHEIGHT, got "${env["BROWSER_VIEWPORT"]}"`);
    return new BrowserService({
      wsUrl: env["BROWSER_WS_URL"] || undefined,
      wsPath: env["BROWSER_WS_PATH"] || undefined,
      cdpUrl: env["BROWSER_CDP_URL"] || undefined,
      executablePath: env["BROWSER_EXECUTABLE"] || undefined,
      channel: env["BROWSER_CHANNEL"] || undefined,
      viewport: { width, height },
      stepTimeoutMs: num(env["BROWSER_STEP_TIMEOUT_MS"], 10_000),
      runBudgetMs: num(env["BROWSER_RUN_BUDGET_MS"], 600_000),
      deny: env["BROWSER_URL_DENY"],
    });
  }

  // ── The primitives ───────────────────────────────────────────────────────

  /** True over CDP: pages in a real Chrome's own profile. */
  get cdp(): boolean {
    return Boolean(this.opts.cdpUrl);
  }

  /** The run's page — context + page created on first use. `viewport` and
   *  `storageState` apply only at creation, and only over a Playwright
   *  server (over CDP both are refused — see the file comment). */
  async page(runId: string, opts: { viewport?: Viewport; storageState?: unknown } = {}): Promise<Page> {
    let run = this.runs.get(runId);
    if (!run) {
      run = { startedAt: Date.now(), exhausted: false };
      this.runs.set(runId, run);
      run.timer = setTimeout(() => {
        run!.exhausted = true;
        void this.closeSession(runId);
      }, this.runBudgetMs);
      run.timer.unref();
    }
    if (run.exhausted) throw new Error(`browser: this run's browser budget (${this.runBudgetMs} ms) is spent`);
    const live = run.session;
    if (live && !live.page.isClosed()) return live.page;
    run.session = undefined;
    await closeQuietly(live);
    const browser = await this.connect();
    let context: BrowserContext;
    if (this.cdp) {
      if (opts.storageState) throw new Error("browser: storageStateSecret does not apply over CDP — the Chrome profile's own logged-in state is the session");
      if (opts.viewport) throw new Error("browser: viewport does not apply over CDP — the window's size is the human's, and a shot is what they see");
      context = browser.contexts()[0] ?? fail("browser: the CDP Chrome has no default context (no window open?)");
    } else {
      context = await browser.newContext({
        viewport: opts.viewport ?? this.viewport,
        ...(opts.storageState ? { storageState: opts.storageState as BrowserContextOptions["storageState"] } : {}),
      });
    }
    const page = await context.newPage();
    const session: Session = { context, own: !this.cdp, page, pages: [page], refs: new Set(), replaced: false, obs: emptyObs() };
    this.watch(session, page);
    run.session = session;
    return page;
  }

  /** Observations and ref resets for one of the run's pages — on the
   *  PAGE, not the context: over CDP the context is the human's, shared. A
   *  popup or target=_blank link replaces the page — the newest page
   *  is THE page; the next snapshot says so. */
  private watch(session: Session, p: Page): void {
    p.setDefaultTimeout(this.stepTimeoutMs);
    const push = (arr: string[], v: string) => {
      arr.push(v);
      if (arr.length > OBS_CAP) arr.shift();
    };
    p.on("console", (m) => {
      if (m.type() === "error") push(session.obs.console, m.text());
    });
    p.on("pageerror", (e) => push(session.obs.pageErrors, String(e.message ?? e)));
    p.on("requestfailed", (r) => push(session.obs.failedRequests, `${r.method()} ${r.url()} — ${r.failure()?.errorText ?? "failed"}`));
    p.on("response", (r) => {
      const s = r.status();
      if (s >= 400) push(session.obs.httpErrors, `${s} ${r.request().method()} ${r.url()}`);
    });
    // A new DOCUMENT resets the refs — `domcontentloaded`, not `framenavigated`,
    // which also fires for pushState / hash changes (a Framer page does both
    // while it scrolls), where every ref still resolves.
    p.on("domcontentloaded", () => session.refs.clear());
    p.on("popup", (popup) => {
      session.page = popup;
      session.pages.push(popup);
      session.refs.clear();
      session.replaced = true;
      this.watch(session, popup);
    });
  }

  /** Resolve a ref from the latest snapshot, or a selector, to a locator. */
  locate(runId: string, target: Target): Locator {
    const s = this.live(runId);
    if (target.ref && target.selector) throw new Error("browser: give ref OR selector, not both");
    if (target.ref) {
      const ref = target.ref.replace(/^@|^ref=/, "");
      if (!s.refs.has(ref)) {
        throw new Error(`browser: unknown ref "${ref}" — call browser/snapshot first (refs reset on every navigation)`);
      }
      return s.page.locator(`aria-ref=${ref}`);
    }
    if (target.selector) return s.page.locator(target.selector);
    throw new Error("browser: give exactly one of ref or selector");
  }

  /** The accessibility tree with `[ref=eN]` tags; records the ref table. */
  async snapshot(runId: string): Promise<{ url: string; title: string; tree: string }> {
    return this.call(runId, "snapshot", undefined, async (s, ms) => {
      const tree = await s.page.ariaSnapshot({ mode: "ai", timeout: ms });
      s.refs = new Set([...tree.matchAll(/\[ref=(e\d+)\]/g)].map((m) => m[1]!));
      const note = s.replaced ? "# a popup / new tab replaced the page; this is the newest page\n" : "";
      s.replaced = false;
      return { url: s.page.url(), title: await s.page.title(), tree: cap(note + tree, TREE_CAP) };
    });
  }

  /** See settle.ts. */
  settle(page: Page, extraMs = 0): Promise<void> {
    return settlePage(page, extraMs);
  }

  /** Drain the observations accumulated since the last call. Empty
   *  when the run has no page. */
  observe(runId: string): Observations {
    const s = this.runs.get(runId)?.session;
    if (!s) return emptyObs();
    const out = s.obs;
    s.obs = emptyObs();
    return out;
  }

  /** Close the run's context (over CDP: only its pages) and forget the run.
   *  Idempotent; the `services.onRunEnd` hook. */
  async dispose(runId: string): Promise<void> {
    const run = this.runs.get(runId);
    if (!run) return;
    this.runs.delete(runId);
    if (run.timer) clearTimeout(run.timer);
    await closeQuietly(run.session);
  }

  /** Milliseconds of browser budget the run has left (the whole budget
   *  before its first `open`). */
  remaining(runId: string): number {
    const run = this.runs.get(runId);
    if (!run) return this.runBudgetMs;
    if (run.exhausted) return 0;
    return Math.max(0, this.runBudgetMs - (Date.now() - run.startedAt));
  }

  /** Throws when the denylist refuses `url`. */
  check(url: string): void {
    const reason = denyReason(url, this.rules);
    if (reason) throw new Error(`browser: refused ${url}: ${reason}`);
  }

  /** Drop the shared connection (tests, shutdown). Every run's context goes
   *  with it; a CDP Chrome is only disconnected from, never closed. */
  async close(): Promise<void> {
    for (const id of [...this.runs.keys()]) await this.dispose(id);
    const b = this.browser;
    this.browser = undefined;
    await b?.close().catch(() => {});
  }

  // ── The verbs — one per browser/* step ───────────────────────────────────

  async open(
    runId: string,
    input: { url?: string; viewport?: Viewport; storageState?: unknown; timeoutMs?: number } = {},
  ): Promise<Navigated> {
    if (input.url) this.check(input.url);
    const page = await this.page(runId, { viewport: input.viewport, storageState: input.storageState });
    if (input.url) return this.goto(runId, { url: input.url, timeoutMs: input.timeoutMs });
    return { url: page.url(), title: await page.title(), status: 0 };
  }

  async goto(runId: string, input: { url: string; wait?: number; timeoutMs?: number }): Promise<Navigated> {
    this.check(input.url);
    return this.call(runId, "goto", input.timeoutMs, async (s, ms) => {
      const resp = await s.page.goto(input.url, { waitUntil: "domcontentloaded", timeout: ms });
      await this.landed(s);
      await settlePage(s.page, input.wait);
      return { url: s.page.url(), title: await s.page.title(), status: resp?.status() ?? 0 };
    }, input.wait);
  }

  async click(runId: string, input: Target & { button?: "left" | "right" | "middle"; double?: boolean; timeoutMs?: number }): Promise<{ url: string }> {
    return this.call(runId, "click", input.timeoutMs, async (s, ms) => {
      const loc = this.locate(runId, input);
      const o = { button: input.button, timeout: ms };
      if (input.double) await loc.dblclick(o);
      else await loc.click(o);
      await settlePage(s.page);
      await this.landed(s);
      return { url: s.page.url() };
    });
  }

  async fill(runId: string, input: Target & { value: string; timeoutMs?: number }): Promise<{ ok: true }> {
    return this.call(runId, "fill", input.timeoutMs, async (s, ms) => {
      await this.locate(runId, input).fill(input.value, { timeout: ms });
      await settlePage(s.page);
      return { ok: true };
    });
  }

  async type(runId: string, input: { text: string; delay?: number; timeoutMs?: number }): Promise<{ ok: true }> {
    return this.call(runId, "type", input.timeoutMs, async (s) => {
      await s.page.keyboard.type(input.text, { delay: input.delay });
      await settlePage(s.page);
      return { ok: true };
    });
  }

  async press(runId: string, input: { key: string; timeoutMs?: number }): Promise<{ url: string }> {
    return this.call(runId, "press", input.timeoutMs, async (s) => {
      await s.page.keyboard.press(input.key);
      await settlePage(s.page);
      await this.landed(s);
      return { url: s.page.url() };
    });
  }

  async select(runId: string, input: Target & { value: string; timeoutMs?: number }): Promise<{ ok: true }> {
    return this.call(runId, "select", input.timeoutMs, async (s, ms) => {
      await this.locate(runId, input).selectOption(input.value, { timeout: ms });
      await settlePage(s.page);
      return { ok: true };
    });
  }

  async hover(runId: string, input: Target & { timeoutMs?: number }): Promise<{ ok: true }> {
    return this.call(runId, "hover", input.timeoutMs, async (s, ms) => {
      await this.locate(runId, input).hover({ timeout: ms });
      await settlePage(s.page);
      return { ok: true };
    });
  }

  async scroll(runId: string, input: Target & { y?: number; timeoutMs?: number }): Promise<{ ok: true }> {
    return this.call(runId, "scroll", input.timeoutMs, async (s, ms) => {
      if (input.ref || input.selector) await this.locate(runId, input).scrollIntoViewIfNeeded({ timeout: ms });
      else if (typeof input.y === "number") await s.page.evaluate((y) => window.scrollBy(0, y), input.y);
      else throw new Error("give ref, selector, or y");
      await settlePage(s.page);
      return { ok: true };
    });
  }

  /** The dynamic-data verb: hold until an element is visible, a text is on
   *  the page, the URL matches, or `ms` pass. Its cap is the run's remaining
   *  budget, not the per-call default — outlasting a slow fetch is its job. */
  async wait(runId: string, input: Target & { text?: string; url?: string; ms?: number; timeoutMs?: number }): Promise<{ ok: true; waitedMs: number }> {
    const budget = input.timeoutMs ?? this.remaining(runId);
    return this.call(runId, "wait", budget, async (s, ms) => {
      const t0 = Date.now();
      if (typeof input.ms === "number") await sleep(Math.min(input.ms, ms));
      else if (input.text) await s.page.locator(`text=${input.text}`).first().waitFor({ state: "visible", timeout: ms });
      else if (input.url) {
        const want = input.url;
        await s.page.waitForURL(want.includes("*") ? want : (u) => u.href.includes(want), { timeout: ms });
      } else if (input.ref || input.selector) await this.locate(runId, input).waitFor({ state: "visible", timeout: ms });
      else throw new Error("give ref, selector, text, url, or ms");
      return { ok: true, waitedMs: Date.now() - t0 };
    });
  }

  async text(runId: string, input: Target & { timeoutMs?: number } = {}): Promise<{ text: string }> {
    return this.call(runId, "text", input.timeoutMs, async (s, ms) => {
      const loc = input.ref || input.selector ? this.locate(runId, input) : s.page.locator("body");
      return { text: cap(await loc.innerText({ timeout: ms }), TEXT_CAP) };
    });
  }

  /** The full frame to `shots/NNN.png` under the run's artifacts; the
   *  0.6× copy rides along as `small` for the step to hand the model. */
  async screenshot(
    runId: string,
    input: Target & { fullPage?: boolean; name?: string; timeoutMs?: number },
    artifacts: ArtifactsCapability,
  ): Promise<Shot & { small: Buffer }> {
    return this.call(runId, "screenshot", input.timeoutMs, async (s, ms) => {
      // scale "css": one pixel per CSS pixel — a headed Chrome on a Retina
      // display would otherwise shoot at device pixels (2×: four times the
      // model's tokens); headless is DPR 1 either way.
      const png =
        input.ref || input.selector
          ? await this.locate(runId, input).screenshot({ type: "png", scale: "css", timeout: ms })
          : await s.page.screenshot({ type: "png", scale: "css", fullPage: input.fullPage, timeout: ms });
      return saveShot(artifacts, runId, png, input.name);
    });
  }

  /** JavaScript IN THE PAGE, never in this host; the JSON result, capped. */
  async evaluate(runId: string, input: { expression: string; timeoutMs?: number }): Promise<{ result: unknown; truncated?: true }> {
    return this.call(runId, "evaluate", input.timeoutMs, async (s) => {
      const result: unknown = await s.page.evaluate(input.expression);
      const json = JSON.stringify(result ?? null);
      if (json.length <= TEXT_CAP) return { result: result ?? null };
      return { result: json.slice(0, TEXT_CAP), truncated: true };
    });
  }

  async back(runId: string, input: { timeoutMs?: number } = {}): Promise<{ url: string; title: string }> {
    return this.call(runId, "back", input.timeoutMs, async (s, ms) => {
      // "commit", not "domcontentloaded": a real Chrome restores history from
      // its back/forward cache, which fires no load events (headless has the
      // cache off). Settle does the waiting either way.
      await s.page.goBack({ waitUntil: "commit", timeout: ms });
      // A bfcache restore (a real Chrome over CDP) fires no domcontentloaded, so
      // the ref-reset handler never runs; clear here — before landed() may throw —
      // or refs from the page we navigated away from survive this navigation.
      s.refs.clear();
      await this.landed(s);
      await settlePage(s.page);
      return { url: s.page.url(), title: await s.page.title() };
    });
  }

  /** The current storage state (cookies + localStorage) to
   *  `state.json` under the run's artifacts, for a human to paste under
   *  Secrets once. */
  async state(runId: string, artifacts: ArtifactsCapability): Promise<{ path: string; url: string }> {
    if (this.cdp) throw new Error("browser/state does not apply over CDP — it would export the human's whole Chrome profile, which is already the session");
    return this.call(runId, "state", undefined, async (s) => {
      const state = await s.context.storageState();
      await artifacts.write(runId, "state.json", JSON.stringify(state, null, 2));
      return { path: "state.json", url: `/artifacts/${runId}/state.json` };
    });
  }

  /** Explicit teardown of the run's context (over CDP: its pages); the
   *  budget clock keeps running. */
  async closeRun(runId: string): Promise<{ ok: true }> {
    await this.closeSession(runId);
    return { ok: true };
  }

  /** The declared compound: open → steps → settle → shoot → drain. A
   *  step that fails fails the whole capture, with the step named. */
  async capture(
    runId: string,
    input: { url: string; steps?: CaptureStep[]; fullPage?: boolean; wait?: number; viewport?: Viewport; timeoutMs?: number },
    artifacts: ArtifactsCapability,
  ): Promise<Shot & { small: Buffer; errors: string[]; pageUrl: string; title: string }> {
    await this.open(runId, { url: input.url, viewport: input.viewport, timeoutMs: input.timeoutMs });
    for (const [i, step] of (input.steps ?? []).entries()) {
      const { do: verb, ...args } = step;
      if (!(CAPTURE_VERBS as readonly string[]).includes(verb)) {
        throw new Error(`capture: step ${i + 1}: unknown verb "${verb}" (one of ${CAPTURE_VERBS.join(", ")})`);
      }
      try {
        await (this[verb as (typeof CAPTURE_VERBS)[number]] as (runId: string, args: unknown) => Promise<unknown>)(runId, args);
      } catch (err) {
        throw new Error(`capture: step ${i + 1} (${verb} ${JSON.stringify(args)}) failed: ${message(err)}`);
      }
    }
    const page = this.live(runId).page;
    await settlePage(page, input.wait);
    const shot = await this.screenshot(runId, { fullPage: input.fullPage, timeoutMs: input.timeoutMs }, artifacts);
    return { ...shot, errors: flattenObs(this.observe(runId)), pageUrl: page.url(), title: await page.title() };
  }

  // ── internals ────────────────────────────────────────────────────────────

  private async connect(): Promise<Browser> {
    if (this.browser?.isConnected()) return this.browser;
    if (!this.connecting) {
      this.connecting = this.launch()
        .then((b) => {
          b.on("disconnected", () => {
            if (this.browser === b) this.browser = undefined;
          });
          this.browser = b;
          return b;
        })
        .finally(() => (this.connecting = undefined));
    }
    return this.connecting;
  }

  private async launch(): Promise<Browser> {
    const { wsUrl, wsPath, cdpUrl, executablePath, channel } = this.opts;
    const { chromium } = await import("playwright-core");
    if (cdpUrl) return chromium.connectOverCDP(cdpUrl, { timeout: 30_000 });
    if (wsUrl) return chromium.connect(`${wsUrl.replace(/\/+$/, "")}/${(wsPath ?? "").replace(/^\/+/, "")}`, { timeout: 30_000 });
    try {
      return await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : { channel: channel ?? "chrome" }) });
    } catch (err) {
      throw new Error(`browser: BROWSER_WS_URL is not set and no local Chrome could be launched: ${message(err)}`);
    }
  }

  private live(runId: string): Session {
    const run = this.runs.get(runId);
    if (run?.exhausted) throw new Error(`browser: this run's browser budget (${this.runBudgetMs} ms) is spent`);
    const s = run?.session;
    if (!s) throw new Error("browser: no page for this run — call browser/open first");
    if (s.page.isClosed() || !s.context.browser()?.isConnected()) {
      run!.session = undefined;
      throw new Error("browser: the run's page was closed (a timeout, the budget, or the browser dropped) — call browser/open to start again");
    }
    return s;
  }

  /** Run `fn` against the run's session under the per-call cap: the
   *  Playwright action gets `ms` as its own timeout (a clean error), and a
   *  hard race a little longer than that closes the context — what aborts a
   *  `screenshot` or `evaluate` stuck on a wedged main thread. */
  private async call<T>(
    runId: string,
    verb: string,
    timeoutMs: number | undefined,
    fn: (s: Session, ms: number) => Promise<T>,
    extraMs = 0,
  ): Promise<T> {
    const s = this.live(runId);
    const ms = timeoutMs ?? this.stepTimeoutMs;
    const lost = Symbol("timeout");
    let timer: NodeJS.Timeout | undefined;
    const race = new Promise<typeof lost>((r) => (timer = setTimeout(() => r(lost), ms + SETTLE_SLACK_MS + Math.max(0, extraMs))));
    const work = fn(s, ms);
    work.catch(() => {}); // the loser of the race must not surface as unhandled
    try {
      const out = await Promise.race([work, race]);
      if (out === lost) {
        await this.closeSession(runId);
        throw new Error(`browser/${verb}: timed out after ${ms} ms; the run's browser context was closed — call browser/open to start again`);
      }
      return out as T;
    } catch (err) {
      const m = strictHint(message(err));
      throw new Error(m.startsWith("browser") || m.startsWith("capture:") ? m : `browser/${verb}: ${m}`);
    } finally {
      clearTimeout(timer);
    }
  }

  /** The denylist on the landed URL (redirects). */
  private async landed(s: Session): Promise<void> {
    const url = s.page.url();
    const reason = denyReason(url, this.rules);
    if (!reason) return;
    await s.page.goto("about:blank").catch(() => {});
    throw new Error(`browser: refused, the page landed on ${url}: ${reason}`);
  }

  private async closeSession(runId: string): Promise<void> {
    const run = this.runs.get(runId);
    const s = run?.session;
    if (!s) return;
    run!.session = undefined;
    await closeQuietly(s);
  }
}

export function flattenObs(o: Observations): string[] {
  return [
    ...o.console.map((s) => `console: ${s}`),
    ...o.pageErrors.map((s) => `pageerror: ${s}`),
    ...o.failedRequests.map((s) => `request failed: ${s}`),
    ...o.httpErrors.map((s) => `http: ${s}`),
  ].slice(-40);
}

function emptyObs(): Observations {
  return { console: [], pageErrors: [], failedRequests: [], httpErrors: [] };
}

/** The run's own context goes as a whole; in a shared (CDP) context only
 *  the run's pages go — the human's tabs and the other runs' pages stay. */
async function closeQuietly(s: Session | undefined): Promise<void> {
  if (!s) return;
  if (s.own) await s.context.close().catch(() => {});
  else await Promise.all(s.pages.map((p) => p.close().catch(() => {})));
}

function fail(msg: string): never {
  throw new Error(msg);
}

function cap(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n)}\n… (truncated at ${n} chars)` : s;
}

function message(err: unknown): string {
  const m = err instanceof Error ? err.message : String(err);
  return m.length > 600 ? `${m.slice(0, 600)}…` : m;
}

/** Playwright's strict-mode error lists the matches but not the way out; a
 *  model that reached for a selector retries with another broad one. */
export function strictHint(m: string): string {
  if (!m.includes("strict mode violation")) return m;
  return `${m.split("\nCall log:")[0]!.trimEnd()}\n→ act by ref instead: browser/snapshot, then the element's [ref=eN] (or narrow the selector, e.g. \`>> nth=1\`)`;
}

function num(v: string | undefined, dflt: number): number {
  if (v == null || v === "") return dflt;
  const n = Number(v);
  if (!Number.isFinite(n) || n <= 0) throw new Error(`expected a positive number, got "${v}"`);
  return n;
}
