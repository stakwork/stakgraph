/**
 * Browser service: a headless-chromium page session that accumulates
 * console/network errors and exposes a snapshot→interact→re-snapshot toolset
 * (the agent-browser pattern). Lifted from `boot-and-exercise.ts`.
 *
 * Sessions are PER-RUN, keyed by runId (a live page is mutable state; the
 * optimize loop runs many evals concurrently, so a singleton page would
 * collide). The manager hands out / reuses one `BrowserSession` per runId and
 * disposes it on `onRunEnd(runId)`.
 */
import { mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";

export interface Observations {
  console: string[];
  pageErrors: string[];
  failedRequests: string[];
  httpErrors: string[];
}

export class BrowserSession {
  private chromium: any;
  private browser: any;
  private page: any;
  private refs = new Map<string, any>();
  obs: Observations = { console: [], pageErrors: [], failedRequests: [], httpErrors: [] };
  available = true;
  note = "";

  constructor(private baseUrl: string, private screenshotDir: string, private timeoutMs: number) {}

  private push(arr: string[], v: string) {
    arr.push(v);
    if (arr.length > 200) arr.shift();
  }

  async ensure(): Promise<boolean> {
    if (this.page) return true;
    try {
      ({ chromium: this.chromium } = await import("@playwright/test"));
    } catch (e) {
      this.available = false;
      this.note = `playwright not available: ${(e as Error).message}`;
      return false;
    }
    try {
      this.browser = await this.chromium.launch();
    } catch (e) {
      this.available = false;
      this.note = `could not launch chromium (browsers installed?): ${(e as Error).message}`;
      return false;
    }
    this.page = await this.browser.newPage();
    this.page.on("console", (m: any) => {
      if (m.type() === "error") this.push(this.obs.console, m.text());
    });
    this.page.on("pageerror", (e: any) => this.push(this.obs.pageErrors, String(e)));
    this.page.on("requestfailed", (r: any) =>
      this.push(this.obs.failedRequests, `${r.method()} ${r.url()} — ${r.failure()?.errorText ?? "failed"}`),
    );
    this.page.on("response", (r: any) => {
      const s = r.status();
      if (s >= 400) this.push(this.obs.httpErrors, `${s} ${r.request().method()} ${r.url()}`);
    });
    return true;
  }

  async open(path: string): Promise<string> {
    if (!(await this.ensure())) return `browser unavailable: ${this.note}`;
    const url = path.startsWith("http") ? path : `${this.baseUrl}${path.startsWith("/") ? "" : "/"}${path}`;
    this.refs.clear();
    try {
      const resp = await this.page
        .goto(url, { waitUntil: "networkidle", timeout: this.timeoutMs })
        .catch(async () => this.page.goto(url, { waitUntil: "domcontentloaded", timeout: this.timeoutMs }));
      const status = resp?.status?.();
      const title = await this.page.title().catch(() => "");
      return `opened ${url} (HTTP ${status ?? "?"}, title: ${JSON.stringify(title)})`;
    } catch (e) {
      return `navigation failed: ${(e as Error).message}`;
    }
  }

  async snapshot(): Promise<string> {
    if (!this.page) return "no page open — call browser_open first";
    this.refs.clear();
    const sel =
      "a, button, input, select, textarea, [role=button], [role=link], [role=tab], [onclick], [type=submit], [type=button]";
    let handles: any[] = [];
    try {
      handles = await this.page.$$(sel);
    } catch (e) {
      return `snapshot failed: ${(e as Error).message}`;
    }
    const lines: string[] = [];
    let i = 1;
    for (const h of handles) {
      let visible = false;
      try {
        visible = await h.isVisible();
      } catch {
        /* detached */
      }
      if (!visible) continue;
      let info: any = {};
      try {
        info = await h.evaluate((el: any) => ({
          tag: el.tagName.toLowerCase(),
          type: el.getAttribute("type") || "",
          role: el.getAttribute("role") || "",
          name: (
            el.getAttribute("aria-label") ||
            el.getAttribute("placeholder") ||
            (el.innerText || el.value || "").trim() ||
            el.getAttribute("name") ||
            ""
          ).slice(0, 80),
        }));
      } catch {
        continue;
      }
      const ref = `e${i++}`;
      this.refs.set(ref, h);
      const desc = info.type
        ? `${info.tag}[type=${info.type}]`
        : info.role
          ? `${info.tag}[role=${info.role}]`
          : info.tag;
      lines.push(`@${ref} ${desc} ${JSON.stringify(info.name)}`);
      if (i > 120) break;
    }
    const cur = await this.page.url().catch(() => "");
    return `URL: ${cur}\n` + (lines.join("\n") || "(no visible interactive elements)");
  }

  async click(ref: string): Promise<string> {
    const h = this.refs.get(ref);
    if (!h) return `unknown ref ${ref} — re-run browser_snapshot (refs reset on navigation)`;
    try {
      await h.click({ timeout: 8000 });
      await this.page.waitForLoadState("networkidle", { timeout: 5000 }).catch(() => {});
      return `clicked ${ref}`;
    } catch (e) {
      return `click failed: ${(e as Error).message}`;
    }
  }

  async fill(ref: string, text: string): Promise<string> {
    const h = this.refs.get(ref);
    if (!h) return `unknown ref ${ref} — re-run browser_snapshot`;
    try {
      await h.fill(text, { timeout: 8000 });
      return `filled ${ref}`;
    } catch (e) {
      return `fill failed: ${(e as Error).message}`;
    }
  }

  async press(key: string): Promise<string> {
    if (!this.page) return "no page open";
    try {
      await this.page.keyboard.press(key);
      await this.page.waitForLoadState("networkidle", { timeout: 5000 }).catch(() => {});
      return `pressed ${key}`;
    } catch (e) {
      return `press failed: ${(e as Error).message}`;
    }
  }

  async screenshot(name = "exercise.png"): Promise<string | undefined> {
    if (!this.page) return undefined;
    mkdirSync(this.screenshotDir, { recursive: true });
    const p = join(this.screenshotDir, name);
    await this.page.screenshot({ path: p, fullPage: true }).catch(() => {});
    return existsSync(p) ? p : undefined;
  }

  /** Drain + clear the accumulated observations (errors since the last drain). */
  drain(): Observations {
    const snap = this.obs;
    this.obs = { console: [], pageErrors: [], failedRequests: [], httpErrors: [] };
    return snap;
  }

  /** Drain + format the observations as a readable summary (so a seeded
   *  tool-step doesn't need to import `summarizeObs`). */
  drainSummary(): string {
    return summarizeObs(this.drain());
  }

  setBaseUrl(url: string) {
    this.baseUrl = url;
  }

  async close() {
    await this.browser?.close().catch(() => {});
    this.browser = undefined;
    this.page = undefined;
    this.refs.clear();
  }
}

export function summarizeObs(o: Observations): string {
  const parts: string[] = [];
  const block = (label: string, arr: string[]) =>
    arr.length ? `${label} (${arr.length}):\n` + arr.slice(0, 15).map((s) => `  - ${s.slice(0, 200)}`).join("\n") : "";
  for (const [label, arr] of [
    ["console errors", o.console],
    ["page errors", o.pageErrors],
    ["failed requests", o.failedRequests],
    ["HTTP 4xx/5xx responses", o.httpErrors],
  ] as const) {
    const b = block(label, arr);
    if (b) parts.push(b);
  }
  return parts.length ? parts.join("\n") : "no console/network errors observed";
}

/** Per-run browser sessions, keyed by runId. */
export class BrowserManager {
  private sessions = new Map<string, BrowserSession>();

  /** Get (or create) the session for a run. `baseUrl`/`screenshotDir` are only
   *  used on first creation; later calls just return the existing session. */
  session(runId: string, baseUrl: string, screenshotDir: string, timeoutMs = 30000): BrowserSession {
    let s = this.sessions.get(runId);
    if (!s) {
      s = new BrowserSession(baseUrl, screenshotDir, timeoutMs);
      this.sessions.set(runId, s);
    }
    return s;
  }

  has(runId: string): boolean {
    return this.sessions.has(runId);
  }

  /** The existing session for a run, or undefined (no auto-create). */
  get(runId: string): BrowserSession | undefined {
    return this.sessions.get(runId);
  }

  /** Dispose a run's session (called from onRunEnd). Idempotent. */
  async dispose(runId: string): Promise<void> {
    const s = this.sessions.get(runId);
    if (!s) return;
    this.sessions.delete(runId);
    await s.close().catch(() => {});
  }
}
