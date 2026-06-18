import { z, defineStep, usageFromResult, computeCost, addUsage } from "vein";
import { spawn } from "node:child_process";
import {
  writeFileSync,
  readFileSync,
  mkdirSync,
  existsSync,
  rmSync,
  readdirSync,
  statSync,
  openSync,
} from "node:fs";
import { join, resolve, isAbsolute, sep, dirname } from "node:path";
import { homedir } from "node:os";
import { createConnection } from "node:net";
import vm from "node:vm";
import yaml from "js-yaml";

/**
 * BOOT-AND-EXERCISE: an autonomous "set up a repo until it actually runs" agent.
 *
 * Where `gitsee/verify-setup` is a READ-ONLY boot GATE (boot once → one
 * screenshot → one vision verdict → score), this step is the PRODUCT loop: a
 * tool-using agent that BOOTS the produced setup, DRIVES the live app in a real
 * headless browser (navigate / snapshot / click / fill), OBSERVES failures the
 * way a QA engineer would (browser console errors, failed requests, 4xx/5xx API
 * responses, AND the server's own logs), DIAGNOSES the root cause (a missing env
 * key, an unmigrated/empty DB, a missing backing service, a cloud dependency that
 * should be mocked, a wrong host-binding/start command), FIXES the
 * pm2.config.js / docker-compose.yml / repo source, REBOOTS, and repeats until
 * the frontend is genuinely functional — then reports what it changed and what
 * (if anything) is still missing.
 *
 * It is allowed to WRITE (that's the whole point — it iterates toward a working
 * setup), so it is NOT an eval signal: don't wire it into the scored optimize
 * loop. Its job is to produce a known-good `setup` + repo `diff`, not to grade
 * the explorer.
 *
 * Layout / runner is pod-faithful, same as verify-setup (staklink reads
 * `.pod-config/.user-dockerfile/pm2.config.js`; the pod-absolute
 * `cwd: /workspaces/<repo>` is rewritten to the local clone). The agent works in
 * LOCAL terms (real on-disk paths); the final deliverable rewrites them back to
 * `/workspaces/<repo>` so it stays pod-portable.
 *
 * Tools: boot, browser_open, browser_snapshot, browser_click, browser_fill,
 * browser_press, browser_observe, assess_ui (vision verdict), read_logs, bash,
 * str_replace_based_edit_tool, final_answer.
 *
 * Output: { booted, working, port, setup, report, diff, changedRepos, changed,
 *   screenshotPath, iterations, logsTail, usage, cost }. `diff` is the per-repo
 *   `git diff` of the agent's source edits (replayable on a fresh pod clone).
 *   Needs `git` + `docker` on PATH, an ANTHROPIC_API_KEY, and `npx playwright
 *   install chromium`; (for useStaklink) network for staklink.
 */

// ── shell helpers ───────────────────────────────────────────────────────────

interface ShResult {
  code: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
}

/** Run a shell command, capture output, never reject. Mirrors staklink's
 *  runner env (CI=1, non-interactive) for parity. */
function sh(
  command: string,
  cwd: string,
  env: Record<string, string>,
  timeoutMs: number,
): Promise<ShResult> {
  return new Promise((resolve) => {
    const child = spawn("sh", ["-c", command], {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, CI: "1", DEBIAN_FRONTEND: "noninteractive", ...env },
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, timeoutMs);
    child.stdout?.on("data", (d) => (stdout += d.toString()));
    child.stderr?.on("data", (d) => (stderr += d.toString()));
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ code, stdout, stderr, timedOut });
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      resolve({ code: -1, stdout, stderr: stderr + String(err), timedOut });
    });
  });
}

/** A neutral listing of the workspace's immediate entries, prepended to the
 *  prompt so the agent knows the layout (and the REAL local path) without burning
 *  steps guessing pod paths like /workspaces/<repo>. Mirrors the core agent's
 *  buildPreamble. */
function buildPreamble(wp: string): string {
  if (!existsSync(wp)) return "";
  const entries = readdirSync(wp, { withFileTypes: true })
    .filter((e) => !e.name.startsWith("."))
    .map((e) => (e.isDirectory() ? `${e.name}/` : e.name))
    .sort();
  if (!entries.length) return "";
  return (
    `Working directory (the workspace root, where the repos are cloned as siblings) is:\n  ${wp}\n` +
    `It contains:\n` +
    entries.map((e) => `- ${e}`).join("\n") +
    `\n\nUse these REAL local paths — the repos are NOT at /workspaces/<repo> here (that's the pod path; the cwd in pm2.config.js is rewritten to the local clone for you). bash runs with this directory as its cwd, so relative paths work.`
  );
}

/** Immediate subdirs of `wp` that are git repos. */
function listRepos(wp: string): string[] {
  if (!existsSync(wp)) return [];
  return readdirSync(wp, { withFileTypes: true })
    .filter((e) => e.isDirectory() && existsSync(join(wp, e.name, ".git")))
    .map((e) => e.name)
    .sort();
}

/** Per-repo `git diff` (intent-to-add so NEW files show as additions) — the
 *  replayable record of the source edits the agent made to boot the app
 *  local-first (flip a mock flag, fix host binding, move a migration). Shipped as
 *  part of the deliverable and re-applied (`git apply`) on a fresh pod clone, just
 *  like gitsee/capture-edits does for the explore path. Inlined here (rather than
 *  reusing that step) so this step keeps ONE rich output — a workflow's output is
 *  its last step's, and capture-edits would drop setup/report/booted/working. */
async function captureRepoDiff(wp: string, maxBytes = 60000): Promise<{ diff: string; changedRepos: string[] }> {
  const parts: string[] = [];
  const changedRepos: string[] = [];
  for (const repo of listRepos(wp)) {
    const dir = join(wp, repo);
    await sh("git add -A -N", dir, {}, 30000).catch(() => {});
    const d = await sh("git diff", dir, {}, 30000);
    if (d.stdout.trim()) {
      changedRepos.push(repo);
      parts.push(`=== ${repo} ===\n${d.stdout}`);
    }
  }
  let diff = parts.join("\n\n");
  if (diff.length > maxBytes) diff = diff.slice(0, maxBytes) + "\n\n[... diff truncated ...]";
  return { diff, changedRepos };
}

/** All docker container IDs (running + stopped), for the snapshot-diff teardown. */
async function dockerContainerIds(): Promise<Set<string>> {
  const r = await sh("docker ps -aq", process.cwd(), {}, 15000);
  return new Set(r.stdout.split("\n").map((s) => s.trim()).filter(Boolean));
}

/** Poll a TCP port until something is listening (the app booted) or timeout. */
function waitForPort(port: number, host: string, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve) => {
    const attempt = () => {
      const socket = createConnection({ port, host });
      socket.setTimeout(2000);
      socket.once("connect", () => {
        socket.destroy();
        resolve(true);
      });
      const retry = () => {
        socket.destroy();
        if (Date.now() >= deadline) resolve(false);
        else setTimeout(attempt, 1500);
      };
      socket.once("error", retry);
      socket.once("timeout", retry);
    };
    attempt();
  });
}

// ── booted-service logs ───────────────────────────────────────────────────────

async function captureAppLogs(wp: string, appName: string): Promise<string> {
  const chunks: string[] = [];
  const inlineLog = join(wp, ".verify", "app.log");
  if (existsSync(inlineLog)) {
    try {
      chunks.push(readFileSync(inlineLog, "utf-8"));
    } catch {
      /* ignore */
    }
  }
  const r = await sh(`npx -y pm2 logs ${appName} --nostream --lines 200`, wp, {}, 30000);
  if (r.stdout.trim()) chunks.push(r.stdout);
  const pm2dir = join(homedir(), ".pm2", "logs");
  for (const f of [`${appName}-out.log`, `${appName}-error.log`]) {
    const p = join(pm2dir, f);
    if (existsSync(p)) {
      try {
        chunks.push(`--- ${f} ---\n${readFileSync(p, "utf-8")}`);
      } catch {
        /* ignore */
      }
    }
  }
  const out = chunks.join("\n");
  return out.length > 20000 ? out.slice(-20000) : out; // keep the tail
}

const LOG_ERROR_RE =
  /(^|\b)(Error:|UnhandledPromiseRejection|unhandledRejection|uncaughtException|EADDRINUSE|ECONNREFUSED|Cannot find module|MODULE_NOT_FOUND|PrismaClientInitializationError|FATAL|panic:|Traceback \(most recent|errored|exited with code [1-9])/i;

function scanLogErrors(logs: string): string[] {
  return logs
    .split("\n")
    .filter((l) => LOG_ERROR_RE.test(l))
    .slice(0, 20);
}

// ── produced-file parsing ─────────────────────────────────────────────────────

interface TwoFiles {
  pm2?: string;
  compose?: string;
}

function splitFiles(doc: string): TwoFiles {
  const out: TwoFiles = {};
  const re = /FILENAME:\s*([^\n]+?)\s*\n+```[a-zA-Z0-9]*\n([\s\S]*?)\n```/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(doc)) !== null) {
    const name = m[1].trim().toLowerCase();
    const body = m[2];
    if (name.includes("pm2")) out.pm2 ??= body;
    else if (name.includes("compose")) out.compose ??= body;
  }
  return out;
}

function hasComposeServices(text: string | undefined): boolean {
  if (!text) return false;
  try {
    const doc = yaml.load(text) as { services?: Record<string, unknown> } | undefined;
    return !!doc?.services && Object.keys(doc.services).length > 0;
  } catch {
    return false;
  }
}

/** A callable, infinitely-chainable no-op for sandboxed require() chains. */
function callableStub(): unknown {
  const fn = function () {
    return stub;
  };
  const stub: unknown = new Proxy(fn, { get: () => stub, apply: () => stub });
  return stub;
}

interface Pm2App {
  name?: string;
  script?: string;
  args?: string;
  cwd?: string;
  env?: Record<string, unknown>;
}

function parsePm2(code: string | undefined): Pm2App[] | null {
  if (!code) return null;
  const sandbox: Record<string, unknown> = {
    module: { exports: {} },
    require: () => callableStub(),
    console: { log() {}, error() {}, warn() {}, info() {} },
    process: { env: {} },
  };
  sandbox.exports = (sandbox.module as { exports: unknown }).exports;
  try {
    vm.createContext(sandbox);
    vm.runInContext(code, sandbox, { timeout: 1000 });
    const exported = (sandbox.module as { exports: unknown }).exports as { apps?: unknown };
    return Array.isArray(exported?.apps) ? (exported.apps as Pm2App[]) : null;
  } catch {
    return null;
  }
}

/**
 * LOCAL emulation of the pod's `$POD_ID` / `$POD_URL` substitution.
 *
 * On the real sandbox the platform expands these placeholders AND its reverse
 * proxy maps `https://<podid>-<port>.<domain>` → `localhost:<port>` inside the
 * pod. Locally there's no proxy/DNS, so plain env expansion can't reproduce it —
 * we rewrite the pod-URL PATTERNS to their `localhost:<port>` equivalents, but
 * ONLY in the staged-for-boot copy. The agent-facing pm2.config.js and the final
 * `setup` keep the `$POD_*` placeholders (the pod contract is the deliverable);
 * only the booted copy is localized, so the app runs correctly here without the
 * agent "fixing" the placeholders away. (This is a local-emulation concern, not a
 * staklink one — staklink does the REAL pod expansion in prod.)
 */
function podSubstituteLocal(pm2Code: string, frontendPort: number): string {
  return pm2Code
    // a sibling service on another port: https://$POD_ID-3001.<domain> → http://localhost:3001
    .replace(/https?:\/\/\$\{?POD_ID\}?-(\d+)\.[A-Za-z0-9.\-]+/g, "http://localhost:$1")
    // this app's base, already-expanded form: https://$POD_ID.<domain> → http://localhost:<frontendPort>
    .replace(/https?:\/\/\$\{?POD_ID\}?\.[A-Za-z0-9.\-]+/g, `http://localhost:${frontendPort}`)
    // $POD_URL (optionally followed by a path/query) → http://localhost:<frontendPort>
    .replace(/\$\{?POD_URL\}?/g, `http://localhost:${frontendPort}`)
    // any leftover bare $POD_ID (rare standalone use)
    .replace(/\$\{?POD_ID\}?/g, "local");
}

/** The frontend app (named "frontend", else the first app) + its port. */
function frontendApp(apps: Pm2App[]): { app: Pm2App; port: number } | null {
  if (!apps.length) return null;
  const app = apps.find((a) => a.name === "frontend") ?? apps[0];
  const port = parseInt(String(app.env?.PORT ?? "3000"), 10) || 3000;
  return { app, port };
}

// ── file editing (str_replace_based_edit_tool) ─────────────────────────────────

const FILE_VIEW_MAX_CHARS = 200_000;

interface TextEditInput {
  command: "view" | "create" | "str_replace" | "insert";
  path: string;
  file_text?: string;
  insert_line?: number;
  new_str?: string;
  insert_text?: string;
  old_str?: string;
  view_range?: number[];
}

function resolveInCwd(p: string, cwd: string): string {
  const target = resolve(isAbsolute(p) ? p : join(cwd, p));
  const root = resolve(cwd);
  if (target !== root && !target.startsWith(root + sep)) {
    throw new Error(`path "${p}" escapes the working directory`);
  }
  return target;
}

function textEdit(input: TextEditInput, cwd: string): string {
  let target: string;
  try {
    target = resolveInCwd(input.path, cwd);
  } catch (e) {
    return `Error: ${(e as Error).message}`;
  }
  switch (input.command) {
    case "view": {
      if (!existsSync(target)) return "Error: File not found";
      if (statSync(target).isDirectory()) {
        const entries = readdirSync(target, { withFileTypes: true })
          .filter((e) => !e.name.startsWith("."))
          .map((e) => (e.isDirectory() ? `${e.name}/` : e.name))
          .sort();
        return entries.length ? entries.join("\n") : "(empty directory)";
      }
      const lines = readFileSync(target, "utf-8").split("\n");
      let start = 1;
      let end = lines.length;
      if (Array.isArray(input.view_range) && input.view_range.length === 2) {
        start = Math.max(1, input.view_range[0]);
        end = input.view_range[1] === -1 ? lines.length : input.view_range[1];
      }
      const out = lines
        .slice(start - 1, end)
        .map((l, i) => `${start + i}: ${l}`)
        .join("\n");
      return out.length > FILE_VIEW_MAX_CHARS
        ? out.slice(0, FILE_VIEW_MAX_CHARS) + "\n\n[... output truncated ...]"
        : out;
    }
    case "create": {
      mkdirSync(dirname(target), { recursive: true });
      writeFileSync(target, input.file_text ?? "");
      return `Successfully created ${input.path}`;
    }
    case "str_replace": {
      if (!existsSync(target)) return "Error: File not found";
      const content = readFileSync(target, "utf-8");
      const old = input.old_str ?? "";
      const count = old ? content.split(old).length - 1 : 0;
      if (count === 0)
        return "Error: No match found for replacement. Please check your text and try again.";
      if (count > 1)
        return `Error: Found ${count} matches for replacement text. Please provide more context to make a unique match.`;
      writeFileSync(target, content.replace(old, input.new_str ?? ""));
      return "Successfully replaced text at exactly one location.";
    }
    case "insert": {
      if (!existsSync(target)) return "Error: File not found";
      const lines = readFileSync(target, "utf-8").split("\n");
      const at = input.insert_line ?? 0;
      if (at < 0 || at > lines.length)
        return `Error: insert_line ${at} is out of range (0-${lines.length})`;
      lines.splice(at, 0, input.insert_text ?? "");
      writeFileSync(target, lines.join("\n"));
      return `Successfully inserted text after line ${at}`;
    }
    default:
      return `Error: unknown command "${(input as { command?: string }).command}"`;
  }
}

// ── live browser session (persistent across tool calls) ────────────────────────

interface Observations {
  console: string[];
  pageErrors: string[];
  failedRequests: string[];
  httpErrors: string[];
}

/** Wraps a single headless chromium page, accumulating console/network errors and
 *  exposing a snapshot→interact→re-snapshot toolset (the agent-browser pattern).
 *  Element refs (@e1…) are invalidated on every snapshot/navigation. */
class BrowserSession {
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

  /** Navigate to a path (relative to the app base). Re-reads networkidle, falls
   *  back to domcontentloaded. */
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

  /** Accessibility-ish list of visible interactive elements with @eN refs. */
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

function summarizeObs(o: Observations): string {
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

// ── vision verdict (the agent's "eyes") ────────────────────────────────────────

async function assessScreenshot(
  pngPath: string,
  url: string,
  obs: Observations,
  logs: string,
  model: string | undefined,
): Promise<{ working: boolean; reason: string; usage: ReturnType<typeof usageFromResult>; cost: number }> {
  const { generateObject } = await import("ai");
  const { anthropic } = await import("@ai-sdk/anthropic");
  const m = anthropic(model ?? process.env["VEIN_LLM_MODEL"] ?? "claude-sonnet-4-6");
  const schema = z.object({
    working: z
      .boolean()
      .describe(
        "true ONLY if the intended app UI rendered (a real, populated, styled page — a login/landing page counts) AND there is no blank/white page, error overlay, stack trace, or 404/500, AND no fatal console/network/server error is breaking the page.",
      ),
    reason: z.string().describe("one or two short sentences: what the screenshot shows + whether the errors look fatal."),
  });
  const image = readFileSync(pngPath);
  const { object, usage: rawUsage } = await generateObject({
    model: m as any,
    schema: schema as any,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: `A web app is booted locally at ${url}. Below is a full-page screenshot, the browser console/network errors, and the server log tail. Did the app's intended UI render and is it functional (not a blank/error/404/500 page, no fatal errors)?

BROWSER OBSERVATIONS:
${summarizeObs(obs)}

SERVER LOGS (tail):
${logs ? logs.slice(-6000) : "(none)"}`,
          },
          { type: "image", image },
        ],
      },
    ],
  });
  const usage = usageFromResult(rawUsage);
  const o = object as { working: boolean; reason: string };
  return { working: o.working, reason: o.reason, usage, cost: computeCost("anthropic", usage) };
}

// ── step ──────────────────────────────────────────────────────────────────────

export default defineStep({
  type: "gitsee/boot-and-exercise",
  description:
    "Autonomous 'set up a repo until it runs' agent. Stages the produced pm2.config.js + docker-compose.yml into the cloned workspace, then runs a tool-using loop that BOOTS the app (docker compose + staklink/pm2), DRIVES it in a real headless browser (open/snapshot/click/fill), OBSERVES failures (console + network 4xx/5xx + server logs), JUDGES the screen with a vision model, FIXES the config/repo (str_replace_based_edit_tool), REBOOTS, and repeats until the frontend is functional. Unlike verify-setup it is allowed to WRITE — so it is NOT a scored eval signal. Captures the agent's repo edits as a replayable per-repo `git diff` (part of the deliverable). Config: workspacePath, setup, checkPath? (default /), maxSteps? (default 120), bootTimeoutMs? (default 420000), useStaklink? (default true), bootCommand?, model?, provider?, keepUp? (default false), enabled? (default true). Output: { booted, working, port, setup, report, diff, changedRepos, changed, screenshotPath, iterations, logsTail, usage, cost }.",
  input: z.object({
    workspacePath: z.string().describe("dir containing the cloned repos as siblings (from gitsee/clone-workspace)"),
    setup: z.string().describe("the initial pm2.config.js + docker-compose.yml two-file string to boot and improve"),
    checkPath: z.string().default("/").describe("path appended to http://localhost:<port> for the first page load"),
    maxSteps: z.number().int().positive().default(120).describe("max agent tool-call iterations (a complex multi-service app + reboots burns a lot)"),
    bootTimeoutMs: z.number().int().positive().default(420000).describe("max wait for the frontend port to bind"),
    useStaklink: z.boolean().default(true).describe("boot via `npx staklink start` (prod-faithful); false = inline pm2-free boot"),
    bootCommand: z.string().default("npx -y staklink@latest start").describe("the staklink boot command (when useStaklink)"),
    model: z.string().optional(),
    provider: z.string().optional(),
    enabled: z.boolean().default(true).describe("false = no-op"),
    keepUp: z.boolean().default(false).describe("skip teardown (leave compose + apps running for debugging)"),
  }),
  output: z.any(),
  async run(cfg) {
    const wp = cfg.workspacePath;
    const logBuf: string[] = [];
    const log = (s: string) => {
      console.log(`[gitsee/boot-exercise] ${s}`);
      logBuf.push(s);
    };
    const fail = (reason: string, extra: Record<string, unknown> = {}) => ({
      booted: false as boolean | null,
      working: null as boolean | null,
      port: null as number | null,
      setup: cfg.setup,
      report: reason,
      iterations: 0,
      logsTail: logBuf.join("\n"),
      cost: 0,
      ...extra,
    });

    if (!cfg.enabled) return fail("disabled (enabled:false)");
    if (!existsSync(wp)) return fail(`workspacePath does not exist: ${wp}`);

    // ── stage the initial files (LOCAL form — agent works in real paths) ────────
    const files = splitFiles(cfg.setup);
    if (!files.pm2) return fail("no pm2.config.js in the provided setup");
    const apps = parsePm2(files.pm2);
    if (!apps) return fail("provided pm2.config.js did not parse");
    const fe = frontendApp(apps);
    if (!fe) return fail("no app found in pm2.config.js");
    let { port } = fe;
    const appName = fe.app.name ?? "frontend";

    // Rewrite pod-absolute cwd → local clone, write the agent-facing files.
    const toLocal = (s: string) => s.replace(/\/workspaces\//g, `${wp}/`);
    const toPod = (s: string) => s.split(`${wp}/`).join("/workspaces/");
    writeFileSync(join(wp, "pm2.config.js"), toLocal(files.pm2));
    if (files.compose) writeFileSync(join(wp, "docker-compose.yml"), toLocal(files.compose));
    log(`staged pm2.config.js (frontend port ${port})${files.compose ? " + docker-compose.yml" : ""}`);

    // Snapshot containers ONCE before any boot, so teardown removes everything the
    // run spun up (compose services + app-spawned stacks like a supabase CLI).
    const preBootContainers = await dockerContainerIds();
    let composeBroughtUp = false;

    const screenshotDir = join(wp, ".exercise");
    const browser = new BrowserSession(`http://localhost:${port}`, screenshotDir, 30000);

    // running tallies for vision-judge calls (folded into the final cost)
    let extraUsage = usageFromResult(undefined);
    let extraCost = 0;
    let lastWorking: boolean | null = null;
    let lastBooted: boolean | null = null;
    let lastScreenshot: string | undefined;

    // ── (re)boot routine, shared by the boot tool ───────────────────────────────
    async function bootApp(): Promise<string> {
      // Tear down a prior app instance so edited config/env is reloaded.
      await sh("npx -y pm2 delete all", wp, {}, 60000).catch(() => {});

      // Re-read the (possibly edited) pm2.config.js, re-derive the port, re-stage.
      const pm2Code = existsSync(join(wp, "pm2.config.js")) ? readFileSync(join(wp, "pm2.config.js"), "utf-8") : "";
      const curApps = parsePm2(pm2Code);
      if (!curApps) return "pm2.config.js did not parse — fix it before booting";
      const curFe = frontendApp(curApps);
      if (!curFe) return "no app in pm2.config.js";
      port = curFe.port;
      browser.setBaseUrl(`http://localhost:${port}`);
      // Localize the pod-URL placeholders ($POD_ID/$POD_URL) for the booted copy
      // ONLY — the agent-facing pm2.config.js keeps them (the pod contract).
      const pm2Staged = podSubstituteLocal(pm2Code, port);
      mkdirSync(join(wp, ".pod-config", ".user-dockerfile"), { recursive: true });
      writeFileSync(join(wp, ".pod-config", ".user-dockerfile", "pm2.config.js"), pm2Staged);

      const composeText = existsSync(join(wp, "docker-compose.yml"))
        ? readFileSync(join(wp, "docker-compose.yml"), "utf-8")
        : undefined;
      const composeServices = hasComposeServices(composeText);

      // Backing services (idempotent — safe to re-run each boot).
      if (composeServices) {
        const up = await sh("docker compose up -d --wait", wp, {}, 300000);
        if (up.code !== 0) {
          const up2 = await sh("docker compose up -d", wp, {}, 300000);
          if (up2.code !== 0) return `docker compose up failed: ${(up.stderr || up2.stderr).slice(0, 600)}`;
        }
        composeBroughtUp = true;
      }

      // Boot the apps.
      if (cfg.useStaklink) {
        const boot = await sh(cfg.bootCommand, wp, {}, 180000);
        if (boot.code !== 0) log(`staklink start returned ${boot.code}: ${boot.stderr.slice(0, 400)}`);
      } else {
        // Inline boot: use the localized (staged) env so $POD_* placeholders are
        // expanded to localhost, matching what staklink would read.
        const app = frontendApp(parsePm2(pm2Staged) ?? curApps)?.app ?? curFe.app;
        const appCwd = (app.cwd ?? "").replace(/\/workspaces\//g, `${wp}/`) || wp;
        const appEnv: Record<string, string> = {};
        for (const [k, v] of Object.entries(app.env ?? {})) appEnv[k] = String(v);
        for (const [key, command] of [
          ["REBUILD_COMMAND", appEnv.REBUILD_COMMAND],
          ["INSTALL_COMMAND", appEnv.INSTALL_COMMAND],
          ["PRE_START_COMMAND", appEnv.PRE_START_COMMAND ?? appEnv.PRE_RUN_COMMAND],
        ] as const) {
          if (!command) continue;
          const r = await sh(command, appCwd, appEnv, 600000);
          if (r.code !== 0) log(`${key} exited ${r.code}: ${r.stderr.slice(0, 300)}`);
        }
        const script = [app.script, app.args].filter(Boolean).join(" ");
        mkdirSync(join(wp, ".verify"), { recursive: true });
        const logFd = openSync(join(wp, ".verify", "app.log"), "w");
        spawn("sh", ["-c", script], {
          cwd: appCwd,
          detached: true,
          stdio: ["ignore", logFd, logFd],
          env: { ...process.env, CI: "1", ...appEnv },
        }).unref();
      }

      const booted = await waitForPort(port, "127.0.0.1", cfg.bootTimeoutMs);
      lastBooted = booted;
      const logs = await captureAppLogs(wp, appName);
      const errs = scanLogErrors(logs);
      return [
        booted ? `BOOTED: frontend port ${port} is listening.` : `NOT BOOTED: port ${port} never bound in ${cfg.bootTimeoutMs}ms.`,
        errs.length ? `Log errors (heuristic):\n${errs.slice(0, 10).join("\n")}` : "No obvious fatal log lines.",
        `Server log tail:\n${logs.slice(-3000)}`,
      ].join("\n\n");
    }

    // ── assemble the agent ──────────────────────────────────────────────────────
    const { ToolLoopAgent, tool, stepCountIs, hasToolCall, generateText } = await import("ai");
    const provider = cfg.provider ?? process.env["VEIN_LLM_PROVIDER"] ?? "anthropic";
    const modelName = cfg.model ?? process.env["VEIN_LLM_MODEL"];
    let model: any;
    let textEditorTool: any;
    let providerOptions: any;
    if (provider === "anthropic") {
      const { anthropic } = await import("@ai-sdk/anthropic");
      model = anthropic(modelName ?? "claude-sonnet-4-6");
      textEditorTool = anthropic.tools.textEditor_20250728({
        execute: async (input: TextEditInput) => textEdit(input, wp),
      });
      providerOptions = { anthropic: { cacheControl: { type: "ephemeral" } } };
    } else if (provider === "openai") {
      const { openai } = await import("@ai-sdk/openai");
      model = openai(modelName ?? "gpt-4o");
    } else {
      throw new Error(`Unknown LLM provider: "${provider}". Supported: anthropic, openai`);
    }

    const tools: Record<string, any> = {
      boot: tool({
        description:
          "Boot (or reboot) the app: re-stage the current pm2.config.js + docker-compose.yml, bring up backing services, start the frontend, and wait for its port. Call after every config/repo edit. Returns whether the port bound + the server log tail + heuristic error lines. Reboots can be slow (install/build/migrate) — batch related edits before calling.",
        inputSchema: z.object({}) as any,
        execute: async () => {
          try {
            return await bootApp();
          } catch (e) {
            return `boot failed: ${(e as Error).message}`;
          }
        },
      }),
      browser_open: tool({
        description:
          "Open a path of the booted app in a real headless browser (default the app root). Returns the HTTP status + page title. Call after boot, and after actions that should navigate.",
        inputSchema: z.object({ path: z.string().default(cfg.checkPath).describe("path like /, /login, /dashboard") }) as any,
        execute: async ({ path }: { path: string }) => browser.open(path ?? cfg.checkPath),
      }),
      browser_snapshot: tool({
        description:
          "List the currently visible interactive elements (links, buttons, inputs…) with @eN refs to click/fill. Refs reset on every navigation — re-snapshot after anything that changes the page.",
        inputSchema: z.object({}) as any,
        execute: async () => browser.snapshot(),
      }),
      browser_click: tool({
        description: "Click an element by its @eN ref from the latest browser_snapshot.",
        inputSchema: z.object({ ref: z.string().describe("an @eN ref, e.g. e3") }) as any,
        execute: async ({ ref }: { ref: string }) => browser.click(ref.replace(/^@/, "")),
      }),
      browser_fill: tool({
        description: "Type text into an input/textarea by its @eN ref (clears first). Use to fill login forms etc.",
        inputSchema: z.object({ ref: z.string(), text: z.string() }) as any,
        execute: async ({ ref, text }: { ref: string; text: string }) => browser.fill(ref.replace(/^@/, ""), text),
      }),
      browser_press: tool({
        description: "Press a keyboard key on the page (e.g. Enter to submit a focused form).",
        inputSchema: z.object({ key: z.string().describe("a key name like Enter, Tab, Escape") }) as any,
        execute: async ({ key }: { key: string }) => browser.press(key),
      }),
      browser_observe: tool({
        description:
          "Drain the browser console errors, failed requests, and 4xx/5xx API responses accumulated since the last call. THE key signal for 'renders but is broken' — a page can look fine while every API call 500s.",
        inputSchema: z.object({}) as any,
        execute: async () => summarizeObs(browser.drain()),
      }),
      assess_ui: tool({
        description:
          "Take a fresh full-page screenshot and have a vision model judge whether the intended UI rendered and looks functional (vs blank/error/404/500), considering the recent browser + server errors. Use it to confirm progress after fixes.",
        inputSchema: z.object({}) as any,
        execute: async () => {
          const shot = await browser.screenshot();
          if (!shot) return `no screenshot (browser unavailable: ${browser.note})`;
          lastScreenshot = shot;
          const logs = await captureAppLogs(wp, appName);
          try {
            const v = await assessScreenshot(shot, `http://localhost:${port}${cfg.checkPath}`, browser.obs, logs, modelName);
            extraUsage = addUsage(extraUsage, v.usage);
            extraCost += v.cost;
            lastWorking = v.working;
            return `working=${v.working} — ${v.reason}`;
          } catch (e) {
            return `vision assessment failed: ${(e as Error).message}`;
          }
        },
      }),
      read_logs: tool({
        description: "Read the booted frontend's recent server logs (stdout+stderr). Use to find the cause of a 500 or a crash loop.",
        inputSchema: z.object({}) as any,
        execute: async () => {
          const logs = await captureAppLogs(wp, appName);
          return logs.slice(-8000) || "(no logs captured)";
        },
      }),
      bash: tool({
        description:
          "Run a bash command in the workspace (cwd = the repos' parent). Use to inspect manifests/lockfiles/.env files, grep (rg) for env-var/service usage, check migrations, run one-off db commands, etc. 10-min timeout.",
        inputSchema: z.object({ command: z.string() }) as any,
        execute: async ({ command }: { command: string }) => {
          const r = await sh(command, wp, {}, 600000);
          const out = `${r.stdout}${r.stderr ? `\n[stderr]\n${r.stderr}` : ""}`.trim();
          return (out || `(exit ${r.code}, no output)`).slice(0, 12000);
        },
      }),
      str_replace_based_edit_tool:
        textEditorTool ??
        tool({
          description:
            "View and edit text files (sandboxed to the workspace). Commands: view (path, optional view_range), create (path, file_text), str_replace (path, old_str EXACTLY once, new_str), insert (path, insert_line, insert_text). Edit pm2.config.js env, docker-compose.yml, or repo source to fix boot/runtime problems.",
          inputSchema: z.object({
            command: z.enum(["view", "create", "str_replace", "insert"]),
            path: z.string(),
            file_text: z.string().optional(),
            insert_line: z.number().int().optional(),
            new_str: z.string().optional(),
            insert_text: z.string().optional(),
            old_str: z.string().optional(),
            view_range: z.array(z.number().int()).optional(),
          }) as any,
          execute: async (input: TextEditInput) => textEdit(input, wp),
        }),
      final_answer: tool({
        description:
          "Call when the app is functional, or when you've hit a wall you genuinely cannot fix. Provide a markdown report with: ## SUMMARY (what you changed + final state), ## WORKING (features you verified), ## MISSING (anything still broken/absent to make it great, each with the evidence and the fix it needs).",
        inputSchema: z.object({ report: z.string() }) as any,
        execute: async ({ report }: { report: string }) => report,
      }),
    };

    const system = `You are an autonomous setup-and-QA engineer. A web app's source is cloned under the working directory, and an initial pm2.config.js + docker-compose.yml have been staged there. Your mission: get the app's FRONTEND genuinely FUNCTIONAL — not just rendering, but with working navigation and core user flows — by ITERATING until it works.

The loop:
1. boot — (re)boot the app. It tells you whether the frontend port bound + the server log tail.
2. browser_open then browser_snapshot — load the app and see what's on the page. Click/fill to exercise REAL flows: follow nav links, try to sign in, submit the primary form, open the main views.
3. browser_observe (console + failed requests + 4xx/5xx API responses) and read_logs (server side). assess_ui for a vision verdict on the current screen. A page that LOOKS fine but whose API calls all 500 is BROKEN — always check browser_observe.
4. Diagnose each failure → root cause: a missing/incorrect env var, an unmigrated or empty database (needs a PRE_START_COMMAND migrate/seed), a missing backing service, a cloud dependency that should be MOCKED (USE_MOCKS / placeholder creds), a wrong host binding (must be 0.0.0.0) or start command.
5. Fix with str_replace_based_edit_tool: edit the pm2.config.js env block, the docker-compose.yml, or the repo source. Prefer the repo's own mock/offline mode. Put FILE changes in the repo; put RUNTIME steps (migrate/seed/reset) in PRE_START_COMMAND.
6. boot again and re-verify.

Keep going until: the port binds, the intended UI renders, primary navigation works, and there are no fatal console/network/server errors — confirmed via assess_ui + browser_observe.

Rules:
- Make MINIMAL, surgical edits. Don't refactor.
- ENV VARS GO IN pm2.config.js. Whenever you set or fix an environment variable, put it in the relevant app's \`env\` block in pm2.config.js — even if you also write it into a repo .env file to make the running app pick it up. Repo .env files (.env.local, .env, …) are usually GITIGNORED, so they are NOT captured in the deliverable diff and would be LOST on a fresh clone; the pm2.config.js env block is the source of truth that ships. Do not rely on a gitignored .env file alone for any var the app needs to boot/run.
- POD URLS ARE AUTOMATIC — KEEP THEM. Env values may use the pod placeholders \`$POD_URL\` (this app's own public base URL) and \`https://$POD_ID-<port>.<domain>\` (a sibling service on another port, e.g. a backend on 3001 or supabase on 54321). This is how the REAL sandbox works, and these are AUTOMATICALLY substituted to the correct \`http://localhost:<port>\` for you when the app is booted locally — so the live app you're testing already has working URLs. KEEP the \`$POD_URL\` / \`$POD_ID-<port>\` placeholders as-is in pm2.config.js; do NOT rewrite them to localhost yourself (that would break the deliverable on the real pod). If a URL is wired wrong, fix WHICH placeholder/port it points at (or a genuinely missing var), not the placeholder syntax.
- Keep the frontend pm2 service named "frontend"; the dev server MUST bind 0.0.0.0.
- Only a primary datastore gets a docker-compose service; mock everything else (no containers for caches/queues/cloud APIs).
- Reboots are expensive — batch related fixes before rebooting.
- Don't loop forever: if you've truly exhausted what you can fix, call final_answer and honestly report what's still missing.`;

    const preamble = buildPreamble(wp);
    const prompt = `${preamble ? preamble + "\n\n" : ""}The workspace is staged and ready. Begin by calling boot, then drive the app and iterate until the frontend is functional. The initial setup you're improving is:\n\n${cfg.setup}`;

    const agent = new ToolLoopAgent({
      model,
      instructions: system,
      tools,
      stopWhen: [hasToolCall("final_answer"), stepCountIs(cfg.maxSteps)],
      ...(providerOptions ? { providerOptions } : {}),
      onStepFinish: (sf: any) => {
        if (!Array.isArray(sf.content)) return;
        for (const c of sf.content) {
          if (c.type === "tool-call" && c.toolName !== "final_answer") {
            console.log("[boot-exercise] TOOL:", c.toolName, JSON.stringify(c.input ?? {}).slice(0, 200));
          }
        }
      },
    });

    let usage = usageFromResult(undefined);
    let cost = 0;
    let report = "";
    let steps = 0;
    try {
      const res = await agent.generate({ prompt });
      steps = (res.steps ?? []).length;
      usage = usageFromResult(res.totalUsage ?? res.usage);
      cost = computeCost(provider, usage);

      // Extract the final_answer text (else salvage the last reasoning text).
      const allSteps = res.steps ?? [];
      let lastText = "";
      for (const step of allSteps)
        for (const item of step.content) if (item.type === "text" && item.text?.trim()) lastText = item.text.trim();
      for (const step of [...allSteps].reverse()) {
        const fa = step.content.find((c: any) => c.type === "tool-result" && c.toolName === "final_answer");
        if (fa) {
          report = String((fa as { output?: unknown }).output ?? "");
          break;
        }
      }
      if (!report) {
        // Ran out of budget without a final_answer — force one no-tools turn.
        try {
          const forced = await generateText({
            model,
            ...(providerOptions ? { providerOptions } : {}),
            messages: [
              ...((res.response?.messages ?? []) as any[]),
              {
                role: "user",
                content:
                  "You've used your budget — do NOT call tools. Write the final report now: ## SUMMARY, ## WORKING, ## MISSING (each with evidence + the fix).",
              },
            ],
          });
          report = (forced.text ?? "").trim();
          const fu = usageFromResult(forced.totalUsage ?? forced.usage);
          usage = addUsage(usage, fu);
          cost += computeCost(provider, fu);
        } catch {
          report = lastText || "(agent produced no final report)";
        }
      }
    } finally {
      await browser.close();
      if (!cfg.keepUp) {
        log("teardown: pm2 delete all + staklink stop + compose down + remove spawned containers");
        await sh("npx -y pm2 delete all", wp, {}, 60000).catch(() => {});
        if (cfg.useStaklink)
          await sh(`${cfg.bootCommand.split(" ").slice(0, -1).join(" ")} stop`, wp, {}, 30000).catch(() => {});
        if (composeBroughtUp) await sh("docker compose down -v", wp, {}, 120000).catch(() => {});
        try {
          const now = await dockerContainerIds();
          const spawned = [...now].filter((id) => !preBootContainers.has(id));
          if (spawned.length) {
            log(`removing ${spawned.length} app-spawned container(s)`);
            await sh(`docker rm -fv ${spawned.join(" ")}`, wp, {}, 120000).catch(() => {});
            await sh("docker network prune -f", wp, {}, 30000).catch(() => {});
          }
        } catch {
          /* ignore */
        }
        // Remove only the staging dir we created; KEEP the edited pm2.config.js +
        // docker-compose.yml + repo edits — they ARE the deliverable.
        try {
          rmSync(join(wp, ".pod-config"), { recursive: true, force: true });
        } catch {
          /* ignore */
        }
      } else {
        log("keepUp:true — leaving services + apps running");
      }
    }

    // Re-read the final (edited) files and re-emit in pod-portable form.
    const finalPm2 = existsSync(join(wp, "pm2.config.js")) ? readFileSync(join(wp, "pm2.config.js"), "utf-8") : toLocal(files.pm2);
    const finalCompose = existsSync(join(wp, "docker-compose.yml"))
      ? readFileSync(join(wp, "docker-compose.yml"), "utf-8")
      : files.compose;
    const finalSetup =
      `FILENAME: pm2.config.js\n\n\`\`\`js\n${toPod(finalPm2)}\n\`\`\`` +
      (finalCompose ? `\n\nFILENAME: docker-compose.yml\n\n\`\`\`yaml\n${toPod(finalCompose)}\n\`\`\`` : "");

    // Capture the agent's repo edits as a replayable diff (part of the
    // deliverable). Teardown only touches pm2/docker/.pod-config (none in a repo
    // tree), so the source edits are intact here.
    const { diff, changedRepos } = await captureRepoDiff(wp);
    if (changedRepos.length) log(`captured repo edits in: ${changedRepos.join(", ")}`);

    usage = addUsage(usage, extraUsage);
    cost += extraCost;
    const logs = logBuf.join("\n");
    log(`done: working=${lastWorking} steps=${steps} changedRepos=${changedRepos.length} $${cost.toFixed(4)}`);

    return {
      booted: lastBooted,
      working: lastWorking,
      port,
      setup: finalSetup,
      report,
      diff,
      changedRepos,
      changed: changedRepos.length > 0,
      screenshotPath: lastScreenshot,
      iterations: steps,
      logsTail: logs.slice(-4000),
      usage,
      cost,
    };
  },
});
