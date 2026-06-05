import { z, defineStep, usageFromResult, computeCost } from "vein";
import { spawn } from "node:child_process";
import { writeFileSync, readFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { createConnection } from "node:net";
import { openSync } from "node:fs";
import vm from "node:vm";
import yaml from "js-yaml";

/**
 * THE BOOT GATE for the gitsee setup-profiler. Takes the produced
 * pm2.config.js + docker-compose.yml pair and ACTUALLY RUNS IT the way the pod
 * does — then proves the frontend is reachable with a headless browser. This is
 * the dominant eval signal: a setup that doesn't boot is a failure no matter how
 * well its file shape matches the gold (see gitsee-eval / score-setup gate).
 *
 * Layout & runner are pod-faithful (see staklink/src/proxy/startup.ts):
 *   - the workspace's repos are already cloned as siblings under `workspacePath`
 *     (by gitsee/clone-workspace, which gitsee-eval runs first);
 *   - we write the produced files where staklink looks: pm2.config.js at the
 *     workspace root (staklink's findPm2Config search list) and at
 *     `.pod-config/.user-dockerfile/pm2.config.js` (its top-priority path),
 *     plus docker-compose.yml at the root;
 *   - the produced pm2 `cwd: /workspaces/<repo>` is rewritten to
 *     `<workspacePath>/<repo>` so it resolves locally (the chosen "clone into a
 *     real workspaces dir + rewrite the cwd prefix" approach);
 *   - `docker compose up -d --wait` brings up the backing services (DB, …);
 *   - staklink boots the apps EXACTLY like prod: REBUILD → INSTALL → PRE_START →
 *     pm2 start → POST_START. (It does NOT run BUILD_COMMAND — dev-mode boot is
 *     the faithful target.) Set `useStaklink:false` for a pm2-free inline boot
 *     (run the env commands + spawn the `script` directly) on CI hosts.
 *
 * Then we poll the frontend's PORT and load `http://localhost:<port><checkPath>`
 * in headless chromium (via @playwright/test), asserting a non-error response
 * and no framework error overlay, and capture a screenshot. Missing browsers
 * degrade gracefully to a boot-only gate (`rendered: null`).
 *
 * Output: { booted, rendered, port, httpStatus, title, reason, logs,
 *   screenshotPath }. `booted`/`rendered` feed score-setup's dominant gate.
 *
 * Self-contained: imports only `vein`, `@playwright/test` (lazy), and Node
 * builtins. Needs `git` (clone, upstream), `docker` (compose), and — when
 * `useStaklink` — network access for `npx staklink`. Playwright browsers must be
 * installed (`npx playwright install chromium`) for the render check.
 */

// ── shell helpers ───────────────────────────────────────────────────────────

interface ShResult {
  code: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
}

/** Run a shell command, capture output, never reject (returns the exit code).
 *  Mirrors staklink's runner env (CI=1, non-interactive) for parity. */
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

/** All docker container IDs (running + stopped). Used to snapshot the container
 *  set before boot so teardown can remove anything the app spun up (a `supabase
 *  start` CLI stack, a minio, etc.) that our `docker compose down` never sees. */
async function dockerContainerIds(): Promise<Set<string>> {
  const r = await sh("docker ps -aq", process.cwd(), {}, 15000);
  return new Set(r.stdout.split("\n").map((s) => s.trim()).filter(Boolean));
}

/** Poll a TCP port until something is listening (the app booted) or we time out. */
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

/** Grab the booted app's recent logs so we can check for fatal errors a
 *  screenshot can't show. Tries the inline-boot log file, then pm2 (staklink
 *  runs apps under pm2), then pm2's on-disk log files. Returns the TAIL. */
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

/** Obvious fatal-error lines in the logs (heuristic; the vision judge also reads
 *  the logs and is authoritative when on). */
function scanLogErrors(logs: string): string[] {
  return logs
    .split("\n")
    .filter((l) => LOG_ERROR_RE.test(l))
    .slice(0, 20);
}

// ── produced-file parsing (mirrors gitsee/score-setup) ────────────────────────

interface TwoFiles {
  pm2?: string;
  compose?: string;
}

/** Split a "FILENAME: <name>\n```lang\n…\n```" doc into the pm2 + compose blocks. */
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

/** Does the compose file declare any real services? A service-less compose (just
 *  a network — the gitsee convention for "no backing services needed") must NOT
 *  be `docker compose up`'d: that errors with "no service selected". */
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

/** Eval `module.exports = {...}` in a locked-down vm. Returns the apps, or null. */
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

/** The frontend app (named "frontend", else the first app) + its port. */
function frontendApp(apps: Pm2App[]): { app: Pm2App; port: number } | null {
  if (!apps.length) return null;
  const app = apps.find((a) => a.name === "frontend") ?? apps[0];
  const port = parseInt(String(app.env?.PORT ?? "3000"), 10) || 3000;
  return { app, port };
}

// ── headless render check ─────────────────────────────────────────────────────

interface RenderResult {
  rendered: boolean | null; // null = could not run the browser (no boot-gate impact)
  httpStatus?: number;
  title?: string;
  screenshotPath?: string;
  note?: string;
  usage?: ReturnType<typeof usageFromResult>;
  cost?: number;
}

/** Vision judge: show the screenshot AND the booted service's log tail to a
 *  multimodal model and ask whether the app rendered + booted cleanly. The
 *  SCREENSHOT is weighted most heavily (the real "is it legit" signal — HTTP 200
 *  + non-empty DOM isn't enough; a white screen or styled error page both pass
 *  those); the LOGS catch fatal errors a screenshot can hide (crash loops,
 *  module-not-found, DB connection refused). anthropic-only. Returns verdict +
 *  call cost. */
async function judgeRender(
  pngPath: string,
  url: string,
  logs: string,
  visionModel: string | undefined,
): Promise<{ working: boolean; reason: string; usage: ReturnType<typeof usageFromResult>; cost: number }> {
  const { generateObject } = await import("ai");
  const { anthropic } = await import("@ai-sdk/anthropic");
  const model = anthropic(visionModel ?? process.env["VEIN_LLM_MODEL"] ?? "claude-sonnet-4-6");
  const schema = z.object({
    working: z
      .boolean()
      .describe("true ONLY if the intended app UI rendered (a real, populated, styled page — a login/landing page counts) AND the logs show no fatal/repeating boot error; false for a blank/white page, an error overlay, a stack trace, an unstyled error, a 404/500/connection-error page, or logs showing a crash loop / fatal startup error."),
    reason: z.string().describe("one short sentence: what the screenshot shows and whether the logs look clean."),
  });
  const image = readFileSync(pngPath);
  const logTail = logs ? logs.slice(-8000) : "(no logs captured)";
  const { object, usage: rawUsage } = await generateObject({
    model: model as any,
    schema: schema as any,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: `We just booted a web app locally for development at ${url}. Below is a full-page screenshot plus the TAIL of the booted server's logs. Did the app's intended UI render SUCCESSFULLY and boot cleanly?

WEIGHT THE SCREENSHOT MOST HEAVILY. Treat as FAILURE: a blank/white page, a framework error overlay (Next.js/Vite/React), a stack trace, raw unstyled error text, a 404/500 page, or a "can't connect" page. Use the LOGS to catch fatal errors the screenshot can hide — a crash loop, uncaught exception, "Cannot find module", DB connection refused, or the process repeatedly errored/exited. A few warnings or a single non-fatal error line is FINE if the UI clearly rendered.

SERVER LOGS (tail):
${logTail}`,
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

/** Load the booted app in headless chromium, screenshot it, then judge whether it
 *  actually rendered — by VISION model when available (the real signal), falling
 *  back to HTTP-status + error-overlay heuristics. */
async function renderCheck(
  url: string,
  screenshotDir: string,
  timeoutMs: number,
  logs: string,
  vision: { use: boolean; model?: string },
): Promise<RenderResult> {
  let chromium: any;
  try {
    ({ chromium } = await import("@playwright/test"));
  } catch (e) {
    return { rendered: null, note: `playwright not available: ${(e as Error).message}` };
  }
  let browser: any;
  try {
    browser = await chromium.launch();
  } catch (e) {
    return { rendered: null, note: `could not launch chromium (browsers installed?): ${(e as Error).message}` };
  }
  try {
    const page = await browser.newPage();
    const resp = await page.goto(url, { waitUntil: "networkidle", timeout: timeoutMs }).catch(async () =>
      page.goto(url, { waitUntil: "domcontentloaded", timeout: timeoutMs }),
    );
    const httpStatus = resp?.status();
    const title = await page.title().catch(() => "");
    const body = (await page.content().catch(() => "")) || "";
    const logErr = scanLogErrors(logs);
    const erroredHeuristic =
      /nextjs__container_errors|__next-error|Application error: a (?:client|server)-side exception|Internal Server Error/i.test(
        body,
      ) || logErr.length > 0;
    mkdirSync(screenshotDir, { recursive: true });
    const screenshotPath = join(screenshotDir, "render.png");
    await page.screenshot({ path: screenshotPath, fullPage: true }).catch(() => {});

    // A hard HTTP error short-circuits to "not rendered".
    if (httpStatus != null && httpStatus >= 400) {
      return { rendered: false, httpStatus, title, screenshotPath, note: `HTTP ${httpStatus}` };
    }

    // The REAL signal: a vision model looks at the screenshot (+ the logs).
    if (vision.use && existsSync(screenshotPath)) {
      try {
        const v = await judgeRender(screenshotPath, url, logs, vision.model);
        return {
          rendered: v.working,
          httpStatus,
          title,
          screenshotPath,
          note: `vision: ${v.reason}`,
          usage: v.usage,
          cost: v.cost,
        };
      } catch (e) {
        // fall through to heuristic if the vision call fails (no key, etc.)
        return {
          rendered: !erroredHeuristic,
          httpStatus,
          title,
          screenshotPath,
          note: `vision judge failed (${(e as Error).message}); heuristic only`,
        };
      }
    }

    return {
      rendered: !erroredHeuristic,
      httpStatus,
      title,
      screenshotPath,
      note: erroredHeuristic
        ? `error overlay/crash detected${logErr.length ? ` (logs: ${logErr[0].trim().slice(0, 120)})` : ""}`
        : "heuristic (vision off)",
    };
  } catch (e) {
    return { rendered: false, note: `navigation failed: ${(e as Error).message}` };
  } finally {
    await browser.close().catch(() => {});
  }
}

// ── step ──────────────────────────────────────────────────────────────────────

export default defineStep({
  type: "gitsee/verify-setup",
  description:
    "Boot gate for the gitsee setup-profiler: stages the produced pm2.config.js + docker-compose.yml into the cloned workspace the pod way (rewriting /workspaces/<repo> cwd to the local clone), `docker compose up -d --wait`, boots the apps via staklink (REBUILD→INSTALL→PRE_START→pm2 start→POST_START; no BUILD_COMMAND, mirroring prod) or an inline pm2-free fallback, polls the frontend PORT, then loads it in headless chromium, screenshots it, and JUDGES THE SCREENSHOT WITH A VISION MODEL (the real 'did it render' signal — HTTP 200 + a non-empty DOM isn't enough; a white screen or styled error page pass those) with an HTTP-status/error-overlay heuristic fallback. Tears down (pm2 delete all + compose down) unless keepUp. Config: workspacePath, setup (the two-file string), checkPath? (default /), bootTimeoutMs?, renderTimeoutMs?, useVision? (default true), visionModel?, useStaklink? (default true), bootCommand?, enabled? (default true), keepUp? (default false). Output: { booted, rendered, port, httpStatus, title, reason, logs, screenshotPath, cost, usage }.",
  input: z.object({
    workspacePath: z.string().describe("dir containing the cloned repos as siblings (from gitsee/clone-workspace)"),
    setup: z.string().describe("the produced pm2.config.js + docker-compose.yml two-file string"),
    checkPath: z.string().default("/").describe("path appended to http://localhost:<port> for the render check"),
    bootTimeoutMs: z.number().int().positive().default(420000).describe("max wait for the frontend port to bind (install+build can be slow)"),
    renderTimeoutMs: z.number().int().positive().default(30000),
    useVision: z.boolean().default(true).describe("judge the screenshot with a vision model (the real 'did it render' signal); false = HTTP-status + error-overlay heuristics only"),
    visionModel: z.string().optional().describe("anthropic vision model for the screenshot judge (default claude-sonnet-4-6)"),
    useStaklink: z.boolean().default(true).describe("boot via `npx staklink start` (prod-faithful); false = inline pm2-free boot"),
    bootCommand: z.string().default("npx -y staklink@latest start").describe("the staklink boot command (when useStaklink)"),
    enabled: z.boolean().default(true).describe("false = no-op (returns booted:null) so the gate is skipped in cheap sweeps"),
    keepUp: z.boolean().default(false).describe("skip teardown (leave compose + apps running for debugging)"),
  }),
  output: z.any(),
  async run(cfg) {
    const logs: string[] = [];
    const log = (s: string) => {
      console.log(`[gitsee/verify] ${s}`);
      logs.push(s);
    };
    const result = (extra: Record<string, unknown>) => ({
      booted: null as boolean | null,
      rendered: null as boolean | null,
      port: null as number | null,
      // vision-judge token cost (0 when vision off / not reached); folded into the
      // eval grand total by gitsee-eval (produce.cost + verify.cost).
      cost: 0,
      logs: logs.join("\n"),
      ...extra,
    });

    if (!cfg.enabled) return result({ reason: "verify disabled (enabled:false)" });

    const wp = cfg.workspacePath;
    if (!existsSync(wp)) return result({ reason: `workspacePath does not exist: ${wp}` });

    // 1. Parse + stage the produced files (pod layout).
    const files = splitFiles(cfg.setup);
    if (!files.pm2) return result({ booted: false, reason: "no pm2.config.js in the produced output" });

    const apps = parsePm2(files.pm2);
    if (!apps) return result({ booted: false, reason: "produced pm2.config.js did not parse" });
    const fe = frontendApp(apps);
    if (!fe) return result({ booted: false, reason: "no app found in pm2.config.js" });
    const { app, port } = fe;

    // Rewrite the pod-absolute cwd (/workspaces/<repo>) to the local clone root.
    const pm2Local = files.pm2.replace(/\/workspaces\//g, `${wp}/`);
    writeFileSync(join(wp, "pm2.config.js"), pm2Local);
    mkdirSync(join(wp, ".pod-config", ".user-dockerfile"), { recursive: true });
    writeFileSync(join(wp, ".pod-config", ".user-dockerfile", "pm2.config.js"), pm2Local);
    if (files.compose) writeFileSync(join(wp, "docker-compose.yml"), files.compose);
    log(`staged pm2.config.js (frontend port ${port}) + ${files.compose ? "docker-compose.yml" : "no compose"}`);

    const appCwd = (app.cwd ?? "").replace(/\/workspaces\//g, `${wp}/`) || wp;
    const appEnv: Record<string, string> = {};
    for (const [k, v] of Object.entries(app.env ?? {})) appEnv[k] = String(v);

    // A service-less compose (just a network) = "no backing services needed":
    // skip compose entirely (docker compose up would error "no service selected").
    const composeServices = hasComposeServices(files.compose);

    // Snapshot the container set BEFORE boot, so teardown can force-remove
    // everything that appeared during this run — including app-spawned stacks our
    // compose file knows nothing about (a `supabase start` CLI project, a minio…).
    const preBootContainers = await dockerContainerIds();

    let booted = false;
    let reason = "";
    try {
      // 2. Backing services.
      if (composeServices) {
        log("docker compose up -d --wait …");
        const up = await sh("docker compose up -d --wait", wp, {}, 300000);
        if (up.code !== 0) {
          // --wait fails if a container has no healthcheck; retry without it.
          const up2 = await sh("docker compose up -d", wp, {}, 300000);
          if (up2.code !== 0) {
            log(`compose up failed: ${(up.stderr || up2.stderr).slice(0, 800)}`);
            return result({ booted: false, port, reason: "docker compose up failed", screenshotPath: undefined });
          }
        }
      } else {
        log(files.compose ? "compose has no services — skipping compose up" : "no docker-compose.yml");
      }

      // 3. Boot the apps.
      if (cfg.useStaklink) {
        log(`booting via staklink: ${cfg.bootCommand}`);
        // staklink launches a pm2-managed proxy that runs startup() (install →
        // pre-start → pm2 start) in the background; we then poll the port.
        const boot = await sh(cfg.bootCommand, wp, {}, 180000);
        if (boot.code !== 0) log(`staklink start returned ${boot.code}: ${boot.stderr.slice(0, 600)}`);
      } else {
        // Inline pm2-free boot: run the env commands, then spawn the script.
        for (const [key, cmd] of [
          ["REBUILD_COMMAND", appEnv.REBUILD_COMMAND],
          ["INSTALL_COMMAND", appEnv.INSTALL_COMMAND],
          ["PRE_START_COMMAND", appEnv.PRE_START_COMMAND ?? appEnv.PRE_RUN_COMMAND],
        ] as const) {
          if (!cmd) continue;
          log(`${key}: ${cmd}`);
          const r = await sh(cmd, appCwd, appEnv, 600000);
          if (r.code !== 0) log(`${key} exited ${r.code}: ${r.stderr.slice(0, 400)}`);
        }
        const script = [app.script, app.args].filter(Boolean).join(" ");
        log(`spawning app: ${script} (cwd ${appCwd})`);
        // Pipe the app's stdout/stderr to a log file so we can inspect it for
        // boot errors (the staklink path uses pm2's logs instead).
        mkdirSync(join(wp, ".verify"), { recursive: true });
        const logFd = openSync(join(wp, ".verify", "app.log"), "w");
        spawn("sh", ["-c", script], {
          cwd: appCwd,
          detached: true,
          stdio: ["ignore", logFd, logFd],
          env: { ...process.env, CI: "1", ...appEnv },
        }).unref();
      }

      // 4. Wait for the port.
      log(`waiting for port ${port} (timeout ${cfg.bootTimeoutMs}ms) …`);
      booted = await waitForPort(port, "127.0.0.1", cfg.bootTimeoutMs);
      reason = booted ? "frontend port bound" : `frontend port ${port} never bound within ${cfg.bootTimeoutMs}ms`;
      log(reason);

      // 5. Render check.
      let render: RenderResult = { rendered: null };
      let appLogs = "";
      let logErrors: string[] = [];
      if (booted) {
        // Read the booted service's logs (so we catch fatal errors a screenshot
        // can't show) and feed them to the judge alongside the screenshot.
        appLogs = await captureAppLogs(wp, app.name ?? "frontend");
        logErrors = scanLogErrors(appLogs);
        if (logErrors.length) log(`log errors (heuristic): ${logErrors.length} line(s) — e.g. ${logErrors[0].trim().slice(0, 120)}`);
        const url = `http://localhost:${port}${cfg.checkPath}`;
        log(`render check: ${url}`);
        render = await renderCheck(url, join(wp, ".verify"), cfg.renderTimeoutMs, appLogs, {
          use: cfg.useVision,
          model: cfg.visionModel,
        });
        log(`rendered=${render.rendered} status=${render.httpStatus ?? "?"} ${render.note ?? ""}`);
      }

      return result({
        booted,
        rendered: render.rendered,
        port,
        httpStatus: render.httpStatus,
        title: render.title,
        screenshotPath: render.screenshotPath,
        cost: render.cost ?? 0,
        usage: render.usage,
        logErrors,
        logsTail: appLogs.slice(-4000),
        reason: render.note ? `${reason}; ${render.note}` : reason,
      });
    } finally {
      // 6. Teardown.
      if (!cfg.keepUp) {
        log("teardown: pm2 delete all + docker compose down + remove app-spawned containers");
        await sh("npx -y pm2 delete all", wp, {}, 60000).catch(() => {});
        if (cfg.useStaklink) await sh(`${cfg.bootCommand.split(" ").slice(0, -1).join(" ")} stop`, wp, {}, 30000).catch(() => {});
        if (composeServices) await sh("docker compose down -v", wp, {}, 120000).catch(() => {});
        // Force-remove anything that appeared since the pre-boot snapshot — this is
        // what catches a `supabase start` stack / minio / any container the app
        // started outside our compose file. `-v` also drops their anonymous volumes.
        try {
          const now = await dockerContainerIds();
          const spawned = [...now].filter((id) => !preBootContainers.has(id));
          if (spawned.length) {
            log(`removing ${spawned.length} app-spawned container(s)`);
            await sh(`docker rm -fv ${spawned.join(" ")}`, wp, {}, 120000).catch(() => {});
            // best-effort prune of the now-dangling networks/volumes they created
            await sh("docker network prune -f", wp, {}, 30000).catch(() => {});
          }
        } catch {
          /* ignore teardown errors */
        }
      } else {
        log("keepUp:true — leaving services + apps running");
      }
    }
  },
});
