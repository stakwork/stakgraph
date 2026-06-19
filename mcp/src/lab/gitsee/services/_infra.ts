/**
 * Shared, pure-ish infra helpers for the gitsee lab services (browser / stack /
 * vision). Lifted near-verbatim from the old `boot-and-exercise.ts` so the
 * services own the machinery and the (future) tool-steps stay thin. In-code only
 * (NOT a seeded step) — free to import node builtins + each other.
 */
import { spawn } from "node:child_process";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { createConnection } from "node:net";
import vm from "node:vm";
import yaml from "js-yaml";

// ── shell ──────────────────────────────────────────────────────────────────

export interface ShResult {
  code: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
}

/** Run a shell command, capture output, never reject. Mirrors staklink's runner
 *  env (CI=1, non-interactive) for parity. */
export function sh(
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

/** All docker container IDs (running + stopped), for the snapshot-diff teardown. */
export async function dockerContainerIds(): Promise<Set<string>> {
  const r = await sh("docker ps -aq", process.cwd(), {}, 15000);
  return new Set(r.stdout.split("\n").map((s) => s.trim()).filter(Boolean));
}

/** Poll a TCP port until something is listening (the app booted) or timeout. */
export function waitForPort(port: number, host: string, timeoutMs: number): Promise<boolean> {
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

// ── booted-service logs ──────────────────────────────────────────────────────

export async function captureAppLogs(wp: string, appName: string): Promise<string> {
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

export function scanLogErrors(logs: string): string[] {
  return logs
    .split("\n")
    .filter((l) => LOG_ERROR_RE.test(l))
    .slice(0, 20);
}

// ── produced-file parsing ─────────────────────────────────────────────────────

export interface TwoFiles {
  pm2?: string;
  compose?: string;
}

export function splitFiles(doc: string): TwoFiles {
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

export function hasComposeServices(text: string | undefined): boolean {
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

export interface Pm2App {
  name?: string;
  script?: string;
  args?: string;
  cwd?: string;
  env?: Record<string, unknown>;
}

export function parsePm2(code: string | undefined): Pm2App[] | null {
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
 * LOCAL emulation of the pod's `$POD_ID` / `$POD_URL` substitution. On the real
 * sandbox the platform expands these AND proxies `https://<podid>-<port>.<domain>`
 * → `localhost:<port>`; locally there's no proxy, so we rewrite the pod-URL
 * PATTERNS to their localhost equivalents in the staged-for-boot copy ONLY.
 */
export function podSubstituteLocal(pm2Code: string, frontendPort: number): string {
  return pm2Code
    .replace(/https?:\/\/\$\{?POD_ID\}?-(\d+)\.[A-Za-z0-9.\-]+/g, "http://localhost:$1")
    .replace(/https?:\/\/\$\{?POD_ID\}?\.[A-Za-z0-9.\-]+/g, `http://localhost:${frontendPort}`)
    .replace(/\$\{?POD_URL\}?/g, `http://localhost:${frontendPort}`)
    .replace(/\$\{?POD_ID\}?/g, "local");
}

/** The frontend app (named "frontend", else the first app) + its port. */
export function frontendApp(apps: Pm2App[]): { app: Pm2App; port: number } | null {
  if (!apps.length) return null;
  const app = apps.find((a) => a.name === "frontend") ?? apps[0];
  const port = parseInt(String(app.env?.PORT ?? "3000"), 10) || 3000;
  return { app, port };
}
