/**
 * Stack service: owns the pod-faithful BOOT lifecycle of a produced setup —
 * stage the `pm2.config.js` + `docker-compose.yml`, bring up backing services
 * (docker compose), start the apps (staklink or an inline pm2-free fallback),
 * wait for the frontend port, read logs, and TEAR DOWN everything on disposal
 * (pm2 + compose + a snapshot-diff of app-spawned containers like a `supabase
 * start` stack). Lifted from `boot-and-exercise.ts`.
 *
 * Sessions are PER-RUN, keyed by runId, and disposed by `onRunEnd(runId)` — so
 * an errored eval in the optimize loop frees its stack without waiting for the
 * whole process to exit. (A hard SIGKILL still needs `cleanup.ts`.)
 */
import {
  writeFileSync,
  readFileSync,
  mkdirSync,
  existsSync,
  rmSync,
  openSync,
} from "node:fs";
import { join } from "node:path";
import { spawn } from "node:child_process";
import { readdirSync } from "node:fs";
import {
  sh,
  splitFiles,
  parsePm2,
  frontendApp,
  hasComposeServices,
  podSubstituteLocal,
  waitForPort,
  dockerContainerIds,
  captureAppLogs,
  scanLogErrors,
} from "./_infra.js";

export interface StackOptions {
  useStaklink?: boolean;
  bootCommand?: string;
  bootTimeoutMs?: number;
  /** Skip teardown on disposal (leave compose + apps up for debugging). */
  keepUp?: boolean;
}

export interface BootResult {
  booted: boolean;
  port: number;
  logsTail: string;
  errors: string[];
  report: string;
}

export class StackSession {
  readonly workspacePath: string;
  readonly useStaklink: boolean;
  readonly bootCommand: string;
  readonly bootTimeoutMs: number;
  readonly keepUp: boolean;

  port = 3000;
  appName = "frontend";
  private composeBroughtUp = false;
  private preBootContainers = new Set<string>();
  private logBuf: string[] = [];
  /** The original produced two-file string (local form), for finalSetup. */
  private initialSetup = "";
  lastBooted: boolean | null = null;
  /** The most recent vision verdict (recorded by gitsee/assess-ui), read by
   *  gitsee/finalize-setup so the deliverable carries it no matter how the
   *  agent loop ends. */
  lastWorking: boolean | null = null;
  lastReason = "";
  lastScreenshot: string | undefined;

  constructor(workspacePath: string, opts: StackOptions = {}) {
    this.workspacePath = workspacePath;
    this.useStaklink = opts.useStaklink ?? true;
    this.bootCommand = opts.bootCommand ?? "npx -y staklink@latest start";
    this.bootTimeoutMs = opts.bootTimeoutMs ?? 420000;
    this.keepUp = opts.keepUp ?? false;
  }

  private log(s: string) {
    console.log(`[gitsee/stack] ${s}`);
    this.logBuf.push(s);
  }

  private toLocal = (s: string) => s.replace(/\/workspaces\//g, `${this.workspacePath}/`);
  private toPod = (s: string) => s.split(`${this.workspacePath}/`).join("/workspaces/");

  /**
   * Stage the produced setup into the workspace (local-path form) and snapshot
   * the running containers so teardown can remove everything this run spins up.
   * Returns the parsed frontend port + app name, or an error string.
   */
  async stage(setup: string): Promise<{ ok: true; port: number; appName: string } | { ok: false; error: string }> {
    this.initialSetup = setup;
    const files = splitFiles(setup);
    if (!files.pm2) return { ok: false, error: "no pm2.config.js in the provided setup" };
    const apps = parsePm2(files.pm2);
    if (!apps) return { ok: false, error: "provided pm2.config.js did not parse" };
    const fe = frontendApp(apps);
    if (!fe) return { ok: false, error: "no app found in pm2.config.js" };

    this.port = fe.port;
    this.appName = fe.app.name ?? "frontend";
    const wp = this.workspacePath;
    writeFileSync(join(wp, "pm2.config.js"), this.toLocal(files.pm2));
    if (files.compose) writeFileSync(join(wp, "docker-compose.yml"), this.toLocal(files.compose));
    this.log(`staged pm2.config.js (frontend port ${this.port})${files.compose ? " + docker-compose.yml" : ""}`);

    this.preBootContainers = await dockerContainerIds();
    return { ok: true, port: this.port, appName: this.appName };
  }

  /** (Re)boot: re-read the (possibly edited) config, bring up compose, start the
   *  apps, wait for the frontend port. Safe to call repeatedly. */
  async boot(): Promise<BootResult> {
    const wp = this.workspacePath;
    await sh("npx -y pm2 delete all", wp, {}, 60000).catch(() => {});

    const pm2Code = existsSync(join(wp, "pm2.config.js")) ? readFileSync(join(wp, "pm2.config.js"), "utf-8") : "";
    const curApps = parsePm2(pm2Code);
    if (!curApps) return this.bootReport(false, "pm2.config.js did not parse — fix it before booting", "");
    const curFe = frontendApp(curApps);
    if (!curFe) return this.bootReport(false, "no app in pm2.config.js", "");
    this.port = curFe.port;

    // Localize pod-URL placeholders in the staged-for-boot copy ONLY.
    const pm2Staged = podSubstituteLocal(pm2Code, this.port);
    mkdirSync(join(wp, ".pod-config", ".user-dockerfile"), { recursive: true });
    writeFileSync(join(wp, ".pod-config", ".user-dockerfile", "pm2.config.js"), pm2Staged);

    const composeText = existsSync(join(wp, "docker-compose.yml"))
      ? readFileSync(join(wp, "docker-compose.yml"), "utf-8")
      : undefined;

    if (hasComposeServices(composeText)) {
      const up = await sh("docker compose up -d --wait", wp, {}, 300000);
      if (up.code !== 0) {
        const up2 = await sh("docker compose up -d", wp, {}, 300000);
        if (up2.code !== 0)
          return this.bootReport(false, `docker compose up failed: ${(up.stderr || up2.stderr).slice(0, 600)}`, "");
      }
      this.composeBroughtUp = true;
    }

    if (this.useStaklink) {
      const boot = await sh(this.bootCommand, wp, {}, 180000);
      if (boot.code !== 0) this.log(`staklink start returned ${boot.code}: ${boot.stderr.slice(0, 400)}`);
    } else {
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
        if (r.code !== 0) this.log(`${key} exited ${r.code}: ${r.stderr.slice(0, 300)}`);
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

    const booted = await waitForPort(this.port, "127.0.0.1", this.bootTimeoutMs);
    this.lastBooted = booted;
    const logs = await captureAppLogs(wp, this.appName);
    const errs = scanLogErrors(logs);
    const header = booted
      ? `BOOTED: frontend port ${this.port} is listening.`
      : `NOT BOOTED: port ${this.port} never bound in ${this.bootTimeoutMs}ms.`;
    return this.bootReport(booted, header, logs, errs);
  }

  private bootReport(booted: boolean, header: string, logs: string, errs: string[] = []): BootResult {
    const report = [
      header,
      errs.length ? `Log errors (heuristic):\n${errs.slice(0, 10).join("\n")}` : "No obvious fatal log lines.",
      `Server log tail:\n${logs.slice(-3000)}`,
    ].join("\n\n");
    return { booted, port: this.port, logsTail: logs.slice(-3000), errors: errs, report };
  }

  async readLogs(): Promise<string> {
    return captureAppLogs(this.workspacePath, this.appName);
  }

  /** Record the latest vision verdict (called by gitsee/assess-ui). */
  recordVerdict(working: boolean | null, reason: string, screenshotPath?: string) {
    this.lastWorking = working;
    this.lastReason = reason;
    if (screenshotPath) this.lastScreenshot = screenshotPath;
  }

  /** Per-repo `git diff` (intent-to-add so NEW files show as additions) — the
   *  replayable record of the agent's source edits. Teardown only touches
   *  pm2/docker/.pod-config (no repo tree), so edits are intact at finalize. */
  async captureRepoDiff(maxBytes = 60000): Promise<{ diff: string; changedRepos: string[] }> {
    const wp = this.workspacePath;
    const repos = existsSync(wp)
      ? readdirSync(wp, { withFileTypes: true })
          .filter((e) => e.isDirectory() && existsSync(join(wp, e.name, ".git")))
          .map((e) => e.name)
          .sort()
      : [];
    const parts: string[] = [];
    const changedRepos: string[] = [];
    for (const repo of repos) {
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

  /** Re-read the final (edited) files and re-emit in pod-portable form. The
   *  deliverable keeps the `$POD_*` placeholders + pod-absolute `cwd`. */
  finalSetup(): string {
    const wp = this.workspacePath;
    const files = splitFiles(this.initialSetup);
    const finalPm2 = existsSync(join(wp, "pm2.config.js"))
      ? readFileSync(join(wp, "pm2.config.js"), "utf-8")
      : this.toLocal(files.pm2 ?? "");
    const finalCompose = existsSync(join(wp, "docker-compose.yml"))
      ? readFileSync(join(wp, "docker-compose.yml"), "utf-8")
      : files.compose;
    return (
      `FILENAME: pm2.config.js\n\n\`\`\`js\n${this.toPod(finalPm2)}\n\`\`\`` +
      (finalCompose ? `\n\nFILENAME: docker-compose.yml\n\n\`\`\`yaml\n${this.toPod(finalCompose)}\n\`\`\`` : "")
    );
  }

  /** Tear down everything this run brought up: pm2, staklink, compose, and — via
   *  the pre-boot container snapshot — any app-spawned stack. Honors `keepUp`
   *  (set at construction). Idempotent. */
  async teardown(): Promise<void> {
    if (this.keepUp) {
      this.log("keepUp:true — leaving services + apps running");
      return;
    }
    const wp = this.workspacePath;
    this.log("teardown: pm2 delete all + staklink stop + compose down + remove spawned containers");
    await sh("npx -y pm2 delete all", wp, {}, 60000).catch(() => {});
    if (this.useStaklink)
      await sh(`${this.bootCommand.split(" ").slice(0, -1).join(" ")} stop`, wp, {}, 30000).catch(() => {});
    if (this.composeBroughtUp) await sh("docker compose down -v", wp, {}, 120000).catch(() => {});
    try {
      const now = await dockerContainerIds();
      const spawned = [...now].filter((id) => !this.preBootContainers.has(id));
      if (spawned.length) {
        this.log(`removing ${spawned.length} app-spawned container(s)`);
        await sh(`docker rm -fv ${spawned.join(" ")}`, wp, {}, 120000).catch(() => {});
        await sh("docker network prune -f", wp, {}, 30000).catch(() => {});
      }
    } catch {
      /* ignore */
    }
    // Remove only the staging dir; KEEP the edited config + repo edits (the
    // deliverable).
    try {
      rmSync(join(wp, ".pod-config"), { recursive: true, force: true });
    } catch {
      /* ignore */
    }
  }
}

/** Per-run stack sessions, keyed by runId. */
export class StackManager {
  private sessions = new Map<string, StackSession>();

  session(runId: string, workspacePath: string, opts?: StackOptions): StackSession {
    let s = this.sessions.get(runId);
    if (!s) {
      s = new StackSession(workspacePath, opts);
      this.sessions.set(runId, s);
    }
    return s;
  }

  has(runId: string): boolean {
    return this.sessions.has(runId);
  }

  /** The existing stack session for a run, or undefined (no auto-create). */
  get(runId: string): StackSession | undefined {
    return this.sessions.get(runId);
  }

  /** Dispose a run's stack (called from onRunEnd). Tears down (honoring the
   *  session's `keepUp`) + drops it. Idempotent. */
  async dispose(runId: string): Promise<void> {
    const s = this.sessions.get(runId);
    if (!s) return;
    this.sessions.delete(runId);
    await s.teardown().catch((e) => console.error(`[gitsee/stack] teardown failed for ${runId}:`, e));
  }
}
