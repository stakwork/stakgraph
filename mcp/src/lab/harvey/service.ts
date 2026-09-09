import { spawn } from "node:child_process";
import { cp, mkdir, readFile, readdir, writeFile, stat } from "node:fs/promises";
import { join, resolve } from "node:path";

/**
 * Harvey LAB (Legal Agent Benchmark) verification service — the lab's
 * hardcoded, NON-EDITABLE grader.
 *
 * Runs the ACTUAL eval from the harvey-labs checkout
 * (`uv run python -m evaluation.run_eval`) as a subprocess — nothing is
 * ported, so scores always come from the real benchmark. The eval logic
 * lives here in the services bag (not in a seeded step) precisely so the
 * workflow-authoring agent can never edit its own grader; the seeded
 * `harvey/*` steps are thin plumbing over `ctx.services.harvey.*`.
 *
 * Integrity invariant: before any run, the checkout must be a CLEAN git tree
 * (no local modifications), and — when `HARVEY_LABS_REV` is set — HEAD must
 * match the pinned rev. A dirty or mispinned checkout refuses to grade.
 * Every result carries the checkout's exact SHA (`benchmarkRev`) so a score
 * is always attributable to a benchmark version.
 *
 * Config (env):
 *   HARVEY_LABS_DIR — path to the harvey-labs checkout (required at call time)
 *   HARVEY_LABS_REV — optional pinned commit SHA (prefix ok)
 * The eval subprocess inherits process.env (it needs ANTHROPIC_API_KEY, and
 * OPENAI_API_KEY too for --dual); harvey-labs also auto-loads its own .env.
 */

export interface ExecResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

/** Injectable subprocess runner (fake it in smokes — no uv/python needed). */
export type ExecFn = (
  cmd: string,
  args: string[],
  opts: { cwd: string; timeoutMs: number; env?: NodeJS.ProcessEnv },
) => Promise<ExecResult>;

export interface HarveyTask {
  task: string;
  title: string;
  work_type?: string;
  tags?: string[];
  instructions: string;
  /** Expected deliverable filenames (from task.json `deliverables`). */
  deliverables: string[];
  /** Absolute path of the task's input documents directory (read-only). */
  documentsDir: string;
  /** Relative paths of the input documents. */
  documents: string[];
  // NOTE: `criteria` (the grading rubric) is DELIBERATELY not returned —
  // the producing agent must never see its own rubric.
}

export interface HarveyEvalArgs {
  /** Task id, e.g. "corporate-ma/review-data-room-red-flag-review". */
  task: string;
  /** Directory whose CONTENTS are the deliverables (staged to
   *  `results/<runId>/output/` in the checkout). */
  sourceDir: string;
  /** Benchmark run id (also the results dir name). Sanitized. */
  runId: string;
  /** Optional producer metrics folded into the score report
   *  (input_tokens, output_tokens, wall_clock_seconds, …). */
  metrics?: Record<string, unknown>;
  /** Single-judge model (default: the harness default, claude-sonnet-4-6). */
  judgeModel?: string;
  /** Dual-judge official-style grading (needs OPENAI_API_KEY too). */
  dual?: boolean;
  /** Concurrent judge calls (harness default 6). */
  parallel?: number;
  timeoutMs?: number;
}

export interface HarveyServices {
  /** Verify the checkout (exists, clean tree, pinned rev) and return
   *  `{ root, rev }`. Cached after first success. */
  verifyCheckout(): Promise<{ root: string; rev: string }>;
  /** Load a task's instructions + document listing — WITHOUT the rubric. */
  getTask(task: string): Promise<HarveyTask>;
  /** Stage deliverables and run the real eval; returns the parsed
   *  scores.json (or scores_dual.json) plus benchmarkRev/runDir/reportPath. */
  evaluate(args: HarveyEvalArgs): Promise<Record<string, unknown>>;
}

const DEFAULT_EVAL_TIMEOUT_MS = 30 * 60_000;

function defaultExec(): ExecFn {
  return (cmd, args, opts) =>
    new Promise<ExecResult>((resolvePromise, reject) => {
      const child = spawn(cmd, args, {
        cwd: opts.cwd,
        env: opts.env ?? process.env,
        stdio: ["ignore", "pipe", "pipe"],
      });
      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (d) => (stdout += d));
      child.stderr.on("data", (d) => (stderr += d));
      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error(`${cmd} timed out after ${opts.timeoutMs}ms`));
      }, opts.timeoutMs);
      child.on("error", (err) => {
        clearTimeout(timer);
        reject(err);
      });
      child.on("close", (code) => {
        clearTimeout(timer);
        resolvePromise({ code, stdout, stderr });
      });
    });
}

/** results dir names must be single, safe path segments. */
function sanitizeRunId(runId: string): string {
  const safe = runId.replace(/[^a-zA-Z0-9._-]/g, "-");
  if (!safe || safe === "." || safe === ".." || safe.startsWith(".")) {
    throw new Error(`harvey: invalid runId "${runId}"`);
  }
  return safe;
}

/** Task ids are slash-joined path segments under tasks/ — no traversal. */
function sanitizeTask(root: string, task: string): string {
  const dir = resolve(root, "tasks", task);
  if (!dir.startsWith(resolve(root, "tasks") + "/")) {
    throw new Error(`harvey: invalid task "${task}"`);
  }
  return dir;
}

export interface BuildHarveyOptions {
  /** Checkout path; defaults to env HARVEY_LABS_DIR (checked at call time). */
  dir?: string;
  /** Pinned commit SHA; defaults to env HARVEY_LABS_REV (optional). */
  rev?: string;
  /** Subprocess runner override (for offline smokes). */
  exec?: ExecFn;
}

export function buildHarveyServices(opts: BuildHarveyOptions = {}): HarveyServices {
  const exec = opts.exec ?? defaultExec();
  // Cached after first successful verification. Re-verified per evaluate()
  // call anyway (cheap: two git commands) so a checkout dirtied mid-session
  // is still caught; the cache only backs getTask.
  let verified: { root: string; rev: string } | undefined;

  async function verifyCheckout(): Promise<{ root: string; rev: string }> {
    const root = opts.dir ?? process.env.HARVEY_LABS_DIR;
    if (!root) {
      throw new Error(
        "harvey: HARVEY_LABS_DIR not configured — point it at the harvey-labs checkout",
      );
    }
    const abs = resolve(root);
    try {
      const s = await stat(abs);
      if (!s.isDirectory()) throw new Error("not a directory");
    } catch {
      throw new Error(`harvey: HARVEY_LABS_DIR does not exist: ${abs}`);
    }

    const head = await exec("git", ["rev-parse", "HEAD"], { cwd: abs, timeoutMs: 10_000 });
    if (head.code !== 0) {
      throw new Error(`harvey: ${abs} is not a git checkout: ${head.stderr.trim()}`);
    }
    const rev = head.stdout.trim();

    // The benchmark must be unmodified: refuse to grade from a dirty tree.
    // (results/ is gitignored upstream, so staged runs don't dirty it.)
    const status = await exec("git", ["status", "--porcelain"], { cwd: abs, timeoutMs: 10_000 });
    if (status.code !== 0) {
      throw new Error(`harvey: git status failed in ${abs}: ${status.stderr.trim()}`);
    }
    const dirty = status.stdout
      .split("\n")
      .filter((l) => l.trim().length > 0)
      .filter((l) => {
        // Untracked entries under results/ are our own staged runs (results/
        // is gitignored upstream; tolerate it untracked too). Anything else
        // counts as a modification.
        const path = l.slice(3);
        return !(l.startsWith("??") && (path.startsWith("results/") || path === "results"));
      });
    if (dirty.length > 0) {
      throw new Error(
        `harvey: benchmark checkout has local modifications — refusing to grade. ` +
        `Restore it to a clean tree first:\n${dirty.slice(0, 10).join("\n")}`,
      );
    }

    const pin = opts.rev ?? process.env.HARVEY_LABS_REV;
    if (pin && !rev.startsWith(pin)) {
      throw new Error(
        `harvey: checkout HEAD ${rev.slice(0, 12)} does not match pinned HARVEY_LABS_REV ${pin} — refusing to grade`,
      );
    }

    verified = { root: abs, rev };
    return verified;
  }

  async function getTask(task: string): Promise<HarveyTask> {
    const { root } = verified ?? (await verifyCheckout());
    const taskDir = sanitizeTask(root, task);
    let config: any;
    try {
      config = JSON.parse(await readFile(join(taskDir, "task.json"), "utf-8"));
    } catch (err) {
      throw new Error(`harvey: task "${task}" not found or unreadable: ${err instanceof Error ? err.message : err}`);
    }
    const documentsDir = join(taskDir, "documents");
    let documents: string[] = [];
    try {
      documents = (await readdir(documentsDir, { recursive: true, withFileTypes: true }))
        .filter((e) => e.isFile())
        .map((e) => join(e.parentPath, e.name).slice(documentsDir.length + 1))
        .sort();
    } catch {
      // task without documents — fine
    }
    const deliverables =
      config.deliverables && typeof config.deliverables === "object"
        ? Object.keys(config.deliverables)
        : [];
    // criteria are DELIBERATELY dropped here — never expose the rubric.
    return {
      task,
      title: config.title,
      work_type: config.work_type,
      tags: config.tags,
      instructions: config.instructions,
      deliverables,
      documentsDir,
      documents,
    };
  }

  async function evaluate(args: HarveyEvalArgs): Promise<Record<string, unknown>> {
    const { root, rev } = await verifyCheckout(); // fresh check every grade
    sanitizeTask(root, args.task);
    const runId = sanitizeRunId(args.runId);

    // Stage deliverables where the harness looks: results/<runId>/output/.
    const runDir = join(root, "results", runId);
    const outputDir = join(runDir, "output");
    await mkdir(outputDir, { recursive: true });
    try {
      await cp(args.sourceDir, outputDir, { recursive: true });
    } catch (err) {
      throw new Error(
        `harvey: could not stage deliverables from ${args.sourceDir}: ${err instanceof Error ? err.message : err}`,
      );
    }
    if (args.metrics) {
      await writeFile(join(runDir, "metrics.json"), JSON.stringify(args.metrics, null, 2));
    }

    const cli = ["run", "python", "-m", "evaluation.run_eval", "--run-id", runId, "--task", args.task];
    if (args.dual) cli.push("--dual");
    else if (args.judgeModel) cli.push("--judge-model", args.judgeModel);
    if (args.parallel) cli.push("--parallel", String(args.parallel));

    const res = await exec("uv", cli, {
      cwd: root,
      timeoutMs: args.timeoutMs ?? DEFAULT_EVAL_TIMEOUT_MS,
      env: process.env,
    });

    // The harness writes scores next to the staged run; read them back.
    const scoresFile = args.dual ? "scores_dual.json" : "scores.json";
    let scores: Record<string, unknown> | undefined;
    try {
      scores = JSON.parse(await readFile(join(runDir, scoresFile), "utf-8"));
    } catch {
      // fall through — combined with exit code below
    }
    if (res.code !== 0 || !scores) {
      throw new Error(
        `harvey: eval failed (exit ${res.code}, ${scoresFile}${scores ? " present" : " missing"}).\n` +
        `stdout tail:\n${res.stdout.slice(-2000)}\nstderr tail:\n${res.stderr.slice(-2000)}`,
      );
    }

    return {
      ...scores,
      benchmarkRev: rev,
      runDir,
      reportPath: join(runDir, "report.html"),
    };
  }

  return { verifyCheckout, getTask, evaluate };
}
