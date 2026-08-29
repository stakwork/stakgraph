import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { ensureGaiaDataset, ensureGaiaPython, SCORER_SHA256 } from "./bootstrap.js";

/**
 * GAIA LAB scoring service — the lab's hardcoded, NON-EDITABLE grader.
 *
 * Scores answers with the REAL benchmark scorer: the leaderboard's
 * `scorer.py` (quasi-exact match with type-aware normalization), run as a
 * python3 subprocess against the validation split's gold answers. Nothing is
 * ported — `question_scorer` executes verbatim, so scores always match what
 * the official leaderboard would produce. The scoring logic lives here in
 * the services bag (not in a seeded/authored step) precisely so the
 * workflow-authoring agent can never edit its own grader; any `gaia/*`
 * steps are thin plumbing over `ctx.services.gaia.*`.
 *
 * Gold isolation: `metadata.jsonl` carries each task's `Final answer`.
 * `getTask`/`listTasks` DELIBERATELY strip it — the producing agent must
 * never see the gold. Only `score()` reads it, inside the subprocess.
 *
 * Integrity invariant: the dataset checkout must be a CLEAN git tree
 * (scorer.py itself is tolerated untracked — it comes from the leaderboard
 * Space, not the dataset repo), and when `GAIA_SCORER_SHA256` is set the
 * scorer file's hash must match. Every result carries the dataset's exact
 * git SHA (`benchmarkRev`) and the scorer's hash (`scorerSha256`) so a
 * score is always attributable to a benchmark + scorer version.
 *
 * Setup is AUTOMATIC (see bootstrap.ts): given an HF token, the dataset is
 * cloned and the leaderboard's scorer.py installed on first use. In the prod
 * image numpy is already on PATH via /usr/src/agent-venv, so a deployment
 * needs exactly one variable — HF_TOKEN.
 *
 * Config (env):
 *   HF_TOKEN           — REQUIRED for a cold start (or HUGGING_FACE_HUB_TOKEN
 *                        / HF_API_TOKEN): read access to the gated dataset.
 *                        The owning account must have accepted GAIA's terms on
 *                        the HF website; that click-through cannot be
 *                        automated. Only an already-populated checkout
 *                        removes the need for it. The rest are optional:
 *   GAIA_DIR           — pin the checkout location (default <cache>/vein/gaia)
 *   GAIA_SCORER_SHA256 — override the in-repo scorer pin (SCORER_SHA256)
 *   GAIA_PYTHON        — python interpreter (default "python3"; needs numpy)
 *   GAIA_AUTO_SETUP=0  — disable auto-setup; GAIA_DIR must be pre-populated
 */

export interface ExecResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

/** Injectable subprocess runner (fake it in smokes — no python needed). */
export type ExecFn = (
  cmd: string,
  args: string[],
  opts: { cwd: string; timeoutMs: number },
) => Promise<ExecResult>;

export interface GaiaTask {
  taskId: string;
  question: string;
  /** 1 | 2 | 3 */
  level: number;
  /** Attachment file name ("" when the task has none). */
  fileName: string;
  /** Absolute path of the attachment (undefined when none). READ-ONLY —
   *  stage a copy into the run's artifacts dir before letting an agent at it. */
  filePath?: string;
  // NOTE: `Final answer` (the gold) is DELIBERATELY not returned — the
  // producing agent must never see it.
}

export interface GaiaScorePair {
  taskId: string;
  /** The model's final answer, verbatim. */
  answer: string;
}

export interface GaiaScoreResult {
  taskId: string;
  level: number | null;
  correct: boolean;
  /** Set when the taskId didn't match a validation task. */
  error?: string;
}

export interface GaiaScoreReport {
  accuracy: number;
  correct: number;
  total: number;
  /** Per-level accuracy, keyed "1"/"2"/"3" (only levels present in pairs). */
  byLevel: Record<string, { correct: number; total: number }>;
  results: GaiaScoreResult[];
  benchmarkRev: string;
  scorerSha256: string;
}

export interface GaiaServices {
  /** Verify the dataset checkout (exists, clean tree, scorer present +
   *  hash-pinned) and return `{ root, rev, scorerSha256 }`. */
  verifyDataset(): Promise<{ root: string; rev: string; scorerSha256: string }>;
  /** Validation task ids + levels + whether they carry a file — no gold. */
  listTasks(opts?: { level?: number }): Promise<Array<{ taskId: string; level: number; hasFile: boolean }>>;
  /** One task's question + attachment path — no gold. */
  getTask(taskId: string): Promise<GaiaTask>;
  /** Score answers with the REAL scorer.py against the validation gold. */
  score(pairs: GaiaScorePair[], opts?: { timeoutMs?: number }): Promise<GaiaScoreReport>;
}

const SPLIT_DIR = join("2023", "validation");
const DEFAULT_SCORE_TIMEOUT_MS = 120_000;

/**
 * The python driver: loads the PINNED scorer.py verbatim, reads the gold
 * from metadata.jsonl, scores each (taskId, answer) pair, prints ONE json
 * object on stdout. question_scorer print()s chatter on some branches, so
 * scoring runs under redirect_stdout — stdout stays pure JSON.
 */
const DRIVER = `
import contextlib, io, json, sys
gaia_dir, pairs_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, gaia_dir)
from scorer import question_scorer
meta = {}
with open("${SPLIT_DIR}/metadata.jsonl".replace("\\\\", "/"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            row = json.loads(line)
            meta[row["task_id"]] = row
with open(pairs_path, encoding="utf-8") as f:
    pairs = json.load(f)
results = []
for p in pairs:
    row = meta.get(p["task_id"])
    if row is None:
        results.append({"taskId": p["task_id"], "level": None, "correct": False,
                        "error": "unknown task_id (not in validation split)"})
        continue
    with contextlib.redirect_stdout(io.StringIO()):
        ok = bool(question_scorer(str(p["answer"]), str(row["Final answer"])))
    results.append({"taskId": p["task_id"], "level": int(row["Level"]), "correct": ok})
print(json.dumps({"results": results}))
`;

function defaultExec(): ExecFn {
  return (cmd, args, opts) =>
    new Promise<ExecResult>((resolvePromise, reject) => {
      const child = spawn(cmd, args, {
        cwd: opts.cwd,
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

interface MetaRow {
  task_id: string;
  Question: string;
  Level: number | string;
  file_name?: string;
  ["Final answer"]?: string;
}

export interface BuildGaiaOptions {
  /** Dataset checkout path; defaults to env GAIA_DIR (checked at call time). */
  dir?: string;
  /** Pinned scorer.py sha256. Defaults to env GAIA_SCORER_SHA256, else the
   *  in-repo SCORER_SHA256 constant — the pin is ALWAYS enforced, so a
   *  deployment can never silently grade with an unidentified scorer. */
  scorerSha256?: string;
  /** Auto-clone the dataset + install scorer.py on first use. Default true
   *  (env GAIA_AUTO_SETUP=0 disables). Smokes pass false to exercise
   *  verifyDataset against a fixture checkout in isolation. */
  autoSetup?: boolean;
  /** HF token; defaults to HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HF_API_TOKEN. */
  hfToken?: string;
  /** Python interpreter; defaults to env GAIA_PYTHON or "python3". */
  python?: string;
  /** Subprocess runner override (for offline smokes). */
  exec?: ExecFn;
}

export function buildGaiaServices(opts: BuildGaiaOptions = {}): GaiaServices {
  const exec = opts.exec ?? defaultExec();
  let verified: { root: string; rev: string; scorerSha256: string } | undefined;
  // metadata parsed once per process — gold stays inside this closure and the
  // score() subprocess; it is never returned to callers.
  let metaCache: Map<string, MetaRow> | undefined;

  async function verifyDataset(): Promise<{ root: string; rev: string; scorerSha256: string }> {
    // Materialise the checkout if it isn't there yet (bootstrap.ts). A no-op
    // costing two stats once it is, so this stays safe on the per-grade path.
    const autoSetup = opts.autoSetup ?? process.env.GAIA_AUTO_SETUP !== "0";
    let root: string | undefined = opts.dir ?? process.env.GAIA_DIR;
    if (autoSetup) {
      ({ root } = await ensureGaiaDataset({
        ...(opts.dir ? { dir: opts.dir } : {}),
        ...(opts.hfToken ? { hfToken: opts.hfToken } : {}),
        ...(opts.scorerSha256 ? { scorerSha256: opts.scorerSha256 } : {}),
      }));
    }
    if (!root) {
      throw new Error(
        "gaia: GAIA_DIR not configured and auto-setup is off — set HF_TOKEN to let the dataset " +
          "bootstrap itself, or point GAIA_DIR at a populated checkout",
      );
    }
    const abs = resolve(root);
    try {
      const s = await stat(abs);
      if (!s.isDirectory()) throw new Error("not a directory");
    } catch {
      throw new Error(`gaia: GAIA_DIR does not exist: ${abs}`);
    }

    const head = await exec("git", ["rev-parse", "HEAD"], { cwd: abs, timeoutMs: 10_000 });
    if (head.code !== 0) {
      throw new Error(`gaia: ${abs} is not a git checkout: ${head.stderr.trim()}`);
    }
    const rev = head.stdout.trim();

    // Importing scorer.py compiles bytecode into __pycache__/ — derived
    // state, but also a shadowing vector: python prefers a matching .pyc over
    // the (sha-pinned) source, so a doctored cache could bypass the pin.
    // Delete it rather than tolerate it. score() runs python -B so it
    // normally never appears; this also heals checkouts dirtied before -B.
    await rm(join(abs, "__pycache__"), { recursive: true, force: true });

    // The dataset must be unmodified (a doctored metadata.jsonl = doctored
    // gold). scorer.py is expected untracked — it comes from the leaderboard
    // Space, not this repo.
    const status = await exec("git", ["status", "--porcelain"], { cwd: abs, timeoutMs: 10_000 });
    if (status.code !== 0) {
      throw new Error(`gaia: git status failed in ${abs}: ${status.stderr.trim()}`);
    }
    const dirty = status.stdout
      .split("\n")
      .filter((l) => l.trim().length > 0)
      .filter((l) => !(l.startsWith("??") && l.slice(3).trim() === "scorer.py"));
    if (dirty.length > 0) {
      throw new Error(
        `gaia: dataset checkout has local modifications — refusing to grade. ` +
          `Restore it to a clean tree first:\n${dirty.slice(0, 10).join("\n")}`,
      );
    }

    let scorerSrc: string;
    try {
      scorerSrc = await readFile(join(abs, "scorer.py"), "utf-8");
    } catch {
      throw new Error(
        `gaia: ${abs}/scorer.py not found — download the official scorer from ` +
          "huggingface.co/spaces/gaia-benchmark/leaderboard (scorer.py) and place it there",
      );
    }
    const scorerSha256 = createHash("sha256").update(scorerSrc).digest("hex");
    // Always pinned: an unidentified scorer makes a score unattributable, so
    // the in-repo constant is the floor, not an opt-in (EVOLVE_SPEC §6).
    const pin = opts.scorerSha256 ?? process.env.GAIA_SCORER_SHA256 ?? SCORER_SHA256;
    if (pin && scorerSha256 !== pin.toLowerCase()) {
      throw new Error(
        `gaia: scorer.py sha256 ${scorerSha256.slice(0, 12)}… does not match pinned GAIA_SCORER_SHA256 — refusing to grade`,
      );
    }

    verified = { root: abs, rev, scorerSha256 };
    return verified;
  }

  async function loadMeta(root: string): Promise<Map<string, MetaRow>> {
    if (metaCache) return metaCache;
    const path = join(root, SPLIT_DIR, "metadata.jsonl");
    let raw: string;
    try {
      raw = await readFile(path, "utf-8");
    } catch {
      throw new Error(`gaia: ${path} not found — is GAIA_DIR a full dataset clone?`);
    }
    const map = new Map<string, MetaRow>();
    for (const line of raw.split("\n")) {
      const t = line.trim();
      if (!t) continue;
      const row = JSON.parse(t) as MetaRow;
      map.set(row.task_id, row);
    }
    metaCache = map;
    return map;
  }

  async function listTasks(o: { level?: number } = {}) {
    const { root } = verified ?? (await verifyDataset());
    const meta = await loadMeta(root);
    const out: Array<{ taskId: string; level: number; hasFile: boolean }> = [];
    for (const row of meta.values()) {
      const level = Number(row.Level);
      if (o.level !== undefined && level !== o.level) continue;
      out.push({ taskId: row.task_id, level, hasFile: Boolean(row.file_name) });
    }
    return out;
  }

  async function getTask(taskId: string): Promise<GaiaTask> {
    const { root } = verified ?? (await verifyDataset());
    const meta = await loadMeta(root);
    const row = meta.get(taskId);
    if (!row) throw new Error(`gaia: task "${taskId}" not in the validation split`);
    // Final answer is DELIBERATELY dropped here — never expose the gold.
    return {
      taskId: row.task_id,
      question: row.Question,
      level: Number(row.Level),
      fileName: row.file_name ?? "",
      ...(row.file_name ? { filePath: join(root, SPLIT_DIR, row.file_name) } : {}),
    };
  }

  async function score(
    pairs: GaiaScorePair[],
    o: { timeoutMs?: number } = {},
  ): Promise<GaiaScoreReport> {
    const autoSetup = opts.autoSetup ?? process.env.GAIA_AUTO_SETUP !== "0";
    const { root, rev, scorerSha256 } = await verifyDataset(); // fresh check every grade
    if (!pairs.length) throw new Error("gaia: score() called with no pairs");

    const tmp = await mkdtemp(join(tmpdir(), "gaia-score-"));
    const pairsPath = join(tmp, "pairs.json");
    await writeFile(
      pairsPath,
      JSON.stringify(pairs.map((p) => ({ task_id: p.taskId, answer: p.answer }))),
    );

    // Resolves to plain `python3` in the prod image (agent venv on PATH has
    // numpy); builds a cached venv on a dev box that lacks it.
    const python = autoSetup
      ? await ensureGaiaPython(opts.python ? { python: opts.python } : {})
      : (opts.python ?? process.env.GAIA_PYTHON ?? "python3");
    // -B: never write __pycache__/ into the checkout (verifyDataset treats a
    // dirty tree as doctored gold and refuses to grade).
    const res = await exec(python, ["-B", "-c", DRIVER, root, pairsPath], {
      cwd: root,
      timeoutMs: o.timeoutMs ?? DEFAULT_SCORE_TIMEOUT_MS,
    });
    if (res.code !== 0) {
      throw new Error(
        `gaia: scorer failed (exit ${res.code}).\nstderr tail:\n${res.stderr.slice(-2000)}`,
      );
    }
    let parsed: { results: GaiaScoreResult[] };
    try {
      parsed = JSON.parse(res.stdout);
    } catch {
      throw new Error(`gaia: scorer emitted non-JSON stdout:\n${res.stdout.slice(0, 2000)}`);
    }

    const results = parsed.results;
    const scored = results.filter((r) => !r.error);
    const correct = scored.filter((r) => r.correct).length;
    const byLevel: Record<string, { correct: number; total: number }> = {};
    for (const r of scored) {
      const key = String(r.level);
      byLevel[key] ??= { correct: 0, total: 0 };
      byLevel[key].total += 1;
      if (r.correct) byLevel[key].correct += 1;
    }
    return {
      accuracy: scored.length ? correct / scored.length : 0,
      correct,
      total: results.length,
      byLevel,
      results,
      benchmarkRev: rev,
      scorerSha256,
    };
  }

  return { verifyDataset, listTasks, getTask, score };
}
