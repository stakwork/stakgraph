import { createHash } from "node:crypto";
import { mkdir, readFile, rename, rm, stat, writeFile, readdir } from "node:fs/promises";
import { homedir } from "node:os";
import { join, resolve, dirname } from "node:path";

/**
 * GAIA dataset BOOTSTRAP — turns "fiddle with three env vars" into "set
 * HF_TOKEN once".
 *
 * The grader (`service.ts`) needs three things on disk, and until now an
 * operator had to produce all three by hand:
 *
 *   1. a git CHECKOUT of the gated `gaia-benchmark/GAIA` dataset (a git
 *      checkout specifically — `verifyDataset` reads HEAD for `benchmarkRev`
 *      and runs `git status` for the clean-tree invariant, so an
 *      `huggingface_hub.snapshot_download` (no `.git`) is NOT a substitute);
 *   2. the leaderboard Space's `scorer.py` at that checkout's root;
 *   3. a python with numpy.
 *
 * This module produces (1) and (2) automatically. (3) is already satisfied in
 * the prod image — `/usr/src/agent-venv/bin` is first on PATH and carries
 * numpy — so `GAIA_PYTHON` only ever needs setting on a dev box.
 *
 * INTEGRITY: auto-setup makes the guarantees STRONGER, not weaker.
 *   - The scorer pin moves from an operator-supplied env var to
 *     `SCORER_SHA256` below — a constant in the repo, reviewable in a diff.
 *     An operator-set pin is self-certifying (you pin whatever you happen to
 *     have); a repo constant is the same class of object as the graders
 *     themselves (EVOLVE_SPEC §6). `GAIA_SCORER_SHA256` still overrides, for
 *     the day the Space publishes a new scorer and someone needs to bump it
 *     before the constant lands.
 *   - A freshly cloned tree is clean by construction, so the clean-tree
 *     invariant starts from a known-good state instead of whatever the
 *     operator's directory had drifted into.
 *
 * LICENSING: the dataset is deliberately NOT baked into the image. Accepting
 * GAIA's terms includes agreeing not to reshare it outside a gated or private
 * repo, and a published container image is neither. It is fetched at runtime
 * into a cache dir (mount it as a volume in prod so the ~205MB download
 * happens once, not per container).
 *
 * GATING: the click-through acceptance on huggingface.co cannot be automated
 * — it is a per-account agreement. The token must belong to an account that
 * has already accepted; a 403 on clone means it has not, and the error below
 * says exactly that.
 *
 * Config (env), all optional:
 *   HF_TOKEN            — (or HUGGING_FACE_HUB_TOKEN / HF_API_TOKEN) a READ
 *                         token. Optional in practice — HF currently serves
 *                         this repo's git endpoints anonymously — but used
 *                         whenever set, since that can change. No username
 *                         is needed: HF takes the token as the password
 *                         with any username.
 *   GAIA_DIR            — pin the checkout location; defaults to
 *                         <cache>/vein/gaia
 *   VEIN_CACHE_DIR      — cache root override (else XDG_CACHE_HOME, else
 *                         ~/.cache)
 *   GAIA_AUTO_SETUP=0   — disable auto-setup; GAIA_DIR must then already be
 *                         populated (the old behaviour)
 *   GAIA_PYTHON         — interpreter override; unset, a numpy-capable python
 *                         is found on PATH or built as a cached venv
 *                         (see ensureGaiaPython below)
 */

/** Public, unauthenticated: the Space is not gated even though the dataset is. */
const SCORER_URL =
  "https://huggingface.co/spaces/gaia-benchmark/leaderboard/raw/main/scorer.py";

const DATASET_REPO = "https://huggingface.co/datasets/gaia-benchmark/GAIA";

/**
 * PINNED dataset revision — and this pin is load-bearing, not hygiene.
 *
 * The repo's default branch NO LONGER CARRIES THE BENCHMARK. As of
 * 682dd723 (main) the `2023/<split>/metadata.jsonl` files — the questions AND the
 * gold — are gone; main is 119 files of attachments only. A plain
 * `git clone` therefore yields a checkout the grader cannot read, failing
 * later and confusingly at `loadMeta`.
 *
 * This SHA is the last revision carrying the full benchmark: 165 validation
 * rows + 300 test rows, matching the published split sizes. It is also the
 * revision every score this lab has produced so far was graded against, so
 * pinning it keeps new numbers comparable to the existing ones.
 *
 * Verified 2026-08-27. Changing it invalidates comparability with every
 * prior `benchmarkRev` — which is exactly why it is a reviewed constant.
 */
export const DATASET_REV = "897f2dfbb5c952b5c3c1509e648381f9c7b70316";

/**
 * sha256 of the leaderboard's `scorer.py` as published at the revision this
 * lab is validated against. Verified 2026-08-27 against
 * huggingface.co/spaces/gaia-benchmark/leaderboard.
 *
 * This is a FIXED POINT (EVOLVE_SPEC §6): a score is only attributable if the
 * scorer that produced it is identified, and identifying it by a constant in
 * a reviewed repo beats identifying it by an env var the operator chose. If
 * the Space republishes the scorer, this constant must be bumped in a diff a
 * human reads — that is the point, not an inconvenience.
 */
export const SCORER_SHA256 =
  "0d44c07f3046eec521697c22e3eaca8719cc81e422a8eaf32695c5f22bdac6e2";

/** Files the checkout must have for the grader to work. */
const VALIDATION_META = join("2023", "validation", "metadata.jsonl");
const SCORER_FILE = "scorer.py";

const CLONE_TIMEOUT_MS = 30 * 60_000; // ~205MB over LFS on a cold cache
const GIT_TIMEOUT_MS = 60_000;

export interface BootstrapExecResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

/** Injectable subprocess runner (faked in smokes — no network, no git-lfs). */
export type BootstrapExecFn = (
  cmd: string,
  args: string[],
  opts: { cwd: string; timeoutMs: number; env?: NodeJS.ProcessEnv },
) => Promise<BootstrapExecResult>;

/** Injectable fetcher for scorer.py (faked in smokes). */
export type FetchTextFn = (url: string) => Promise<string>;

export interface EnsureGaiaOptions {
  /** Target checkout dir. Defaults to env GAIA_DIR, else <cache>/vein/gaia. */
  dir?: string;
  /** HF token. Defaults to HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HF_API_TOKEN. */
  hfToken?: string;
  /** Set false (or GAIA_AUTO_SETUP=0) to require a pre-populated GAIA_DIR. */
  autoSetup?: boolean;
  /** Expected scorer hash; defaults to GAIA_SCORER_SHA256 else SCORER_SHA256. */
  scorerSha256?: string;
  exec?: BootstrapExecFn;
  fetchText?: FetchTextFn;
  /** Progress sink; defaults to console.log (a cold clone takes minutes). */
  log?: (msg: string) => void;
}

/** Where a checkout lands when GAIA_DIR is unset. */
export function defaultGaiaDir(): string {
  const cache =
    process.env["VEIN_CACHE_DIR"] ??
    process.env["XDG_CACHE_HOME"] ??
    join(homedir(), ".cache");
  return join(cache, "vein", "gaia");
}

function resolveToken(explicit?: string): string | undefined {
  return (
    explicit ??
    process.env["HF_TOKEN"] ??
    process.env["HUGGING_FACE_HUB_TOKEN"] ??
    process.env["HF_API_TOKEN"] ??
    undefined
  );
}

async function exists(p: string): Promise<boolean> {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

function defaultExec(): BootstrapExecFn {
  return async (cmd, args, opts) => {
    const { spawn } = await import("node:child_process");
    return new Promise<BootstrapExecResult>((res, rej) => {
      const child = spawn(cmd, args, {
        cwd: opts.cwd,
        stdio: ["ignore", "pipe", "pipe"],
        ...(opts.env ? { env: opts.env } : {}),
      });
      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (d) => (stdout += d));
      child.stderr.on("data", (d) => (stderr += d));
      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        rej(new Error(`${cmd} timed out after ${opts.timeoutMs}ms`));
      }, opts.timeoutMs);
      child.on("error", (err) => {
        clearTimeout(timer);
        rej(err);
      });
      child.on("close", (code) => {
        clearTimeout(timer);
        res({ code, stdout, stderr });
      });
    });
  };
}

const defaultFetchText: FetchTextFn = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`GET ${url} → HTTP ${r.status}`);
  return r.text();
};

/**
 * Is `dir` a usable GAIA checkout? Deliberately checks the two files the
 * grader actually opens rather than merely "the directory exists" — a clone
 * killed halfway leaves a directory behind, and a silent half-checkout would
 * surface later as a confusing scorer crash.
 */
async function isPopulated(dir: string): Promise<boolean> {
  return (
    (await exists(join(dir, ".git"))) &&
    (await exists(join(dir, VALIDATION_META))) &&
    (await exists(join(dir, SCORER_FILE)))
  );
}

/**
 * Guard against the silent-LFS-pointer failure: without git-lfs, `git clone`
 * SUCCEEDS and every attachment is a ~130-byte text pointer. The agent then
 * "reads" a spreadsheet that is actually a pointer stub and the task fails for
 * a reason nothing reports. Cheaper to detect here than to debug there.
 */
async function assertNoLfsPointers(dir: string): Promise<void> {
  const split = join(dir, "2023", "validation");
  let names: string[];
  try {
    names = await readdir(split);
  } catch {
    return;
  }
  const attachment = names.find((n) => n !== "metadata.jsonl" && n.includes("."));
  if (!attachment) return;
  const head = await readFile(join(split, attachment), "utf-8").catch(() => "");
  if (head.startsWith("version https://git-lfs.github.com/spec")) {
    throw new Error(
      `gaia: attachments in ${split} are unresolved git-lfs pointers — the clone ran without git-lfs. ` +
        `Install git-lfs (apt install git-lfs && git lfs install), delete ${dir}, and retry.`,
    );
  }
}

async function assertGitLfs(exec: BootstrapExecFn, cwd: string): Promise<void> {
  const r = await exec("git", ["lfs", "version"], { cwd, timeoutMs: GIT_TIMEOUT_MS }).catch(
    () => ({ code: 1, stdout: "", stderr: "git-lfs not on PATH" }),
  );
  if (r.code !== 0) {
    throw new Error(
      "gaia: git-lfs is required — GAIA's attachments (xlsx/pdf/mp3/png) are LFS-backed, and " +
        "cloning without it yields pointer stubs instead of files. Install it " +
        "(Debian: apt install git-lfs; macOS: brew install git-lfs) then run `git lfs install`.",
    );
  }
}

/**
 * Fetch the leaderboard's scorer.py and verify it against the pin BEFORE it
 * lands in the checkout. A mismatch writes nothing — a wrong scorer on disk
 * would be refused by `verifyDataset` on every subsequent grade anyway, and
 * leaving it there just makes the failure harder to read.
 */
async function installScorer(
  dir: string,
  expected: string,
  fetchText: FetchTextFn,
  log: (m: string) => void,
): Promise<string> {
  log(`gaia: fetching scorer.py from the leaderboard Space`);
  const src = await fetchText(SCORER_URL);
  const sha = createHash("sha256").update(src).digest("hex");
  if (sha !== expected.toLowerCase()) {
    throw new Error(
      `gaia: downloaded scorer.py sha256 ${sha.slice(0, 12)}… != expected ${expected.slice(0, 12)}… — ` +
        `refusing to install. The leaderboard Space may have republished the scorer; ` +
        `verify the change and bump SCORER_SHA256 in mcp/src/lab/gaia/bootstrap.ts (or set GAIA_SCORER_SHA256).`,
    );
  }
  await writeFile(join(dir, SCORER_FILE), src, "utf-8");
  return sha;
}

/**
 * Materialise the dataset at DATASET_REV.
 *
 * NOT `git clone`: the pinned revision is not the tip of any branch (see
 * DATASET_REV), so this is init → remote add → `fetch --depth 1 <sha>` →
 * `checkout FETCH_HEAD`. HF's git server allows fetching an arbitrary SHA,
 * and the shallow fetch keeps this to one revision (~210MB with LFS).
 *
 * The token, WHEN PRESENT, reaches git through a one-shot inline credential
 * helper reading it from the CHILD ENV: it is never written to `.git/config`
 * (which a plain `https://user:token@…` remote would do, persisting a live
 * credential inside the checkout) and never appears in argv (visible to any
 * `ps` on the host). HF accepts any username with the token as the password,
 * so the username is a constant.
 *
 * The token is OPTIONAL: this repo's git endpoints currently serve anonymous
 * fetches even though its `resolve` HTTP endpoint returns 401. That may
 * change without notice, so a token is used whenever one is configured.
 */
async function cloneDataset(
  target: string,
  token: string | undefined,
  exec: BootstrapExecFn,
  log: (m: string) => void,
): Promise<void> {
  const parent = dirname(target);
  await mkdir(parent, { recursive: true });
  const staging = `${target}.partial`;
  await rm(staging, { recursive: true, force: true });

  log(
    `gaia: fetching ${DATASET_REPO} @ ${DATASET_REV.slice(0, 12)} (~210MB with LFS) → ${target}` +
      (token ? "" : " [anonymous — no HF token configured]"),
  );
  await mkdir(staging, { recursive: true });

  // Token, when present, is supplied by an env-reading helper; no token means
  // a plain anonymous fetch. GIT_TERMINAL_PROMPT=0 keeps a credential prompt
  // from hanging a headless container forever.
  const helper = '!f() { echo "username=hf"; echo "password=$GAIA_HF_TOKEN"; }; f';
  const auth = token ? ["-c", `credential.helper=${helper}`] : [];
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    GIT_TERMINAL_PROMPT: "0",
    ...(token ? { GAIA_HF_TOKEN: token } : {}),
  };

  const steps: Array<{ args: string[]; timeoutMs: number }> = [
    { args: ["init", "-q", "."], timeoutMs: GIT_TIMEOUT_MS },
    { args: ["remote", "add", "origin", DATASET_REPO], timeoutMs: GIT_TIMEOUT_MS },
    // The pinned SHA is not a branch tip, so fetch it by object name.
    { args: [...auth, "fetch", "--depth", "1", "origin", DATASET_REV], timeoutMs: CLONE_TIMEOUT_MS },
    // Checkout runs the LFS smudge filter — this is where attachments become
    // real files rather than pointers, hence the git-lfs precondition.
    { args: [...auth, "checkout", "-q", "FETCH_HEAD"], timeoutMs: CLONE_TIMEOUT_MS },
  ];

  for (const step of steps) {
    const res = await exec("git", step.args, { cwd: staging, timeoutMs: step.timeoutMs, env });
    if (res.code === 0) continue;
    await rm(staging, { recursive: true, force: true });
    const tail = res.stderr.slice(-1500);
    // HF surfaces gating several ways depending on the endpoint git hits:
    // a 401/403 on the HTTP layer, "restricted"/"Please log in" prose, or —
    // when a blob fetch is refused mid-negotiation — a bare
    // "expected 'packfile'". Match all of them, or a gated failure gets
    // reported as a generic git error and the operator has nothing to act on.
    const denied =
      /40[13]|forbidden|unauthor|restricted|gated|please log in|expected 'packfile'|could not fetch/i.test(
        tail,
      );
    throw new Error(
      denied
        ? `gaia: access to the dataset was refused${token ? " with the configured HF token" : " (no HF token configured)"}. ` +
          (token
            ? `The owning account has probably not accepted GAIA's terms — that click-through cannot be automated. ` +
              `Visit ${DATASET_REPO}, accept, then retry.`
            : `Set HF_TOKEN to a read token from an account that has accepted GAIA's terms at ${DATASET_REPO}.`) +
          `\n${tail}`
        : `gaia: git ${step.args.filter((a) => a !== "-c" && !a.startsWith("credential.")).join(" ")} failed (exit ${res.code}).\n${tail}`,
    );
  }

  // The pin exists because main lost the metadata — so prove this revision
  // actually has it before publishing the checkout, rather than failing far
  // away in loadMeta with "is GAIA_DIR a full dataset clone?".
  if (!(await exists(join(staging, VALIDATION_META)))) {
    await rm(staging, { recursive: true, force: true });
    throw new Error(
      `gaia: ${DATASET_REV.slice(0, 12)} has no ${VALIDATION_META} — the pinned revision no longer carries the ` +
        `benchmark metadata. DATASET_REV in bootstrap.ts needs to point at a revision that does.`,
    );
  }

  // Atomic publish: the target only ever exists fully-formed, so a killed
  // bootstrap can't leave a half-checkout that looks populated to the next run.
  await rename(staging, target);
}

/** In-process de-dup: concurrent evals (`eval/optimize` fans out per dataset
 *  entry) must not race two clones into the same directory. Keyed BY RESOLVED
 *  DIR, not a bare singleton — two services pointed at different checkouts in
 *  one process must not be handed each other's root. */
const inflight = new Map<string, Promise<{ root: string }>>();

/**
 * Ensure a usable GAIA checkout exists; returns its resolved root.
 * Idempotent and cheap on the hot path (two stats) — safe to call per grade.
 */
export async function ensureGaiaDataset(
  opts: EnsureGaiaOptions = {},
): Promise<{ root: string }> {
  const key = resolve(opts.dir ?? process.env["GAIA_DIR"] ?? defaultGaiaDir());
  const existing = inflight.get(key);
  if (existing) return existing;
  const started = (async () => {
    const root = key;
    const log = opts.log ?? ((m: string) => console.log(m));
    const exec = opts.exec ?? defaultExec();
    const fetchText = opts.fetchText ?? defaultFetchText;
    const expected =
      opts.scorerSha256 ?? process.env["GAIA_SCORER_SHA256"] ?? SCORER_SHA256;

    if (await isPopulated(root)) {
      await assertNoLfsPointers(root);
      return { root };
    }

    const autoSetup =
      opts.autoSetup ?? process.env["GAIA_AUTO_SETUP"] !== "0";
    if (!autoSetup) {
      throw new Error(
        `gaia: ${root} is not a populated GAIA checkout and GAIA_AUTO_SETUP=0 — ` +
          `populate it manually or re-enable auto-setup.`,
      );
    }

    // Optional: HF currently serves this repo's git endpoints anonymously.
    // Used when configured, and the failure path above tells the operator to
    // set it if access is ever refused.
    const token = resolveToken(opts.hfToken);

    await assertGitLfs(exec, dirname(root));

    // A checkout can be present-but-incomplete (interrupted clone, or a dir
    // holding only scorer.py). Start clean rather than trying to repair it.
    const hasGit = await exists(join(root, ".git"));
    const hasMeta = await exists(join(root, VALIDATION_META));
    if (!hasGit || !hasMeta) {
      if (await exists(root)) {
        log(`gaia: ${root} exists but is not a complete checkout — re-cloning`);
        await rm(root, { recursive: true, force: true });
      }
      await cloneDataset(root, token, exec, log);
      await assertNoLfsPointers(root);
    }

    if (!(await exists(join(root, SCORER_FILE)))) {
      await installScorer(root, expected, fetchText, log);
    }

    log(`gaia: dataset ready at ${root}`);
    return { root };
  })();
  inflight.set(key, started);
  try {
    return await started;
  } catch (e) {
    inflight.delete(key); // let a later call retry a transient failure
    throw e;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Python
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The scorer's only dependency is numpy (`scorer.py` imports it for the
 * numeric-answer branch). Resolution order, cheapest first:
 *
 *   1. GAIA_PYTHON / opts.python — an explicit operator choice is trusted as-is.
 *   2. `python3` on PATH, IF it can import numpy. This is the prod image: the
 *      agent venv at /usr/src/agent-venv is first on PATH and already carries
 *      numpy, so a container resolves here and never builds anything.
 *   3. A cached venv at <cache>/vein/gaia-venv, created on demand. This is the
 *      dev-box path — macOS system python has no numpy, which is the only
 *      reason GAIA_PYTHON ever had to be set by hand.
 *
 * Memoised per process: step 2 costs a subprocess, and score() runs per grade.
 */
let pythonPromise: Promise<string> | undefined;

async function importsNumpy(python: string, exec: BootstrapExecFn): Promise<boolean> {
  const r = await exec(python, ["-c", "import numpy"], {
    cwd: process.cwd(),
    timeoutMs: GIT_TIMEOUT_MS,
  }).catch(() => ({ code: 1, stdout: "", stderr: "" }));
  return r.code === 0;
}

export interface EnsurePythonOptions {
  /** Explicit interpreter; defaults to env GAIA_PYTHON. Trusted without probing. */
  python?: string;
  /** Venv location when one must be built. Defaults to <cache>/vein/gaia-venv. */
  venvDir?: string;
  exec?: BootstrapExecFn;
  log?: (msg: string) => void;
}

/** Resolve (building if needed) a python interpreter that can run scorer.py. */
export async function ensureGaiaPython(opts: EnsurePythonOptions = {}): Promise<string> {
  if (pythonPromise) return pythonPromise;
  pythonPromise = (async () => {
    const exec = opts.exec ?? defaultExec();
    const log = opts.log ?? ((m: string) => console.log(m));

    const explicit = opts.python ?? process.env["GAIA_PYTHON"];
    if (explicit) return explicit;

    if (await importsNumpy("python3", exec)) return "python3";

    const venv = opts.venvDir ?? join(dirname(defaultGaiaDir()), "gaia-venv");
    const venvPython = join(venv, "bin", "python");
    if ((await exists(venvPython)) && (await importsNumpy(venvPython, exec))) {
      return venvPython;
    }

    log(`gaia: python3 has no numpy (scorer.py needs it) — building a venv at ${venv}`);
    await mkdir(dirname(venv), { recursive: true });
    await rm(venv, { recursive: true, force: true });

    const mk = await exec("python3", ["-m", "venv", venv], {
      cwd: dirname(venv),
      timeoutMs: 5 * 60_000,
    });
    if (mk.code !== 0) {
      throw new Error(
        `gaia: could not create a venv at ${venv} (exit ${mk.code}).\n${mk.stderr.slice(-800)}\n` +
          `Point GAIA_PYTHON at an interpreter that has numpy instead.`,
      );
    }
    const pip = await exec(join(venv, "bin", "pip"), ["install", "--no-cache-dir", "numpy"], {
      cwd: venv,
      timeoutMs: 10 * 60_000,
    });
    if (pip.code !== 0) {
      throw new Error(
        `gaia: numpy install failed in ${venv} (exit ${pip.code}).\n${pip.stderr.slice(-800)}`,
      );
    }
    if (!(await importsNumpy(venvPython, exec))) {
      throw new Error(`gaia: built ${venv} but it still cannot import numpy`);
    }
    log(`gaia: scorer python ready at ${venvPython}`);
    return venvPython;
  })();
  try {
    return await pythonPromise;
  } catch (e) {
    pythonPromise = undefined;
    throw e;
  }
}

/** Test seam: drop the memoised bootstrap (smokes run several scenarios). */
export function resetGaiaBootstrap(): void {
  inflight.clear();
  pythonPromise = undefined;
}
