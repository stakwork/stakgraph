/**
 * git_pr.ts — composite PR-landing pipeline
 *
 * All git invocations use spawn("git", args, { shell: false }) — never
 * executeBashCommand (which uses shell: true, never rejects on non-zero exit,
 * and has an unsuitable default timeout for push).
 *
 * Public surface:
 *   runGit, GitHubClient, OctokitGitHubClient,
 *   resolveIdentity, gitEnv, branchName,
 *   acquireWorktree, releaseWorktree, landChange
 */

import { spawn, ChildProcess } from "child_process";
import fs from "fs";
import path from "path";
import os from "os";
import { Octokit } from "@octokit/rest";
import { withRepoLock } from "./repo_lock.js";
import { gitleaksProtect } from "./gitleaks.js";
import { redactCredentials } from "./utils.js";

// ---------------------------------------------------------------------------
// runGit — shared git execution primitive
// ---------------------------------------------------------------------------

export interface RunGitResult {
  code: number;
  stdout: string;
  stderr: string;
}

export interface RunGitOpts {
  cwd: string;
  env: NodeJS.ProcessEnv;
  timeoutMs: number;
  signal?: AbortSignal;
}

/**
 * Spawn git with an explicit arg array (shell: false) and resolve with
 * { code, stdout, stderr }.  Never throws on non-zero exit — callers
 * classify.  Honours AbortSignal and timeoutMs independently.
 */
export function runGit(
  args: string[],
  opts: RunGitOpts
): Promise<RunGitResult> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const settle = (v: RunGitResult) => {
      if (settled) return;
      settled = true;
      resolve(v);
    };
    const fail = (e: Error) => {
      if (settled) return;
      settled = true;
      reject(e);
    };

    let child: ChildProcess;
    try {
      child = spawn("git", args, {
        shell: false,
        cwd: opts.cwd,
        env: opts.env,
      });
    } catch (e) {
      return fail(e instanceof Error ? e : new Error(String(e)));
    }

    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];

    child.stdout?.on("data", (d: Buffer) => stdoutChunks.push(d));
    child.stderr?.on("data", (d: Buffer) => stderrChunks.push(d));

    const collect = () => ({
      stdout: Buffer.concat(stdoutChunks).toString("utf8").trim(),
      stderr: Buffer.concat(stderrChunks).toString("utf8").trim(),
    });

    const timer = setTimeout(() => {
      if (!settled) {
        child.kill("SIGKILL");
        fail(new Error(`git ${args[0]} timed out after ${opts.timeoutMs}ms`));
      }
    }, opts.timeoutMs);

    let abortHandler: (() => void) | undefined;
    if (opts.signal) {
      abortHandler = () => {
        if (!settled) {
          child.kill("SIGKILL");
          const { stdout, stderr } = collect();
          settle({ code: -1, stdout, stderr });
        }
      };
      if (opts.signal.aborted) {
        clearTimeout(timer);
        child.kill("SIGKILL");
        return settle({ code: -1, stdout: "", stderr: "aborted" });
      }
      opts.signal.addEventListener("abort", abortHandler);
    }

    child.on("error", (err) => {
      clearTimeout(timer);
      if (abortHandler) opts.signal?.removeEventListener("abort", abortHandler);
      fail(err);
    });

    child.on("close", (code) => {
      clearTimeout(timer);
      if (abortHandler) opts.signal?.removeEventListener("abort", abortHandler);
      const { stdout, stderr } = collect();
      settle({ code: code ?? -1, stdout, stderr });
    });
  });
}

// ---------------------------------------------------------------------------
// GitHubClient interface + OctokitGitHubClient implementation
// ---------------------------------------------------------------------------

export interface GitHubUser {
  login: string;
  id: number;
  email: string | null;
  name: string | null;
}

export interface GitHubRepo {
  default_branch: string;
  permissions?: {
    push?: boolean;
    admin?: boolean;
  };
}

export interface GitHubPR {
  number: number;
  html_url: string;
}

export interface GitHubClient {
  users: {
    getAuthenticated(): Promise<{ data: GitHubUser }>;
  };
  repos: {
    get(params: { owner: string; repo: string }): Promise<{ data: GitHubRepo }>;
  };
  pulls: {
    create(params: {
      owner: string;
      repo: string;
      title: string;
      body: string;
      head: string;
      base: string;
    }): Promise<{ data: GitHubPR }>;
  };
}

/** Production implementation backed by @octokit/rest. */
export class OctokitGitHubClient implements GitHubClient {
  private octokit: Octokit;

  constructor(pat: string) {
    this.octokit = new Octokit({ auth: pat });
  }

  get users() {
    return {
      getAuthenticated: async () => {
        const r = await this.octokit.users.getAuthenticated();
        return {
          data: {
            login: r.data.login,
            id: r.data.id,
            email: r.data.email ?? null,
            name: r.data.name ?? null,
          } as GitHubUser,
        };
      },
    };
  }

  get repos() {
    return {
      get: async (params: { owner: string; repo: string }) => {
        const r = await this.octokit.repos.get(params);
        return {
          data: {
            default_branch: r.data.default_branch,
            permissions: r.data.permissions,
          } as GitHubRepo,
        };
      },
    };
  }

  get pulls() {
    return {
      create: async (params: {
        owner: string;
        repo: string;
        title: string;
        body: string;
        head: string;
        base: string;
      }) => {
        const r = await this.octokit.pulls.create(params);
        return {
          data: {
            number: r.data.number,
            html_url: r.data.html_url,
          } as GitHubPR,
        };
      },
    };
  }
}

// ---------------------------------------------------------------------------
// resolveIdentity
// ---------------------------------------------------------------------------

export interface AgentIdentity {
  login: string;
  name: string;
  email: string;
  id: number;
}

export type ResolveIdentityResult =
  | { ok: true; identity: AgentIdentity }
  | { ok: false; failure: "identity_mismatch"; error: string };

/**
 * Resolve the GitHub identity for a PAT.  The token's own login is
 * authoritative — claimedUsername, if supplied, must match or we refuse.
 * Email falls back to the GitHub no-reply address when not public.
 */
export async function resolveIdentity(
  githubClient: GitHubClient,
  _pat: string,
  claimedUsername?: string
): Promise<ResolveIdentityResult> {
  const { data: user } = await githubClient.users.getAuthenticated();
  if (claimedUsername && claimedUsername !== user.login) {
    return {
      ok: false,
      failure: "identity_mismatch",
      error: `Token login '${user.login}' does not match supplied username '${claimedUsername}'`,
    };
  }
  const email =
    user.email && user.email.trim().length > 0
      ? user.email.trim()
      : `${user.id}+${user.login}@users.noreply.github.com`;
  const identity: AgentIdentity = {
    login: user.login,
    name: user.name ?? user.login,
    email,
    id: user.id,
  };
  return { ok: true, identity };
}

// ---------------------------------------------------------------------------
// gitEnv — explicit allowlist, never spreads process.env
// ---------------------------------------------------------------------------

/**
 * Build the child-process environment for git invocations.
 * Uses an explicit allowlist — never spreads process.env — so ambient
 * credentials (GITHUB_TOKEN, GH_TOKEN, PAT) from the container env cannot
 * leak into the child.
 */
export function gitEnv(
  identity: AgentIdentity,
  pat: string,
  runHome: string
): NodeJS.ProcessEnv {
  const b64 = Buffer.from(`x-access-token:${pat}`).toString("base64");
  return {
    // Author/committer identity
    GIT_AUTHOR_NAME: identity.name,
    GIT_AUTHOR_EMAIL: identity.email,
    GIT_COMMITTER_NAME: identity.name,
    GIT_COMMITTER_EMAIL: identity.email,
    // Per-run credential via env, not argv or .git/config
    GIT_CONFIG_COUNT: "1",
    GIT_CONFIG_KEY_0: "http.https://github.com/.extraheader",
    GIT_CONFIG_VALUE_0: `Authorization: Basic ${b64}`,
    // Isolation: no global/system config, no credential helpers, no prompts
    GIT_CONFIG_GLOBAL: "/dev/null",
    GIT_CONFIG_SYSTEM: "/dev/null",
    GIT_TERMINAL_PROMPT: "0",
    GIT_ASKPASS: "",
    HOME: runHome,
    // Minimum required runtime vars
    PATH: process.env.PATH ?? "/usr/local/bin:/usr/bin:/bin",
    LANG: process.env.LANG ?? "en_US.UTF-8",
  };
}

// ---------------------------------------------------------------------------
// branchName
// ---------------------------------------------------------------------------

// Hint must start and end with alphanumeric and contain only safe chars.
// A leading dash is git option injection; trailing dash is just ugly.
const HINT_RE = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,58}[a-zA-Z0-9]$|^[a-zA-Z0-9]$/;

function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40) || "change";
}

/**
 * Construct a safe branch name.  branchHint is validated against
 * HINT_RE; on failure or absence the title is slugified.  The final
 * name is always swarm/<segment>-<first-8-chars-of-runId>.  Throws if
 * the result equals the baseBranchName.
 */
export function branchName(
  runId: string,
  title: string,
  branchHint?: string,
  baseBranchName?: string
): string {
  const segment =
    branchHint && HINT_RE.test(branchHint) ? branchHint : slugify(title);
  const name = `swarm/${segment}-${runId.slice(0, 8)}`;
  if (baseBranchName && name === baseBranchName) {
    throw new Error(
      `Constructed branch name '${name}' equals the base branch '${baseBranchName}'`
    );
  }
  return name;
}

// ---------------------------------------------------------------------------
// WorktreeHandle
// ---------------------------------------------------------------------------

export interface WorktreeHandle {
  worktreePath: string;
  baseDir: string;
  baseSha: string;
  baseName: string;
  branch: string;
  owner: string;
  repo: string;
  runId: string;
  runHome: string;
  /** child processes tracked for this run (killed on release) */
  _children: Set<ChildProcess>;
  /** true once releaseWorktree has fully executed */
  _released: boolean;
}

// ---------------------------------------------------------------------------
// acquireWorktree
// ---------------------------------------------------------------------------

export interface AcquireWorktreeOpts {
  baseDir: string;
  owner: string;
  repo: string;
  runId: string;
  /** PAT used for fetching */
  pat: string;
  githubClient: GitHubClient;
  /** Explicit base branch name; resolved from GitHub API if omitted */
  base?: string;
  /** Pin a specific commit sha instead of the branch head */
  commit?: string;
  signal?: AbortSignal;
}

export type AcquireWorktreeResult =
  | { ok: true; handle: WorktreeHandle }
  | { ok: false; failure: "base_repo_vanished"; error: string };

const SWARM_WORK_ROOT = "/tmp/.swarm-work";

/**
 * Acquire a per-run git worktree at
 * /tmp/.swarm-work/<runId>/<owner>/<repo>, serialized via withRepoLock.
 */
export async function acquireWorktree(
  opts: AcquireWorktreeOpts
): Promise<AcquireWorktreeResult> {
  return withRepoLock(opts.baseDir, () => _acquireWorktree(opts));
}

async function _acquireWorktree(
  opts: AcquireWorktreeOpts
): Promise<AcquireWorktreeResult> {
  const { baseDir, owner, repo, runId, pat, githubClient, base, commit, signal } = opts;

  // Path traversal guard
  const worktreePath = path.resolve(
    path.join(SWARM_WORK_ROOT, runId, owner, repo)
  );
  const expectedParent = path.resolve(path.join(SWARM_WORK_ROOT, runId));
  // Must be exactly two segments below the runId dir
  const rel = path.relative(expectedParent, worktreePath);
  if (rel.startsWith("..") || rel.split(path.sep).length !== 2) {
    return {
      ok: false,
      failure: "base_repo_vanished",
      error: `Path traversal rejected: ${worktreePath}`,
    };
  }

  // Verify baseDir is a valid git repo
  if (!fs.existsSync(baseDir)) {
    return {
      ok: false,
      failure: "base_repo_vanished",
      error: `Base directory does not exist: ${baseDir}`,
    };
  }
  const gitDir = path.join(baseDir, ".git");
  if (!fs.existsSync(gitDir)) {
    return {
      ok: false,
      failure: "base_repo_vanished",
      error: `Not a git repository (no .git): ${baseDir}`,
    };
  }

  // A minimal env for rev-parse (no credentials needed — local only)
  const localEnv: NodeJS.ProcessEnv = {
    PATH: process.env.PATH ?? "/usr/local/bin:/usr/bin:/bin",
    HOME: os.tmpdir(),
    GIT_TERMINAL_PROMPT: "0",
    GIT_ASKPASS: "",
  };

  const revParseCheck = await runGit(["rev-parse", "--git-dir"], {
    cwd: baseDir,
    env: localEnv,
    timeoutMs: 10_000,
    signal,
  });
  if (revParseCheck.code !== 0) {
    return {
      ok: false,
      failure: "base_repo_vanished",
      error: `git rev-parse failed in ${baseDir}: ${revParseCheck.stderr}`,
    };
  }

  // Resolve base branch name
  let baseName: string;
  if (base) {
    baseName = base;
  } else {
    const { data: repoData } = await githubClient.repos.get({ owner, repo });
    baseName = repoData.default_branch;
  }

  // Build a credential env for fetch
  const runHome = path.join(SWARM_WORK_ROOT, runId, ".home");
  fs.mkdirSync(runHome, { mode: 0o700, recursive: true });

  // Create a dummy identity for fetching (will be overwritten by gitEnv in landChange)
  const fetchEnv: NodeJS.ProcessEnv = {
    PATH: process.env.PATH ?? "/usr/local/bin:/usr/bin:/bin",
    HOME: runHome,
    GIT_TERMINAL_PROMPT: "0",
    GIT_ASKPASS: "",
    GIT_CONFIG_COUNT: "1",
    GIT_CONFIG_KEY_0: "http.https://github.com/.extraheader",
    GIT_CONFIG_VALUE_0: `Authorization: Basic ${Buffer.from(
      `x-access-token:${pat}`
    ).toString("base64")}`,
    GIT_CONFIG_GLOBAL: "/dev/null",
    GIT_CONFIG_SYSTEM: "/dev/null",
  };

  let baseSha: string;

  if (commit) {
    // Resolve the explicit commit
    const res = await runGit(["rev-parse", "--verify", commit], {
      cwd: baseDir,
      env: localEnv,
      timeoutMs: 15_000,
      signal,
    });
    if (res.code !== 0) {
      return {
        ok: false,
        failure: "base_repo_vanished",
        error: `Cannot resolve commit ${commit}: ${res.stderr}`,
      };
    }
    baseSha = res.stdout.trim();
  } else {
    // Fetch the base branch explicitly — the base clone may be single-branch
    const remoteUrl = `https://github.com/${owner}/${repo}.git`;
    const fetchRes = await runGit(
      [
        "fetch",
        "--no-tags",
        remoteUrl,
        `+refs/heads/${baseName}:refs/remotes/origin/${baseName}`,
      ],
      { cwd: baseDir, env: fetchEnv, timeoutMs: 120_000, signal }
    );
    if (fetchRes.code !== 0) {
      return {
        ok: false,
        failure: "base_repo_vanished",
        error: `fetch failed: ${fetchRes.stderr}`,
      };
    }

    // Resolve the sha
    const shaRes = await runGit(
      ["rev-parse", `refs/remotes/origin/${baseName}`],
      { cwd: baseDir, env: localEnv, timeoutMs: 10_000, signal }
    );
    if (shaRes.code !== 0) {
      return {
        ok: false,
        failure: "base_repo_vanished",
        error: `Cannot resolve refs/remotes/origin/${baseName}: ${shaRes.stderr}`,
      };
    }
    baseSha = shaRes.stdout.trim();
  }

  // Determine branch name
  const branch = branchName(runId, "swarm-change", undefined, baseName);

  // Create the worktree dir with mode 0700
  fs.mkdirSync(worktreePath, { mode: 0o700, recursive: true });
  // worktree add requires the directory to not exist; remove if empty
  // (git will recreate it)
  try {
    fs.rmdirSync(worktreePath);
  } catch {
    // non-empty or doesn't exist — git will fail with its own message
  }

  const worktreeRes = await runGit(
    ["worktree", "add", "--detach", worktreePath, baseSha],
    { cwd: baseDir, env: localEnv, timeoutMs: 30_000, signal }
  );
  if (worktreeRes.code !== 0) {
    fs.rmSync(worktreePath, { recursive: true, force: true });
    return {
      ok: false,
      failure: "base_repo_vanished",
      error: `git worktree add failed: ${worktreeRes.stderr}`,
    };
  }
  // Fix permissions on created directory
  try { fs.chmodSync(worktreePath, 0o700); } catch { /* best-effort */ }

  // Create the branch inside the worktree
  const switchRes = await runGit(["switch", "-c", branch], {
    cwd: worktreePath,
    env: localEnv,
    timeoutMs: 10_000,
    signal,
  });
  if (switchRes.code !== 0) {
    // cleanup
    await runGit(["worktree", "remove", "--force", worktreePath], {
      cwd: baseDir,
      env: localEnv,
      timeoutMs: 15_000,
    });
    fs.rmSync(worktreePath, { recursive: true, force: true });
    return {
      ok: false,
      failure: "base_repo_vanished",
      error: `git switch -c failed: ${switchRes.stderr}`,
    };
  }

  const handle: WorktreeHandle = {
    worktreePath,
    baseDir,
    baseSha,
    baseName,
    branch,
    owner,
    repo,
    runId,
    runHome,
    _children: new Set(),
    _released: false,
  };
  return { ok: true, handle };
}

// ---------------------------------------------------------------------------
// releaseWorktree — idempotent cleanup
// ---------------------------------------------------------------------------

/**
 * Remove this repo's worktree and its run-scoped files.  Safe to call twice
 * or when the worktree was already removed.
 *
 * A single runId may host multiple repos at
 * /tmp/.swarm-work/<runId>/<owner>/<repo>, all sharing the run's `.home`.
 * We therefore only remove THIS repo's worktree directory (and its now-empty
 * owner dir), and delete the shared run directory — including `.home` — only
 * once no other repo remains under it.  Nuking the whole run dir here would
 * destroy sibling repos' worktrees out from under their git metadata.
 */
export async function releaseWorktree(handle: WorktreeHandle): Promise<void> {
  if (handle._released) return;
  handle._released = true;

  // Drop this run's landed-result entry — the run is over, so the
  // already_landed guard no longer needs it and its diff can be freed.
  landedResults.delete(handle.runId);

  // Kill any tracked child processes
  for (const child of handle._children) {
    try { child.kill("SIGKILL"); } catch { /* ignore */ }
  }
  handle._children.clear();

  const localEnv: NodeJS.ProcessEnv = {
    PATH: process.env.PATH ?? "/usr/local/bin:/usr/bin:/bin",
    HOME: os.tmpdir(),
    GIT_TERMINAL_PROMPT: "0",
    GIT_ASKPASS: "",
  };

  // git worktree remove --force
  if (fs.existsSync(handle.baseDir)) {
    await runGit(
      ["worktree", "remove", "--force", handle.worktreePath],
      { cwd: handle.baseDir, env: localEnv, timeoutMs: 15_000 }
    ).catch(() => {});

    // git worktree prune
    await runGit(["worktree", "prune"], {
      cwd: handle.baseDir,
      env: localEnv,
      timeoutMs: 10_000,
    }).catch(() => {});
  }

  const runDir = path.join(SWARM_WORK_ROOT, handle.runId);

  // Remove only this repo's worktree dir (git usually removed it already) and
  // then prune its now-empty owner dir.
  fs.rmSync(handle.worktreePath, { recursive: true, force: true });
  const ownerDir = path.dirname(handle.worktreePath);
  // Only prune the owner dir when it lives under this run dir (guards against
  // pathological owner values collapsing the path elsewhere).
  if (path.dirname(ownerDir) === runDir) {
    try {
      if (fs.readdirSync(ownerDir).length === 0) fs.rmdirSync(ownerDir);
    } catch { /* non-empty or already gone — leave it */ }
  }

  // Delete the shared run dir (including `.home`) only if no sibling repo
  // remains — i.e. nothing left but the `.home` scratch dir.
  try {
    const remaining = fs
      .readdirSync(runDir)
      .filter((entry) => entry !== ".home");
    if (remaining.length === 0) {
      fs.rmSync(runDir, { recursive: true, force: true });
    }
  } catch {
    // runDir already gone — nothing to do.
  }
}

// ---------------------------------------------------------------------------
// landChange
// ---------------------------------------------------------------------------

export interface LandChangeOpts {
  handle: WorktreeHandle;
  identity: AgentIdentity;
  env: NodeJS.ProcessEnv;
  githubClient: GitHubClient;
  signal?: AbortSignal;
  title: string;
  body: string;
  /** Configurable limits */
  maxFiles?: number;
  maxBytes?: number;
}

export type LandChangeSuccess = {
  ok: true;
  url: string;
  number: number;
  branch: string;
  base: string;
  headSha: string;
  diff: string;
  filesChanged: number;
};

export type LandChangeFailure = {
  ok: false;
  failure:
    | "patch_conflict"
    | "push_rejected"
    | "pr_create_failed"
    | "base_repo_vanished"
    | "no_changes"
    | "secrets_detected"
    | "change_too_large"
    | "identity_mismatch"
    | "no_push_permission"
    | "aborted"
    | "already_landed";
  diff: string;
  error: string;
};

export type LandChangeResult = LandChangeSuccess | LandChangeFailure;

// Per-run already-landed tracking (keyed by runId).
//
// Entries are normally evicted by releaseWorktree when a run finishes. The
// size cap is a backstop against runs that store a result but never release
// (crash, early return): the Map preserves insertion order, so exceeding the
// cap evicts the oldest entries first. Without this, every landed run would
// retain its full diff (up to DEFAULT_MAX_BYTES) for the process lifetime.
const landedResults = new Map<string, LandChangeResult>();
const MAX_LANDED_RESULTS = 512;

/** Record a run's result, evicting the oldest entries past the cap. */
function recordLandedResult(runId: string, result: LandChangeResult): void {
  landedResults.set(runId, result);
  while (landedResults.size > MAX_LANDED_RESULTS) {
    const oldest = landedResults.keys().next().value;
    if (oldest === undefined) break;
    landedResults.delete(oldest);
  }
}

const DEFAULT_MAX_FILES = 200;
const DEFAULT_MAX_BYTES = 2 * 1024 * 1024; // 2 MB

/**
 * Stage, scan, commit, push, and open a PR.  Returns a discriminated result
 * carrying the diff on both success and failure.  A second call for the
 * same runId returns `already_landed`.
 */
export async function landChange(
  opts: LandChangeOpts
): Promise<LandChangeResult> {
  const { handle, identity, env, githubClient, signal, title, body } = opts;
  const maxFiles = opts.maxFiles ?? DEFAULT_MAX_FILES;
  const maxBytes = opts.maxBytes ?? DEFAULT_MAX_BYTES;

  // already_landed guard
  const prior = landedResults.get(handle.runId);
  if (prior) return { ...prior, failure: "already_landed" } as LandChangeFailure;

  let stagedDiff = "";

  const fail = (
    failure: LandChangeFailure["failure"],
    error: string
  ): LandChangeFailure => {
    const result: LandChangeFailure = {
      ok: false,
      failure,
      diff: stagedDiff,
      error: redactCredentials(error, opts.env?.GIT_CONFIG_VALUE_0?.replace(
        /^Authorization: Basic /, ""
      )),
    };
    recordLandedResult(handle.runId, result);
    return result;
  };

  // 1. Abort check
  if (signal?.aborted) return fail("aborted", "Aborted before git add");

  // 2. git add -A
  const addRes = await runGit(["add", "-A"], {
    cwd: handle.worktreePath,
    env,
    timeoutMs: 30_000,
    signal,
  });
  if (addRes.code !== 0) {
    return fail("patch_conflict", `git add failed: ${addRes.stderr}`);
  }

  // 3. Check for staged changes
  const statusRes = await runGit(
    ["diff", "--cached", "--name-only"],
    { cwd: handle.worktreePath, env, timeoutMs: 10_000 }
  );
  const stagedFiles = statusRes.stdout
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (stagedFiles.length === 0) {
    return fail("no_changes", "No staged changes after git add -A");
  }

  // 4. Capture staged diff (pre-commit)
  const diffCachedRes = await runGit(
    ["diff", "--cached"],
    { cwd: handle.worktreePath, env, timeoutMs: 30_000 }
  );
  stagedDiff = diffCachedRes.stdout;

  // 5. Size ceiling
  if (stagedFiles.length > maxFiles) {
    return fail(
      "change_too_large",
      `Staged ${stagedFiles.length} files, limit is ${maxFiles}`
    );
  }

  // Count bytes in staged diff as a proxy for total change size
  const stagedBytes = Buffer.byteLength(stagedDiff, "utf8");
  if (stagedBytes > maxBytes) {
    return fail(
      "change_too_large",
      `Staged diff is ${stagedBytes} bytes, limit is ${maxBytes}`
    );
  }

  // 6. Secret scan (gitleaks protect)
  try {
    const findings = gitleaksProtect(handle.worktreePath);
    if (findings.length > 0) {
      const summary = findings
        .map((f) => `${f.File}:${f.StartLine} (${f.RuleID})`)
        .join(", ");
      return fail("secrets_detected", `Secret scan found ${findings.length} finding(s): ${summary}`);
    }
  } catch (scanErr: unknown) {
    // If gitleaks binary is absent or any unexpected error, fail closed
    const msg =
      scanErr instanceof Error ? scanErr.message : String(scanErr);
    // Distinguish "not found" from a scan error
    if (
      msg.includes("ENOENT") ||
      msg.includes("not found") ||
      msg.includes("No such file")
    ) {
      return fail(
        "secrets_detected",
        "gitleaks binary not found — failing closed on secret scan"
      );
    }
    // Re-throw unexpected scan errors
    throw scanErr;
  }

  // 7. Write commit message to temp file (never pass via -m to avoid shell)
  const msgFile = path.join(handle.runHome, "COMMIT_MSG");
  fs.writeFileSync(msgFile, `${title}\n\n${body}`, { mode: 0o600 });

  const commitRes = await runGit(["commit", "-F", msgFile], {
    cwd: handle.worktreePath,
    env,
    timeoutMs: 30_000,
    signal,
  });
  if (commitRes.code !== 0) {
    return fail("patch_conflict", `git commit failed: ${commitRes.stderr}`);
  }

  // 8. Abort check (between commit and push)
  if (signal?.aborted) return fail("aborted", "Aborted after commit, before push");

  // 9. Capture post-commit diff
  const diffRes = await runGit(
    ["diff", `${handle.baseSha}..HEAD`],
    { cwd: handle.worktreePath, env, timeoutMs: 30_000 }
  );
  const fullDiff = diffRes.stdout || stagedDiff;

  // Resolve head sha
  const headRes = await runGit(["rev-parse", "HEAD"], {
    cwd: handle.worktreePath,
    env,
    timeoutMs: 10_000,
  });
  const headSha = headRes.stdout.trim();

  // 10. Push — explicit remote URL + fully-qualified refspec, never origin
  const remoteUrl = `https://github.com/${handle.owner}/${handle.repo}.git`;
  const refspec = `refs/heads/${handle.branch}:refs/heads/${handle.branch}`;

  const pushRes = await runGit(
    ["push", remoteUrl, refspec],
    { cwd: handle.worktreePath, env, timeoutMs: 120_000, signal }
  );
  if (pushRes.code !== 0) {
    const stderr = pushRes.stderr;
    // Classify push failure
    if (signal?.aborted) return fail("aborted", `Push aborted: ${stderr}`);
    if (stderr.includes("rejected") || stderr.includes("non-fast-forward")) {
      return fail("push_rejected", `Push rejected: ${stderr}`);
    }
    return fail("push_rejected", `Push failed (code ${pushRes.code}): ${stderr}`);
  }

  // 11. Create the PR
  let prNumber: number;
  let prUrl: string;
  try {
    const { data: pr } = await githubClient.pulls.create({
      owner: handle.owner,
      repo: handle.repo,
      title,
      body,
      head: handle.branch,
      base: handle.baseName,
    });
    prNumber = pr.number;
    prUrl = pr.html_url;
  } catch (prErr: unknown) {
    const errMsg = prErr instanceof Error ? prErr.message : String(prErr);
    // Best-effort: delete the remote branch so nothing is left behind
    await runGit(
      ["push", remoteUrl, `--delete`, `refs/heads/${handle.branch}`],
      { cwd: handle.worktreePath, env, timeoutMs: 30_000 }
    ).catch(() => {});

    const result: LandChangeFailure = {
      ok: false,
      failure: "pr_create_failed",
      diff: fullDiff,
      error: redactCredentials(errMsg),
    };
    recordLandedResult(handle.runId, result);
    return result;
  }

  const success: LandChangeSuccess = {
    ok: true,
    url: prUrl,
    number: prNumber,
    branch: handle.branch,
    base: handle.baseName,
    headSha,
    diff: fullDiff,
    filesChanged: stagedFiles.length,
  };
  recordLandedResult(handle.runId, success);
  return success;
}

/** Clear the already-landed state for a runId (for testing). */
export function _clearLandedState(runId: string): void {
  landedResults.delete(runId);
}
