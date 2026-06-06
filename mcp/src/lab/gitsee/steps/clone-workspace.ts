import { z, defineStep } from "vein";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, readdir, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

/**
 * SELF-CONTAINED clone of a WORKSPACE — a set of repos cloned as SIBLINGS under
 * one dir (mirroring the pod's `/workspaces/<repo>` layout). A workspace usually
 * has one app to run (the "frontend") plus local-dependency repos it builds
 * against. No internal deps — only `vein` + Node builtins + the `git` binary.
 * Idempotent: existing clones are reused (per workspace + rev). `token` falls
 * back to the GITHUB_TOKEN env for private repos.
 *
 * Output: { workspacePath, repos } — workspacePath is the dir containing each
 * repo as a subdir (`<workspacePath>/<repo>`); repos is the list of dir names.
 */
function git(args: string[], cwd: string, timeoutMs = 120000): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn("git", args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`git ${args[0]} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    child.stderr.on("data", (d) => (stderr += d.toString()));
    child.on("close", (code) => {
      clearTimeout(timer);
      code === 0 ? resolve() : reject(new Error(`git ${args[0]} failed (${code}): ${stderr}`));
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
}

export default defineStep({
  type: "gitsee/clone-workspace",
  description:
    "Self-contained clone of a WORKSPACE (a set of repos cloned as siblings under one dir, like the pod's /workspaces/<repo>). No internal deps; needs `git`. Idempotent. Config: workspace (dir/label), repos ([{owner, repo, rev?}], rev pins a commit), token? (falls back to GITHUB_TOKEN env), clean? (default true — `git reset --hard && git clean -fd` a reused clone so each run starts from a pristine working tree, discarding the prior agent's edits/new files (keeps gitignored node_modules); ALSO prunes stale sibling dirs no longer in `repos`). Output: { workspacePath, repos }.",
  input: z.object({
    workspace: z.string().describe("workspace name — the clone dir under the lab tmp root"),
    repos: z
      .array(
        z.object({
          owner: z.string(),
          repo: z.string(),
          rev: z.string().optional().describe("commit/branch/tag to pin (defaults to default-branch HEAD)"),
        }),
      )
      .min(1)
      .describe("the repos to clone as siblings; usually the frontend app + its local-dependency repos"),
    token: z.string().optional(),
    clean: z
      .boolean()
      .default(true)
      .describe(
        "reset a REUSED clone to a pristine working tree (git reset --hard + git clean -fd) so each run/optimizer generation starts fresh — discards the prior explore agent's edits and any files it created. Keeps gitignored deps (node_modules) for speed. Also PRUNES stale sibling dirs in the reused workspace that aren't in the current `repos` list (e.g. a repo dropped from the dataset) so the explore agent can't pick an orphaned repo as the frontend. Set false to preserve a dirty tree for debugging.",
      ),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const baseDir = join(tmpdir(), "gitsee-lab");
    const wsDir = join(baseDir, cfg.workspace);
    await mkdir(wsDir, { recursive: true });
    const token = cfg.token || process.env.GITHUB_TOKEN;
    const auth = token ? `${token}@` : "";

    const repos: string[] = [];
    for (const r of cfg.repos) {
      const rev = r.rev; // local const so narrowing survives into the .catch closure
      const dir = join(wsDir, r.repo);
      const url = `https://${auth}github.com/${r.owner}/${r.repo}.git`;

      if (existsSync(join(dir, ".git"))) {
        console.log(`[gitsee] reusing existing clone at ${dir}`);
        // Pristine slate: drop the prior explore agent's edits + untracked files
        // (best-effort; keeps gitignored node_modules). Done before any rev
        // checkout so a dirty tree can't block it.
        if (cfg.clean) {
          await git(["reset", "--hard"], dir).catch(() => {});
          await git(["clean", "-fd"], dir).catch(() => {});
        }
      } else {
        console.log(`[gitsee] cloning ${r.owner}/${r.repo}${rev ? " @ " + rev : ""} → ${dir}`);
        await git(["clone", "--depth", "1", url, dir], wsDir);
      }

      if (rev) {
        await git(["fetch", "--depth", "1", "origin", rev], dir).catch(() => git(["fetch", "origin", rev], dir));
        await git(["checkout", "FETCH_HEAD"], dir);
      }
      repos.push(r.repo);
    }

    // Prune STALE entries: anything in the reused workspace that isn't a repo in
    // the current list — orphaned repo dirs (a repo dropped from the dataset) AND
    // stray root files (e.g. a prior verify-setup's staged pm2.config.js /
    // docker-compose.yml). The workspace should contain only the listed repo
    // subdirs; otherwise the explore agent lists the workspace, sees the cruft,
    // and may pick an orphaned repo as the "frontend" — the cross-version
    // contamination bug.
    if (cfg.clean) {
      const keep = new Set(repos);
      for (const entry of await readdir(wsDir, { withFileTypes: true })) {
        if (keep.has(entry.name)) continue;
        console.log(`[gitsee] pruning stale workspace entry ${join(wsDir, entry.name)}`);
        await rm(join(wsDir, entry.name), { recursive: true, force: true }).catch(() => {});
      }
    }

    console.log(`[gitsee] workspace "${cfg.workspace}" ready at ${wsDir} (${repos.length} repo(s): ${repos.join(", ")})`);
    return { workspacePath: wsDir, repos };
  },
});
