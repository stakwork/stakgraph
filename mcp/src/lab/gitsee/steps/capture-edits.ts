import { z, defineStep } from "vein";
import { spawn } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";

/**
 * Capture the agent's REPO EDITS as a replayable `git diff` — the precise form of
 * "what the agent changed to make this workspace boot local-first" (e.g. moving a
 * misplaced Supabase migration into the framework's auto-migrations dir, flipping
 * a USE_MOCKS default, patching a hardcoded cloud URL). The setup-profiler's
 * deliverable is the pm2.config.js + docker-compose.yml PLUS this diff: in a
 * fresh-cloned pod the diff is re-applied (`git apply`) so the loose working-tree
 * edits survive, while runtime steps (db reset/migrate) stay in PRE_START_COMMAND.
 *
 * Runs as the LAST step of gitsee-explore-services and PASSES THROUGH the agent's
 * `result`/`usage`/`cost`/`steps` (a workflow's output is its last step's output),
 * so `produce.result`/`produce.usage`/`produce.cost` keep resolving and
 * `produce.diff` is added. Captures the diff BEFORE any boot/install pollutes the
 * tree (the explore clone is install-free, so the diff is pure source edits).
 *
 * Per repo (each git-repo subdir of workspacePath): `git add -A -N` (so NEW files
 * the agent created show as additions) then `git diff`. Output `{ result, usage,
 * cost, steps, diff, changedRepos, changed }`.
 *
 * Self-contained: imports only `vein` + Node builtins. Needs `git` on PATH.
 */

function git(args: string[], cwd: string, timeoutMs = 30000): Promise<string> {
  return new Promise((resolve) => {
    const child = spawn("git", args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      resolve(stdout);
    }, timeoutMs);
    child.stdout?.on("data", (d) => (stdout += d.toString()));
    child.on("close", () => {
      clearTimeout(timer);
      resolve(stdout);
    });
    child.on("error", () => {
      clearTimeout(timer);
      resolve(stdout);
    });
  });
}

/** Immediate subdirs of `cwd` that are git repos. */
function listRepos(cwd: string): string[] {
  if (!existsSync(cwd)) return [];
  return readdirSync(cwd, { withFileTypes: true })
    .filter((e) => e.isDirectory() && existsSync(join(cwd, e.name, ".git")))
    .map((e) => e.name)
    .sort();
}

export default defineStep({
  type: "gitsee/capture-edits",
  description:
    "Capture the agent's repo edits as a replayable `git diff` per repo (the deliverable form of 'what it changed to boot local-first'). Runs `git add -A -N` + `git diff` in each git-repo subdir of workspacePath, BEFORE any install/boot pollutes the tree. Passes through the explorer agent's result/usage/cost/steps (so it can be the last step of gitsee-explore-services) and adds { diff, changedRepos, changed }. Config: workspacePath, result?, usage?, cost?, steps?, maxBytes? (default 60000). Needs `git`.",
  input: z.object({
    workspacePath: z.string().describe("dir containing the cloned repos as siblings"),
    // Passthrough of the explorer agent's output (this is the last step, and a
    // workflow's output is its last step's output).
    result: z.string().optional(),
    usage: z.any().optional(),
    cost: z.number().optional(),
    steps: z.number().optional(),
    maxBytes: z.number().int().positive().default(60000).describe("cap on the combined diff size"),
  }),
  output: z.any(),
  async run(cfg) {
    const repos = listRepos(cfg.workspacePath);
    const parts: string[] = [];
    const changedRepos: string[] = [];
    for (const repo of repos) {
      const dir = join(cfg.workspacePath, repo);
      // Intent-to-add so brand-new files show up as additions in the diff.
      await git(["add", "-A", "-N"], dir);
      const diff = await git(["diff"], dir);
      if (diff.trim()) {
        changedRepos.push(repo);
        parts.push(`=== ${repo} ===\n${diff}`);
      }
    }
    let diff = parts.join("\n\n");
    if (diff.length > cfg.maxBytes) diff = diff.slice(0, cfg.maxBytes) + "\n\n[... diff truncated ...]";
    if (changedRepos.length)
      console.log(`[gitsee/capture-edits] captured edits in ${changedRepos.length} repo(s): ${changedRepos.join(", ")}`);

    // Pass the explorer agent's fields through unchanged + add the diff.
    return {
      result: cfg.result,
      usage: cfg.usage,
      cost: cfg.cost,
      steps: cfg.steps,
      diff,
      changedRepos,
      changed: changedRepos.length > 0,
    };
  },
});
