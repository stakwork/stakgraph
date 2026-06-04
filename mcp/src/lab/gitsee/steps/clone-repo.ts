import { z, defineStep } from "vein";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

/**
 * SELF-CONTAINED shallow git clone (no internal deps — only `vein` + Node
 * builtins + the `git` binary). Clones a public/private GitHub repo into a temp
 * dir and returns its local path for `gitsee/explore-services` to explore.
 * Idempotent: an existing clone is reused as-is (cheap re-runs).
 *
 * Output: { repoPath }.
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
  type: "gitsee/clone-repo",
  description:
    "Self-contained shallow git clone of a GitHub repo into a temp dir (no internal deps; needs the `git` binary). Idempotent — reuses an existing clone. Config: owner, repo, token?. Output: { repoPath }.",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
    token: z.string().optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const baseDir = join(tmpdir(), "gitsee-lab");
    await mkdir(baseDir, { recursive: true });
    const repoPath = join(baseDir, `${cfg.owner}-${cfg.repo}`);

    if (existsSync(join(repoPath, ".git"))) {
      console.log(`[gitsee] reusing existing clone at ${repoPath}`);
      return { repoPath };
    }

    const auth = cfg.token ? `${cfg.token}@` : "";
    const url = `https://${auth}github.com/${cfg.owner}/${cfg.repo}.git`;
    console.log(`[gitsee] cloning ${cfg.owner}/${cfg.repo} → ${repoPath}`);
    await git(["clone", "--depth", "1", url, repoPath], baseDir);
    return { repoPath };
  },
});
