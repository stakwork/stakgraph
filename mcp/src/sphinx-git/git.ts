import { spawn, spawnSync } from "node:child_process";

/**
 * Spawn `git` with the given args, inheriting stdio (so editors, pagers, and
 * TTY behavior all work transparently). Forwards exit code and signals.
 *
 * Returns a promise that resolves once the process is fully reaped, but the
 * common usage is to let it call process.exit directly.
 */
export function spawnGit(
  args: string[],
  extraEnv?: Record<string, string>,
): Promise<number> {
  return new Promise((resolve) => {
    const child = spawn("git", args, {
      stdio: "inherit",
      env: { ...process.env, ...(extraEnv ?? {}) },
    });
    child.on("exit", (code, signal) => {
      if (signal) {
        // Re-raise the signal on ourselves so the parent shell sees a
        // signal-terminated process, not just a non-zero exit.
        process.kill(process.pid, signal);
        return;
      }
      const exit = code ?? 1;
      resolve(exit);
      process.exit(exit);
    });
    child.on("error", (err) => {
      process.stderr.write(`sphinx-git: failed to spawn git: ${err.message}\n`);
      process.exit(127);
    });
  });
}

/**
 * Synchronous capture of `git <args>` stdout. Returns null on non-zero exit.
 * Used for read-only inspection (rev-parse, cat-file) where we don't need a
 * TTY.
 */
export function gitCapture(args: string[]): string | null {
  const r = spawnSync("git", args, { encoding: "utf8" });
  if (r.status !== 0) return null;
  return r.stdout;
}
