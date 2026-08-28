/**
 * Shared child-process helpers for every place vein shells out (the agent
 * step's repo tools + bash, the chat builder's bash). One implementation so
 * timeout semantics, output capping, and — critically — env scrubbing are
 * identical everywhere.
 *
 * **Env scrubbing.** Children get a MINIMAL environment (`minimalEnv`), not
 * `process.env`: the vein server's env holds exactly the credentials the
 * secrets boundary exists to keep away from models (ANTHROPIC_API_KEY,
 * VEIN_SECRET_KEY, provider keys, …) — a naive spawn would hand them to any
 * model-authored `env`/`printenv` one-liner. Steps get credentials via
 * `ctx.services.secrets`, never ambient env, so scrubbing costs adapters
 * nothing.
 */

import { spawn } from "node:child_process";

/** Allowlisted env vars a shell needs to behave normally (run tools from
 *  PATH, resolve ~, write temp files, keep git/locale sane). Everything
 *  else — API keys above all — is withheld. */
const ENV_ALLOWLIST = [
  "PATH",
  "HOME",
  "TMPDIR",
  "TMP",
  "TEMP",
  "LANG",
  "LC_ALL",
  "TERM",
  "USER",
  "LOGNAME",
  "SHELL",
] as const;

/** The scrubbed child env: allowlisted vars from `process.env` only. */
export function minimalEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {};
  for (const k of ENV_ALLOWLIST) {
    if (process.env[k] !== undefined) env[k] = process.env[k];
  }
  return env;
}

/** Spawn a child, capture stdout with a timeout + output cap. Exit 1 with no
 *  stderr → "No matches found" (grep/rg/find idiom). */
export function capture(
  child: ReturnType<typeof spawn>,
  timeoutMs: number,
  maxBytes: number,
): Promise<string> {
  return new Promise((resolve, reject) => {
    let stdout = "";
    let stderr = "";
    let done = false;
    const cap = (s: string) =>
      s.length > maxBytes ? s.slice(0, maxBytes) + "\n\n[... output truncated ...]" : s;
    const finish = (fn: () => void) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      fn();
    };
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      finish(() => reject(new Error(`Command timed out after ${timeoutMs}ms`)));
    }, timeoutMs);
    child.stdout?.on("data", (d) => {
      stdout += d.toString();
      if (stdout.length > maxBytes) {
        child.kill("SIGKILL");
        finish(() => resolve(cap(stdout)));
      }
    });
    child.stderr?.on("data", (d) => (stderr += d.toString()));
    child.on("close", (code) =>
      finish(() => {
        if (code === 0) resolve(cap(stdout));
        else if (code === 1 && !stderr) resolve(cap(stdout) || "No matches found");
        else reject(new Error(`Command failed (${code}): ${stderr || stdout || "Unknown error"}`));
      }),
    );
    child.on("error", (err) => finish(() => reject(err)));
  });
}

/** Run a program with explicit args (NO shell) — safe for untrusted args like a
 *  search query (no quoting/escaping/injection). Scrubbed env. */
export const runCmd = (cmd: string, args: string[], cwd: string, timeoutMs = 10000, maxBytes = 10000) =>
  capture(
    spawn(cmd, args, { cwd, stdio: ["ignore", "pipe", "pipe"], env: minimalEnv() }),
    timeoutMs,
    maxBytes,
  );

/** Run an arbitrary shell command string (the `bash` tools need a full shell).
 *  Scrubbed env — model-authored commands never see the server's API keys.
 *
 *  `extraEnv` is the ONE sanctioned widening of the scrubbed env: the agent
 *  step's `secretsEnv` config resolves named secrets via ctx.services.secrets
 *  and injects the VALUES here — into the subprocess env only, never into a
 *  prompt or log (the model writes `$NAME`; the shell expands it at exec
 *  time, and the agent step masks the values out of every tool output before
 *  the model or the event log sees them). Callers other than that path should
 *  not pass it. */
export const runShell = (
  command: string,
  cwd: string,
  timeoutMs = 15000,
  maxBytes = 10000,
  extraEnv?: Record<string, string>,
) =>
  capture(
    spawn(command, {
      cwd,
      shell: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: extraEnv && Object.keys(extraEnv).length ? { ...minimalEnv(), ...extraEnv } : minimalEnv(),
    }),
    timeoutMs,
    maxBytes,
  );
