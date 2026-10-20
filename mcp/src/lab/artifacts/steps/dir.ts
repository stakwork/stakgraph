import { z, defineStep, type StepContext, type StrutCapabilities } from "strut";
import { isAbsolute, join, resolve, sep } from "node:path";

/**
 * Resolve THIS run's artifacts directory (creating it) and return its
 * absolute path. The bridge between the artifacts capability (keyed by
 * ctx.runId, which workflow expressions can't see) and steps that take a
 * path — point an `agent` step's `cwd` at `{{ dir.path }}` so its file tools
 * write into the run's artifact store, servable at GET /artifacts/:runId.
 *
 * Optional `sub`: a RELATIVE subdirectory to create under the run's dir and
 * return instead. Subflows share their parent's runId (one run = one
 * artifacts dir), so several produce subflows in one batch run would
 * otherwise collide on `./output/` — a per-item `sub` (e.g. the task id)
 * keeps them apart. The subdirectory is made through `ctx.services.shell`
 * (`mkdir -p` under the run's own dir): a step reaches the disk through the
 * services bag, never `node:fs` (strut src/step-lint.ts), and the artifacts
 * capability has no mkdir of its own.
 */
export default defineStep({
  type: "artifacts/dir",
  description:
    "Return this run's artifacts directory (absolute path, created on demand). " +
    "Use as an early workflow step and point an agent step's cwd at its `path` output " +
    "so produced files land in the run's artifact store (browsable at GET /artifacts/:runId). " +
    "Optional `sub`: relative subdirectory to create under the run's dir and return instead — " +
    "use a per-item value (e.g. the task id) when several produce subflows share one parent " +
    "run (same runId → same artifacts dir) so they don't collide on ./output/.",
  input: z.object({
    sub: z
      .string()
      .optional()
      .describe(
        "Relative subdirectory of the run's artifacts dir to create and return (nested paths ok). Absolute paths and '..' segments are rejected.",
      ),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const c = ctx as StepContext<StrutCapabilities>;
    const artifacts = c.services?.artifacts;
    if (!artifacts) throw new Error("artifacts capability unavailable");
    const base = await artifacts.dir(c.runId);
    if (!cfg.sub) return { path: base, runId: c.runId };
    if (isAbsolute(cfg.sub) || cfg.sub.split(/[\\/]/).includes("..")) {
      throw new Error(`artifacts/dir: sub must be a relative path without '..': "${cfg.sub}"`);
    }
    const path = resolve(join(base, cfg.sub));
    if (path !== base && !path.startsWith(base + sep)) {
      throw new Error(`artifacts/dir: sub escapes the run's artifacts dir: "${cfg.sub}"`);
    }
    const shell = c.services?.shell;
    if (!shell) throw new Error("shell capability unavailable");
    const made = await shell({ cmd: "mkdir", args: ["-p", path], cwd: base });
    if (made.code !== 0) throw new Error(`artifacts/dir: could not create "${cfg.sub}": ${made.stderr.trim() || `exit ${made.code}`}`);
    return { path, runId: c.runId, sub: cfg.sub };
  },
});
