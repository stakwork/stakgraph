import { z, defineStep, type StepContext, type VeinCapabilities } from "vein";

/**
 * Resolve THIS run's artifacts directory (creating it) and return its
 * absolute path. The bridge between the artifacts capability (keyed by
 * ctx.runId, which workflow expressions can't see) and steps that take a
 * path — point an `agent` step's `cwd` at `{{ dir.path }}` so its file tools
 * write into the run's artifact store, servable at GET /artifacts/:runId.
 */
export default defineStep({
  type: "artifacts/dir",
  description:
    "Return this run's artifacts directory (absolute path, created on demand). " +
    "Use as an early workflow step and point an agent step's cwd at its `path` output " +
    "so produced files land in the run's artifact store (browsable at GET /artifacts/:runId).",
  input: z.object({}),
  output: z.any(),
  async run(_cfg, ctx) {
    const c = ctx as StepContext<VeinCapabilities>;
    const artifacts = c.services?.artifacts;
    if (!artifacts) throw new Error("artifacts capability unavailable");
    return { path: await artifacts.dir(c.runId), runId: c.runId };
  },
});
