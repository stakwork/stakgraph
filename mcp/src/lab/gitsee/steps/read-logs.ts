import { z, defineStep } from "vein";
import type { GitseeServices } from "../services/index.js";

/** Read the booted frontend's recent server logs (stdout+stderr). Use to find
 *  the cause of a 500 or a crash loop. Per-run stack keyed by `ctx.runId`.
 *  Workflow step OR agent tool. */
export default defineStep({
  type: "gitsee/read-logs",
  description:
    "Read the booted frontend's recent server logs (stdout+stderr). Use to find the cause of a 500 or a crash loop. Requires a staged+booted stack.",
  input: z.object({}),
  output: z.any(),
  async run(_cfg, ctx) {
    const { gitsee } = ctx.services as { gitsee: GitseeServices };
    const stack = gitsee.stack.get(ctx.runId);
    if (!stack) return "no staged stack — run gitsee/stage-setup + gitsee/boot first";
    const logs = await stack.readLogs();
    return logs.slice(-8000) || "(no logs captured)";
  },
});
