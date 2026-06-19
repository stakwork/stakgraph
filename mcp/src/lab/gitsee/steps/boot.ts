import { z, defineStep } from "vein";
import type { GitseeServices } from "../services/index.js";

/**
 * (Re)boot the staged app: re-read the (possibly edited) pm2.config.js +
 * docker-compose.yml, bring up backing services, start the apps (staklink or
 * inline), and wait for the frontend port. Call after every config/repo edit.
 * Usable both as a workflow step AND as an agent tool (it derives the per-run
 * stack from `ctx.runId`, so it needs no config). Requires gitsee/stage-setup to
 * have run first.
 *
 * Output: { booted, port, logsTail, errors, report }.
 */
export default defineStep({
  type: "gitsee/boot",
  description:
    "Boot (or reboot) the staged app: re-stage current config, bring up compose services, start the frontend, wait for its port. Call after every config/repo edit. Returns whether the port bound + the server log tail + heuristic error lines. Requires gitsee/stage-setup first. No config. Output: { booted, port, logsTail, errors, report }.",
  input: z.object({}),
  output: z.any(),
  async run(_cfg, ctx) {
    const { gitsee } = ctx.services as { gitsee: GitseeServices };
    const stack = gitsee.stack.get(ctx.runId);
    if (!stack) return { booted: false, report: "no staged stack — run gitsee/stage-setup first" };
    const res = await stack.boot();
    // Point the (lazily-created) browser session at the booted port.
    gitsee.browser.get(ctx.runId)?.setBaseUrl(`http://localhost:${res.port}`);
    return res;
  },
});
