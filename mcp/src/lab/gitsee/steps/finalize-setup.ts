import { z, defineStep } from "vein";
import type { GitseeServices } from "../services/index.js";

/**
 * Produce the boot-and-exercise deliverable from the per-run stack session
 * (keyed by `ctx.runId`): the final pod-portable `setup` (re-read edited config,
 * `$POD_*` placeholders kept), the replayable per-repo `git diff` of the agent's
 * source edits, and the boot/working verdict. The LAST step of the QA loop.
 *
 * The teardown of the booted stack is handled by the lab's `onRunEnd` — this
 * step is read-only against the (still-live) workspace, so the diff is intact.
 *
 * Output: { booted, working, port, setup, report, diff, changedRepos, changed }.
 */
export default defineStep({
  type: "gitsee/finalize-setup",
  description:
    "Produce the deliverable from the per-run stack: final pod-portable setup (pm2.config.js + docker-compose.yml), the replayable per-repo git diff, and the boot/working verdict. Config: report? (the agent's markdown report), working? (last assess verdict). Output: { booted, working, port, setup, report, diff, changedRepos, changed }.",
  input: z.object({
    report: z.string().default(""),
    working: z.boolean().nullable().default(null),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { gitsee } = ctx.services as { gitsee: GitseeServices };
    const stack = gitsee.stack.get(ctx.runId);
    if (!stack) {
      return { booted: null, working: cfg.working, port: null, setup: "", report: cfg.report, diff: "", changedRepos: [], changed: false };
    }
    const setup = stack.finalSetup();
    const { diff, changedRepos } = await stack.captureRepoDiff();
    return {
      booted: stack.lastBooted,
      working: cfg.working,
      port: stack.port,
      setup,
      report: cfg.report,
      diff,
      changedRepos,
      changed: changedRepos.length > 0,
    };
  },
});
