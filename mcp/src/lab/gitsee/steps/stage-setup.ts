import { z, defineStep } from "vein";
import type { GitseeServices } from "../services/index.js";

/**
 * Stage a produced `pm2.config.js` + `docker-compose.yml` (FILENAME two-file
 * string) into the cloned workspace, creating the PER-RUN stack session that the
 * boot / browser / read-logs / finalize steps share (keyed by `ctx.runId`). The
 * session also snapshots running containers so teardown removes everything this
 * run spins up — disposed automatically by the lab's `onRunEnd`.
 *
 * Thin: all machinery lives in `ctx.services.gitsee.stack`. Output:
 * { staged, port, appName }.
 */
export default defineStep({
  type: "gitsee/stage-setup",
  description:
    "Stage the produced pm2.config.js + docker-compose.yml into the cloned workspace and create the per-run stack session (shared by gitsee/boot, browser, read-logs, finalize). Config: workspacePath, setup (the two-file string), useStaklink? (default true), bootCommand?, bootTimeoutMs? (default 420000), keepUp? (default false). Output: { staged, port, appName, error? }.",
  input: z.object({
    workspacePath: z.string(),
    setup: z.string(),
    useStaklink: z.boolean().default(true),
    bootCommand: z.string().default("npx -y staklink@latest start"),
    bootTimeoutMs: z.number().int().positive().default(420000),
    keepUp: z.boolean().default(false),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { gitsee } = ctx.services as { gitsee: GitseeServices };
    const stack = gitsee.stack.session(ctx.runId, cfg.workspacePath, {
      useStaklink: cfg.useStaklink,
      bootCommand: cfg.bootCommand,
      bootTimeoutMs: cfg.bootTimeoutMs,
      keepUp: cfg.keepUp,
    });
    const res = await stack.stage(cfg.setup);
    if (!res.ok) return { staged: false, error: res.error };
    return { staged: true, port: res.port, appName: res.appName };
  },
});
