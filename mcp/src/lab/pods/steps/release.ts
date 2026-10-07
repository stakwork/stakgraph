import { z, defineStep } from "strut";
import { brief, hiveApi, type PodCtx } from "./_shared.js";

export default defineStep({
  type: "pod/release",
  description:
    "Release a claimed hive pod back to the pool and drop the job's hold on it. Do this when the job is done with the pod — the change is merged, " +
    "or the person says so; a pod the job still needs stays claimed (strut releases it itself when the job is deleted or idle too long). " +
    "A pod hive no longer knows counts as released. Output: { podId, released: true }.",
  input: z.object({
    workspace: z.string().min(1).describe("The hive workspace id the pod was claimed for"),
    podId: z.string().min(1).describe("The pod, from pod/claim"),
  }),
  output: z.object({ podId: z.string(), released: z.literal(true) }),
  async run(cfg, ctx: PodCtx) {
    const { base, headers } = await hiveApi(ctx);
    const q = new URLSearchParams({ podId: cfg.podId });
    const url = `${base}/api/pool-manager/drop-pod/${encodeURIComponent(cfg.workspace)}?${q}`;
    const res = await ctx.services.http(url, { method: "POST", headers });
    // 404: hive already let it go (recycled, or released by hand) — nothing to hold on to.
    if (!res.ok && res.status !== 404) {
      throw new Error(`hive drop-pod for pod "${cfg.podId}" in workspace "${cfg.workspace}": HTTP ${res.status} — ${brief(res.body)} (an org key releases only its own org's pods)`);
    }
    if (ctx.job && ctx.services?.jobs) await ctx.services.jobs.release(ctx.job, cfg.podId);
    return { podId: cfg.podId, released: true as const };
  },
});
