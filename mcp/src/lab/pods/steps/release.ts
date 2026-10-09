import { z, defineStep } from "strut";
import { brief, hiveApi, type PodCtx } from "./_shared.js";

export default defineStep({
  type: "pod/release",
  description:
    "Release a claimed hive pod back to the pool and drop the job's hold on it. Do this when the job is done with the pod — the change is merged, " +
    "or the person says so; a pod the job still needs stays claimed (strut releases it itself when the job is deleted or idle too long). " +
    "A pod hive no longer knows, or has since handed to someone else, counts as released: it is not this job's any more. Output: { podId, released: true }.",
  input: z.object({
    workspace: z.string().min(1).describe("The hive workspace the pod was claimed for, as given to pod/claim (slug or id)"),
    podId: z.string().min(1).describe("The pod, from pod/claim"),
  }),
  output: z.object({ podId: z.string(), released: z.literal(true) }),
  async run(cfg, ctx: PodCtx) {
    const { base, headers } = await hiveApi(ctx);
    const q = new URLSearchParams({ podId: cfg.podId });
    // As the job (strut plans/job-artifact-events.md §2): hive drops the pod
    // only while this job is still its claimant — never from under a task or
    // another job it was handed to since. The sweep's release runs under a
    // context with the job, so a late sweep goes through the same check.
    if (ctx.job) q.set("job", ctx.job);
    const url = `${base}/api/pool-manager/drop-pod/${encodeURIComponent(cfg.workspace)}?${q}`;
    const res = await ctx.services.http(url, { method: "POST", headers });
    // 404: hive already let it go (recycled, or released by hand). 409
    // `reassigned`: hive gave it to someone else since. Either way the pod
    // is not this job's any more — nothing to hold on to.
    if (!res.ok && res.status !== 404 && res.status !== 409) {
      throw new Error(`hive drop-pod for pod "${cfg.podId}" in workspace "${cfg.workspace}": HTTP ${res.status} — ${brief(res.body)} (an org key releases only its own org's pods)`);
    }
    if (ctx.job && ctx.services?.jobs) await ctx.services.jobs.release(ctx.job, cfg.podId);
    return { podId: cfg.podId, released: true as const };
  },
});
