import { z, defineStep } from "strut";
import { brief, hiveApi, seal, type PodCtx } from "./_shared.js";

export default defineStep({
  type: "pod/claim",
  description:
    "Claim a hive pod for a workspace: a sandbox with the workspace's repositories, a dev server (`frontend`), an IDE (`ide`) and a control server (`control`) " +
    "that runs a coding agent, resets, diffs, pushes and tests — the other pod/* steps. Output: { podId, workspace, frontend, ide, control, sealed }. " +
    "`sealed` is the pod's password, sealed: pass it to every other pod/* step; the password itself is never shown. " +
    "On a run with a job the pod is a HOLD on the job: it stays claimed across turns until pod/release, or until the job is deleted or idle past " +
    "STRUT_WORKDIR_TTL_DAYS, when strut releases it. Claim once per job and report `frontend` and `ide` as artifacts of kind `url`, so the person can watch; " +
    "claim again only when a pod step says the pod rejected its password. Fails `no pod available` when the pool is empty: try later. " +
    "On a run with a job hive records the job as the pod's claimant (its pod list shows the job and who started it), and pod/release lets go of it only while it still is.",
  input: z.object({
    workspace: z.string().min(1).describe("The hive workspace the pod belongs to, by its slug (no `@`) or id: the workspace whose code is changing. A job's first line names its own: `Hive workspace: @<slug>`"),
  }),
  output: z.object({
    podId: z.string(),
    workspace: z.string(),
    frontend: z.string().nullable(),
    ide: z.string().nullable(),
    control: z.string().nullable(),
    sealed: z.string(),
  }),
  async run(cfg, ctx: PodCtx) {
    const { base, headers } = await hiveApi(ctx);
    // A run with a job claims AS the job (strut plans/job-artifact-events.md
    // §2): hive stamps the pod `job:<id>` — the claimant column a task's id
    // goes in, so its pod list shows the job — and keeps the run as the
    // reason; `drop-pod?job=` then releases only a pod still marked by this
    // job. Without a job nothing is sent, as before.
    const q = new URLSearchParams();
    if (ctx.job) {
      q.set("job", ctx.job);
      q.set("run", ctx.runId);
    }
    const qs = q.toString();
    const url = `${base}/api/pool-manager/claim-pod/${encodeURIComponent(cfg.workspace)}${qs ? `?${qs}` : ""}`;
    const res = await ctx.services.http(url, { method: "POST", headers });
    if (res.status === 503) throw new Error(`no pod available in workspace "${cfg.workspace}" right now — try again in a few minutes`);
    if (!res.ok) {
      throw new Error(`hive claim-pod for workspace "${cfg.workspace}": HTTP ${res.status} — ${brief(res.body)} (is the workspace right — a slug or id of this org — and HIVE_API_KEY an org key for its org?)`);
    }
    const b: any = res.body ?? {};
    const podId = b.podId ?? b.pod_id;
    if (!podId) throw new Error(`hive claim-pod returned no podId: ${brief(b)}`);
    if (!b.password) throw new Error(`hive returned pod ${podId} with no password; nothing can drive it`);
    const out = {
      podId: String(podId),
      workspace: cfg.workspace,
      frontend: (b.frontend as string | undefined) ?? null,
      ide: (b.ide as string | undefined) ?? (b.pod_url as string | undefined) ?? null,
      control: (b.control as string | undefined) ?? null,
      sealed: await seal(ctx, String(b.password)),
    };
    if (ctx.job && ctx.services?.jobs) {
      await ctx.services.jobs.hold(ctx.job, {
        id: out.podId,
        kind: "pod",
        release: { type: "pod/release", input: { workspace: cfg.workspace, podId: out.podId } },
        ...(out.frontend ? { note: out.frontend } : {}),
      });
    }
    return out;
  },
});
