import { z, defineStep } from "strut";
import type { JobListing, JobView } from "strut";
import { brief, dropPod, hiveApi, seal, type PodCtx } from "./_shared.js";

/** How long another job must sit with no run in flight before its pod may go
 *  to a claim that found the pool empty. Long enough that a person still in
 *  the conversation — reading the reply, opening the pod, answering a
 *  question — keeps the pod and the agent session on it; short enough that
 *  a job waiting hours on a pull request review gives its pod up to the
 *  next person who needs one, instead of holding a two-pod pool for days
 *  (peel-4, 2026-11-02: two finished jobs held both pods and every claim
 *  failed for a night). `Pod.md` names the number to the agent: change both. */
export const IDLE_MINUTES = 15;

/**
 * The pool is empty (hive's 503): free ONE pod of `workspace` that another
 * job of this strut holds while it sits idle — no run of the job in flight,
 * and nothing done for IDLE_MINUTES — the one idle longest. It goes the way
 * the hold's own release would: hive's drop-pod AS that job (so never a pod
 * hive has handed to someone else since), then the hold off the job (which
 * tells the job's host). The job's next turn finds its pod gone and claims
 * again (the `Pod` page says so). Null when there is none, or this strut's
 * jobs capability cannot list jobs.
 */
export async function takeIdlePod(ctx: PodCtx, workspace: string, now = Date.now()): Promise<{ job: string; podId: string } | null> {
  const jobs = ctx.services?.jobs;
  if (typeof jobs?.list !== "function" || typeof jobs.get !== "function") return null;
  const idle: Array<{ job: string; podId: string; since: number }> = [];
  for (const row of (await jobs.list()) as JobListing[]) {
    if (!row.holds || row.busy || row.job === ctx.job) continue;
    const view = (await jobs.get(row.job, { threads: false })) as JobView | null;
    // A run in flight — the turn holding the job's directory, or any run of
    // the job without a summary yet — keeps the pod, however long ago it began.
    if (!view || view.busy || view.runs.some((r) => r.status === "running")) continue;
    // Last used: the record's last write (a run starting, a hold taken or
    // let go) or its last run's end, whichever is later.
    const ends = view.runs.map((r) => (Date.parse(r.at) || 0) + (r.durationMs ?? 0));
    const since = Math.max(Date.parse(view.usedAt) || 0, ...ends);
    if (now - since < IDLE_MINUTES * 60_000) continue;
    for (const h of view.holds) {
      const input = h.release?.input as { workspace?: unknown } | undefined;
      if (h.kind === "pod" && h.release?.type === "pod/release" && input?.workspace === workspace) idle.push({ job: row.job, podId: h.id, since });
    }
  }
  idle.sort((a, b) => a.since - b.since);
  for (const c of idle) {
    const res = await dropPod(ctx, workspace, c.podId, c.job);
    if (!res.ok && res.status !== 404 && res.status !== 409) {
      console.warn(`pod/claim: could not take pod ${c.podId} from idle job ${c.job}: HTTP ${res.status} — ${brief(res.body)}`);
      return null;
    }
    // Off the job either way, as pod/release has it: dropped now, or not the
    // job's any more (404: hive let it go; 409: hive gave it to someone else).
    await jobs.release(c.job, c.podId);
    if (res.ok) {
      const mins = Math.round((now - c.since) / 60_000);
      console.log(`pod/claim: took pod ${c.podId} of workspace ${workspace} from job ${c.job} (idle ${mins} min) for ${ctx.job ? `job ${ctx.job}` : `run ${ctx.runId}`}`);
      return { job: c.job, podId: c.podId };
    }
  }
  return null;
}

export default defineStep({
  type: "pod/claim",
  description:
    "Claim a hive pod for a workspace: a sandbox with the workspace's repositories, a dev server (`frontend`), an IDE (`ide`) and a control server (`control`) " +
    "that runs a coding agent, resets, diffs, pushes and tests — the other pod/* steps. Output: { podId, workspace, frontend, ide, control, sealed, took }. " +
    "On a run with a job the pod is a HOLD on the job, its `frontend` the hold's url (a check on the job reads it as {{ input.urls.pod }}), and the other pod/* steps " +
    "find it there: pass them no podId, control or sealed for the job's own pod. On a run without a job, pass `control` and `sealed` (the pod's password, sealed; " +
    "the password itself is never shown) to every other pod/* step. A held pod stays claimed across turns until pod/release, or until the job is deleted or idle past " +
    `STRUT_WORKDIR_TTL_DAYS, when strut releases it — or until another job's claim finds the pool empty while this job has had no run for ${IDLE_MINUTES} minutes: ` +
    "then that claim takes the pod (`took` names the job it came from). Claim once per job and report `frontend` and `ide` as artifacts of kind `url`, so the person can watch; " +
    "claim again when a pod step says the job no longer holds its pod or the pod rejected its password. Fails `no pod available` when the pool is empty and no idle job's pod can be taken: try later. " +
    "`fresh: true` claims only a pod created after the workspace's last pool-config push, built from the config as it is now — the pod a promoted fix is " +
    "checked on (pod/config-set), a second pod beside the job's own; it fails `no fresh pod ready yet` until the pool has made one: try again shortly. " +
    "`generation` says which a pod is: `new` (created after the last push), `old`, or null from a hive that does not say. " +
    "On a run with a job hive records the job as the pod's claimant (its pod list shows the job and who started it), and pod/release lets go of it only while it still is.",
  input: z.object({
    workspace: z.string().min(1).describe("The hive workspace the pod belongs to, by its slug (no `@`) or id: the workspace whose code is changing. A job's first line names its own: `Hive workspace: @<slug>`"),
    fresh: z.boolean().default(false).describe("Only a pod created after the workspace's last pool-config push: to check a change pod/config-set made"),
  }),
  output: z.object({
    podId: z.string(),
    workspace: z.string(),
    frontend: z.string().nullable(),
    ide: z.string().nullable(),
    control: z.string().nullable(),
    sealed: z.string(),
    generation: z.enum(["new", "old"]).nullable(),
    /** The idle job whose pod was freed for this claim, when the pool was empty. */
    took: z.object({ job: z.string(), podId: z.string() }).nullable(),
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
    // A pod built from the config as it is now (strut plans/pod-config.md
    // §4): hive hands out only a pod created after the last push, or 409.
    if (cfg.fresh) q.set("generation", "new");
    const qs = q.toString();
    const url = `${base}/api/pool-manager/claim-pod/${encodeURIComponent(cfg.workspace)}${qs ? `?${qs}` : ""}`;
    let res = await ctx.services.http(url, { method: "POST", headers });
    // An empty pool may be one held by jobs that finished their work: free an
    // idle one and claim once more. (A fresh claim never sees a 503 — hive
    // answers an empty pool with 409 — and a borrowed pod is no reason to
    // take a job's.)
    const took = res.status === 503 ? await takeIdlePod(ctx, cfg.workspace) : null;
    if (took) res = await ctx.services.http(url, { method: "POST", headers });
    if (res.status === 503) {
      throw new Error(
        `no pod available in workspace "${cfg.workspace}" right now — ` +
          (took
            ? `pod ${took.podId}, freed from idle job ${took.job}, went to another claim first`
            : `and no pod another job here has held idle for ${IDLE_MINUTES} minutes could be freed`) +
          "; try again in a few minutes",
      );
    }
    if (cfg.fresh && res.status === 409) {
      throw new Error(`no fresh pod ready yet in workspace "${cfg.workspace}" — ${brief((res.body as any)?.error ?? res.body)}; the pool makes one after a config push: try again in a minute or two`);
    }
    if (!res.ok) {
      throw new Error(`hive claim-pod for workspace "${cfg.workspace}": HTTP ${res.status} — ${brief(res.body)} (is the workspace right — a slug or id of this org — and HIVE_API_KEY an org key for its org?)`);
    }
    const b: any = res.body ?? {};
    const podId = b.podId ?? b.pod_id;
    if (!podId) throw new Error(`hive claim-pod returned no podId: ${brief(b)}`);
    if (!b.password) throw new Error(`hive returned pod ${podId} with no password; nothing can drive it`);
    const generation = b.generation === "new" || b.generation === "old" ? (b.generation as "new" | "old") : null;
    if (cfg.fresh && generation !== "new") {
      // A hive that predates `generation=new` hands out any pod: give it back
      // rather than check a config change on a pod built before it.
      const drop = await dropPod(ctx, cfg.workspace, String(podId), ctx.job);
      if (!drop.ok && drop.status !== 404 && drop.status !== 409) {
        console.warn(`pod/claim: could not release pod ${podId}: HTTP ${drop.status} — ${brief(drop.body)}`);
      }
      throw new Error(`hive gave pod ${podId} for a fresh claim but did not say it is new (generation: ${JSON.stringify(b.generation ?? null)}) — this hive cannot hand out a pod built after the last push; the pod was released`);
    }
    const out = {
      podId: String(podId),
      workspace: cfg.workspace,
      frontend: (b.frontend as string | undefined) ?? null,
      ide: (b.ide as string | undefined) ?? (b.pod_url as string | undefined) ?? null,
      control: (b.control as string | undefined) ?? null,
      sealed: await seal(ctx, String(b.password)),
      generation,
      took,
    };
    if (ctx.job && ctx.services?.jobs) {
      // `url` is where the pod is reached: strut folds it into the job's
      // `urls.pod`, so a claim's check says `{{ input.urls.pod }}/settings`
      // and survives the pod being released and claimed again. A fresh pod
      // is only borrowed to check a config change, never the job's pod: its
      // hold carries no url, so `urls.pod` stays the job's own.
      //
      // `control` + `sealed` ride in the release input — strut keeps it as
      // given and pod/release's schema drops the two — so every other pod/*
      // step on the job reads them from here (podAccess), never from what a
      // model copied. `GET /jobs/:id` serves them to the readers of the run
      // log, which already carries `sealed` in this step's output; `job.update`
      // carries none of the input. Sealed under HIVE_API_KEY either way.
      await ctx.services.jobs.hold(ctx.job, {
        id: out.podId,
        kind: "pod",
        release: {
          type: "pod/release",
          input: { workspace: cfg.workspace, podId: out.podId, ...(out.control ? { control: out.control, sealed: out.sealed } : {}) },
        },
        ...(out.frontend ? { note: out.frontend, ...(cfg.fresh ? {} : { url: out.frontend }) } : {}),
      });
    }
    return out;
  },
});
