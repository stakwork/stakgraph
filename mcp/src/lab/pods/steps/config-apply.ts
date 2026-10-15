import { z, defineStep } from "strut";
import { failure, settle, submit, type PodCtx } from "./_shared.js";
import { FILES, readStored } from "./_pool.js";

export default defineStep({
  type: "pod/config-apply",
  description:
    "Make a claimed pod run its workspace's POOL CONFIG as hive stores it now: hive's pm2.config.js, docker-compose.yml and Dockerfile are written " +
    "into the pod, the compose services are brought up, and the apps reload (install, build, start) — in this pod, without waiting for the pool to " +
    "build a new one. Use it after pod/config-set to run the change in the pod you hold (that is how a first config is checked: pod/config-set " +
    "{ wait: false }, then this, then look at the app), or to put back the pool config after trying things in the pod. It is not the fresh-pod " +
    "check: whatever was made by hand in this pod (a bucket, a table) is still here. A changed Dockerfile is written but reaches a pod only when the " +
    "pod is created (`rebuildNeeded`): for that, pod/config-set { wait: true }, then pod/claim { fresh: true }. The files never reach this step's " +
    "output (they carry the env values). Waits for the reload (default 30 min). Never fails on a compose or app failure — read `success`, " +
    "`compose`, `reload` and `processes`. " +
    "Output: { workspace, applied, written, unchanged, success, compose: { up, project, services, skipped?, error? }, reload: { ok, error? }, " +
    "rebuildNeeded, processes: [{ name, status }], request_id }.",
  input: z.object({
    workspace: z.string().min(1).describe("The hive workspace, by its slug (no `@`) — the one given to pod/claim"),
    control: z.string().min(1).describe("The pod's control URL (pod/claim)"),
    sealed: z.string().min(1).describe("The pod's sealed password (pod/claim)"),
    pollMs: z.number().int().positive().default(5000),
    timeoutMs: z.number().int().positive().default(1_800_000),
  }),
  output: z.object({
    workspace: z.string(),
    applied: z.array(z.enum(FILES)).describe("hive's files sent to the pod"),
    written: z.array(z.string()).describe("files whose content changed on the pod"),
    unchanged: z.array(z.string()),
    success: z.boolean().describe("the compose services and the reload both went through; whether each app runs is `processes`"),
    compose: z.object({
      up: z.boolean(),
      project: z.string().nullable(),
      services: z.array(z.string()),
      skipped: z.string().optional(),
      error: z.string().optional(),
    }),
    reload: z.object({ ok: z.boolean(), error: z.string().optional() }),
    rebuildNeeded: z.boolean().describe("the Dockerfile changed: this pod still runs the old base image"),
    processes: z.array(z.object({ name: z.string(), status: z.string() })),
    request_id: z.string(),
  }),
  async run(cfg, ctx: PodCtx) {
    // Read in code and sent on: the stored files carry the env values in
    // clear, so they go from hive to the pod and never into the output.
    const stored = await readStored(ctx, cfg.workspace);
    const files = FILES.flatMap((name) => (stored.files[name] ? [{ name, content: stored.files[name]! }] : []));
    if (!files.some((f) => f.name === "pm2.config.js")) {
      throw new Error(`pod/config-apply: hive stores no pm2.config.js for workspace "${cfg.workspace}" — nothing to run; pod/config shows what it has`);
    }
    let first;
    try {
      first = await submit(ctx, cfg.control, cfg.sealed, "/pod-config", "PUT", { files });
    } catch (err) {
      if (err instanceof Error && /^PUT \/pod-config: 404\b/.test(err.message)) {
        throw new Error(
          "PUT /pod-config: 404 — this pod's staklink predates PUT /pod-config (stakwork/staklink#146); the config reaches it only when the pod is " +
            "recreated: pod/config-set { wait: true }, then pod/claim { fresh: true }",
        );
      }
      throw err;
    }
    const p = await settle(ctx, cfg.control, cfg.sealed, first, {
      pollMs: cfg.pollMs,
      timeoutMs: cfg.timeoutMs,
      what: "pod/config-apply",
      stillRunning: "the apps are still installing or building; look at them with the pod's process list before applying again",
    });
    if (p.status === "failed") throw new Error(`pod/config-apply: the pod could not apply the config: ${failure(p.body)}`);
    const r: any = p.body?.result ?? {};
    const compose = r.compose ?? {};
    return {
      workspace: cfg.workspace,
      applied: files.map((f) => f.name),
      written: Array.isArray(r.written) ? r.written.map(String) : [],
      unchanged: Array.isArray(r.unchanged) ? r.unchanged.map(String) : [],
      success: r.success === true,
      compose: {
        up: compose.up === true,
        project: typeof compose.project === "string" ? compose.project : null,
        services: Array.isArray(compose.services) ? compose.services.map(String) : [],
        ...(typeof compose.skipped === "string" ? { skipped: compose.skipped } : {}),
        ...(typeof compose.error === "string" ? { error: compose.error } : {}),
      },
      reload: { ok: r.reload?.ok === true, ...(typeof r.reload?.error === "string" ? { error: r.reload.error } : {}) },
      rebuildNeeded: r.rebuildNeeded === true,
      processes: Array.isArray(r.processes)
        ? r.processes.map((x: any) => ({ name: String(x?.name ?? ""), status: String(x?.status ?? "unknown") }))
        : [],
      request_id: first.requestId,
    };
  },
});
