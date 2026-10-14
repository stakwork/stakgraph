import { z, defineStep } from "strut";
import type { PodCtx } from "./_shared.js";
import { applyEdits, checkEnvValues, poolName, readStored, writeStored, type Service } from "./_pool.js";

const ServiceInput = z.object({
  port: z.union([z.number().int().positive(), z.string().min(1)]).describe("The port it listens on: its PORT"),
  scripts: z
    .object({
      start: z.string().min(1).describe("What starts it (the pm2 app's script): a development command, with hot reload where the app has one"),
      install: z.string().optional().describe("INSTALL_COMMAND"),
      build: z.string().optional().describe("BUILD_COMMAND"),
      test: z.string().optional().describe("TEST_COMMAND"),
      e2eTest: z.string().optional().describe("E2E_TEST_COMMAND"),
      preStart: z.string().optional().describe("PRE_START_COMMAND: run before the apps start, every start"),
      postStart: z.string().optional().describe("POST_START_COMMAND: run after they start, every start — state a fresh pod needs (a bucket, seed data) goes here"),
      rebuild: z.string().optional().describe("REBUILD_COMMAND"),
      reset: z.string().optional().describe("RESET_COMMAND"),
    })
    .describe("Its commands; hive turns each into the pm2 app's env var named beside it"),
  env: z
    .record(z.string(), z.string())
    .optional()
    .describe("Its own env vars, replacing the stored ones; leave out to keep them. A masked value as pod/config shows it keeps that stored value"),
  cwd: z.string().optional().describe("Where it runs: a repository name, or a path in one (`app/web`)"),
  interpreter: z.string().optional(),
  language: z.string().optional(),
  dev: z.boolean().optional(),
  advanced: z.record(z.string(), z.union([z.string(), z.number(), z.boolean()])).optional().describe("Any other pm2 field of the app"),
});

export default defineStep({
  type: "pod/config-set",
  description:
    "Change a workspace's POOL CONFIG in hive — what every new pod is built from — and wait for the pool to recreate its pods. Outward-facing: every " +
    "UNUSED pod of the workspace is recreated (pods in use are recreated when released), everyone's; ask the person first. Only what changes, by name: " +
    "`services` and `env` are maps — a name set to a value is written (a service as a whole), `null` removes it, a name left out stays as stored; " +
    "`containerFiles` takes `docker-compose.yml` (or `.yaml`, stored as `.yml`) and `Dockerfile` as whole texts. pm2.config.js is generated from " +
    "`services` and never written; devcontainer.json is never written. `env` is the workspace's, every service's. Masked values (`ab******yz`, as " +
    "pod/config shows them) keep the stored value; a tagged one (`…#3fa9c1`, from a file view) is refused. Hive's stored config is read right before " +
    "the write, so an edit someone made since pod/config survives. Then it waits (default 20 min) until the change — or a later one, whose push carries " +
    "this one — is ready: a pod created after the push is running, for pod/claim { fresh: true }. Output: { workspace, changeId, state: ready | " +
    "rolling (wait: false) | unchanged (nothing to push), settledBy, written, removed, rollout }.",
  input: z.object({
    workspace: z.string().min(1).describe("The hive workspace, by its slug (no `@`) — the one given to pod/claim"),
    services: z.record(z.string(), ServiceInput.nullable()).optional().describe("Service edits by name: the whole service, or null to remove it"),
    env: z.record(z.string(), z.string().nullable()).optional().describe("Workspace-wide env var edits by name: the value, or null to remove it"),
    containerFiles: z.record(z.string(), z.string().min(1)).optional().describe("Whole files by name: docker-compose.yml (or .yaml) and Dockerfile"),
    wait: z.boolean().default(true).describe("Wait until a pod built from the change is ready; false returns once hive has pushed it"),
    pollMs: z.number().int().positive().default(10_000),
    timeoutMs: z.number().int().positive().default(1_200_000),
  }),
  output: z.object({
    workspace: z.string(),
    changeId: z.string().nullable(),
    state: z.enum(["ready", "rolling", "unchanged"]),
    settledBy: z.string().nullable(),
    written: z.object({ services: z.array(z.string()), env: z.array(z.string()), containerFiles: z.array(z.string()) }),
    removed: z.object({ services: z.array(z.string()), env: z.array(z.string()) }),
    rollout: z.any().nullable(),
  }),
  async run(cfg, ctx: PodCtx) {
    const serviceEdits = cfg.services ?? {};
    const envEdits = cfg.env ?? {};
    const files: Record<string, string> = {};
    for (const [name, text] of Object.entries(cfg.containerFiles ?? {})) {
      const n = poolName(name);
      if (n === "pm2.config.js") throw new Error("pod/config-set: pm2.config.js is generated from `services` — write the change as `services` edits");
      if (!n) throw new Error(`pod/config-set: "${name}" is not a file this writes — docker-compose.yml (or .yaml) and Dockerfile only; devcontainer.json is never edited`);
      if (files[n]) throw new Error("pod/config-set: two compose files given — the compose file is one file, docker-compose.yml");
      files[n] = Buffer.from(text, "utf-8").toString("base64");
    }
    const editsLists = Object.keys(serviceEdits).length > 0 || Object.keys(envEdits).length > 0;
    if (!editsLists && !Object.keys(files).length) throw new Error("pod/config-set: nothing to change — give `services`, `env` or `containerFiles`");

    // Read right before writing: hive replaces the services and env lists
    // whole, so the lists sent are what is stored NOW with these edits.
    const stored = await readStored(ctx, cfg.workspace);
    const body: Record<string, unknown> = {};
    const written = { services: [] as string[], env: [] as string[], containerFiles: Object.keys(files) };
    const removed = { services: [] as string[], env: [] as string[] };
    if (editsLists) {
      const storedEnv = Object.fromEntries(stored.env.map((e) => [String(e.name), String(e.value)]));
      const setEnv = Object.fromEntries(Object.entries(envEdits).filter((kv): kv is [string, string] => kv[1] !== null));
      checkEnvValues(setEnv, storedEnv, "env");
      for (const [name, svc] of Object.entries(serviceEdits)) {
        if (svc?.env) checkEnvValues(svc.env, stored.services.find((s) => s.name === name)?.env ?? {}, `services.${name}.env`);
      }
      const services = applyEdits<Service>(stored.services, serviceEdits as Record<string, Service | null>, "service");
      const env = applyEdits(
        stored.env.map((e) => ({ name: String(e.name), value: String(e.value) })),
        Object.fromEntries(Object.entries(envEdits).map(([name, value]) => [name, value === null ? null : { name, value }])),
        "env var",
      );
      // Both lists, whichever was edited: hive regenerates pm2.config.js from
      // them in the write, so the stored file and the push agree — sent alone,
      // an env change would be overridden by the old file's values at push.
      body["services"] = services.list;
      body["environmentVariables"] = env.list;
      written.services = services.written;
      written.env = env.written;
      removed.services = services.removed;
      removed.env = env.removed;
    }
    if (Object.keys(files).length) body["containerFiles"] = files;

    const change = await writeStored(ctx, cfg.workspace, body);
    const result = { workspace: cfg.workspace, changeId: change.changeId, settledBy: null as string | null, written, removed };
    if (!change.changeId) return { ...result, state: "unchanged" as const, rollout: null };
    if (change.state === "failed") {
      throw new Error(
        `pod/config-set: hive saved change ${change.changeId} but its push to the pool failed (${change.error ?? "no reason given"}): no pod was recreated. ` +
          "The change is in hive's config; tell the person, who can retry it from hive.",
      );
    }
    if (!cfg.wait) return { ...result, state: "rolling" as const, rollout: null };

    const deadline = Date.now() + cfg.timeoutMs;
    let rollout: any = null;
    for (;;) {
      if (Date.now() >= deadline) {
        throw new Error(
          `pod/config-set: change ${change.changeId} was not ready after ${Math.round(cfg.timeoutMs / 60_000)} min (${rollout?.state ?? "rolling"}) — ` +
            `pod/config { workspace: "${cfg.workspace}" } shows its rollout; claim a fresh pod once it is ready`,
        );
      }
      await new Promise((r) => setTimeout(r, cfg.pollMs));
      await ctx.control?.checkpoint();
      rollout = (await readStored(ctx, cfg.workspace)).rollout;
      const id = typeof rollout?.changeId === "string" ? rollout.changeId : null;
      if (!id) throw new Error(`pod/config-set: hive no longer records change ${change.changeId} (its rollout reads ${JSON.stringify(rollout?.state ?? null)})`);
      // A later change replaced ours as the one hive tracks; its push sent
      // hive's whole stored config, this change included, so its rollout is ours.
      const which = id === change.changeId ? `change ${id}` : `change ${id} (later than ${change.changeId}, and carrying it)`;
      if (rollout.state === "ready") return { ...result, state: "ready" as const, settledBy: id === change.changeId ? null : id, rollout };
      if (rollout.state === "failed") throw new Error(`pod/config-set: the rollout of ${which} failed — ${rollout.error ?? "no reason given"}`);
    }
  },
});
