import { z, defineStep } from "strut";
import { capText, type PodCtx } from "./_shared.js";
import { FILES, mask, maskPm2, readPod, readStored, tagger, unifiedDiff, type FileName } from "./_pool.js";

const File = z.object({
  name: z.enum(FILES),
  hive: z.string().nullable(),
  pod: z.string().nullable(),
  path: z.string().nullable(),
  status: z.enum(["same", "changed", "hive only", "pod only"]).nullable(),
  diff: z.string().nullable(),
});

export default defineStep({
  type: "pod/config",
  description:
    "Read a workspace's POOL CONFIG — what hive builds every new pod from — and, given a claimed pod (`control` + `sealed`), the files that pod runs from, " +
    "with a diff per file (hive → pod): what a fix made in the pod would change for every pod. The config is `services` (pm2.config.js is generated from " +
    "them: a service's `port` and `scripts` become its PORT and *_COMMAND env vars), the workspace-wide `env`, `docker-compose.yml` and `Dockerfile`; " +
    "devcontainer.json is left out. Env values are masked: `env` and each service's `env` as hive masks them (`ab******yz` — write them back as they " +
    "are, or leave them out, and the stored value is kept), and in the pm2 files with a tag (`ab******yz#3fa9c1`) that is equal for equal values within " +
    "one read, so a changed value shows in the diff — such a tagged value is never a value to write. A pod older than the last push also shows that push, " +
    "reversed, in its diff. `rollout` is the last config change's state (pod/config-set). " +
    "Output: { workspace, services, env, files: [{ name, hive, pod, path, status, diff }], rollout }.",
  input: z.object({
    workspace: z.string().min(1).describe("The hive workspace, by its slug (no `@`) — the one given to pod/claim"),
    control: z.string().min(1).optional().describe("A claimed pod's control URL (pod/claim), to diff its files against hive's"),
    sealed: z.string().min(1).optional().describe("That pod's sealed password (pod/claim)"),
    maxChars: z.number().int().positive().default(12_000).describe("Cap on each file's text and each diff"),
  }),
  output: z.object({
    workspace: z.string(),
    services: z.array(z.any()),
    env: z.record(z.string(), z.string()),
    files: z.array(File),
    rollout: z.any().nullable(),
  }),
  async run(cfg, ctx: PodCtx) {
    if (!cfg.control !== !cfg.sealed) throw new Error("pod/config: give both `control` and `sealed` (a claimed pod), or neither (hive's config alone)");
    const stored = await readStored(ctx, cfg.workspace);
    const pod = cfg.control && cfg.sealed ? await readPod(ctx, cfg.control, cfg.sealed) : null;
    // One tagger for both sides of every diff: equal values, equal tags.
    const tag = tagger();
    const view = (name: FileName, text: string | undefined) =>
      text === undefined ? null : name === "pm2.config.js" ? maskPm2(text, tag) : text;
    const files = FILES.map((name) => {
      const hive = view(name, stored.files[name]);
      const onPod = pod?.find((f) => f.name === name);
      const mine = pod ? view(name, onPod?.content) : null;
      const status = !pod
        ? null
        : hive === null && mine === null
          ? null
          : hive === null
            ? ("pod only" as const)
            : mine === null
              ? ("hive only" as const)
              : hive === mine
                ? ("same" as const)
                : ("changed" as const);
      const diff = pod && (hive !== null || mine !== null) ? unifiedDiff(hive ?? "", mine ?? "", `hive/${name}`, `pod/${name}`) : null;
      return {
        name,
        hive: hive === null ? null : capText(hive, cfg.maxChars),
        pod: mine === null ? null : capText(mine, cfg.maxChars),
        path: onPod?.path ?? null,
        status,
        diff: diff === null ? null : capText(diff, cfg.maxChars),
      };
    });
    // Hive masks these for a key caller already; masking again is a no-op on
    // its masks and keeps a value in clear from ever leaving the step.
    const services = stored.services.map((s) =>
      s.env ? { ...s, env: Object.fromEntries(Object.entries(s.env).map(([k, v]) => [k, mask(String(v))])) } : s,
    );
    const env = Object.fromEntries(stored.env.map((e) => [String(e.name), mask(String(e.value))]));
    return { workspace: cfg.workspace, services, env, files, rollout: stored.rollout };
  },
});
