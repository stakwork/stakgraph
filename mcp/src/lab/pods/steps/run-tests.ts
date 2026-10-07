import { z, defineStep } from "strut";
import { capText, failure, settle, submit, type PodCtx } from "./_shared.js";

export default defineStep({
  type: "pod/run-tests",
  description:
    "Run the pod's configured test commands (one per app) and report per app whether they passed, with the tail of each output; the full logs go to the " +
    "run's artifacts (`log_path`). Never fails on red tests — read `success`. `ran: 0` means no app on the pod has a test command. Waits (default 1 h).",
  input: z.object({
    control: z.string().min(1).describe("The pod's control URL (pod/claim)"),
    sealed: z.string().min(1).describe("The pod's sealed password (pod/claim)"),
    label: z.string().default("run").describe("Prefix for the log files, e.g. baseline / after"),
    tailChars: z.number().int().positive().default(4000),
    pollMs: z.number().int().positive().default(5000),
    timeoutMs: z.number().int().positive().default(3_600_000),
  }),
  output: z.object({
    success: z.boolean(),
    ran: z.number(),
    apps: z.array(z.object({ name: z.string(), ok: z.boolean(), error: z.string().nullable(), stdout_tail: z.string(), stderr_tail: z.string(), log_path: z.string().nullable() })),
    request_id: z.string(),
    error: z.string().nullable(),
  }),
  async run(cfg, ctx: PodCtx) {
    const first = await submit(ctx, cfg.control, cfg.sealed, "/test", "PUT");
    const p = await settle(ctx, cfg.control, cfg.sealed, first, { pollMs: cfg.pollMs, timeoutMs: cfg.timeoutMs, what: "pod/run-tests" });
    if (p.status === "failed") {
      return { success: false, ran: 0, apps: [], request_id: first.requestId, error: `the pod could not run its tests: ${failure(p.body)}` };
    }
    const results: Record<string, any> = p.body?.result?.results ?? {};
    const tail = (s: unknown) => capText(String(s ?? ""), cfg.tailChars);
    const apps = [];
    for (const [name, r] of Object.entries(results)) {
      // A non-zero exit stores { stderr: "Command failed with exit code N\n…" } and no stdout key.
      const ok = r != null && Object.prototype.hasOwnProperty.call(r, "stdout");
      const rel = `pod-test/${cfg.label}-${name}.log`;
      let log_path: string | null = null;
      if (ctx.services?.artifacts) {
        await ctx.services.artifacts.write(ctx.runId, rel, `# ${name}\n## stdout\n${r?.stdout ?? ""}\n## stderr\n${r?.stderr ?? ""}\n`);
        log_path = rel;
      }
      apps.push({ name, ok, error: ok ? null : String(r?.stderr ?? "no output").split("\n")[0] ?? null, stdout_tail: tail(r?.stdout), stderr_tail: tail(r?.stderr), log_path });
    }
    const ran = apps.length;
    return { success: ran > 0 && apps.every((a) => a.ok), ran, apps, request_id: first.requestId, error: ran === 0 ? "no app on this pod has a test command — nothing ran" : null };
  },
});
