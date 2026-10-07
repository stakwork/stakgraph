import { z, defineStep } from "strut";
import { agentOutcome, progress, type PodCtx } from "./_shared.js";

export default defineStep({
  type: "pod/agent-status",
  description:
    "How a task started with pod/agent-start is doing: one look, no waiting. `status` is pending, completed or failed; when completed, `output` is what the " +
    "agent said (long outputs are cut in the middle) and `summary` its own summary when it wrote one; when failed, `error` says why. Never fails on a " +
    "failed task — read `status`.",
  input: z.object({
    control: z.string().min(1).describe("The pod's control URL (pod/claim)"),
    sealed: z.string().min(1).describe("The pod's sealed password (pod/claim)"),
    request_id: z.string().min(1).describe("From pod/agent-start"),
    outputChars: z.number().int().positive().default(20_000).describe("Cap on `output`"),
  }),
  output: z.object({
    status: z.enum(["pending", "completed", "failed"]),
    output: z.string().nullable(),
    summary: z.string().nullable(),
    usage: z.any().nullable(),
    model: z.string().nullable(),
    error: z.string().nullable(),
  }),
  async run(cfg, ctx: PodCtx) {
    return agentOutcome(await progress(ctx, cfg.control, cfg.sealed, cfg.request_id), cfg.outputChars);
  },
});
