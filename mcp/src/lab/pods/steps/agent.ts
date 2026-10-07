import { z, defineStep } from "strut";
import { agentBody, agentOutcome, settle, submit, type PodCtx } from "./_shared.js";

export default defineStep({
  type: "pod/agent",
  description:
    "Run the pod's coding agent on a task and WAIT for it (default up to 45 min; a cancelled run stops waiting). pod/agent-start + pod/agent-status, " +
    "in one step, for a task you will wait for; a failed task fails the step. The session is this run's job unless `session` is given, so on a job the " +
    "agent remembers earlier turns; the prompt must still stand alone. Output: { request_id, session, status: completed, output, summary, usage, model }.",
  input: z.object({
    control: z.string().min(1).describe("The pod's control URL (pod/claim)"),
    sealed: z.string().min(1).describe("The pod's sealed password (pod/claim)"),
    prompt: z.string().min(1).describe("The task, standing alone"),
    system: z.string().optional().describe("A system prompt for the agent"),
    repoName: z.string().optional().describe("The repository to work in (its name) when the pod has several"),
    model: z.string().optional().describe("A model, in aieo's form; the pod's default when omitted"),
    session: z.string().optional().describe("The agent's session id; this run's job when omitted"),
    outputChars: z.number().int().positive().default(20_000).describe("Cap on `output`"),
    pollMs: z.number().int().positive().default(5000),
    timeoutMs: z.number().int().positive().default(2_700_000),
  }),
  output: z.object({
    request_id: z.string(),
    session: z.string().nullable(),
    status: z.literal("completed"),
    output: z.string().nullable(),
    summary: z.string().nullable(),
    usage: z.any().nullable(),
    model: z.string().nullable(),
  }),
  async run(cfg, ctx: PodCtx) {
    const { body, session } = await agentBody(ctx, cfg);
    const first = await submit(ctx, cfg.control, cfg.sealed, "/agent", "POST", body);
    const p = await settle(ctx, cfg.control, cfg.sealed, first, { pollMs: cfg.pollMs, timeoutMs: cfg.timeoutMs, what: "pod/agent" });
    const out = agentOutcome(p, cfg.outputChars);
    if (p.status === "failed") throw new Error(`pod/agent: the pod's agent failed — ${out.error}`);
    return { request_id: first.requestId, session, status: "completed" as const, output: out.output, summary: out.summary, usage: out.usage, model: out.model };
  },
});
