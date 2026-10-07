import { z, defineStep } from "strut";
import { agentBody, submit, type PodCtx } from "./_shared.js";

export default defineStep({
  type: "pod/agent-start",
  description:
    "Start the pod's coding agent on a task, in the pod's repositories, and return at once with a `request_id`: check on it with pod/agent-status, " +
    "or wait for it with pod/agent instead of this. The agent's session is this run's job unless `session` is given, so on a job it REMEMBERS what it " +
    "did in earlier turns; the prompt must still stand alone — name the repository, the files, the change and what to check. Its model calls are " +
    "billed to this run through the LLM gateway. Start-and-end-your-turn suits a long task the person will watch in the pod's frontend; pod/agent suits " +
    "one you will wait for. Output: { request_id, session, status }.",
  input: z.object({
    control: z.string().min(1).describe("The pod's control URL (pod/claim)"),
    sealed: z.string().min(1).describe("The pod's sealed password (pod/claim)"),
    prompt: z.string().min(1).describe("The task, standing alone"),
    system: z.string().optional().describe("A system prompt for the agent"),
    repoName: z.string().optional().describe("The repository to work in (its name, e.g. `hive`) when the pod has several"),
    model: z.string().optional().describe("A model, in aieo's form (`sonnet`, `anthropic/claude-sonnet-5-5`); the pod's default when omitted"),
    session: z.string().optional().describe("The agent's session id; this run's job when omitted"),
  }),
  output: z.object({
    request_id: z.string(),
    session: z.string().nullable(),
    status: z.enum(["pending", "completed", "failed"]),
  }),
  async run(cfg, ctx: PodCtx) {
    const { body, session } = await agentBody(ctx, cfg);
    const first = await submit(ctx, cfg.control, cfg.sealed, "/agent", "POST", body);
    return { request_id: first.requestId, session, status: first.status };
  },
});
