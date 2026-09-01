import { z, defineStep, type StepContext } from "vein";

/**
 * The PINNED graph research sub-agent (the lab equivalent of the repo-agent's
 * `graph_sub_agent` tool). A thin wrapper over the vein-core `agent` step with
 * the child's configuration FIXED in code: read-only jarvis grants, a pinned
 * system frame, and no recursion (the child is never granted `agent` or this
 * step). The parent LLM supplies ONLY the research question — it cannot widen
 * the child's tool grants, swap its system prompt, or nest further sub-agents.
 *
 * Grant THIS step via `agentTools: ["harvey/graph-sub-agent"]` wherever a role
 * agent (ingestion, cross-check, drafter, …) should be able to delegate a
 * knowledge-graph research question without holding the jarvis read tools
 * itself.
 */
const READ_ONLY_JARVIS = [
  "jarvis/get-ontology",
  "jarvis/get-ontology-type",
  "jarvis/graph-search",
  "jarvis/graph-get",
  "jarvis/graph-get-batched",
  "jarvis/graph-neighbors",
];

const SYSTEM = `You are a knowledge-graph research sub-agent. Answer the research question
using ONLY the jarvis graph tools — you have no file or shell access and must
not invent graph contents. Method: search (jarvis_graph_search) or enter via
known roots, expand with jarvis_graph_neighbors, then read the relevant nodes
in full with jarvis_graph_get_batched. Prefer breadth first (list candidates),
then depth on the few that matter. Report findings with each node's ref_id so
the caller can cite or link them, and say clearly when the graph does NOT
contain something — an honest gap beats a guess.`;

export default defineStep({
  type: "harvey/graph-sub-agent",
  description:
    "Delegate a research question to a READ-ONLY knowledge-graph sub-agent. It searches and walks the " +
    "Jarvis graph (search → neighbors → batched reads) and returns a findings report with ref_ids. It " +
    "cannot write to the graph, read files, or run commands. Use it for questions like 'what checklist " +
    "concepts exist under the Practice Area roots?' or 'list every ComputedFigure ingested for this task'.",
  input: z.object({
    question: z
      .string()
      .min(1)
      .describe("The research question, with any known context (namespace, node names, ref_ids)."),
    namespace: z
      .string()
      .optional()
      .describe("Graph namespace the question concerns (task data partition). Concepts live in the default namespace."),
    model: z.string().optional().describe("Model override for the sub-agent (default claude-sonnet-5)."),
    maxSteps: z.number().int().positive().max(60).optional().describe("Tool-call budget (default 25)."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const registry = (ctx as StepContext & { registry?: Record<string, { input: z.ZodTypeAny; run: Function }> })
      .registry;
    const agentDef = registry?.["agent"];
    if (!agentDef) {
      throw new Error("harvey/graph-sub-agent: core `agent` step not found — requires the runner-populated ctx.registry");
    }

    const namespaceNote = cfg.namespace
      ? `\n\nTask data lives in graph namespace "${cfg.namespace}" (pass it as the \`namespace\` arg on graph reads scoped to this task). Shared Concept methodology lives in the DEFAULT namespace — omit \`namespace\` when reading Concepts.`
      : "";

    const childCfg = agentDef.input.parse({
      // The sub-agent never touches files; cwd is still required by the agent
      // step, so hand it the current process dir and grant no file tools.
      cwd: process.cwd(),
      system: SYSTEM + namespaceNote,
      prompt: cfg.question,
      model: cfg.model ?? "claude-sonnet-5",
      maxSteps: cfg.maxSteps ?? 25,
      // Non-empty on purpose: an empty toolFilter means ALL built-ins (bash,
      // file editing, web_search). repo_overview is the least-capable built-in
      // — effectively "no built-ins" while satisfying the subset contract.
      toolFilter: ["repo_overview"],
      agentTools: READ_ONLY_JARVIS,
      finalAnswer:
        "Your final research report: findings with ref_ids, organized by relevance, plus an explicit list of what the graph does NOT contain.",
    });

    return agentDef.run(childCfg, ctx);
  },
});
