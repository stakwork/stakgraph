import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";
import { buildDecisionPrompt } from "../pipeline.js";
import { normalizeUsage } from "../../../aieo/src/usage.js";

/**
 * Ask the LLM which concept(s) a change belongs to. The system prompt and
 * decision guidelines are overridable via config — this is the swappable
 * "prompt experiment" seam. Output: { decision, usage }.
 */
export default defineStep({
  type: "concepts/decide",
  description: "Ask the LLM how a change maps to concepts (add/create/update/ignore). Optional config: systemPrompt, guidelines (override defaults). Output: { decision, usage }.",
  input: z.object({
    change: z.any(),
    markdown: z.string().nullable().optional(),
    skipped: z.boolean().optional(),
    owner: z.string(),
    repo: z.string(),
    systemPrompt: z.string().optional(),
    guidelines: z.string().optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { storage, llm } = ctx.services as ConceptServices;

    if (cfg.skipped || !cfg.markdown) {
      return {
        decision: {
          actions: ["ignore"],
          summary: "Skipped (maintenance/trivial)",
          reasoning: "Filtered by maintenance heuristic",
        },
        usage: normalizeUsage(),
        skipped: true,
      };
    }

    const repoId = `${cfg.owner}/${cfg.repo}`;
    const prompt = await buildDecisionPrompt(storage, repoId, cfg.markdown, {
      systemPrompt: cfg.systemPrompt,
      guidelines: cfg.guidelines,
    });

    const change = cfg.change;
    const label =
      change.type === "pr"
        ? `concept decision: PR #${change.data.number}`
        : `concept decision: commit ${String(change.data.sha).substring(0, 7)}`;

    const { decision, usage } = await llm.decide(prompt, undefined, undefined, label);
    return { decision, usage };
  },
});
