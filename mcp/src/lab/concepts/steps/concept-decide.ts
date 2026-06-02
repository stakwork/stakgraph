import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";
import type { Concept } from "../types.js";

/**
 * Ask the LLM which concept(s) a change belongs to.
 *
 * Pure mechanism: this step assembles the prompt (system + concept/theme
 * context + change markdown + guidelines) and calls the LLM. The prompt text
 * itself — `systemPrompt` and `guidelines` — is NOT baked in here; it's the
 * experiment surface and lives in the calling workflow's `params` block
 * (see `workflows/process-change.yaml`), passed in via config. Sweep prompt
 * variants by overriding `params` per run; promote a winner by editing the
 * workflow's `params` default. Output: { decision, usage }.
 */

function formatConceptContext(concepts: Concept[]): string {
  if (concepts.length === 0) return "## Current Concepts\n\nNo concepts yet.";
  const list = concepts
    .slice()
    .sort((a, b) => b.lastUpdated.getTime() - a.lastUpdated.getTime())
    .map((f) => {
      const prCount = f.prNumbers.length;
      const commitCount = (f.commitShas || []).length;
      const summary =
        commitCount > 0 ? `[${prCount} PRs, ${commitCount} commits]` : `[${prCount} PRs]`;
      return `- **${f.name}** (\`${f.id}\`): ${f.description} ${summary}`;
    })
    .join("\n");
  return `## Current Concepts\n\n${list}`;
}

function formatThemeContext(themes: string[]): string {
  if (themes.length === 0) return "## Recent Technical Themes\n\nNo recent themes.";
  const list = themes.slice().reverse().slice(0, 100).join(", ");
  return `## Recent Technical Themes (last 100 of ${themes.length})\n\n${list}`;
}

function emptyUsage() {
  return {
    input: 0,
    cache_read: 0,
    cache_write: 0,
    output: 0,
    total: 0,
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
  };
}

export default defineStep({
  type: "concepts/decide",
  description: "Ask the LLM how a change maps to concepts (add/create/update/ignore). Required config: systemPrompt, guidelines — supplied by the calling workflow's `params` block (the prompt experiment surface). Output: { decision, usage }.",
  input: z.object({
    change: z.any(),
    markdown: z.string().nullable().optional(),
    skipped: z.boolean().optional(),
    owner: z.string(),
    repo: z.string(),
    systemPrompt: z.string(),
    guidelines: z.string(),
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
        usage: emptyUsage(),
        skipped: true,
      };
    }

    const repoId = `${cfg.owner}/${cfg.repo}`;
    const concepts = await storage.getAllConcepts(repoId);
    const themes = await storage.getRecentThemes(repoId);
    const system = cfg.systemPrompt;
    const guidelines = cfg.guidelines;

    const prompt = `${system}

${formatConceptContext(concepts)}

${formatThemeContext(themes)}

${cfg.markdown}

${guidelines}`;

    const change = cfg.change;
    const label =
      change.type === "pr"
        ? `concept decision: PR #${change.data.number}`
        : `concept decision: commit ${String(change.data.sha).substring(0, 7)}`;

    const { decision, usage } = await llm.decide(prompt, undefined, undefined, label);
    return { decision, usage };
  },
});
