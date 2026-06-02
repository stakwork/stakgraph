import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";
import type { Concept } from "../types.js";

/**
 * Collect the repo's final Concept set into a single, scoreable artifact —
 * the **eval output**. The concepts pipeline writes Concepts to the graph
 * (Neo4j) rather than returning them, so without this step the run's recorded
 * output is the last pipeline step (link-files), not the concepts themselves.
 *
 * Placing this as the FINAL step of an eval target workflow means its output
 * automatically becomes the run output (recorded in run.json + rendered in the
 * UI via the `markdown` field) — so an eval just reads the run output and
 * scores `markdown` / `concepts` against the expected gold set. No out-of-band
 * graph query needed.
 *
 * Output: { count, concepts, markdown }.
 */

function activity(c: Concept): number {
  return (c.prNumbers?.length ?? 0) + (c.commitShas?.length ?? 0);
}

function renderMarkdown(repoId: string, concepts: Concept[]): string {
  const header = `# Concepts: ${repoId}\n\n${concepts.length} concept${concepts.length === 1 ? "" : "s"}.`;
  if (concepts.length === 0) return `${header}\n\n_No concepts._`;
  const body = concepts
    .map((c) => {
      const prs = c.prNumbers?.length ?? 0;
      const commits = c.commitShas?.length ?? 0;
      const meta =
        commits > 0 ? `_${prs} PRs, ${commits} commits_` : `_${prs} PRs_`;
      return `## ${c.name}\n\n${c.description}\n\n${meta}`;
    })
    .join("\n\n");
  return `${header}\n\n${body}`;
}

export default defineStep({
  type: "concepts/collect-for-eval",
  description:
    "Collect the repo's final Concept set from the graph into a single scoreable artifact (the eval output). Use as the LAST step of an eval target so its output becomes the recorded, UI-visible run output. Output: { count, concepts, markdown }.",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { storage } = ctx.services as ConceptServices;
    const repoId = `${cfg.owner}/${cfg.repo}`;

    const all = await storage.getAllConcepts(repoId);
    // Sort by activity (most-touched first), then name — a stable "top N" order.
    const concepts = all
      .slice()
      .sort((a, b) => activity(b) - activity(a) || a.name.localeCompare(b.name));

    return {
      count: concepts.length,
      concepts: concepts.map((c) => ({
        id: c.id,
        name: c.name,
        description: c.description,
        prNumbers: c.prNumbers ?? [],
        commitShas: c.commitShas ?? [],
      })),
      markdown: renderMarkdown(repoId, concepts),
    };
  },
});
