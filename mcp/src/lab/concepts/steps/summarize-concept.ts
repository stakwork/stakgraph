import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";

/**
 * (Re)generate documentation for the given concepts from their PR/commit
 * history. Output: { usage }.
 */
export default defineStep({
  type: "concepts/summarize",
  description: "Generate/update documentation for the given concept ids from their change history. Output: { usage }.",
  input: z.object({
    conceptIds: z.array(z.string()),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { summarizer } = ctx.services as ConceptServices;
    const usage = await summarizer.summarizeModifiedConcepts(cfg.conceptIds);
    return { usage };
  },
});
