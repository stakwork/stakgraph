import { z, defineStep } from "vein";

/**
 * The harvey_lab_filter_contested_criteria equivalent: drop rubric criteria
 * whose EvalRequirement graph node carries `contested: true` (a durable,
 * cross-run "this criterion definition is bad — don't score it" signal,
 * distinct from the per-run `flagged` dispute of a fail verdict).
 *
 * FAILS OPEN by design: any surprise in the graph data (error strings from
 * the jarvis read steps, missing properties, shape drift) filters NOTHING —
 * a broken filter must never block scoring. Matching is by EvalRequirement
 * id ("<evalsetId>-<criterionId>", how merge_requirements writes them) with
 * a bare-criterion-id fallback.
 */
export default defineStep({
  type: "harvey/filter-contested",
  description:
    "Drop rubric criteria whose EvalRequirement node has contested=true. Pass the rubric plus the " +
    "requirement nodes (jarvis/graph-get-batched output). FAILS OPEN: on any data surprise the full " +
    "rubric is returned. Output: { rubric, dropped: [criterion ids], kept, total }.",
  input: z.object({
    rubric: z.array(z.any()).describe("Rubric criteria: [{ id, title, match_criteria, deliverables }]."),
    requirements: z
      .any()
      .optional()
      .describe("EvalRequirement nodes (jarvis/graph-get-batched output; any shape tolerated)."),
    evalsetId: z.string().optional().describe("EvalSet id — requirement ids are '<evalsetId>-<criterionId>'."),
  }),
  output: z.any(),
  async run(cfg) {
    const dropped: string[] = [];
    let rubric = cfg.rubric;
    try {
      const contested = new Set<string>();
      const nodes = Array.isArray(cfg.requirements) ? cfg.requirements : [];
      for (const n of nodes) {
        const props = (n as Record<string, any>)?.properties;
        if (props && typeof props === "object" && props.contested === true && typeof props.id === "string") {
          contested.add(props.id);
        }
      }
      if (contested.size > 0) {
        rubric = cfg.rubric.filter((c) => {
          const id = String((c as Record<string, any>)?.id ?? "");
          const hit = contested.has(id) || (cfg.evalsetId ? contested.has(`${cfg.evalsetId}-${id}`) : false);
          if (hit) dropped.push(id);
          return !hit;
        });
      }
    } catch {
      // fail open: never block scoring on filter trouble
      rubric = cfg.rubric;
      dropped.length = 0;
    }
    return { rubric, dropped, kept: rubric.length, total: cfg.rubric.length };
  },
});
