import { z, defineStep } from "vein";

/**
 * 58313's guard_first_run: the first EvalTrigger ever linked to an EvalSet
 * hangs off HAS_BASELINE_TRIGGER; every later one off HAS_TRIGGER. Input is
 * graph/graph-neighbors' output for the EvalSet (edge_type filtered to the
 * two trigger edges). This run's own trigger is excluded by ref_id in case
 * it is already linked (a retried write). An unreadable hop (error string)
 * fails toward HAS_TRIGGER — never claim a baseline on no evidence.
 */
export default defineStep({
  type: "wfbench/trigger-edge",
  description:
    "Decide the EvalSet→EvalTrigger edge: HAS_BASELINE_TRIGGER when the EvalSet has no prior EvalTrigger neighbors (excluding this run's trigger_ref_id), else HAS_TRIGGER; an unreadable hop yields HAS_TRIGGER. Output: { edge_type, is_baseline, prior_triggers, readable }.",
  input: z.object({
    neighbors: z.any().describe("graph/graph-neighbors output for the EvalSet (array, or an error string)."),
    trigger_ref_id: z.string().optional().describe("This run's EvalTrigger ref_id, excluded from the prior count."),
  }),
  output: z.any(),
  async run(cfg) {
    const readable = Array.isArray(cfg.neighbors);
    const prior = readable
      ? (cfg.neighbors as any[]).filter(
          (n) => n && n.node_type === "EvalTrigger" && (!cfg.trigger_ref_id || n.ref_id !== cfg.trigger_ref_id),
        ).length
      : 0;
    const is_baseline = readable && prior === 0;
    return {
      edge_type: is_baseline ? "HAS_BASELINE_TRIGGER" : "HAS_TRIGGER",
      is_baseline,
      prior_triggers: prior,
      readable,
    };
  },
});
