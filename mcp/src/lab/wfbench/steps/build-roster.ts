import { z, defineStep, type StepContext } from "vein";

/**
 * The graph ROSTER for one benchmark run (stakwork 58313 steps 2–6), as
 * write-ready payloads — the writes themselves are graph/create-node and
 * graph/create-batch-triplet. Node ids follow 58313 exactly:
 *
 *   EvalSet         id = task_slug                        (55741)
 *   EvalRequirement id = <task_slug>-<criterion_id>       (58114, one per criterion,
 *                                                          EvalSet -HAS_REQUIREMENT-> it)
 *   EvalTrigger     id = <task_slug>-<project_id>         (55741; project_id = this runId)
 *
 * Properties are limited to what the jarvis ontology declares for each
 * type (vein's graph backend rejects undeclared attributes): 58313's
 * EvalSet.project_id (an int there) and EvalTrigger.name are therefore
 * omitted; the trigger's `agent` carries the harness name so the node has
 * a title. The EvalSet→EvalTrigger edge (HAS_TRIGGER vs
 * HAS_BASELINE_TRIGGER) is decided later by wfbench/trigger-edge.
 */
export default defineStep({
  type: "wfbench/build-roster",
  description:
    "Build the eval roster payloads for one run with 58313's id conventions: EvalSet { id: task_slug }, EvalRequirement { id: <slug>-<criterion_id> } triplets (HAS_REQUIREMENT, edge order), EvalTrigger { id: <slug>-<runId>, workflow_id, workflow_version_id, workflow_input, project_id }. Output: { run_id, evalset_id, trigger_id, evalset: { node_type, node_data }, requirement_triplets, requirement_ids, trigger: { node_type, node_data } }.",
  input: z.object({
    task_slug: z.string().min(1),
    task_title: z.string(),
    instructions: z.string(),
    criteria: z.array(z.any()).describe("wfbench/normalize-task's criteria."),
    workflow: z.string().describe("The harness workflow name (recorded as EvalTrigger.workflow_id / agent)."),
    workflow_version: z.any().optional().describe("The harness workflow's version (EvalTrigger.workflow_version_id)."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const runId = (ctx as StepContext)?.runId || "no-run";
    const slug = cfg.task_slug;
    const evalset = { node_type: "EvalSet", node_data: { id: slug, name: cfg.task_title } };

    const requirement_triplets = cfg.criteria.map((c: any, i: number) => ({
      source_type: "EvalSet",
      source_data: { id: slug },
      target_type: "EvalRequirement",
      target_data: {
        id: `${slug}-${c.id}`,
        name: String(c.title ?? ""),
        description: String(c.match_criteria ?? ""),
        deliverables: Array.isArray(c.deliverables) ? c.deliverables : [],
      },
      edge_type: "HAS_REQUIREMENT",
      edge_data: { order: i },
    }));

    const trigger_id = `${slug}-${runId}`;
    const trigger = {
      node_type: "EvalTrigger",
      node_data: {
        id: trigger_id,
        agent: cfg.workflow,
        source: "vein",
        environment: "vein-lab",
        workflow_id: cfg.workflow,
        workflow_version_id: cfg.workflow_version == null ? "" : String(cfg.workflow_version),
        workflow_input: cfg.instructions,
        project_id: runId,
        run_count: 1,
      },
    };

    return {
      run_id: runId,
      evalset_id: slug,
      trigger_id,
      evalset,
      requirement_triplets,
      requirement_ids: requirement_triplets.map((t) => t.target_data.id),
      trigger,
    };
  },
});
