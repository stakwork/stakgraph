import { z, defineStep } from "vein";

/**
 * 58313's wfbench_classify_run_result.py: three-way classification of the
 * produced workflow's launch (meta/run-workflow's { runId, status, output,
 * error } — or { error } when the meta surface refused, or nothing at all
 * when the launch gate skipped it):
 *   launch_ok + completed  → score it
 *   launch_ok + failed     → still score (partial credit: the artifact exists)
 *   no runId               → harness error, never score
 */
export default defineStep({
  type: "wfbench/classify-run",
  description:
    "Classify the produced workflow's run: { launch_ok, execution_status: completed|failed|none, project_id (the child runId), error_type, error, run_output }. No runId (skipped launch, refused, or crashed before starting) → launch_ok=false with the gate's error_type.",
  input: z.object({
    run: z.any().optional().describe("meta/run-workflow output (undefined when the launch was skipped)."),
    gate_error_type: z.any().optional().describe("Why the launch was skipped (check-input-keys' error_type)."),
    gate_error: z.any().optional(),
  }),
  output: z.any(),
  async run(cfg) {
    const r = cfg.run && typeof cfg.run === "object" ? cfg.run : null;
    const runId = typeof r?.runId === "string" && r.runId ? r.runId : null;
    if (!runId) {
      const refused = typeof r?.error === "string" ? r.error : typeof cfg.gate_error === "string" ? cfg.gate_error : null;
      return {
        launch_ok: false,
        execution_status: "none",
        project_id: null,
        error_type: typeof r?.error === "string" ? "launch_refused" : (cfg.gate_error_type ?? "not_launched"),
        error: refused ?? "the produced workflow was not launched",
        run_output: null,
      };
    }
    const completed = r.status === "success";
    const err = r.error && typeof r.error === "object" ? r.error.message : r.error;
    return {
      launch_ok: true,
      execution_status: completed ? "completed" : "failed",
      project_id: runId,
      error_type: completed ? null : "produced_workflow_failed",
      error: completed ? null : (typeof err === "string" ? err : `run status ${String(r.status)}`),
      run_output: r.output ?? null,
    };
  },
});
