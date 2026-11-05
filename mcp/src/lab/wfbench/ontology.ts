/**
 * The jarvis types wfbench writes, as jarvis declares them: each type's
 * attributes and the edges between them. An excerpt of the jarvis schema
 * library strut used to bundle (`fixtures/jarvis-ontology.ts`, version
 * m125-code-description); strut no longer carries jarvis's library
 * (stakwork/strut plans/ontology.md), so the smoke checks its graph
 * payloads against this. Refresh it from jarvis-backend's schema library
 * when wfbench writes a new attribute or edge.
 */
export const WFBENCH_ONTOLOGY: { types: Readonly<Record<string, readonly string[]>>; edges: ReadonlyArray<readonly [string, string, string]> } = {
  types: {
    EvalSet: ["created_at", "description", "id", "name", "project_id", "recursion", "updated_at"],
    EvalRequirement: ["contested", "created_at", "deliverables", "description", "id", "name", "negative_cases", "positive_cases", "prompt_snippet", "updated_at"],
    EvalTrigger: ["agent", "body", "change_type", "end_point", "endpoint_url", "environment", "id", "negative_cases", "positive_cases", "project_id", "prompt_id", "prompt_version_id", "prompts", "run_count", "source", "start_point", "workflow_id", "workflow_input", "workflow_version_id"],
    EvalTriggerOutput: ["attempt_number", "endpoint_url", "id", "judge_model", "judge_notes", "max_score", "n_passed", "n_total", "report_url", "result", "score", "verdict"],
    CriterionResult: ["agent_thinking_excerpt", "contested", "criterion_id", "document_excerpt", "flagged", "id", "llm_flag_reason", "reasoning", "title", "verdict"],
  },
  edges: [
    ["EvalRequirement", "HAS_CRITERION_RESULT", "CriterionResult"],
    ["EvalRequirement", "HAS_TRIGGER", "EvalTrigger"],
    ["EvalSet", "HAS_BASELINE_TRIGGER", "EvalTrigger"],
    ["EvalSet", "HAS_REQUIREMENT", "EvalRequirement"],
    ["EvalSet", "HAS_SUBSET", "EvalSet"],
    ["EvalSet", "HAS_TRIGGER", "EvalTrigger"],
    ["EvalTrigger", "HAS_OUTPUT", "EvalTriggerOutput"],
    ["EvalTriggerOutput", "HAS_CRITERION_RESULT", "CriterionResult"],
  ],
};
