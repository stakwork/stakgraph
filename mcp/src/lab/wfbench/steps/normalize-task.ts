import { z, defineStep } from "vein";

/**
 * Normalize one benchmark task (the Hive payload / stakwork 58313 set_var
 * shape) into the harness's canonical form. Fails LOUD on a malformed task
 * — bad criteria or an unparseable workflow_input_json is a harness error
 * before any agent budget is spent, exactly where 58313's set_var would
 * have choked.
 *
 * task_slug is kept VERBATIM: it is the EvalSet id, and Hive's roster upsert
 * + rubrics reader (eval-nodes.ts / fetchTaskRubricRoster) key the EvalSet on
 * the exact corpus slug, e.g. "wfbench/generate-capital-city" — slugifying it
 * would create a second roster Hive never finds. task_key is the slugified
 * form, used only where a name must be filesystem/URL safe (the candidate
 * workflow name).
 *
 * Output: { task_slug, task_key, task_title, instructions, criteria, n_criteria,
 *           workflow_input, workflow_input_keys, rerun_expected_output }
 *   criteria: [{ id, title, match_criteria, deliverables }] (rubric order kept)
 */
const slugify = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

const parseMaybeJson = (v: unknown): unknown => {
  if (typeof v !== "string") return v;
  const t = v.trim();
  if (!t) return undefined;
  if (!/^[\[{]/.test(t)) return v;
  try {
    return JSON.parse(t);
  } catch {
    return v;
  }
};

export default defineStep({
  type: "wfbench/normalize-task",
  description:
    "Normalize a benchmark task (task_slug, task_title, instructions, criteria as array or JSON string, workflow_input_json as object or JSON string, rerun_expected_output) into the harness shape. task_slug is kept verbatim (the EvalSet id Hive looks up); task_key is its slugified form for workflow names. Throws on empty/invalid criteria or a non-object workflow input. Output: { task_slug, task_key, task_title, instructions, criteria: [{ id, title, match_criteria, deliverables }], n_criteria, workflow_input, workflow_input_keys, rerun_expected_output }.",
  input: z.object({
    task_slug: z.string().min(1).describe("Task id, kept verbatim — it is the EvalSet id (Hive: the corpus slug, e.g. 'wfbench/generate-capital-city')."),
    task_title: z.string().optional().describe("Human title (defaults to the slug)."),
    instructions: z.string().min(1).describe("The one-line (or longer) English task the author builds a workflow from."),
    criteria: z.any().describe("Rubric: JSON array (or JSON string) of { id, title, match_criteria, deliverables? }."),
    workflow_input_json: z
      .any()
      .optional()
      .describe("The input the PRODUCED workflow is launched with — object or JSON string. Its keys are the input-key contract."),
    rerun_expected_output: z.any().optional().describe("Optional expected output of the rerun, handed to the judge."),
  }),
  output: z.any(),
  async run(cfg) {
    const task_slug = cfg.task_slug.trim();
    if (!task_slug) throw new Error("wfbench/normalize-task: task_slug is blank");
    const task_key = slugify(task_slug);
    if (!task_key) throw new Error(`wfbench/normalize-task: task_slug "${cfg.task_slug}" slugifies to nothing`);
    const task_title = (cfg.task_title ?? "").trim() || cfg.task_slug;

    const rawCriteria = parseMaybeJson(cfg.criteria);
    if (!Array.isArray(rawCriteria) || rawCriteria.length === 0) {
      throw new Error("wfbench/normalize-task: criteria must be a non-empty JSON array of rubric criteria");
    }
    const seen = new Set<string>();
    const criteria = rawCriteria.map((c: any, i: number) => {
      const obj = c && typeof c === "object" ? c : {};
      let id = String(obj.id ?? obj.criterion_id ?? `c${i + 1}`);
      if (seen.has(id)) id = `${id}-${i + 1}`;
      seen.add(id);
      return {
        id,
        title: String(obj.title ?? obj.name ?? id),
        match_criteria: String(obj.match_criteria ?? obj.description ?? ""),
        deliverables: Array.isArray(obj.deliverables) ? obj.deliverables : [],
      };
    });

    const wi = parseMaybeJson(cfg.workflow_input_json) ?? {};
    if (typeof wi !== "object" || wi === null || Array.isArray(wi)) {
      throw new Error("wfbench/normalize-task: workflow_input_json must be a JSON object (the produced workflow's input)");
    }
    const workflow_input = wi as Record<string, unknown>;

    return {
      task_slug,
      task_key,
      task_title,
      instructions: cfg.instructions,
      criteria,
      n_criteria: criteria.length,
      workflow_input,
      workflow_input_keys: Object.keys(workflow_input).sort(),
      rerun_expected_output: parseMaybeJson(cfg.rerun_expected_output) ?? null,
    };
  },
});
