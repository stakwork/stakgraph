import { z, defineStep } from "vein";

/**
 * 58313's wfbench_build_produced_materials.py: turn the produced artifacts
 * into judge-ready materials. PRODUCED materials (what n_materials counts —
 * zero means the judge has nothing to grade and the harness reports
 * no_materials_produced instead of a fake 0/N): the workflow body and every
 * custom step the author wrote. CONTEXT materials (always attached): the
 * engine's static validation of the workflow (ok / errors / warnings — the
 * evidence for "structurally valid" criteria), the launch payload, the
 * rerun's output + status, and the expected output when the task carries one. `materials_text` is the single markdown block the
 * judge reads — the judge needs no tools.
 */
const clip = (s: string, max: number) => (s.length > max ? `${s.slice(0, max)}\n…[truncated ${s.length - max} chars]` : s);
const json = (v: unknown) => {
  try {
    return JSON.stringify(v, null, 2) ?? "null";
  } catch {
    return String(v);
  }
};

export default defineStep({
  type: "wfbench/build-materials",
  description:
    "Assemble judge materials from the produced workflow YAML, its custom step sources (meta/get-step outputs), the engine's static validation (meta/validate-workflow output), the rerun output/status, the launch payload and the expected output. Output: { n_materials (produced only), materials: [{ type, name, content }], materials_text, task_desc, warnings }.",
  input: z.object({
    workflow: z.string().describe("Produced workflow name."),
    version: z.any().optional(),
    workflow_yaml: z.string().describe("Produced workflow YAML ('' when nothing was published)."),
    custom_steps: z.array(z.any()).optional().describe("meta/get-step outputs for the steps the author created."),
    validation: z.any().optional().describe("meta/validate-workflow output for the produced YAML ({ ok, errors, warnings, summary })."),
    run_output: z.any().optional(),
    execution_status: z.string().optional(),
    project_id: z.any().optional(),
    rerun_expected_output: z.any().optional(),
    launch_payload: z.any().optional(),
    instructions: z.string().describe("The task instructions (becomes task_desc)."),
    maxChars: z.number().int().positive().default(60_000).describe("Per-material content cap."),
  }),
  output: z.any(),
  async run(cfg) {
    const warnings: string[] = [];
    const produced: Array<{ type: string; name: string; content: string }> = [];
    const context: Array<{ type: string; name: string; content: string }> = [];

    if (cfg.workflow_yaml.trim()) {
      produced.push({
        type: "WORKFLOW",
        name: `${cfg.workflow}${cfg.version ? `@${cfg.version}` : ""}`,
        content: clip(cfg.workflow_yaml, cfg.maxChars),
      });
    } else {
      warnings.push("no workflow body");
    }
    for (const s of cfg.custom_steps ?? []) {
      const o = s && typeof s === "object" ? (s as Record<string, any>) : {};
      const src = [o.source, o.code, o.sourceCode].find((v) => typeof v === "string" && v.trim());
      const name = String(o.type ?? o.name ?? "step");
      if (!src) {
        warnings.push(`custom step ${name}: no source (${typeof o.error === "string" ? o.error : "unreadable"})`);
        continue;
      }
      produced.push({ type: "STEP", name, content: clip(src, cfg.maxChars) });
    }

    if (cfg.validation && typeof cfg.validation === "object") {
      const v = cfg.validation as Record<string, any>;
      const nErr = Array.isArray(v.errors) ? v.errors.length : 0;
      const nWarn = Array.isArray(v.warnings) ? v.warnings.length : 0;
      context.push({
        type: "VALIDATION",
        name: `static validation: ${v.ok === true ? "ok" : "INVALID"} (${nErr} error(s), ${nWarn} warning(s))`,
        content: json({ ok: v.ok, errors: v.errors ?? [], warnings: v.warnings ?? [], summary: v.summary ?? null }),
      });
    }
    context.push({ type: "LAUNCH_PAYLOAD", name: "workflow_input", content: json(cfg.launch_payload ?? {}) });
    context.push({
      type: "RUN_OUTPUT",
      name: `run ${cfg.project_id ?? "none"} (${cfg.execution_status ?? "none"})`,
      content: clip(json(cfg.run_output ?? null), cfg.maxChars),
    });
    if (cfg.rerun_expected_output != null && cfg.rerun_expected_output !== "") {
      context.push({
        type: "EXPECTED_OUTPUT",
        name: "rerun_expected_output",
        content: clip(typeof cfg.rerun_expected_output === "string" ? cfg.rerun_expected_output : json(cfg.rerun_expected_output), cfg.maxChars),
      });
    }

    const materials = [...produced, ...context];
    const fence = (type: string) => (type === "WORKFLOW" ? "yaml" : type === "STEP" ? "ts" : "json");
    const materials_text = materials
      .map((m) => `### ${m.type}: ${m.name}\n\n\`\`\`${fence(m.type)}\n${m.content}\n\`\`\``)
      .join("\n\n");

    return {
      n_materials: produced.length,
      materials,
      materials_text,
      task_desc: cfg.instructions,
      warnings,
    };
  },
});
