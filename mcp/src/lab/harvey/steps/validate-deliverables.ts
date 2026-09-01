import { z, defineStep } from "vein";
import { stat } from "node:fs/promises";
import { join } from "node:path";

/**
 * The validate_deliverable_names equivalent — the HARD GATE between drafting
 * and scoring. Every deliverable filename referenced by the rubric (plus the
 * task's canonical list) must exist as a non-empty file in outputDir with its
 * EXACT name. Throws (fails the run) on any miss: judging a criterion whose
 * deliverable never materialized would grade an empty answer as a real one.
 */
export default defineStep({
  type: "harvey/validate-deliverables",
  description:
    "Hard gate before scoring: verify every deliverable filename (from the rubric criteria and the " +
    "canonical list) exists as a non-empty file in outputDir under its EXACT name. Throws listing every " +
    "missing/empty file. Output: { ok: true, files: [{ file, bytes }] }.",
  input: z.object({
    outputDir: z.string().describe("Absolute path of the final deliverables directory (the aggregator's ./output)."),
    deliverables: z.array(z.string()).default([]).describe("Canonical deliverable filenames."),
    rubric: z
      .array(z.any())
      .default([])
      .describe("Rubric criteria — each may carry a `deliverables` filename list; the union is validated."),
  }),
  output: z.any(),
  async run(cfg) {
    const expected = new Set<string>(cfg.deliverables);
    for (const c of cfg.rubric) {
      const list = (c as Record<string, any>)?.deliverables;
      if (Array.isArray(list)) for (const d of list) if (typeof d === "string" && d) expected.add(d);
    }
    if (expected.size === 0) {
      throw new Error("harvey/validate-deliverables: no expected deliverable names (empty rubric + deliverables)");
    }

    const files: Array<{ file: string; bytes: number }> = [];
    const problems: string[] = [];
    for (const file of [...expected].sort()) {
      try {
        const s = await stat(join(cfg.outputDir, file));
        if (!s.isFile()) problems.push(`${file} (not a file)`);
        else if (s.size === 0) problems.push(`${file} (empty)`);
        else files.push({ file, bytes: s.size });
      } catch {
        problems.push(`${file} (missing)`);
      }
    }
    if (problems.length > 0) {
      throw new Error(
        `harvey/validate-deliverables: ${problems.length} deliverable(s) failed in ${cfg.outputDir}: ${problems.join(", ")}`,
      );
    }
    return { ok: true, files };
  },
});
