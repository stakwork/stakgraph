import { z, defineStep } from "vein";
import { stat } from "node:fs/promises";
import { extname, join } from "node:path";

/**
 * Intake normalization for the harvey-deliver pipeline (the lab-shaped port
 * of stakwork's harvey-lab-normalize-documents script). Input documents are
 * LOCAL files from the harvey-labs task checkout (harvey/get-task's
 * documentsDir + relative listing) — no URLs, no fetching. This step:
 *
 *  - derives the run's graph NAMESPACE from the task slug (also used as the
 *    EvalSet id): "practice-area/task-slug" → "practice-area-task-slug"
 *  - verifies every listed document actually exists (stat) and records size
 *  - classifies by extension and flags spreadsheet sources (.xlsx) — the
 *    flag_spreadsheet_sources equivalent
 *  - hard-fails when the task has NO readable documents and `requireDocuments`
 *    is set (the guard_missing_docs equivalent) — a deliver run without a
 *    record to work from would just hallucinate
 */
const SPREADSHEET_EXTS = new Set([".xlsx", ".xls", ".csv"]);

export default defineStep({
  type: "harvey/normalize-documents",
  description:
    "Normalize a Harvey deliver run's input documents (LOCAL files from harvey/get-task): derive the " +
    "graph namespace from the task slug, verify each document exists, flag spreadsheet sources, and " +
    "hard-fail when documents are required but missing. Output: { namespace, documents: [{ file, path, " +
    "ext, isSpreadsheet, bytes }], count, hasSpreadsheets, missing }.",
  input: z.object({
    task: z.string().describe("Task id ('practice-area/task-slug') — slugified into the namespace/EvalSet id."),
    documentsDir: z.string().describe("Absolute path of the task's read-only documents directory."),
    documents: z
      .array(z.string())
      .default([])
      .describe("Relative document paths inside documentsDir (harvey/get-task's `documents`)."),
    requireDocuments: z
      .boolean()
      .default(true)
      .describe("Throw when no readable documents are found (guard_missing_docs). Set false for doc-less tasks."),
  }),
  output: z.any(),
  async run(cfg) {
    const namespace = cfg.task
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
    if (!namespace) throw new Error(`harvey/normalize-documents: task "${cfg.task}" yields an empty namespace`);

    const documents: Array<{ file: string; path: string; ext: string; isSpreadsheet: boolean; bytes: number }> = [];
    const missing: string[] = [];
    for (const file of cfg.documents) {
      const path = join(cfg.documentsDir, file);
      try {
        const s = await stat(path);
        if (!s.isFile()) {
          missing.push(file);
          continue;
        }
        const ext = extname(file).toLowerCase();
        documents.push({ file, path, ext, isSpreadsheet: SPREADSHEET_EXTS.has(ext), bytes: s.size });
      } catch {
        missing.push(file);
      }
    }

    if (missing.length > 0) {
      // Listed-but-unreadable is always fatal: the caller believes these
      // documents exist, and silently dropping one plants a completeness hole.
      throw new Error(
        `harvey/normalize-documents: ${missing.length} listed document(s) missing under ${cfg.documentsDir}: ${missing.join(", ")}`,
      );
    }
    if (cfg.requireDocuments && documents.length === 0) {
      throw new Error(`harvey/normalize-documents: task "${cfg.task}" has no input documents (requireDocuments=true)`);
    }

    return {
      namespace,
      documents,
      count: documents.length,
      hasSpreadsheets: documents.some((d) => d.isSpreadsheet),
      missing,
    };
  },
});
