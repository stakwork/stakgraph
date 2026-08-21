import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceManager } from "vein";

/**
 * Google Sheets steps, seeded into the vein workspace. Each is a
 * self-contained port of the matching mcp repo-agent tool
 * (`mcp/src/repo/toolsGoogleSheets.ts`) speaking the same Sheets/Drive REST
 * contract — but routed through `ctx.services.http` + `ctx.services.secrets`
 * (GOOGLE_SERVICE_ACCOUNT_JSON / GOOGLE_DRIVE_FOLDER_ID, env-backed) so runs
 * are cassette-recordable and credentials stay scrubbed. Reconciled by
 * content hash on boot (edits via the vein UI publish a new active version).
 *
 * Grant them to an agent step with `agentTools: ["sheets/*"]` (glob), or an
 * explicit subset (e.g. a read-only child gets just `sheets/get-values`).
 *
 * Steps are ALWAYS seeded (deterministic workspace, deterministic registry);
 * a deployment without GOOGLE_SERVICE_ACCOUNT_JSON gets a loud per-run error
 * instead of a silently missing tool.
 */
const SEED_STEPS: Array<{ file: string; type: string }> = [
  { file: "create-spreadsheet.ts", type: "sheets/create-spreadsheet" },
  { file: "update-values.ts", type: "sheets/update-values" },
  { file: "batch-update-values.ts", type: "sheets/batch-update-values" },
  { file: "get-values.ts", type: "sheets/get-values" },
  { file: "add-sheet.ts", type: "sheets/add-sheet" },
  { file: "import-spreadsheet.ts", type: "sheets/import-spreadsheet" },
];

const HERE = dirname(fileURLToPath(import.meta.url));

export async function seedSheetsSteps(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "steps");
  for (const { file, type } of SEED_STEPS) {
    try {
      const code = await readFile(join(dir, file), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "sheets-seed");
      if (changed) console.log(`[sheets] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(
        `[sheets] could not seed step "${type}":`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}
