import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceManager } from "vein";

/** Generic artifact plumbing steps (NOT an experiment). Today just
 *  `artifacts/dir` — see steps/dir.ts. */
const SEED_STEPS: Array<{ file: string; type: string }> = [
  { file: "dir.ts", type: "artifacts/dir" },
];

const HERE = dirname(fileURLToPath(import.meta.url));

export async function seedArtifactSteps(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "steps");
  for (const { file, type } of SEED_STEPS) {
    try {
      const code = await readFile(join(dir, file), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "artifacts-seed");
      if (changed) console.log(`[artifacts] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(`[artifacts] could not seed step "${type}":`, err instanceof Error ? err.message : err);
    }
  }
}
