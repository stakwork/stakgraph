import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceManager } from "vein";

/**
 * Jarvis knowledge-graph steps, seeded into the vein workspace. Each is a
 * self-contained port of the matching mcp repo-agent tool
 * (`mcp/src/repo/toolsJarvis.ts`) speaking the same Jarvis HTTP contract —
 * but routed through `ctx.services.http` + `ctx.services.secrets`
 * (JARVIS_URL / API_TOKEN, env-backed) so runs are cassette-recordable and
 * credentials stay scrubbed. Reconciled by content hash on boot (edits via
 * the vein UI publish a new active version).
 *
 * Grant them to an agent step with `agentTools: ["jarvis/*"]` (glob), or a
 * read-only subset by listing the read steps explicitly.
 *
 * Steps are ALWAYS seeded (deterministic workspace, deterministic registry);
 * a deployment without JARVIS_URL gets a loud per-run error instead of a
 * silently missing tool. The ontology CRUD family (schema editing) is
 * deliberately not ported — schema changes stay a human/setup activity.
 */
const SEED_STEPS: Array<{ file: string; type: string }> = [
  // reads
  { file: "get-ontology.ts", type: "jarvis/get-ontology" },
  { file: "get-ontology-type.ts", type: "jarvis/get-ontology-type" },
  { file: "graph-search.ts", type: "jarvis/graph-search" },
  { file: "graph-get.ts", type: "jarvis/graph-get" },
  { file: "graph-get-batched.ts", type: "jarvis/graph-get-batched" },
  { file: "graph-neighbors.ts", type: "jarvis/graph-neighbors" },
  // writes
  { file: "create-node.ts", type: "jarvis/create-node" },
  { file: "edit-node.ts", type: "jarvis/edit-node" },
  { file: "create-triplet.ts", type: "jarvis/create-triplet" },
  { file: "create-batch-triplet.ts", type: "jarvis/create-batch-triplet" },
];

const HERE = dirname(fileURLToPath(import.meta.url));

export async function seedJarvisSteps(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "steps");
  for (const { file, type } of SEED_STEPS) {
    try {
      const code = await readFile(join(dir, file), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "jarvis-seed");
      if (changed) console.log(`[jarvis] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(
        `[jarvis] could not seed step "${type}":`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}
