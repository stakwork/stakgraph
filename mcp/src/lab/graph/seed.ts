import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceManager } from "vein";

/**
 * Vein-native knowledge-graph steps, seeded into the vein workspace — the
 * `graph/*` twins of the `jarvis/*` steps. Same input/output shapes, but
 * backed by vein's own Neo4j-over-bolt graph layer (`vein/src/graph/*`,
 * plans/jarvis-graph-compat.md) instead of the Jarvis HTTP API, so a
 * workflow swaps backends by step type. Connection via
 * `ctx.services.secrets` (NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD, optional
 * VEIN_GRAPH_NAMESPACE / VEIN_GRAPH_EMBEDDINGS). Reconciled by content hash
 * on boot (edits via the vein UI publish a new active version).
 *
 * Grant them to an agent step with `agentTools: ["graph/*"]` (glob), or a
 * read-only subset by listing the read steps explicitly.
 *
 * Steps are ALWAYS seeded (deterministic workspace, deterministic registry);
 * a deployment without NEO4J_URI gets a loud per-run error instead of a
 * silently missing tool.
 */
const SEED_STEPS: Array<{ file: string; type: string }> = [
  // reads
  { file: "get-ontology.ts", type: "graph/get-ontology" },
  { file: "get-ontology-type.ts", type: "graph/get-ontology-type" },
  { file: "graph-search.ts", type: "graph/graph-search" },
  { file: "graph-get.ts", type: "graph/graph-get" },
  { file: "graph-get-batched.ts", type: "graph/graph-get-batched" },
  { file: "graph-neighbors.ts", type: "graph/graph-neighbors" },
  // writes
  { file: "register-namespace.ts", type: "graph/register-namespace" },
  { file: "create-node.ts", type: "graph/create-node" },
  { file: "edit-node.ts", type: "graph/edit-node" },
  { file: "create-triplet.ts", type: "graph/create-triplet" },
  { file: "create-batch-triplet.ts", type: "graph/create-batch-triplet" },
];

const HERE = dirname(fileURLToPath(import.meta.url));

export async function seedGraphSteps(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "steps");
  for (const { file, type } of SEED_STEPS) {
    try {
      const code = await readFile(join(dir, file), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "graph-seed");
      if (changed) console.log(`[graph] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(
        `[graph] could not seed step "${type}":`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}
