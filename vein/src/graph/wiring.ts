/**
 * Env-driven wiring for the graph-backed workspace: set
 * `VEIN_WORKSPACE_BACKEND=graph` (plus the `NEO4J_*` connection vars) and
 * the default server keeps workflows/steps in Neo4j instead of the
 * filesystem. Runs, chats, secrets, artifacts, and cassettes stay local
 * under `dataDir` (`VEIN_WORKSPACE`) — the run/chat projector
 * (`projector.ts`, or the `graph/project` step) builds their graph view.
 */
import { join } from "node:path";
import type { GraphBackend, GraphBackendOptions } from "./backend.js";
import { openGraphBackendFromEnv } from "./backend.js";
import { Neo4jWorkspaceStore, type Neo4jWorkspaceStoreOptions } from "./workspace-store.js";

export function graphWorkspaceRequested(env: Record<string, string | undefined> = process.env): boolean {
  return (env["VEIN_WORKSPACE_BACKEND"] ?? "").toLowerCase() === "graph";
}

/** Same default as the mcp host's own Neo4j client and the `graph/*` steps:
 *  a local Neo4j needs nothing configured. */
export const DEFAULT_NEO4J_HOST = "localhost:7687";

/**
 * Where the graph store materializes active custom steps for the module
 * loader: INSIDE the data dir, beside where a file workspace would keep
 * `steps/custom`. Custom steps `import "vein"` (and "zod"), and Node resolves
 * that by walking up from the FILE's directory — so the dir must sit in the
 * same tree as the file store's, never under the OS temp dir. Distinct from
 * `steps/custom` so switching backends on one dir can't prune the other's
 * files.
 */
export function graphMaterializeDir(dataDir: string): string {
  return join(dataDir, "steps", "_graph");
}

/**
 * Open the graph backend from env and wrap it in a `Neo4jWorkspaceStore`.
 * Connection: `NEO4J_URI`, else `bolt://<NEO4J_HOST>` (default
 * `localhost:7687`); `NEO4J_USER` / `NEO4J_PASSWORD` default to
 * neo4j / testtest — the same resolution as the mcp host, so a deployment's
 * existing vars just work and a local Neo4j needs none. `dataDir` (default
 * `VEIN_WORKSPACE` / `./workspace`) is where custom steps are materialized.
 */
export async function graphWorkspaceFromEnv(
  env: Record<string, string | undefined> = process.env,
  opts: { dataDir?: string; backend?: GraphBackendOptions; store?: Neo4jWorkspaceStoreOptions } = {},
): Promise<{ backend: GraphBackend; workspace: Neo4jWorkspaceStore }> {
  const backend = await openGraphBackendFromEnv({ NEO4J_HOST: DEFAULT_NEO4J_HOST, ...env }, opts.backend)!;
  const dataDir = opts.dataDir ?? env["VEIN_WORKSPACE"] ?? "./workspace";
  const store = { materializeDir: graphMaterializeDir(dataDir), ...opts.store };
  return { backend, workspace: new Neo4jWorkspaceStore(backend, store) };
}
