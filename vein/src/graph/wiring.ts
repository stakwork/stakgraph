/**
 * Env-driven wiring for the graph-backed workspace: set
 * `VEIN_WORKSPACE_BACKEND=graph` (plus the `NEO4J_*` connection vars) and
 * the default server keeps workflows/steps in Neo4j instead of the
 * filesystem. Runs, chats, secrets, artifacts, and cassettes stay local
 * under `dataDir` (`VEIN_WORKSPACE`) — the run/chat projector
 * (`projector.ts`, or the `graph/project` step) builds their graph view.
 */
import type { GraphBackend, GraphBackendOptions } from "./backend.js";
import { openGraphBackendFromEnv } from "./backend.js";
import { Neo4jWorkspaceStore, type Neo4jWorkspaceStoreOptions } from "./workspace-store.js";

export function graphWorkspaceRequested(env: Record<string, string | undefined> = process.env): boolean {
  return (env["VEIN_WORKSPACE_BACKEND"] ?? "").toLowerCase() === "graph";
}

/**
 * Open the graph backend from env and wrap it in a `Neo4jWorkspaceStore`.
 * Throws when no connection is configured — a deployment that asked for
 * the graph backend must not silently fall back to files.
 */
export async function graphWorkspaceFromEnv(
  env: Record<string, string | undefined> = process.env,
  opts: { backend?: GraphBackendOptions; store?: Neo4jWorkspaceStoreOptions } = {},
): Promise<{ backend: GraphBackend; workspace: Neo4jWorkspaceStore }> {
  const pending = openGraphBackendFromEnv(env, opts.backend);
  if (!pending) {
    throw new Error("VEIN_WORKSPACE_BACKEND=graph needs NEO4J_URI (or NEO4J_HOST) — no Neo4j connection configured");
  }
  const backend = await pending;
  return { backend, workspace: new Neo4jWorkspaceStore(backend, opts.store) };
}
