// Default strut server — a thin wrapper over createStrut() that keeps
// workflows/steps in Neo4j by default (see src/graph/wiring.ts; NEO4J_*
// vars, localhost:7687 when unset) while runs/chats/blobs stay under
// STRUT_WORKSPACE. Set STRUT_WORKSPACE_BACKEND=fs to boot with the
// filesystem-backed defaults instead (FileRunStore, workspace loaded from
// STRUT_WORKSPACE, registry built by scanning steps/) — no Neo4j needed.
//
// This file is the canonical "library usage" example: anything you
// see here, your own consumer code can do too. Pass your own
// `services`, your own in-code `registry`, your own store — and mount
// the returned `app` wherever you like (under another Hono router,
// behind Express, or just call `strut.listen(port)`).

import { createStrut, type Strut } from "./createStrut.js";
import { FileRunStore } from "./store.js";
import { FileChatStore } from "./chat-store.js";
import { FileSecretStore } from "./secret-store.js";
import { graphWorkspaceRequested } from "./graph/wiring.js";

let strutInstance: Strut | null = null;

/** Lazily build the default Strut instance. Kept lazy so importing this
 *  module doesn't kick off filesystem I/O at module-load time. */
async function getDefault(): Promise<Strut> {
  if (!strutInstance) {
    if (graphWorkspaceRequested()) {
      const { graphWorkspaceFromEnv } = await import("./graph/wiring.js");
      // dataDir keeps its file default (STRUT_WORKSPACE / ./workspace): runs,
      // chats, secrets, artifacts, cassettes, and the materialized custom
      // steps stay local. Explicit file stores, since a non-file workspace
      // would otherwise default to memory.
      const dataDir = process.env["STRUT_WORKSPACE"] ?? "./workspace";
      const { backend, workspace } = await graphWorkspaceFromEnv(process.env, { dataDir });
      strutInstance = await createStrut({
        workspace,
        // Same backend for the chat builder's read-only graph_query tool.
        graph: backend,
        dataDir,
        store: new FileRunStore(dataDir),
        chatStore: new FileChatStore(dataDir),
        secretStore: new FileSecretStore(dataDir),
      });
    } else {
      strutInstance = await createStrut();
    }
  }
  return strutInstance;
}

/**
 * Hono app for the default strut. Returns the same
 * instance on repeated calls. Most code should call `createStrut()`
 * directly and mount `strut.app` — this helper exists for ergonomic
 * scripting and backwards compatibility.
 */
export async function getApp() {
  return (await getDefault()).app;
}

/**
 * Boot the default strut server on `port` (defaults to
 * `STRUT_PORT` or `3000`). Equivalent to:
 *
 * ```ts
 * const strut = await createStrut();
 * await strut.listen(port);
 * ```
 */
export async function startServer(port?: number, host?: string): Promise<number> {
  const strut = await getDefault();
  return strut.listen(port, host);
}

// Run directly when invoked as a script.
const isMain =
  process.argv[1]?.endsWith("server.ts") ||
  process.argv[1]?.endsWith("server.js");
if (isMain) {
  startServer();
}
