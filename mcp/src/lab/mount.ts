import type { Express, Request, Response, NextFunction } from "express";
import { getRequestListener } from "@hono/node-server";
import { createLabVein } from "./createLabVein.js";

/**
 * Bridge a (lazily-built) vein Hono app into Express. The instance is
 * created on the first request to its mount path, so mcp boot is never
 * coupled to an experiment's Neo4j / LLM-key dependencies — and a broken
 * experiment can't take down the whole server at startup.
 */
type NodeListener = (req: any, res: any) => void;

function bridge(factory: () => Promise<{ app: { fetch: any } }>) {
  let listenerP: Promise<NodeListener> | null = null;
  return (req: Request, res: Response, next: NextFunction) => {
    const p =
      listenerP ??
      (listenerP = factory().then(
        (vein) => getRequestListener(vein.app.fetch) as NodeListener,
      ));
    p.then((listener) => listener(req, res)).catch(next);
  };
}

/**
 * Mount the single lab vein under `/lab` (API + run-streaming SSE). All
 * experiments share this one instance — they're groups of workflows
 * inside it, not separate servers.
 *
 * UI note: `serveUi` is false — vein's web UI assumes root-absolute
 * asset/API paths and can't be served under a sub-path without a
 * base-path rebuild; run the vein UI separately during experimentation.
 *
 * Registration MUST happen before `express.json()` so vein receives the
 * raw request stream (same constraint as the graph SSE routes).
 */
export function mountLab(app: Express): void {
  app.use("/lab", bridge(() => createLabVein({ serveUi: false })));
}
