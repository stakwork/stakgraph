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
 * The vein UI is served too: its build uses relative asset paths and a
 * runtime-derived API base, so it works under `/lab` as long as we
 * redirect `/lab` → `/lab/` (so relative `./assets/...` resolve under the
 * mount dir).
 *
 * Registration MUST happen before `express.json()` so vein receives the
 * raw request stream (same constraint as the graph SSE routes).
 */
/**
 * Gate every /lab route behind the mcp-wide API_TOKEN (unset = dev mode =
 * open, the same posture as the /events route). Two accepted credentials:
 * HTTP Basic `admin:<API_TOKEN>` — the browser prompts once for the UI and
 * then attaches it to every request including EventSource streams, which
 * cannot carry custom headers — and the `x-api-token` header, matching the
 * rest of mcp for server-to-server callers.
 */
function labAuth(req: Request, res: Response, next: NextFunction): void {
  const apiToken = process.env.API_TOKEN;
  if (!apiToken) return next();
  if (req.header("x-api-token") === apiToken) return next();
  const header = req.header("authorization") ?? "";
  if (header.startsWith("Basic ")) {
    const decoded = Buffer.from(header.slice(6), "base64").toString();
    const sep = decoded.indexOf(":");
    const user = decoded.slice(0, sep);
    const pass = decoded.slice(sep + 1);
    if (sep > 0 && user === "admin" && pass === apiToken) return next();
  }
  res.set("WWW-Authenticate", 'Basic realm="stakgraph-lab"');
  res.status(401).json({ error: "Unauthorized" });
}

export function mountLab(app: Express): void {
  // Trailing slash so the SPA's relative asset URLs resolve under /lab/.
  // Express routing is non-strict, so `/lab` also matches `/lab/`; guard
  // against redirecting `/lab/` to itself (an infinite 308 loop) by only
  // redirecting the exact, slash-less path and letting `/lab/` fall through
  // to the vein bridge below.
  app.get("/lab", (req, res, next) => {
    if (req.path === "/lab/") return next();
    res.redirect(308, "/lab/");
  });
  app.use("/lab", labAuth, bridge(() => createLabVein({ serveUi: true })));
}
