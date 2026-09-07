import type { Express, Request, Response, NextFunction } from "express";
import type { IncomingMessage, Server } from "node:http";
import type { Duplex } from "node:stream";
import { getRequestListener } from "@hono/node-server";
import { createAudioUpgradeHandler, type AudioUpgradeHandler } from "vein";
import { createLabVein } from "./createLabVein.js";

/** The one lab vein, built on first use (HTTP request or dictation upgrade). */
let labVeinP: ReturnType<typeof createLabVein> | null = null;
function labVein() {
  return (labVeinP ??= createLabVein({ serveUi: true }));
}

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

/** Does this request carry the lab credential? (`labAuth` without the
 *  response side, so the WebSocket upgrade can apply the same rule.) */
function labAuthorized(req: { header(name: string): string | undefined }): boolean {
  const apiToken = process.env.API_TOKEN;
  if (!apiToken) return true;
  if (req.header("x-api-token") === apiToken) return true;
  const header = req.header("authorization") ?? "";
  if (header.startsWith("Basic ")) {
    const decoded = Buffer.from(header.slice(6), "base64").toString();
    const sep = decoded.indexOf(":");
    const user = decoded.slice(0, sep);
    const pass = decoded.slice(sep + 1);
    if (sep > 0 && user === "admin" && pass === apiToken) return true;
  }
  return false;
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
  if (labAuthorized(req)) return next();
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
  app.use("/lab", labAuth, bridge(labVein));
}

const LAB_AUDIO_STREAM = "/lab/audio/stream";

/**
 * Dictation over `/lab/audio/stream` (vein `src/audio/ws.ts`). A WebSocket
 * upgrade never enters Express, so the bridge above can't carry it: hook the
 * Node server's `upgrade` event, apply the lab credential (browsers resend
 * cached Basic auth on same-origin handshakes, so the UI's one-time prompt
 * covers it), then hand the socket to vein. Built lazily like the bridge —
 * the first dictation boots the lab vein if a request hasn't already.
 * Other upgrade paths get a 404 rather than a socket left hanging.
 */
export function attachLabAudio(server: Server): void {
  let handlerP: Promise<AudioUpgradeHandler | null> | null = null;
  const handler = () =>
    (handlerP ??= labVein().then((vein) =>
      vein.stt
        ? createAudioUpgradeHandler(vein.stt, { basePath: "/lab", authorize: () => true })
        : null,
    ));
  const reject = (socket: Duplex, status: string) => {
    socket.write(`HTTP/1.1 ${status}\r\nConnection: close\r\n\r\n`);
    socket.destroy();
  };
  server.on("upgrade", (req: IncomingMessage, socket: Duplex, head: Buffer) => {
    const url = new URL(req.url ?? "/", "http://localhost");
    if (url.pathname !== LAB_AUDIO_STREAM) return reject(socket, "404 Not Found");
    const header = (name: string) => req.headers[name.toLowerCase()] as string | undefined;
    if (!labAuthorized({ header })) return reject(socket, "401 Unauthorized");
    handler()
      .then((h) => {
        if (!h) return reject(socket, "501 Not Implemented");
        if (socket.destroyed) return;
        h.handle(req, socket, head);
      })
      .catch((e) => {
        console.error("[lab] dictation upgrade failed:", e);
        reject(socket, "500 Internal Server Error");
      });
  });
}
