import type { Express, Request, Response, NextFunction } from "express";
import type { IncomingMessage, Server } from "node:http";
import type { Duplex } from "node:stream";
import { getRequestListener } from "@hono/node-server";
import { createAudioUpgradeHandler, type AudioUpgradeHandler } from "strut";
import { createLabStrut } from "./createLabStrut.js";
import { stashLabActor } from "./actor.js";
import { verifyApiToken, type ApiTokenPayload } from "../repo/events.js";

/** The one lab strut, built on first use (HTTP request or dictation upgrade). */
let labStrutP: ReturnType<typeof createLabStrut> | null = null;
function labStrut() {
  return (labStrutP ??= createLabStrut({ serveUi: true }));
}

/**
 * Bridge a (lazily-built) strut Hono app into Express. The instance is
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
        (strut) => getRequestListener(strut.app.fetch) as NodeListener,
      ));
    p.then((listener) => listener(req, res)).catch(next);
  };
}

/** The payload of a live `/mint-token` JWT — `undefined` when the token is
 *  missing, malformed, expired, or of another scope. */
function isEmbedJwt(token: string | null | undefined): ApiTokenPayload | undefined {
  if (!token) return undefined;
  try {
    return verifyApiToken(token);
  } catch {
    return undefined;
  }
}

interface LabCredentials {
  header(name: string): string | undefined;
  /** `?key=` — how an embedding host hands the strut UI its key on first
   *  load, and where the UI puts it on the dictation WebSocket (neither can
   *  set a header). */
  key?: string | null;
}

/** What an accepted credential grants: entry, plus — when the credential
 *  says who is asking — the strut `actor` the request is attributed to
 *  (plans/mothership-cost-control.md §5). */
interface LabGrant {
  /** JWT → its `sub`. `x-api-token` → hive's own `x-strut-actor` header,
   *  trusted because the token proves the caller is hive. Basic → none. */
  actor?: string;
}

const grant = (actor: string | undefined): LabGrant => {
  const v = actor?.trim();
  return v ? { actor: v } : {};
};

/** Does this request carry the lab credential, and for whom? `undefined`
 *  when it does not. (`labAuth` without the response side, so the WebSocket
 *  upgrade can apply the same rule.) */
function labAuthorized(req: LabCredentials): LabGrant | undefined {
  const apiToken = process.env.API_TOKEN;
  // Dev mode: open — and with nothing to prove who a caller is, nobody's
  // word on who should pay is taken either (strut's own default is the same).
  if (!apiToken) return {};
  if (req.header("x-api-token") === apiToken) return grant(req.header("x-strut-actor"));
  const viaKey = isEmbedJwt(req.key);
  if (viaKey) return grant(viaKey.sub);
  const header = req.header("authorization") ?? "";
  if (header.startsWith("Bearer ")) {
    const viaBearer = isEmbedJwt(header.slice(7).trim());
    if (viaBearer) return grant(viaBearer.sub);
  }
  if (header.startsWith("Basic ")) {
    const decoded = Buffer.from(header.slice(6), "base64").toString();
    const sep = decoded.indexOf(":");
    const user = decoded.slice(0, sep);
    const pass = decoded.slice(sep + 1);
    if (sep > 0 && user === "admin" && pass === apiToken) return {};
  }
  return undefined;
}

/**
 * Mount the single lab strut under `/lab` (API + run-streaming SSE). All
 * experiments share this one instance — they're groups of workflows
 * inside it, not separate servers.
 *
 * The strut UI is served too: its build uses relative asset paths and a
 * runtime-derived API base, so it works under `/lab` as long as we
 * redirect `/lab` → `/lab/` (so relative `./assets/...` resolve under the
 * mount dir).
 *
 * Registration MUST happen before `express.json()` so strut receives the
 * raw request stream (same constraint as the graph SSE routes).
 */
/**
 * Gate every /lab route behind the mcp-wide API_TOKEN (unset = dev mode =
 * open, the same posture as the /events route). Three accepted credentials:
 * HTTP Basic `admin:<API_TOKEN>` — the browser prompts once for the UI and
 * then attaches it to every request including EventSource streams, which
 * cannot carry custom headers — the `x-api-token` header, matching the
 * rest of mcp for server-to-server callers, and a `/mint-token` JWT for
 * iframe embeds: the host loads `/lab/?key=<jwt>`, and the strut UI stashes
 * the key and replays it as `Authorization: Bearer` on every fetch (its
 * streams are fetch-based) and as `?key=` on the dictation WebSocket.
 *
 * An accepted credential may also name the strut `actor` (the JWT's `sub`,
 * or hive's `x-strut-actor` beside `x-api-token`). It is stashed on the Node
 * request for strut's `resolveActor` hook — a header rewrite would not
 * survive the Hono bridge (see actor.ts).
 */
export function labAuth(req: Request, res: Response, next: NextFunction): void {
  if (isUiAsset(req)) return next();
  const key = typeof req.query.key === "string" ? req.query.key : null;
  const granted = labAuthorized({ header: (name) => req.header(name), key });
  if (granted) {
    stashLabActor(req, granted.actor);
    return next();
  }
  // A client that showed up with a (now bad or expired) JWT is an embed, not
  // a person at a browser: a Basic challenge would pop a login dialog inside
  // the host's iframe. Give it a plain 401 and let the host re-mint.
  const isEmbed = key !== null || (req.header("authorization") ?? "").startsWith("Bearer ");
  if (!isEmbed) res.set("WWW-Authenticate", 'Basic realm="stakgraph-lab"');
  res.status(401).json({ error: "Unauthorized" });
}

/**
 * The UI's hashed JS/CSS bundles. Public, like mcp's own `/assets`: they
 * hold no secrets, and a `<script>`/`<link>` tag can't carry the embed JWT
 * (only cached Basic credentials ride along on their own).
 *
 * The path is parsed the way strut's Hono bridge will parse it (origin +
 * raw url through WHATWG URL: resolves `..`, `%2e%2e`, `\`), NOT read off
 * `req.path` — otherwise `/lab/assets/../secrets` would pass a prefix check
 * here and then be routed as `/secrets` past the gate.
 */
function isUiAsset(req: Request): boolean {
  if (req.method !== "GET" && req.method !== "HEAD") return false;
  try {
    return new URL(`http://localhost${req.url}`).pathname.startsWith("/assets/");
  } catch {
    return false;
  }
}

export function mountLab(app: Express): void {
  // Trailing slash so the SPA's relative asset URLs resolve under /lab/.
  // Express routing is non-strict, so `/lab` also matches `/lab/`; guard
  // against redirecting `/lab/` to itself (an infinite 308 loop) by only
  // redirecting the exact, slash-less path and letting `/lab/` fall through
  // to the strut bridge below.
  app.get("/lab", (req, res, next) => {
    if (req.path === "/lab/") return next();
    res.redirect(308, "/lab/");
  });
  app.use("/lab", labAuth, bridge(labStrut));
}

const LAB_AUDIO_STREAM = "/lab/audio/stream";

/**
 * Dictation over `/lab/audio/stream` (strut `src/audio/ws.ts`). A WebSocket
 * upgrade never enters Express, so the bridge above can't carry it: hook the
 * Node server's `upgrade` event, apply the lab credential (browsers resend
 * cached Basic auth on same-origin handshakes, so the UI's one-time prompt
 * covers it), then hand the socket to strut. Built lazily like the bridge —
 * the first dictation boots the lab strut if a request hasn't already.
 * Other upgrade paths get a 404 rather than a socket left hanging. No actor:
 * dictation makes no LLM call and launches nothing.
 */
export function attachLabAudio(server: Server): void {
  let handlerP: Promise<AudioUpgradeHandler | null> | null = null;
  const handler = () =>
    (handlerP ??= labStrut().then((strut) =>
      strut.stt
        ? createAudioUpgradeHandler(strut.stt, { basePath: "/lab", authorize: () => true })
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
    if (!labAuthorized({ header, key: url.searchParams.get("key") })) {
      return reject(socket, "401 Unauthorized");
    }
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
