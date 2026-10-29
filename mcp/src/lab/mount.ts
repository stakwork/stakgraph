import type { Express, Request, Response, NextFunction } from "express";
import type { IncomingMessage, Server } from "node:http";
import type { Duplex } from "node:stream";
import { getRequestListener } from "@hono/node-server";
import { createAudioUpgradeHandler, scrubProcessEnv, type AudioUpgradeHandler, type Strut } from "strut";
import { createLabStrut } from "./createLabStrut.js";
import { env, freezeEnv } from "../env.js";
import { stashLabActor, stashLabScope, type LabScope } from "./actor.js";
import { verifyApiToken, type ApiTokenPayload, type ApiTokenScope } from "../repo/events.js";

/** The one lab strut, built on first use (HTTP request, dictation upgrade,
 *  or a `/gitree/process` / `/repo/describe` launch — lab/proxy.ts). The
 *  env is scrubbed first (below): nothing runs a step before the lab is
 *  built, and strut must be built over the scrubbed env (see `scrubEnv`). */
let labStrutP: ReturnType<typeof createLabStrut> | null = null;
export function labStrut() {
  return (labStrutP ??= (async () => {
    scrubEnv();
    return createLabStrut({ serveUi: true });
  })());
}

/** What mcp sets on `process.env` itself, on purpose, for every git child
 *  anything in this process spawns (repo/clone.ts at import: never prompt
 *  for credentials, never hang a headless container) — git behavior, not a
 *  credential, so the scrub leaves it where every spawn finds it. */
const KEEP_FOR_GIT = ["GIT_TERMINAL_PROMPT", "GIT_ASKPASS"];

/**
 * The env boundary (strut src/env.ts, stakwork/strut#228): a workspace step
 * is TypeScript running in this process, so `process.env` — where the
 * deployment's keys arrive — is readable by any step and by any child it
 * spawns with it. The default strut server empties `process.env` at boot; a
 * host that embeds `createStrut` does it itself. mcp does it when the lab
 * boots, BEFORE `createLabStrut`: strut's own code reads its boot snapshot
 * (`bootEnv()`), but `standardServices` takes that snapshot by reference
 * when the strut is built, and until the scrub runs the snapshot IS
 * `process.env` — a scrub after `createStrut` would empty the very object
 * the secrets fallback holds, and `ctx.services.secrets` would resolve no
 * key the deployment set in env. Scrubbed first, every construction-time
 * read — strut's through `bootEnv()`, the lab's own through `env`
 * (src/env.ts: the workspace path, NEO4J_*, GITHUB_TOKEN, the provider key,
 * GAIA_*) — sees the copy, and a step's `process.env.X` sees nothing.
 *
 * mcp's own request-time reads (`API_TOKEN` on every route, `NEO4J_*`,
 * aieo's provider keys, `GITHUB_TOKEN`) go through `env` too, frozen here
 * right before the scrub so it holds the same copy.
 *
 * `STRUT_ENV_SCRUB=0` turns it off (strut's own switch, the same name), for
 * a swarm where something unexpected still reads the env; `STRUT_ENV_KEEP=A,B`
 * keeps named variables, as in strut.
 */
function scrubEnv(): void {
  if (env["STRUT_ENV_SCRUB"] === "0") {
    console.log("[env] STRUT_ENV_SCRUB=0: process.env left as it is");
    return;
  }
  freezeEnv();
  const gone = scrubProcessEnv({ keep: KEEP_FOR_GIT });
  if (gone.length) console.log(`[env] ${gone.length} variable(s) taken out of process.env (kept in the boot snapshot): ${gone.join(", ")}`);
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

/** The payload of a live `/mint-token` JWT of an `accept`ed scope —
 *  `undefined` when the token is missing, malformed, expired, or of another
 *  scope. */
function isEmbedJwt(
  token: string | null | undefined,
  accept: readonly ApiTokenScope[] = ["api"],
): ApiTokenPayload | undefined {
  if (!token) return undefined;
  try {
    return verifyApiToken(token, accept);
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
 *  (plans/mothership-cost-control.md §5), and for a peer, its scope. */
interface LabGrant {
  /** JWT → its `sub`. `x-api-token` → hive's own `x-strut-actor` header,
   *  trusted because the token proves the caller is hive. A `lab:peer` JWT
   *  → its `sub`, else the caller's `x-strut-actor`: the person the calling
   *  strut's run is for. Basic → none. */
  actor?: string;
  /** `peer` for a `lab:peer` JWT — another strut (strut plans/federation.md
   *  §3) — and `peer:person` for one whose `sub` names the person (hive's
   *  pairing; strut plans/chained-orchestrators.md §2): its actor is the
   *  token's own, so strut lets it launch a `peer: person` workflow (the
   *  orchestrator), open that person's own sessions and join their room.
   *  Strut decides what each may do; absent = everything, as before. */
  scope?: LabScope;
}

const grant = (actor: string | undefined): LabGrant => {
  const v = actor?.trim();
  return v ? { actor: v } : {};
};

/** Does this request carry the lab credential, and for whom? `undefined`
 *  when it does not. (`labAuth` without the response side, so the WebSocket
 *  upgrade can apply the same rule.) */
function labAuthorized(req: LabCredentials): LabGrant | undefined {
  const apiToken = env.API_TOKEN;
  // Dev mode: open — and with nothing to prove who a caller is, nobody's
  // word on who should pay is taken either (strut's own default is the same).
  if (!apiToken) return {};
  if (req.header("x-api-token") === apiToken) return grant(req.header("x-strut-actor"));
  const viaKey = isEmbedJwt(req.key);
  if (viaKey) return grant(viaKey.sub);
  const header = req.header("authorization") ?? "";
  if (header.startsWith("Bearer ")) {
    // A peer token is long-lived and held by a machine, so it is a header
    // only: never `?key=`, where it would sit in a URL.
    const viaBearer = isEmbedJwt(header.slice(7).trim(), ["api", "lab:peer"]);
    if (viaBearer?.scope === "lab:peer") {
      // A `sub` names a PERSON: the actor is the token's own. Without one
      // the token stands for a strut, whose run names its person by header.
      if (viaBearer.sub) return { ...grant(viaBearer.sub), scope: "peer:person" };
      return { ...grant(req.header("x-strut-actor")), scope: "peer" };
    }
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
 *
 * A fourth, for ANOTHER STRUT: a `lab:peer` JWT (`/mint-token { scope:
 * "lab:peer" }`), as Bearer only. Its request is stashed as strut's `peer`
 * scope (`resolveScope`), and strut answers it as one — every read, a
 * launch, control of a run a peer launched, 403 for the rest (strut
 * plans/federation.md §3). Its actor is its `sub`, else the caller's
 * `x-strut-actor` — the person the calling strut's run is for; with a `sub`
 * the token is a PERSON's and its scope `peer:person` (strut
 * plans/chained-orchestrators.md §2).
 *
 * One kind of request passes with none of the three: a read carrying one
 * of strut's own read tokens as `?t=` — a FILE TOKEN on a run's artifacts
 * or a job's files, a CATALOG TOKEN on the published catalog and its
 * archives (`isTokenRead`) — strut judges that one.
 */
export function labAuth(req: Request, res: Response, next: NextFunction): void {
  if (isUiAsset(req) || isTokenRead(req)) return next();
  const key = typeof req.query.key === "string" ? req.query.key : null;
  const granted = labAuthorized({ header: (name) => req.header(name), key });
  if (granted) {
    stashLabActor(req, granted.actor);
    stashLabScope(req, granted.scope);
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

/**
 * A GET/HEAD of one run's artifacts (`/artifacts/:runId[/…]`) or one job's
 * files (`/jobs/:id/files[/…]`) carrying strut's FILE TOKEN as `?t=` (strut
 * `src/auth.ts`). Handed straight to strut, which refuses any `?t=` it
 * cannot vouch for whatever its gate's mode: the lab strut has no API key,
 * but signs under STRUT_SECRET_KEY (index.ts derives it from API_TOKEN), so
 * a token is as good as the credential that fetched the listing it came
 * from, and worth only that run's files. The strut UI's artifact links
 * carry this instead of the embed JWT — an HTML artifact read by its token
 * runs script under `sandbox allow-scripts`, and a page can read its own
 * URL, so the JWT must never be in one.
 *
 * And a read of this lab's published package CATALOG (`/packages/catalog`)
 * or one of its archives (`/packages/:publisher/:name/archive`) carrying a
 * CATALOG TOKEN (strut plans/marketplace.md §12.1): what a workspace's
 * strut that subscribes to the org's central strut holds — hive mints it
 * here with `POST /publisher/tokens` — worth the packages and nothing else.
 * Or the same two routes marked `?public=1` (§13, "One publisher, two
 * keys"): the PUBLIC catalog, sealed by the publisher's second key, and the
 * archives it lists — open to anyone, and strut serves nothing but what that
 * card lists whatever its gate's mode.
 *
 * The path is parsed as `isUiAsset` parses it, so a path that resolves
 * elsewhere does not pass.
 */
function isTokenRead(req: Request): boolean {
  if (req.method !== "GET" && req.method !== "HEAD") return false;
  try {
    const url = new URL(`http://localhost${req.url}`);
    const packages = /^\/packages\/(?:catalog|[^/]+\/[^/]+\/archive)$/.test(url.pathname);
    if (packages && url.searchParams.has("public")) return true;
    if (!url.searchParams.has("t")) return false;
    return /^\/artifacts\/[^/]+(\/|$)/.test(url.pathname) || /^\/jobs\/[^/]+\/files(\/|$)/.test(url.pathname) || packages;
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
const LAB_ROOMS = "/lab/rooms";

const rejectUpgrade = (socket: Duplex, status: string) => {
  socket.write(`HTTP/1.1 ${status}\r\nConnection: close\r\n\r\n`);
  socket.destroy();
};

/**
 * A ROOM over `/lab/rooms` (strut `src/rooms.ts`, plans/chained-orchestrators.md
 * §3): the socket a person's laptop holds open here to hear the runs it
 * launched — `run.end` and `job.update`, a launch's `callback` / `jobCallback`
 * `{ room: true }`. The lab credential is applied and stashed as `labAuth`
 * does for HTTP, then the socket goes to strut's `upgrade`, whose own gate
 * and hooks (`resolveLabActor` / `resolveLabScope` off `env.incoming`)
 * decide the room: the actor of a full credential, or a `peer:person`
 * token's `sub` — never a strut's `lab:peer` token, whose person is a header.
 */
export function upgradeLabRoom(
  req: IncomingMessage,
  socket: Duplex,
  head: Buffer,
  strut: () => Promise<Pick<Strut, "upgrade">> = labStrut,
): void {
  const url = new URL(req.url ?? "/", "http://localhost");
  const header = (name: string) => req.headers[name.toLowerCase()] as string | undefined;
  const granted = labAuthorized({ header, key: url.searchParams.get("key") });
  if (!granted) return rejectUpgrade(socket, "401 Unauthorized");
  stashLabActor(req, granted.actor);
  stashLabScope(req, granted.scope);
  strut()
    .then((s) => {
      if (socket.destroyed) return;
      if (!s.upgrade(req, socket, head, { basePath: "/lab" })) rejectUpgrade(socket, "404 Not Found");
    })
    .catch((e) => {
      console.error("[lab] room upgrade failed:", e);
      rejectUpgrade(socket, "500 Internal Server Error");
    });
}

/**
 * The lab's WebSockets. A WebSocket upgrade never enters Express, so the
 * bridge above can't carry it: hook the Node server's `upgrade` event.
 * `/lab/rooms` → `upgradeLabRoom`. Dictation over `/lab/audio/stream`
 * (strut `src/audio/ws.ts`): apply the lab credential (browsers resend
 * cached Basic auth on same-origin handshakes, so the UI's one-time prompt
 * covers it), then hand the socket to strut. Built lazily like the bridge —
 * the first upgrade boots the lab strut if a request hasn't already. Other
 * upgrade paths get a 404 rather than a socket left hanging. Dictation takes
 * no actor: it makes no LLM call and launches nothing.
 */
export function attachLabSockets(server: Server): void {
  let handlerP: Promise<AudioUpgradeHandler | null> | null = null;
  const handler = () =>
    (handlerP ??= labStrut().then((strut) =>
      strut.stt
        ? createAudioUpgradeHandler(strut.stt, { basePath: "/lab", authorize: () => true })
        : null,
    ));
  const reject = rejectUpgrade;
  server.on("upgrade", (req: IncomingMessage, socket: Duplex, head: Buffer) => {
    const url = new URL(req.url ?? "/", "http://localhost");
    if (url.pathname === LAB_ROOMS) return upgradeLabRoom(req, socket, head);
    if (url.pathname !== LAB_AUDIO_STREAM) return reject(socket, "404 Not Found");
    const header = (name: string) => req.headers[name.toLowerCase()] as string | undefined;
    const granted = labAuthorized({ header, key: url.searchParams.get("key") });
    if (!granted) return reject(socket, "401 Unauthorized");
    // Dictation is not what a peer is for (read, launch, control its runs).
    if (granted.scope) return reject(socket, "403 Forbidden");
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
