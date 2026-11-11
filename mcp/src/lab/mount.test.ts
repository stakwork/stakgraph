import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomBytes } from "node:crypto";
import express from "express";
import { getRequestListener } from "@hono/node-server";
import { createStrut, MemoryRunStore, PUBLIC_POST, WorkspaceManager, type Strut } from "strut";
import { labAuth, retryable, upgradeLabChannel, upgradeLabRoom } from "./mount.js";
import { labActorOf, resolveLabActor, resolveLabScope } from "./actor.js";
import { mintToken, signApiToken, signEventsToken, verifyApiToken } from "../repo/events.js";

const API_TOKEN = "test-api-token";

type RoomMsg = Record<string, unknown>;

/** A raw WebSocket handshake on `/lab/rooms` (mcp has no `ws` of its own):
 *  the status it was refused with, or 101 and the text messages the server
 *  sends (unmasked frames, server → client). */
function openRoom(
  port: number,
  headers: Record<string, string>,
  path = "/lab/rooms",
): Promise<{ status: number; until: (pred: (m: RoomMsg) => boolean) => Promise<RoomMsg>; send: (m: RoomMsg) => void; close: () => void }> {
  return new Promise((resolve, reject) => {
    const req = http.request({
      port,
      path,
      headers: {
        connection: "Upgrade",
        upgrade: "websocket",
        "sec-websocket-version": "13",
        "sec-websocket-key": randomBytes(16).toString("base64"),
        ...headers,
      },
    });
    req.on("response", (res) => {
      res.resume();
      resolve({ status: res.statusCode ?? 0, until: () => Promise.reject(new Error("refused")), send() {}, close() {} });
    });
    req.on("upgrade", (_res, socket) => {
      const messages: RoomMsg[] = [];
      let buf = Buffer.alloc(0);
      socket.on("data", (chunk: Buffer) => {
        buf = Buffer.concat([buf, chunk]);
        while (buf.length >= 2) {
          const op = buf[0]! & 0x0f;
          let len = buf[1]! & 0x7f;
          let off = 2;
          if (len === 126) {
            if (buf.length < 4) break;
            len = buf.readUInt16BE(2);
            off = 4;
          } else if (len === 127) {
            if (buf.length < 10) break;
            len = Number(buf.readBigUInt64BE(2));
            off = 10;
          }
          if (buf.length < off + len) break;
          if (op === 1) messages.push(JSON.parse(buf.subarray(off, off + len).toString("utf8")) as RoomMsg);
          buf = buf.subarray(off + len);
        }
      });
      const until = async (pred: (m: RoomMsg) => boolean) => {
        for (let i = 0; i < 400; i++) {
          const m = messages.find(pred);
          if (m) return m;
          await new Promise((r) => setTimeout(r, 10));
        }
        throw new Error("the room never got it");
      };
      // A client frame is masked (RFC 6455); these are small text frames.
      const send = (m: RoomMsg) => {
        const payload = Buffer.from(JSON.stringify(m));
        const mask = randomBytes(4);
        const head = payload.length < 126 ? Buffer.from([0x81, 0x80 | payload.length]) : Buffer.from([0x81, 0x80 | 126, payload.length >> 8, payload.length & 0xff]);
        socket.write(Buffer.concat([head, mask, Buffer.from(payload.map((b, i) => b ^ mask[i % 4]!))]));
      };
      resolve({ status: 101, until, send, close: () => socket.destroy() });
    });
    req.on("error", reject);
    req.end();
  });
}

/** Raw request — `fetch` would normalize `..` out of the path client-side. */
function get(
  port: number,
  path: string,
  headers: Record<string, string> = {},
  method = "GET",
  body?: string,
): Promise<{ status: number; body: string; headers: http.IncomingHttpHeaders }> {
  return new Promise((resolve, reject) => {
    const req = http.request({ port, path, method, headers }, (res) => {
      let body = "";
      res.on("data", (c) => (body += c));
      res.on("end", () => resolve({ status: res.statusCode ?? 0, body, headers: res.headers }));
    });
    req.on("error", reject);
    req.end(body);
  });
}

describe("labAuth", () => {
  let server: http.Server;
  let port: number;
  let prevToken: string | undefined;
  let prevStrutKey: string | undefined;
  let prevSecretKey: string | undefined;
  let strut: Strut;
  let strutDir: string;

  before(async () => {
    prevToken = process.env.API_TOKEN;
    process.env.API_TOKEN = API_TOKEN;
    // The lab strut runs open behind labAuth: no key of its own.
    prevStrutKey = process.env.STRUT_API_KEY;
    delete process.env.STRUT_API_KEY;
    // ...but it signs its tokens (file tokens, channel tickets) under
    // STRUT_SECRET_KEY, which index.ts derives from API_TOKEN.
    prevSecretKey = process.env.STRUT_SECRET_KEY;
    process.env.STRUT_SECRET_KEY = API_TOKEN;
    const app = express();
    app.use("/lab", labAuth, (req, res) => {
      // What strut's `resolveActor` / `resolveScope` see: the Hono bridge
      // hands them the Express `req` as `c.env.incoming` (see actor.ts).
      const env = { env: { incoming: req } };
      res.json({ url: req.url, actor: resolveLabActor(env) ?? null, scope: resolveLabScope(env) ?? null });
    });
    // A real strut behind labAuth and the real bridge, wired as
    // createLabStrut wires the lab's: what strut does with a peer's request.
    strutDir = await mkdtemp(join(tmpdir(), "lab-peer-"));
    strut = await createStrut({
      workspace: new WorkspaceManager(strutDir),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
      // The lab's MCP doors are on (it keeps chat's default), among them a
      // public app's, `POST /public/mcp`.
      enableMcp: true,
      scheduler: false,
      autoResume: false,
      resolveActor: resolveLabActor,
      resolveScope: resolveLabScope,
    });
    // `peer: true`: strut lets a peer launch only a workflow that says so.
    await strut.workspace.publishWorkflow("hello", "v1", {
      peer: true,
      steps: [{ id: "a", type: "log", config: { message: "hi" } }],
    });
    // `peer: person`: only a token whose `sub` names a person launches it
    // (the orchestrator's posture, strut plans/chained-orchestrators.md §2).
    await strut.workspace.publishWorkflow("chat", "v1", {
      peer: "person",
      steps: [{ id: "a", type: "log", config: { message: "hi" } }],
    });
    app.use("/strut", labAuth, getRequestListener(strut.app.fetch));
    // The real bridge (mount.ts): @hono/node-server's listener, which hands
    // the fetch handler `{ incoming, outgoing }` as its env — the same
    // object `labAuth` stashed the actor on.
    app.use(
      "/bridged",
      labAuth,
      getRequestListener((_request: Request, env: unknown) =>
        Response.json({ actor: resolveLabActor({ env }) ?? null }),
      ),
    );
    server = app.listen(0);
    await new Promise((r) => server.once("listening", r));
    port = (server.address() as AddressInfo).port;
  });

  after(async () => {
    if (prevToken === undefined) delete process.env.API_TOKEN;
    else process.env.API_TOKEN = prevToken;
    if (prevStrutKey !== undefined) process.env.STRUT_API_KEY = prevStrutKey;
    if (prevSecretKey === undefined) delete process.env.STRUT_SECRET_KEY;
    else process.env.STRUT_SECRET_KEY = prevSecretKey;
    await new Promise((r) => server.close(r));
    await strut.close();
    await rm(strutDir, { recursive: true, force: true });
  });

  it("rejects a request with no credential, prompting for Basic", async () => {
    const res = await get(port, "/lab/workflows");
    assert.equal(res.status, 401);
    assert.match(res.headers["www-authenticate"] ?? "", /^Basic /);
  });

  it("does not Basic-challenge an embed whose JWT went bad (no login dialog in the iframe)", async () => {
    const expired = signApiToken(-10);
    const viaBearer = await get(port, "/lab/workflows", { authorization: `Bearer ${expired}` });
    assert.equal(viaBearer.status, 401);
    assert.equal(viaBearer.headers["www-authenticate"], undefined);
    const viaKey = await get(port, `/lab/?key=${expired}`);
    assert.equal(viaKey.status, 401);
    assert.equal(viaKey.headers["www-authenticate"], undefined);
  });

  it("accepts x-api-token", async () => {
    const res = await get(port, "/lab/workflows", { "x-api-token": API_TOKEN });
    assert.equal(res.status, 200);
  });

  it("accepts Basic admin:<API_TOKEN>", async () => {
    const basic = Buffer.from(`admin:${API_TOKEN}`).toString("base64");
    const res = await get(port, "/lab/workflows", { authorization: `Basic ${basic}` });
    assert.equal(res.status, 200);
  });

  it("accepts a mint-token JWT as ?key= (iframe document load)", async () => {
    const res = await get(port, `/lab/?key=${signApiToken("1h")}`);
    assert.equal(res.status, 200);
  });

  it("accepts a mint-token JWT as Bearer (the strut UI's fetches)", async () => {
    const res = await get(port, "/lab/workflows", {
      authorization: `Bearer ${signApiToken("1h")}`,
    });
    assert.equal(res.status, 200);
  });

  it("rejects the raw API_TOKEN as ?key= — only a JWT rides in a URL", async () => {
    const res = await get(port, `/lab/?key=${API_TOKEN}`);
    assert.equal(res.status, 401);
  });

  it("rejects an expired JWT", async () => {
    const res = await get(port, "/lab/workflows", {
      authorization: `Bearer ${signApiToken(-10)}`,
    });
    assert.equal(res.status, 401);
  });

  it("rejects a JWT of another scope (per-request events token)", async () => {
    const res = await get(port, `/lab/?key=${signEventsToken("req-1")}`);
    assert.equal(res.status, 401);
  });

  it("serves UI assets without a credential", async () => {
    const res = await get(port, "/lab/assets/index-abc123.js");
    assert.equal(res.status, 200);
  });

  it("only opens assets for GET/HEAD", async () => {
    const res = await get(port, "/lab/assets/index-abc123.js", {}, "POST");
    assert.equal(res.status, 401);
  });

  it("hands a file read, or a catalog read, carrying strut's own token through — strut judges it", async () => {
    for (const path of [
      "/lab/artifacts/123?t=abc",
      "/lab/artifacts/123/report/page.html?t=abc",
      "/lab/jobs/j-1/files?t=abc",
      "/lab/jobs/j-1/files/plan.md?t=abc",
      "/lab/packages/catalog?t=abc",
      "/lab/packages/acme/linear/archive?digest=sha256:0&t=abc",
    ]) {
      const res = await get(port, path);
      assert.equal(res.status, 200, path);
      assert.equal(JSON.parse(res.body).actor, null, path);
    }
    // A read only, of those routes only, with ?t= only — and never a path
    // that resolves elsewhere.
    for (const [path, method] of [
      ["/lab/artifacts/123/page.html", "GET"],
      ["/lab/artifacts/123/page.html?t=abc", "POST"],
      ["/lab/workflows?t=abc", "GET"],
      ["/lab/workflows/x/runs/1/events?t=abc", "GET"],
      ["/lab/artifacts/123/../../secrets?t=abc", "GET"],
      ["/lab/jobs/j-1/holds?t=abc", "GET"],
      ["/lab/packages?t=abc", "GET"],
      ["/lab/packages/catalog?t=abc", "POST"],
      ["/lab/packages/acme/linear?t=abc", "GET"],
      ["/lab/packages/acme/linear/archive/x?t=abc", "GET"],
      ["/lab/packages/../workflows?t=abc", "GET"],
      ["/lab/packages/catalog", "GET"],
    ] as const) {
      const res = await get(port, path, {}, method);
      assert.equal(res.status, 401, `${method} ${path}`);
    }
  });

  it("hands strut's public view to anyone: its reads, a preflight, a public app's door", async () => {
    for (const [path, method] of [
      ["/lab/public", "GET"],
      ["/lab/public/", "GET"],
      ["/lab/public/workflows", "GET"],
      ["/lab/public/workflows/x/runs/1/events", "GET"],
      ["/lab/public/apps/bench/document", "HEAD"],
      ["/lab/public/workflows", "OPTIONS"],
      ["/lab/public/mcp", "OPTIONS"],
      ["/lab/public/mcp", "POST"],
      // The visitor's page: strut's UI shell, opened on a public app.
      ["/lab/?app=bench&solo=1&public=1", "GET"],
      ["/lab/index.html?app=bench&solo=1&public=1", "HEAD"],
    ] as const) {
      const res = await get(port, path, {}, method);
      assert.equal(res.status, 200, `${method} ${path}`);
      if (method !== "HEAD") assert.equal(JSON.parse(res.body).actor, null, `${method} ${path}`);
    }
    // Reads only, under `/public` only, and strut's public POSTs (below) —
    // never the record itself (`/public-view`, a person's), never a path
    // that resolves elsewhere.
    for (const [path, method] of [
      ["/lab/public/../workflows", "GET"],
      ["/lab/public/%2e%2e/workflows", "GET"],
      ["/lab/public/.%2E/workflows", "GET"],
      ["/lab/public\\..\\workflows", "GET"],
      ["/lab/public%2F..%2Fworkflows", "GET"],
      ["/lab//public", "GET"],
      ["/lab/publicx", "GET"],
      ["/lab/public-view", "GET"],
      ["/lab/public-view", "PUT"],
      ["/lab/public-view/needs?app=bench", "GET"],
      ["/lab/public", "PUT"],
      ["/lab/public/workflows", "DELETE"],
      ["/lab/public/workflows", "POST"],
      ["/lab/public/other", "POST"],
      ["/lab/public/mcp/x", "POST"],
      ["/lab/public/mcp/../../workflows/hello/run", "POST"],
      // The UI shell only with `public=1`, only by GET/HEAD, and nothing
      // else carrying the flag.
      ["/lab/", "GET"],
      ["/lab/?app=bench&solo=1", "GET"],
      ["/lab/?public=0", "GET"],
      ["/lab/?public=1", "POST"],
      ["/lab/?public=1", "OPTIONS"],
      ["/lab/workflows?public=1", "GET"],
      ["/lab/apps?public=1", "GET"],
      ["/lab/x/index.html?public=1", "GET"],
    ] as const) {
      const res = await get(port, path, {}, method);
      assert.equal(res.status, 401, `${method} ${path}`);
    }
  });

  it("passes every POST strut's PUBLIC_POST matches, and no other, with no credential", async () => {
    // One example per route strut opens to anyone (strut plans/paid-runs.md
    // §3, §4, §6): a public app's door, buying credit and claiming it, a
    // public launch and its cancel, a runnable tool's call — a step type
    // may hold slashes.
    const open = [
      "/public/mcp",
      "/public/credit",
      "/public/credit/claim",
      "/public/workflows/x/run",
      "/public/workflows/x/runs/r-1/cancel",
      "/public/steps/x/run",
      "/public/steps/browser/open/run",
    ];
    for (const path of open) {
      // The gate follows strut's matcher: a route strut renames fails here.
      assert.ok(PUBLIC_POST.test(path), `strut's PUBLIC_POST no longer matches ${path}`);
      const res = await get(port, `/lab${path}`, {}, "POST");
      assert.equal(res.status, 200, `POST /lab${path}`);
      const seen = JSON.parse(res.body);
      assert.equal(seen.url, path, `POST /lab${path}`);
      assert.equal(seen.actor, null, `POST /lab${path}`);
      assert.equal(seen.scope, null, `POST /lab${path}`);
    }
    // And `llms.txt` (plans/paid-runs.md §10): what `GET /public` shows, as
    // a page. Read only, at the root only.
    for (const method of ["GET", "HEAD"]) {
      assert.equal((await get(port, "/lab/llms.txt", {}, method)).status, 200, `${method} /lab/llms.txt`);
    }
    for (const [path, method] of [
      // Look-alikes: one segment too many or too few, or a path that
      // resolves out of `/public`.
      ["/lab/public/workflows/x/run/extra", "POST"],
      ["/lab/public/workflows/run", "POST"],
      ["/lab/public/workflows/x/runs/r-1/cancel/x", "POST"],
      ["/lab/public/workflows/x/runs/r-1", "POST"],
      ["/lab/public/workflows/x/runs/cancel", "POST"],
      ["/lab/public/steps/run", "POST"],
      ["/lab/public/steps/x/run/x", "POST"],
      ["/lab/public/credit/claim/x", "POST"],
      ["/lab/public/credit/x", "POST"],
      ["/lab/public/workflows/x/../../../workflows/x/run", "POST"],
      ["/lab/public/%2e%2e/workflows/x/run", "POST"],
      ["/lab/public/credit?x=/public/mcp", "PUT"],
      ["/lab/public/workflows/x/run", "PUT"],
      ["/lab/public/credit", "DELETE"],
      // The private doors those mirror.
      ["/lab/workflows/x/run", "POST"],
      ["/lab/workflows/x/runs/r-1/cancel", "POST"],
      ["/lab/steps/x/run", "POST"],
      ["/lab/chat", "POST"],
      ["/lab/mcp", "POST"],
      ["/lab/public-view", "POST"],
      // llms.txt by GET/HEAD at the root alone.
      ["/lab/llms.txt", "POST"],
      ["/lab/llms.txt", "OPTIONS"],
      ["/lab/x/llms.txt", "GET"],
      ["/lab/llms.txt/../workflows", "GET"],
    ] as const) {
      const res = await get(port, path, {}, method);
      assert.equal(res.status, 401, `${method} ${path}`);
    }
  });

  it("strut answers an anonymous launch, credit purchase and tool call itself, through the real bridge", async () => {
    // Nothing is runnable or priced here: each is strut's own 404, never the
    // gate's 401.
    const json = { "content-type": "application/json" };
    for (const [path, body] of [
      ["/strut/public/credit", JSON.stringify({ usd: 1 })],
      ["/strut/public/credit/claim", JSON.stringify({ macaroon: "x" })],
      ["/strut/public/workflows/hello/run", JSON.stringify({ input: {} })],
      ["/strut/public/workflows/hello/runs/r-1/cancel", ""],
      ["/strut/public/steps/log/run", JSON.stringify({ config: { message: "hi" } })],
    ] as const) {
      const res = await get(port, path, json, "POST", body);
      assert.equal(res.status, 404, `POST ${path}: ${res.body}`);
      assert.notDeepEqual(JSON.parse(res.body), { error: "Unauthorized" }, `POST ${path}`);
    }
    // llms.txt with nothing public: strut's 404, not the gate's 401.
    assert.equal((await get(port, "/strut/llms.txt")).status, 404);
  });

  it("strut serves an anonymous visitor its public view alone, through the real bridge", async () => {
    // Nothing on the record: nothing served — `hello` is published, not public.
    const empty = await get(port, "/strut/public");
    assert.equal(empty.status, 200, empty.body);
    // `runnable`: what anyone may run, and for how much (plans/paid-runs.md §2).
    assert.deepEqual(JSON.parse(empty.body), { workflows: [], steps: [], concepts: [], types: [], apps: [], runnable: [] });
    assert.equal(empty.headers["access-control-allow-origin"], "*");
    assert.equal((await get(port, "/strut/public/workflows/hello")).status, 404);
    // An encoded slash is no separator, for the gate or for strut: it stays
    // under `/public`, and is not the private `/workflows`.
    assert.equal((await get(port, "/strut/public/..%2Fworkflows")).status, 404);
    // The record is a person's, behind the credential.
    assert.equal((await get(port, "/strut/public-view")).status, 401);
    const hive = { "x-api-token": API_TOKEN, "content-type": "application/json" };
    const put = (body: unknown) => get(port, "/strut/public-view", hive, "PUT", JSON.stringify(body));
    assert.equal((await put({ workflows: [{ name: "hello" }] })).status, 200);
    try {
      const hello = await get(port, "/strut/public/workflows/hello");
      assert.equal(hello.status, 200, hello.body);
      assert.equal(JSON.parse(hello.body).name, "hello");
      // Its runs were not made public with it.
      assert.equal((await get(port, "/strut/public/workflows/hello/runs")).status, 404);
      assert.equal((await get(port, "/strut/public/workflows/chat")).status, 404);
    } finally {
      assert.equal((await put({})).status, 200);
    }
    // A public app's door answers with no credential: an MCP server whose
    // one tool is `step_run`.
    const mcp = await get(
      port,
      "/strut/public/mcp",
      { "content-type": "application/json", accept: "application/json, text/event-stream" },
      "POST",
      JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/list" }),
    );
    assert.equal(mcp.status, 200, mcp.body);
    assert.match(mcp.body, /"step_run"/);
  });

  // ── The actor (plans/mothership-cost-control.md §5) ───────────────────

  it("a JWT's `sub` is the actor, via Bearer and via ?key=", async () => {
    const token = signApiToken("1h", "octocat-42");
    const viaBearer = await get(port, "/lab/workflows", { authorization: `Bearer ${token}` });
    assert.equal(viaBearer.status, 200);
    assert.equal(JSON.parse(viaBearer.body).actor, "octocat-42");
    const viaKey = await get(port, `/lab/?key=${token}`);
    assert.equal(viaKey.status, 200);
    assert.equal(JSON.parse(viaKey.body).actor, "octocat-42");
  });

  it("a JWT without `sub` names no actor", async () => {
    const res = await get(port, "/lab/workflows", {
      authorization: `Bearer ${signApiToken("1h")}`,
    });
    assert.equal(res.status, 200);
    assert.equal(JSON.parse(res.body).actor, null);
  });

  it("x-api-token trusts hive's x-strut-actor", async () => {
    const res = await get(port, "/lab/workflows", {
      "x-api-token": API_TOKEN,
      "x-strut-actor": "  octocat-42 ",
    });
    assert.equal(res.status, 200);
    assert.equal(JSON.parse(res.body).actor, "octocat-42");
  });

  it("x-strut-actor alone is not a credential", async () => {
    const res = await get(port, "/lab/workflows", { "x-strut-actor": "octocat-42" });
    assert.equal(res.status, 401);
  });

  it("x-strut-actor is ignored beside Basic auth and beside a JWT", async () => {
    const basic = Buffer.from(`admin:${API_TOKEN}`).toString("base64");
    const viaBasic = await get(port, "/lab/workflows", {
      authorization: `Basic ${basic}`,
      "x-strut-actor": "octocat-42",
    });
    assert.equal(viaBasic.status, 200);
    assert.equal(JSON.parse(viaBasic.body).actor, null);
    const viaJwt = await get(port, "/lab/workflows", {
      authorization: `Bearer ${signApiToken("1h")}`,
      "x-strut-actor": "octocat-42",
    });
    assert.equal(viaJwt.status, 200);
    assert.equal(JSON.parse(viaJwt.body).actor, null);
  });

  it("the actor reaches strut through the real Hono bridge (c.env.incoming)", async () => {
    const viaJwt = await get(port, "/bridged/workflows", {
      authorization: `Bearer ${signApiToken("1h", "octocat-42")}`,
    });
    assert.equal(viaJwt.status, 200);
    assert.equal(JSON.parse(viaJwt.body).actor, "octocat-42");
    const viaHive = await get(port, "/bridged/workflows", {
      "x-api-token": API_TOKEN,
      "x-strut-actor": "hive-7",
    });
    assert.equal(viaHive.status, 200);
    assert.equal(JSON.parse(viaHive.body).actor, "hive-7");
    const basic = Buffer.from(`admin:${API_TOKEN}`).toString("base64");
    const viaBasic = await get(port, "/bridged/workflows", { authorization: `Basic ${basic}` });
    assert.equal(viaBasic.status, 200);
    assert.equal(JSON.parse(viaBasic.body).actor, null);
  });

  it("the actor is the JWT's `sub` claim, round-tripped by signApiToken", () => {
    assert.equal(verifyApiToken(signApiToken("1h", "octocat-42")).sub, "octocat-42");
    assert.equal(verifyApiToken(signApiToken("1h")).sub, undefined);
    assert.equal(labActorOf(undefined), undefined);
    assert.equal(resolveLabActor({}), undefined);
  });

  // ── A peer: another strut (strut plans/federation.md §3) ──────────────

  it("a lab:peer JWT is a peer's, as Bearer only — never ?key=", async () => {
    const token = signApiToken("60d", undefined, "lab:peer");
    const res = await get(port, "/lab/workflows", { authorization: `Bearer ${token}` });
    assert.equal(res.status, 200);
    assert.equal(JSON.parse(res.body).scope, "peer");
    assert.equal((await get(port, `/lab/?key=${token}`)).status, 401);
    // Every other credential is strut's full scope, as before.
    const embed = await get(port, "/lab/workflows", { authorization: `Bearer ${signApiToken("1h")}` });
    assert.equal(JSON.parse(embed.body).scope, null);
    const hive = await get(port, "/lab/workflows", { "x-api-token": API_TOKEN });
    assert.equal(JSON.parse(hive.body).scope, null);
  });

  it("a peer's actor is its token's `sub`, else the caller's x-strut-actor", async () => {
    const named = signApiToken("60d", "octocat-42", "lab:peer");
    const viaSub = await get(port, "/lab/workflows", { authorization: `Bearer ${named}`, "x-strut-actor": "mallory-9" });
    assert.equal(JSON.parse(viaSub.body).actor, "octocat-42");
    const anyone = signApiToken("60d", undefined, "lab:peer");
    const viaHeader = await get(port, "/lab/workflows", { authorization: `Bearer ${anyone}`, "x-strut-actor": " alice-1 " });
    assert.equal(JSON.parse(viaHeader.body).actor, "alice-1");
    const nobody = await get(port, "/lab/workflows", { authorization: `Bearer ${anyone}` });
    assert.equal(JSON.parse(nobody.body).actor, null);
  });

  it("a lab:peer JWT with a `sub` is a PERSON's: `peer:person`, its actor the sub whatever the header says", async () => {
    const person = signApiToken("60d", "octocat-42", "lab:peer");
    const viaSub = JSON.parse((await get(port, "/lab/workflows", { authorization: `Bearer ${person}`, "x-strut-actor": "mallory-9" })).body);
    assert.deepEqual([viaSub.scope, viaSub.actor], ["peer:person", "octocat-42"]);
    // Without one it stands for a strut, naming its person by header.
    const strutToken = signApiToken("60d", undefined, "lab:peer");
    const viaHeader = JSON.parse((await get(port, "/lab/workflows", { authorization: `Bearer ${strutToken}`, "x-strut-actor": "octocat-42" })).body);
    assert.deepEqual([viaHeader.scope, viaHeader.actor], ["peer", "octocat-42"]);
  });

  it("strut opens a `peer: person` workflow to a person's token, and refuses a strut's naming the same person", async () => {
    const json = { "content-type": "application/json" };
    const person = { ...json, authorization: `Bearer ${signApiToken("60d", "octocat-42", "lab:peer")}` };
    const byStrut = { ...json, authorization: `Bearer ${signApiToken("60d", undefined, "lab:peer")}`, "x-strut-actor": "octocat-42" };
    const refused = await get(port, "/strut/workflows/chat/run", byStrut, "POST", "{}");
    assert.equal(refused.status, 403);
    assert.match(refused.body, /person's credential only/);
    const launched = await get(port, "/strut/workflows/chat/run", person, "POST", "{}");
    assert.equal(launched.status, 202, launched.body);
    const { runId } = JSON.parse(launched.body) as { runId: string };
    let start;
    for (let i = 0; i < 200 && !start; i++) {
      start = (await strut.store.getRunEvents("chat", runId)).find((e) => e.type === "run.start");
      if (!start) await new Promise((r) => setTimeout(r, 10));
    }
    assert.deepEqual([start?.origin, start?.peerPerson, start?.actor], ["peer", true, "octocat-42"]);
  });

  it("a room: /lab/rooms takes the lab credential and strut's hooks decide it — a person's token joins and hears its run, a strut's is refused", async () => {
    const sockets = http.createServer();
    sockets.on("upgrade", (req, socket, head) => upgradeLabRoom(req, socket, head, async () => strut));
    await new Promise<void>((r) => sockets.listen(0, r));
    const sp = (sockets.address() as AddressInfo).port;
    const person = `Bearer ${signApiToken("60d", "octocat-42", "lab:peer")}`;
    try {
      assert.equal((await openRoom(sp, {})).status, 401);
      const byStrut = await openRoom(sp, { authorization: `Bearer ${signApiToken("60d", undefined, "lab:peer")}`, "x-strut-actor": "octocat-42" });
      assert.equal(byStrut.status, 403);
      const room = await openRoom(sp, { authorization: person });
      assert.equal(room.status, 101);
      try {
        const launched = await get(
          port,
          "/strut/workflows/hello/run",
          { "content-type": "application/json", authorization: person },
          "POST",
          JSON.stringify({ callback: { room: true } }),
        );
        assert.equal(launched.status, 202, launched.body);
        const { runId } = JSON.parse(launched.body) as { runId: string };
        const end = await room.until((m) => m.event === "run.end" && m.runId === runId);
        assert.equal(end.status, "success");
      } finally {
        room.close();
      }
    } finally {
      await new Promise((r) => sockets.close(r));
    }
  });

  it("channels: /lab/channels — a ticket minted through the lab is the credential; neither, or a forged one, is a 401; a person's credential plus ?app= works, a strut's is refused", async () => {
    const sockets = http.createServer();
    sockets.on("upgrade", (req, socket, head) => upgradeLabChannel(req, socket, head, async () => strut));
    await new Promise<void>((r) => sockets.listen(0, r));
    const sp = (sockets.address() as AddressInfo).port;
    const published = await strut.workspace.publishApp("board", "hash-1", { title: "Board" } as never);
    const version = published.version;
    try {
      // Hive's shape: the swarm key + the member as x-strut-actor mints the ticket.
      const minted = await get(
        port,
        "/strut/apps/board/channel-ticket",
        { "content-type": "application/json", "x-api-token": API_TOKEN, "x-strut-actor": "octocat-42" },
        "POST",
        JSON.stringify({ version }),
      );
      assert.equal(minted.status, 200, minted.body);
      const { ticket } = JSON.parse(minted.body) as { ticket: string };
      assert.ok(ticket);
      // 1. The ticket alone, no credential: in, as the ticket's actor.
      const viaTicket = await openRoom(sp, {}, `/lab/channels?ticket=${encodeURIComponent(ticket)}`);
      assert.equal(viaTicket.status, 101);
      try {
        viaTicket.send({ op: "join", channel: "main" });
        const joined = await viaTicket.until((m) => m.op === "joined");
        const members = joined.members as Array<{ actor?: string }>;
        assert.deepEqual(members.map((m) => m.actor), ["octocat-42"]);
      } finally {
        viaTicket.close();
      }
      // 2. Neither a credential nor a ticket (nor an empty one).
      assert.equal((await openRoom(sp, {}, "/lab/channels")).status, 401);
      assert.equal((await openRoom(sp, {}, "/lab/channels?ticket=")).status, 401);
      // 3. A forged ticket is strut's 401: one well formed but unsigned,
      // naming someone else (what a strut with no secret would take), and
      // one that is not a ticket at all.
      const claims = Buffer.from(JSON.stringify({ actor: "mallory", app: "board", version, exp: Date.now() + 60_000 })).toString("base64url");
      assert.equal((await openRoom(sp, {}, `/lab/channels?ticket=${encodeURIComponent(`strut-channel.${claims}.`)}`)).status, 401);
      assert.equal((await openRoom(sp, {}, "/lab/channels?ticket=forged.ticket")).status, 401);
      // 4. A person's credential plus ?app=<name>@<version>: in, as the
      // token's `sub` (the peer relay's shape).
      const person = `Bearer ${signApiToken("60d", "hubot-7", "lab:peer")}`;
      const viaCred = await openRoom(sp, { authorization: person }, `/lab/channels?app=board@${version}`);
      assert.equal(viaCred.status, 101);
      try {
        viaCred.send({ op: "join", channel: "relay" });
        const joined = await viaCred.until((m) => m.op === "joined");
        assert.deepEqual((joined.members as Array<{ actor?: string }>).map((m) => m.actor), ["hubot-7"]);
      } finally {
        viaCred.close();
      }
      // 5. A strut's peer token (no `sub`) names its person by header: strut
      // refuses it a channel.
      const strutPeer = `Bearer ${signApiToken("60d", undefined, "lab:peer")}`;
      const refused = await openRoom(sp, { authorization: strutPeer, "x-strut-actor": "octocat-42" }, `/lab/channels?app=board@${version}`);
      assert.equal(refused.status, 403);
    } finally {
      await new Promise((r) => sockets.close(r));
    }
  });

  it("a lab:peer token opens nothing outside the lab", () => {
    const token = signApiToken("60d", undefined, "lab:peer");
    // mcp's own API (graph/routes.ts authMiddleware) verifies with the default.
    assert.throws(() => verifyApiToken(token), /scope/);
    assert.equal(verifyApiToken(token, ["api", "lab:peer"]).scope, "lab:peer");
  });

  it("strut answers the lab's peer as one, through the real bridge", async () => {
    const json = { "content-type": "application/json" };
    const peer = { ...json, authorization: `Bearer ${signApiToken("60d", "alice-1", "lab:peer")}` };
    const hive = { ...json, "x-api-token": API_TOKEN };
    assert.equal((await get(port, "/strut/workflows", peer)).status, 200);
    assert.equal((await get(port, "/strut/secrets/X", peer, "PUT", '{"value":"v"}')).status, 403);
    assert.equal((await get(port, "/strut/secrets/X", hive, "PUT", '{"value":"v"}')).status, 200);

    // A launch passes, stamped as a peer's and billed to the token's person.
    const launched = await get(port, "/strut/workflows/hello/run", peer, "POST", "{}");
    assert.equal(launched.status, 202, launched.body);
    const { runId } = JSON.parse(launched.body) as { runId: string };
    let start;
    for (let i = 0; i < 200 && !start; i++) {
      start = (await strut.store.getRunEvents("hello", runId)).find((e) => e.type === "run.start");
      if (!start) await new Promise((r) => setTimeout(r, 10));
    }
    assert.equal(start?.origin, "peer");
    assert.equal(start?.actor, "alice-1");

    // It controls the run it launched, and no other.
    const mine = await get(port, `/strut/workflows/hello/runs/${runId}/cancel`, peer, "POST");
    assert.notEqual(mine.status, 403, mine.body);
    const theirs = JSON.parse((await get(port, "/strut/workflows/hello/run", hive, "POST", "{}")).body).runId;
    assert.equal((await get(port, `/strut/workflows/hello/runs/${theirs}/cancel`, peer, "POST")).status, 403);
  });

  // Hono resolves dot segments when it builds the request URL, so each of
  // these would be ROUTED as /secrets — the gate must see them the same way.
  for (const path of [
    "/lab/assets/../secrets",
    "/lab/assets/%2e%2e/secrets",
    "/lab/assets/.%2E/secrets",
    "/lab/assets\\..\\secrets",
    "/lab//assets/../secrets",
  ]) {
    it(`does not open ${path}`, async () => {
      const res = await get(port, path);
      assert.equal(res.status, 401);
    });
  }
});

describe("mintToken", () => {
  let server: http.Server;
  let port: number;
  let prevToken: string | undefined;

  before(async () => {
    prevToken = process.env.API_TOKEN;
    process.env.API_TOKEN = API_TOKEN;
    const app = express();
    app.use(express.json());
    app.post("/mint-token", mintToken);
    server = app.listen(0);
    await new Promise((r) => server.once("listening", r));
    port = (server.address() as AddressInfo).port;
  });

  after(async () => {
    if (prevToken === undefined) delete process.env.API_TOKEN;
    else process.env.API_TOKEN = prevToken;
    await new Promise((r) => server.close(r));
  });

  const mint = (body: unknown, headers: Record<string, string> = { "x-api-token": API_TOKEN }) =>
    get(port, "/mint-token", { "content-type": "application/json", ...headers }, "POST", JSON.stringify(body));

  it("mints an embed's token by default: scope api, an hour", async () => {
    const res = await mint({ sub: "octocat-42" });
    assert.equal(res.status, 200);
    const body = JSON.parse(res.body);
    assert.equal(body.scope, "api");
    assert.equal(body.expires_in, "1h");
    const claims = verifyApiToken(body.token);
    assert.equal(claims.sub, "octocat-42");
    assert.equal(claims.exp! - claims.iat!, 3600);
  });

  it("mints a peer's token: scope lab:peer, sixty days unless told otherwise", async () => {
    const body = JSON.parse((await mint({ scope: "lab:peer", sub: "octocat-42" })).body);
    assert.equal(body.scope, "lab:peer");
    assert.equal(body.expires_in, "60d");
    const claims = verifyApiToken(body.token, ["lab:peer"]);
    assert.equal(claims.sub, "octocat-42");
    assert.equal(claims.exp! - claims.iat!, 60 * 86400);
    const short = JSON.parse((await mint({ scope: "lab:peer", expires_in: "1d" })).body);
    const shortClaims = verifyApiToken(short.token, ["lab:peer"]);
    assert.equal(shortClaims.exp! - shortClaims.iat!, 86400);
  });

  it("refuses an unknown scope, and anyone without the raw API_TOKEN", async () => {
    assert.equal((await mint({ scope: "admin" })).status, 400);
    const jwt = signApiToken("1h");
    assert.equal((await mint({}, { authorization: `Bearer ${jwt}` })).status, 401);
    const peerJwt = signApiToken("60d", undefined, "lab:peer");
    assert.equal((await mint({ scope: "lab:peer" }, { authorization: `Bearer ${peerJwt}` })).status, 401, "a peer token never renews itself");
  });
});

describe("retryable", () => {
  it("makes the promise once and shares it, pending and resolved", async () => {
    let made = 0;
    let resolve!: (v: string) => void;
    const get = retryable(() => {
      made++;
      return new Promise<string>((r) => (resolve = r));
    });
    const a = get();
    const b = get();
    assert.equal(a, b, "callers while it is pending share one attempt");
    resolve("lab");
    assert.equal(await a, "lab");
    assert.equal(await get(), "lab");
    assert.equal(made, 1);
  });

  it("forgets a rejection: the next call makes it again (a lab boot that failed while Neo4j was down)", async () => {
    let made = 0;
    let neo4jUp = false;
    const get = retryable(async () => {
      made++;
      if (!neo4jUp) throw new Error("The database has encountered a critical error");
      return "lab";
    });
    const first = get();
    assert.equal(get(), first, "one attempt in flight at a time");
    await assert.rejects(first, /critical error/);
    await assert.rejects(get(), /critical error/);
    neo4jUp = true;
    assert.equal(await get(), "lab");
    assert.equal(await get(), "lab");
    assert.equal(made, 3, "two failed attempts, one that held");
  });

  it("a caller retrying from its own catch gets a new attempt", async () => {
    let made = 0;
    const get = retryable(async () => {
      if (made++ === 0) throw new Error("down");
      return "up";
    });
    const retried = await get().catch(() => get());
    assert.equal(retried, "up");
  });
});
