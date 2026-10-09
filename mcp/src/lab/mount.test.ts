import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import express from "express";
import { getRequestListener } from "@hono/node-server";
import { createStrut, MemoryRunStore, WorkspaceManager, type Strut } from "strut";
import { labAuth } from "./mount.js";
import { labActorOf, resolveLabActor, resolveLabScope } from "./actor.js";
import { mintToken, signApiToken, signEventsToken, verifyApiToken } from "../repo/events.js";

const API_TOKEN = "test-api-token";

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
  let strut: Strut;
  let strutDir: string;

  before(async () => {
    prevToken = process.env.API_TOKEN;
    process.env.API_TOKEN = API_TOKEN;
    // The lab strut runs open behind labAuth: no key of its own.
    prevStrutKey = process.env.STRUT_API_KEY;
    delete process.env.STRUT_API_KEY;
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
      scheduler: false,
      autoResume: false,
      resolveActor: resolveLabActor,
      resolveScope: resolveLabScope,
    });
    await strut.workspace.publishWorkflow("hello", "v1", {
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

  it("hands a run's or a job's file read carrying strut's file token through — strut judges it", async () => {
    for (const path of [
      "/lab/artifacts/123?t=abc",
      "/lab/artifacts/123/report/page.html?t=abc",
      "/lab/jobs/j-1/files?t=abc",
      "/lab/jobs/j-1/files/plan.md?t=abc",
    ]) {
      const res = await get(port, path);
      assert.equal(res.status, 200, path);
      assert.equal(JSON.parse(res.body).actor, null, path);
    }
    // A read only, under those two scopes only, with ?t= only — and never
    // a path that resolves elsewhere.
    for (const [path, method] of [
      ["/lab/artifacts/123/page.html", "GET"],
      ["/lab/artifacts/123/page.html?t=abc", "POST"],
      ["/lab/workflows?t=abc", "GET"],
      ["/lab/workflows/x/runs/1/events?t=abc", "GET"],
      ["/lab/artifacts/123/../../secrets?t=abc", "GET"],
      ["/lab/jobs/j-1/holds?t=abc", "GET"],
    ] as const) {
      const res = await get(port, path, {}, method);
      assert.equal(res.status, 401, `${method} ${path}`);
    }
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
