import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";
import express from "express";
import { getRequestListener } from "@hono/node-server";
import { labAuth } from "./mount.js";
import { labActorOf, resolveLabActor } from "./actor.js";
import { signApiToken, signEventsToken, verifyApiToken } from "../repo/events.js";

const API_TOKEN = "test-api-token";

/** Raw request — `fetch` would normalize `..` out of the path client-side. */
function get(
  port: number,
  path: string,
  headers: Record<string, string> = {},
  method = "GET",
): Promise<{ status: number; body: string; headers: http.IncomingHttpHeaders }> {
  return new Promise((resolve, reject) => {
    const req = http.request({ port, path, method, headers }, (res) => {
      let body = "";
      res.on("data", (c) => (body += c));
      res.on("end", () => resolve({ status: res.statusCode ?? 0, body, headers: res.headers }));
    });
    req.on("error", reject);
    req.end();
  });
}

describe("labAuth", () => {
  let server: http.Server;
  let port: number;
  let prevToken: string | undefined;

  before(async () => {
    prevToken = process.env.API_TOKEN;
    process.env.API_TOKEN = API_TOKEN;
    const app = express();
    app.use("/lab", labAuth, (req, res) => {
      // What strut's `resolveActor` sees: the Hono bridge hands it the
      // Express `req` as `c.env.incoming` (see actor.ts).
      res.json({ url: req.url, actor: resolveLabActor({ env: { incoming: req } }) ?? null });
    });
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
    await new Promise((r) => server.close(r));
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
