import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";
import express from "express";
import { labAuth } from "./mount.js";
import { signApiToken, signEventsToken } from "../repo/events.js";

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
      res.json({ url: req.url });
    });
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
