import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { Hono } from "hono";
import { requireApiKey, _resetAuthState } from "./auth.js";

/**
 * Tests for the deployment-scoped shared-secret middleware.
 *
 * We attach `requireApiKey` to a minimal Hono app rather than importing the
 * real `server.ts`, which has module-level state (workspace, registry) we
 * don't want polluting the dev workspace.
 */
function buildApp(): Hono {
  const app = new Hono();
  app.post("/steps", requireApiKey, (c) => c.json({ ok: true, hit: "post" }));
  app.delete("/steps", requireApiKey, (c) => c.json({ ok: true, hit: "delete-bulk" }));
  app.delete("/steps/:name{.+}", requireApiKey, (c) =>
    c.json({ ok: true, hit: "delete-one", name: c.req.param("name") }),
  );
  app.get("/steps", (c) => c.json({ ok: true, hit: "get" })); // unprotected
  return app;
}

describe("requireApiKey middleware", () => {
  const originalKey = process.env["VEIN_API_KEY"];

  beforeEach(() => {
    _resetAuthState();
    delete process.env["VEIN_API_KEY"];
  });

  afterEach(() => {
    if (originalKey === undefined) delete process.env["VEIN_API_KEY"];
    else process.env["VEIN_API_KEY"] = originalKey;
    _resetAuthState();
  });

  // ── Permissive / dev mode ────────────────────────────────────────────────

  describe("when VEIN_API_KEY is unset", () => {
    it("allows POST /steps without any auth header", async () => {
      const app = buildApp();
      const res = await app.request("/steps", { method: "POST" });
      assert.equal(res.status, 200);
      const body = (await res.json()) as { hit: string };
      assert.equal(body.hit, "post");
    });

    it("allows DELETE /steps without any auth header", async () => {
      const app = buildApp();
      const res = await app.request("/steps", { method: "DELETE" });
      assert.equal(res.status, 200);
    });

    it("allows DELETE /steps/:name without any auth header", async () => {
      const app = buildApp();
      const res = await app.request("/steps/gitree/save-pr", { method: "DELETE" });
      assert.equal(res.status, 200);
      const body = (await res.json()) as { name: string };
      assert.equal(body.name, "gitree/save-pr");
    });

    it("treats empty-string VEIN_API_KEY the same as unset", async () => {
      process.env["VEIN_API_KEY"] = "";
      const app = buildApp();
      const res = await app.request("/steps", { method: "POST" });
      assert.equal(res.status, 200);
    });
  });

  // ── Enforced ─────────────────────────────────────────────────────────────

  describe("when VEIN_API_KEY is set", () => {
    const KEY = "k_test_abc_123";

    beforeEach(() => {
      process.env["VEIN_API_KEY"] = KEY;
    });

    it("accepts a request with the matching bearer token", async () => {
      const app = buildApp();
      const res = await app.request("/steps", {
        method: "POST",
        headers: { authorization: `Bearer ${KEY}` },
      });
      assert.equal(res.status, 200);
    });

    it("accepts case-insensitive 'Bearer' scheme", async () => {
      const app = buildApp();
      const res = await app.request("/steps", {
        method: "POST",
        headers: { authorization: `bearer ${KEY}` },
      });
      assert.equal(res.status, 200);
    });

    it("rejects a request with no Authorization header", async () => {
      const app = buildApp();
      const res = await app.request("/steps", { method: "POST" });
      assert.equal(res.status, 401);
      const body = (await res.json()) as { error: string };
      assert.ok(body.error.toLowerCase().includes("unauthorized"));
    });

    it("rejects a request with a wrong bearer token", async () => {
      const app = buildApp();
      const res = await app.request("/steps", {
        method: "POST",
        headers: { authorization: "Bearer wrong-key" },
      });
      assert.equal(res.status, 401);
    });

    it("rejects a request that omits the Bearer scheme", async () => {
      const app = buildApp();
      const res = await app.request("/steps", {
        method: "POST",
        headers: { authorization: KEY },
      });
      assert.equal(res.status, 401);
    });

    it("rejects an empty Bearer token", async () => {
      const app = buildApp();
      const res = await app.request("/steps", {
        method: "POST",
        headers: { authorization: "Bearer " },
      });
      assert.equal(res.status, 401);
    });

    it("gates DELETE /steps the same way", async () => {
      const app = buildApp();

      const bad = await app.request("/steps", { method: "DELETE" });
      assert.equal(bad.status, 401);

      const good = await app.request("/steps?publisher=mcp", {
        method: "DELETE",
        headers: { authorization: `Bearer ${KEY}` },
      });
      assert.equal(good.status, 200);
    });

    it("gates DELETE /steps/:name the same way", async () => {
      const app = buildApp();

      const bad = await app.request("/steps/gitree/save-pr", {
        method: "DELETE",
      });
      assert.equal(bad.status, 401);

      const good = await app.request("/steps/gitree/save-pr", {
        method: "DELETE",
        headers: { authorization: `Bearer ${KEY}` },
      });
      assert.equal(good.status, 200);
    });

    it("does NOT gate GET /steps (reads remain public)", async () => {
      const app = buildApp();
      const res = await app.request("/steps", { method: "GET" });
      assert.equal(res.status, 200);
    });
  });
});
