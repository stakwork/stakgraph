import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdir, rm, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";

import {
  FileSecretStore,
  MemorySecretStore,
  isValidSecretName,
} from "./secret-store.js";
import { secretsCapability } from "./capabilities.js";
import { createVein } from "./createVein.js";
import { WorkspaceManager } from "./workspace.js";
import { MemoryRunStore } from "./store.js";

// ── name validation ─────────────────────────────────────────────────────────

describe("isValidSecretName", () => {
  it("accepts env-var-style names", () => {
    assert.ok(isValidSecretName("GITHUB_TOKEN"));
    assert.ok(isValidSecretName("_x"));
    assert.ok(isValidSecretName("a1_b2"));
  });
  it("rejects invalid names", () => {
    assert.ok(!isValidSecretName("1leading"));
    assert.ok(!isValidSecretName("has-dash"));
    assert.ok(!isValidSecretName("has space"));
    assert.ok(!isValidSecretName("dot.name"));
    assert.ok(!isValidSecretName(""));
  });
});

// ── MemorySecretStore ───────────────────────────────────────────────────────

describe("MemorySecretStore", () => {
  it("set / get / delete / list round-trips", async () => {
    const s = new MemorySecretStore();
    assert.equal(await s.get("A"), undefined);

    await s.set("A", "secret-a");
    await s.set("B", "secret-b");
    assert.equal(await s.get("A"), "secret-a");

    const list = await s.list();
    assert.deepEqual(
      list.map((x) => x.name),
      ["A", "B"],
    );
    // list never leaks values
    assert.ok(!JSON.stringify(list).includes("secret-a"));

    assert.equal(await s.delete("A"), true);
    assert.equal(await s.delete("A"), false);
    assert.equal(await s.get("A"), undefined);
  });

  it("preserves createdAt across overwrite, bumps updatedAt", async () => {
    const s = new MemorySecretStore();
    await s.set("K", "v1");
    const [first] = await s.list();
    await new Promise((r) => setTimeout(r, 5));
    await s.set("K", "v2");
    const [second] = await s.list();
    assert.equal(await s.get("K"), "v2");
    assert.equal(first.createdAt, second.createdAt);
    assert.notEqual(first.updatedAt, second.updatedAt);
  });

  it("rejects invalid names on set", async () => {
    const s = new MemorySecretStore();
    await assert.rejects(() => s.set("bad-name", "x"), /invalid secret name/);
  });
});

// ── FileSecretStore ─────────────────────────────────────────────────────────

describe("FileSecretStore", () => {
  let tempDir: string;
  beforeEach(async () => {
    tempDir = join(tmpdir(), `vein-secrets-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
  });
  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("persists encrypted; plaintext never hits disk; round-trips across instances", async () => {
    const a = new FileSecretStore(tempDir);
    await a.set("API_KEY", "super-secret-value");

    // A fresh instance reads the same persisted value (decrypts correctly).
    const b = new FileSecretStore(tempDir);
    assert.equal(await b.get("API_KEY"), "super-secret-value");

    // The on-disk file must NOT contain the plaintext.
    const raw = await readFile(join(tempDir, "secrets.json"), "utf-8");
    assert.ok(!raw.includes("super-secret-value"), "value must be encrypted at rest");
    assert.ok(raw.includes("API_KEY"), "name is stored in clear (it's not the secret)");
  });

  it("list returns names + metadata only, no values", async () => {
    const s = new FileSecretStore(tempDir);
    await s.set("TOK", "abc123");
    const list = await s.list();
    assert.equal(list.length, 1);
    assert.equal(list[0].name, "TOK");
    assert.ok(!JSON.stringify(list).includes("abc123"));
  });
});

// ── secretsCapability over a store ──────────────────────────────────────────

describe("secretsCapability (store-backed)", () => {
  it("reads store first, falls back to env", async () => {
    const store = new MemorySecretStore();
    await store.set("FROM_STORE", "store-val");
    const cap = secretsCapability(store, {
      envFallback: { FROM_ENV: "env-val", FROM_STORE: "should-not-win" },
    });
    assert.equal(await cap.get("FROM_STORE"), "store-val");
    assert.equal(await cap.get("FROM_ENV"), "env-val");
    assert.equal(await cap.get("MISSING"), undefined);
  });

  it("still supports a flat source (back-compat)", async () => {
    const cap = secretsCapability({ X: "1" });
    assert.equal(await cap.get("X"), "1");
    assert.equal(await cap.get("Y"), undefined);
  });
});

// ── /secrets endpoints ──────────────────────────────────────────────────────

describe("/secrets endpoints", () => {
  let tempDir: string;
  beforeEach(async () => {
    tempDir = join(tmpdir(), `vein-secrets-api-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
  });
  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  async function makeVein() {
    return createVein({
      workspace: new WorkspaceManager(tempDir),
      store: new MemoryRunStore(),
      secretStore: new MemorySecretStore(),
      serveUi: false,
      enableChat: false,
    });
  }

  it("PUT then GET returns names only (never the value)", async () => {
    const vein = await makeVein();

    const put = await vein.app.request("/secrets/GITHUB_TOKEN", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ value: "ghp_topsecret" }),
    });
    assert.equal(put.status, 200);

    const get = await vein.app.request("/secrets");
    assert.equal(get.status, 200);
    const body = (await get.json()) as { secrets: { name: string }[] };
    assert.deepEqual(
      body.secrets.map((s) => s.name),
      ["GITHUB_TOKEN"],
    );
    assert.ok(!JSON.stringify(body).includes("ghp_topsecret"));
  });

  it("the stored value is readable by the secrets capability", async () => {
    const vein = await makeVein();
    await vein.app.request("/secrets/MY_KEY", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ value: "v-123" }),
    });
    const svc = vein.services as { secrets: { get(n: string): Promise<string | undefined> } };
    assert.equal(await svc.secrets.get("MY_KEY"), "v-123");
  });

  it("rejects an invalid name (400) and a missing value (400)", async () => {
    const vein = await makeVein();

    const badName = await vein.app.request("/secrets/bad-name", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ value: "x" }),
    });
    assert.equal(badName.status, 400);

    const noValue = await vein.app.request("/secrets/GOOD", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({}),
    });
    assert.equal(noValue.status, 400);
  });

  it("DELETE removes a secret; deleting a missing one is 404", async () => {
    const vein = await makeVein();
    await vein.app.request("/secrets/TMP", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ value: "x" }),
    });

    const del = await vein.app.request("/secrets/TMP", { method: "DELETE" });
    assert.equal(del.status, 200);

    const again = await vein.app.request("/secrets/TMP", { method: "DELETE" });
    assert.equal(again.status, 404);
  });

  it("returns 501 when the consumer injected their own secrets capability", async () => {
    const vein = await createVein({
      workspace: new WorkspaceManager(tempDir),
      store: new MemoryRunStore(),
      services: { secrets: { async get() { return "injected"; } } },
      serveUi: false,
      enableChat: false,
    });
    const get = await vein.app.request("/secrets");
    assert.equal(get.status, 501);
  });
});
