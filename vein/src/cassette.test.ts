import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  httpCapability,
  secretsCapability,
  standardServices,
  type FetchLike,
} from "./capabilities.js";
import {
  withCassette,
  emptyCassette,
  type Cassette,
} from "./cassette.js";

// ── a fake fetch that records calls and returns canned responses ───────────

function fakeFetch(
  handler: (url: string, init: any) => { status?: number; headers?: Record<string, string>; body: unknown },
): FetchLike & { calls: Array<{ url: string; init: any }> } {
  const calls: Array<{ url: string; init: any }> = [];
  const fn = (async (url: string, init: any = {}) => {
    calls.push({ url, init });
    const r = handler(url, init);
    const headers = r.headers ?? { "content-type": "application/json" };
    const text = typeof r.body === "string" ? r.body : JSON.stringify(r.body);
    return {
      status: r.status ?? 200,
      ok: (r.status ?? 200) < 400,
      headers: {
        forEach(cb: (value: string, key: string) => void) {
          for (const [k, v] of Object.entries(headers)) cb(v, k);
        },
      },
      async text() {
        return text;
      },
    };
  }) as any;
  fn.calls = calls;
  return fn;
}

// ── capabilities ───────────────────────────────────────────────────────────

describe("httpCapability", () => {
  it("issues a GET and parses a JSON response into a plain object", async () => {
    const f = fakeFetch((url) => {
      assert.equal(url, "https://api.example.com/things?limit=2");
      return { body: { data: [{ id: "a" }, { id: "b" }] } };
    });
    const http = httpCapability(f);
    const res = await http("https://api.example.com/things", { query: { limit: 2 } });
    assert.equal(res.status, 200);
    assert.equal(res.ok, true);
    assert.deepEqual(res.body, { data: [{ id: "a" }, { id: "b" }] });
    assert.equal(f.calls[0]!.init.method, "GET");
  });

  it("JSON-encodes object bodies and defaults to POST", async () => {
    const f = fakeFetch((_url, init) => {
      assert.equal(init.method, "POST");
      assert.equal(init.headers["content-type"], "application/json");
      assert.deepEqual(JSON.parse(init.body), { name: "x" });
      return { body: { ok: true } };
    });
    const http = httpCapability(f);
    const res = await http("https://api.example.com/things", { body: { name: "x" } });
    assert.deepEqual(res.body, { ok: true });
  });
});

describe("secretsCapability", () => {
  it("reads from the provided source", async () => {
    const s = secretsCapability({ STRIPE_KEY: "sk_live_secret" });
    assert.equal(await s.get("STRIPE_KEY"), "sk_live_secret");
    assert.equal(await s.get("MISSING"), undefined);
  });
});

// ── cassette: record then replay ───────────────────────────────────────────

describe("withCassette record/replay", () => {
  function stripeServices(fetchImpl: FetchLike) {
    return standardServices({
      fetchImpl,
      secretsSource: { STRIPE_KEY: "sk_live_supersecret" },
    });
  }

  // The adapter logic under test — calls secrets + http through services.
  async function listCharges(services: ReturnType<typeof stripeServices>, customer: string) {
    const key = await services.secrets.get("STRIPE_KEY");
    const res = await services.http("https://api.stripe.com/v1/charges", {
      query: { customer },
      headers: { authorization: `Bearer ${key}` },
    });
    return (res.body as { data: Array<{ id: string }> }).data.map((c) => c.id);
  }

  it("records a live call, then replays it offline without hitting fetch", async () => {
    const cassette = emptyCassette();

    // 1. RECORD against the "real" API.
    const liveFetch = fakeFetch(() => ({ body: { data: [{ id: "ch_1" }, { id: "ch_2" }] } }));
    const recording = withCassette(stripeServices(liveFetch) as any, { mode: "record", cassette });
    const recorded = await listCharges(recording as any, "cus_123");
    assert.deepEqual(recorded, ["ch_1", "ch_2"]);
    assert.equal(liveFetch.calls.length, 1);

    // 2. REPLAY with a fetch that throws — proves no network is touched.
    const deadFetch = fakeFetch(() => {
      throw new Error("network should NOT be called during replay");
    });
    const replaying = withCassette(stripeServices(deadFetch) as any, { mode: "replay", cassette });
    const replayed = await listCharges(replaying as any, "cus_123");
    assert.deepEqual(replayed, ["ch_1", "ch_2"]);
    assert.equal(deadFetch.calls.length, 0);
  });

  it("never writes the real secret to the cassette", async () => {
    const cassette = emptyCassette();
    const liveFetch = fakeFetch(() => ({ body: { data: [] } }));
    const recording = withCassette(stripeServices(liveFetch) as any, { mode: "record", cassette });
    await listCharges(recording as any, "cus_123");

    const serialized = JSON.stringify(cassette);
    assert.ok(!serialized.includes("sk_live_supersecret"), "secret leaked into cassette");
    assert.ok(serialized.includes("{{secret:STRIPE_KEY}}"), "secret token missing");
    // secrets.get itself is not recorded — only the http call is.
    assert.equal(cassette.entries.length, 1);
    assert.equal(cassette.entries[0]!.key, "http");
  });

  it("replay matches even with no real credentials present", async () => {
    // Record with the real secret...
    const cassette = emptyCassette();
    const liveFetch = fakeFetch(() => ({ body: { data: [{ id: "ch_9" }] } }));
    await listCharges(
      withCassette(stripeServices(liveFetch) as any, { mode: "record", cassette }) as any,
      "cus_x",
    );

    // ...replay in an environment where the secret is absent (undefined).
    const credlessServices = standardServices({
      fetchImpl: fakeFetch(() => {
        throw new Error("no network in replay");
      }),
      secretsSource: {}, // STRIPE_KEY missing
    });
    const replaying = withCassette(credlessServices as any, { mode: "replay", cassette });
    const out = await listCharges(replaying as any, "cus_x");
    assert.deepEqual(out, ["ch_9"]);
  });

  it("walks repeated identical calls in recorded order (pagination)", async () => {
    const cassette: Cassette = {
      entries: [
        { key: "http", args: ["https://x/feed", {}], result: { ok: 1 } },
        { key: "http", args: ["https://x/feed", {}], result: { ok: 2 } },
      ],
    };
    const services = withCassette(
      standardServices({ fetchImpl: fakeFetch(() => ({ body: {} })) }) as any,
      { mode: "replay", cassette },
    );
    const a = await (services as any).http("https://x/feed", {});
    const b = await (services as any).http("https://x/feed", {});
    assert.deepEqual(a, { ok: 1 });
    assert.deepEqual(b, { ok: 2 });
  });

  it("throws a clear error when replay has no matching entry", async () => {
    const services = withCassette(
      standardServices({ fetchImpl: fakeFetch(() => ({ body: {} })) }) as any,
      { mode: "replay", cassette: emptyCassette() },
    );
    await assert.rejects(
      () => (services as any).http("https://x/missing", {}),
      /no recorded call for "http"/,
    );
  });
});
