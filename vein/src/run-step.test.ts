import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { defineStep, type StepRegistry } from "./core.js";
import { coreRegistry } from "./steps/registry.js";
import { runSingleStep, cassettePath } from "./run-step.js";
import { standardServices, type FetchLike } from "./capabilities.js";

// A fake fetch returning canned JSON, recording how many times it was called.
function fakeFetch(
  handler: (url: string, init: any) => unknown,
): FetchLike & { count: number } {
  const fn = (async (url: string, init: any = {}) => {
    fn.count++;
    const body = handler(url, init);
    const text = JSON.stringify(body);
    return {
      status: 200,
      ok: true,
      headers: {
        forEach(cb: (value: string, key: string) => void) {
          cb("application/json", "content-type");
        },
      },
      async text() {
        return text;
      },
    };
  }) as any;
  fn.count = 0;
  return fn;
}

// An adapter-style step: reads a secret + calls http through ctx.services.
const listCharges = defineStep({
  type: "stripe/list-charges",
  input: z.object({ customer: z.string() }),
  output: z.any(),
  async run(cfg, ctx) {
    const svc = ctx.services as ReturnType<typeof standardServices>;
    const key = await svc.secrets.get("STRIPE_KEY");
    const res = await svc.http("https://api.stripe.com/v1/charges", {
      query: { customer: cfg.customer },
      headers: { authorization: `Bearer ${key}` },
    });
    return { ids: (res.body as { data: Array<{ id: string }> }).data.map((c) => c.id) };
  },
});

function registry(): StepRegistry {
  return { "stripe/list-charges": listCharges } as StepRegistry;
}

describe("runSingleStep", () => {
  it("runs a step in isolation and returns output + events", async () => {
    const services = standardServices({
      fetchImpl: fakeFetch(() => ({ data: [{ id: "ch_1" }] })),
      secretsSource: { STRIPE_KEY: "sk_test" },
    });
    const res = await runSingleStep("stripe/list-charges", registry(), services, {
      config: { customer: "{{ input.customer }}" },
      input: { customer: "cus_1" },
    });
    assert.equal(res.status, "success");
    assert.deepEqual(res.output, { ids: ["ch_1"] });
    assert.ok(res.events.some((e) => e.type === "step.end"));
  });

  it("errors clearly for an unknown step type", async () => {
    const res = await runSingleStep("nope", registry(), {});
    assert.equal(res.status, "error");
    assert.match(res.error!.message, /not found/);
  });

  it("records a fixture (secret scrubbed), then replays offline", async () => {
    const dir = await mkdtemp(join(tmpdir(), "vein-cassette-"));
    const path = cassettePath(dir, "stripe/list-charges");

    // RECORD — live fetch hit once, fixture written.
    const live = fakeFetch(() => ({ data: [{ id: "ch_7" }, { id: "ch_8" }] }));
    const recRes = await runSingleStep(
      "stripe/list-charges",
      registry(),
      standardServices({ fetchImpl: live, secretsSource: { STRIPE_KEY: "sk_live_secret" } }),
      { config: { customer: "cus_42" }, cassette: { mode: "record", path } },
    );
    assert.equal(recRes.status, "success");
    assert.deepEqual(recRes.output, { ids: ["ch_7", "ch_8"] });
    assert.equal(live.count, 1);
    assert.equal(recRes.recorded, 1); // one http call captured (secrets.get isn't recorded)

    // Fixture on disk must not contain the real secret.
    const raw = await readFile(path, "utf-8");
    assert.ok(!raw.includes("sk_live_secret"), "secret leaked to cassette file");
    assert.ok(raw.includes("{{secret:STRIPE_KEY}}"));

    // REPLAY — fetch that throws proves no network is touched.
    const dead = fakeFetch(() => {
      throw new Error("network must not be called in replay");
    });
    const repRes = await runSingleStep(
      "stripe/list-charges",
      registry(),
      standardServices({ fetchImpl: dead, secretsSource: {} }), // no creds present
      { config: { customer: "cus_42" }, cassette: { mode: "replay", path } },
    );
    assert.equal(repRes.status, "success");
    assert.deepEqual(repRes.output, { ids: ["ch_7", "ch_8"] });
    assert.equal(dead.count, 0);
  });

  it("the built-in http step routes through services.http (so it's recordable)", async () => {
    const dir = await mkdtemp(join(tmpdir(), "vein-http-"));
    const path = cassettePath(dir, "http");

    // RECORD — fake fetch behind services.http; the core http step uses it.
    const live = fakeFetch(() => ({ hello: "world" }));
    const rec = await runSingleStep("http", coreRegistry(), standardServices({ fetchImpl: live }), {
      config: { url: "https://api.example.com/data" },
      cassette: { mode: "record", path },
    });
    assert.equal(rec.status, "success");
    assert.deepEqual(rec.output, { status: 200, body: { hello: "world" } });
    assert.equal(rec.recorded, 1);
    assert.equal(live.count, 1);

    // REPLAY — offline, no network touched.
    const dead = fakeFetch(() => {
      throw new Error("no network in replay");
    });
    const rep = await runSingleStep("http", coreRegistry(), standardServices({ fetchImpl: dead }), {
      config: { url: "https://api.example.com/data" },
      cassette: { mode: "replay", path },
    });
    assert.equal(rep.status, "success");
    assert.deepEqual(rep.output, { status: 200, body: { hello: "world" } });
    assert.equal(dead.count, 0);
  });
});
