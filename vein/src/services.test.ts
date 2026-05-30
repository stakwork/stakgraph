import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { z } from "zod";
import { flow, step, defineStep, type StepRegistry } from "./core.js";
import { runWorkflow } from "./runner.js";

// ── Services as capability interfaces ──────────────────────────────────────
//
// These tests document the intended usage: consumers define a typed
// `Services` interface (graph store, file system, …), implement it
// concretely per environment (prod, test, in-memory), and inject it via
// `runWorkflow({ services })`. The same workflow runs against any
// implementation — that's the whole point.

interface GraphStore {
  query(cypher: string): Promise<{ rows: unknown[] }>;
}

interface FileSystem {
  readFile(path: string): Promise<string>;
}

interface MyServices {
  graph: GraphStore;
  fs: FileSystem;
}

/** A step typed against `MyServices` — `ctx.services` is fully typed. */
const graphQueryStep = defineStep<"graph.query", z.ZodObject<{ cypher: z.ZodString }>, z.ZodAny, MyServices>({
  type: "graph.query",
  input: z.object({ cypher: z.string() }),
  output: z.any(),
  async run(cfg, ctx) {
    return ctx.services.graph.query(cfg.cypher);
  },
});

const readFileStep = defineStep<"fs.read", z.ZodObject<{ path: z.ZodString }>, z.ZodString, MyServices>({
  type: "fs.read",
  input: z.object({ path: z.string() }),
  output: z.string(),
  async run(cfg, ctx) {
    return ctx.services.fs.readFile(cfg.path);
  },
});

function makeRegistry(): StepRegistry {
  return {
    "graph.query": graphQueryStep,
    "fs.read": readFileStep,
  } as StepRegistry;
}

// ── Fake implementations ───────────────────────────────────────────────────

class InMemoryGraph implements GraphStore {
  public calls: string[] = [];
  constructor(private readonly fixtures: Record<string, unknown[]>) {}
  async query(cypher: string): Promise<{ rows: unknown[] }> {
    this.calls.push(cypher);
    return { rows: this.fixtures[cypher] ?? [] };
  }
}

class InMemoryFs implements FileSystem {
  constructor(private readonly files: Record<string, string>) {}
  async readFile(path: string): Promise<string> {
    const v = this.files[path];
    if (v === undefined) throw new Error(`no such file: ${path}`);
    return v;
  }
}

// ── Tests ──────────────────────────────────────────────────────────────────

describe("services injection", () => {
  it("exposes the injected services bag on ctx.services", async () => {
    const wf = flow("graph-test", {
      input: z.object({ cypher: z.string() }),
      steps: [
        step("q", "graph.query", { cypher: "{{ input.cypher }}" }),
      ],
    });

    const graph = new InMemoryGraph({
      "MATCH (n) RETURN n": [{ id: 1 }, { id: 2 }],
    });

    const result = await runWorkflow<MyServices>(
      wf,
      { cypher: "MATCH (n) RETURN n" },
      makeRegistry(),
      {
        services: { graph, fs: new InMemoryFs({}) },
      },
    );

    assert.equal(result.status, "success");
    assert.deepEqual(result.output, { rows: [{ id: 1 }, { id: 2 }] });
    assert.deepEqual(graph.calls, ["MATCH (n) RETURN n"]);
  });

  it("runs the same workflow against different service implementations", async () => {
    const wf = flow("read", {
      input: z.object({ path: z.string() }),
      steps: [step("r", "fs.read", { path: "{{ input.path }}" })],
    });
    const reg = makeRegistry();

    // "prod" fixture
    const prodFs = new InMemoryFs({ "/etc/conf": "prod-value" });
    const prod = await runWorkflow<MyServices>(wf, { path: "/etc/conf" }, reg, {
      services: { graph: new InMemoryGraph({}), fs: prodFs },
    });
    assert.equal(prod.output, "prod-value");

    // "test" fixture — same workflow, same registry, different services
    const testFs = new InMemoryFs({ "/etc/conf": "test-value" });
    const test = await runWorkflow<MyServices>(wf, { path: "/etc/conf" }, reg, {
      services: { graph: new InMemoryGraph({}), fs: testFs },
    });
    assert.equal(test.output, "test-value");
  });

  it("defaults services to {} when no services are passed", async () => {
    // A step that tolerates a missing service field.
    const peek = defineStep({
      type: "peek",
      input: z.object({}),
      output: z.any(),
      async run(_cfg, ctx) {
        return ctx.services;
      },
    });

    const wf = flow("peek", {
      input: z.object({}),
      steps: [step("p", "peek", {})],
    });

    const result = await runWorkflow(wf, {}, { peek } as StepRegistry);
    assert.equal(result.status, "success");
    assert.deepEqual(result.output, {});
  });

  it("threads services through control flow (foreach + subflow)", async () => {
    // A child workflow that uses graph.query.
    const child = flow("child", {
      input: z.object({ cypher: z.string() }),
      steps: [step("q", "graph.query", { cypher: "{{ input.cypher }}" })],
    });

    // Parent fans out queries via foreach + subflow.
    const parent = flow("parent", {
      input: z.object({ queries: z.array(z.string()) }),
      steps: [
        step("all", "foreach", {
          items: "{{ input.queries }}",
          body: step("one", "subflow", {
            workflow: "child",
            input: { cypher: "{{ $current }}" },
          }),
        }),
      ],
    });

    const graph = new InMemoryGraph({
      "Q1": [{ a: 1 }],
      "Q2": [{ b: 2 }],
    });

    const result = await runWorkflow<MyServices>(
      parent,
      { queries: ["Q1", "Q2"] },
      makeRegistry(),
      {
        services: { graph, fs: new InMemoryFs({}) },
        workspace: {
          async getWorkflow(name) {
            if (name === "child") return child;
            throw new Error(`unknown workflow: ${name}`);
          },
          async getWorkflowVersion(name) {
            if (name === "child") return child;
            throw new Error(`unknown workflow: ${name}`);
          },
        },
      },
    );

    assert.equal(result.status, "success");
    assert.deepEqual(result.output, [
      { rows: [{ a: 1 }] },
      { rows: [{ b: 2 }] },
    ]);
    assert.deepEqual(graph.calls, ["Q1", "Q2"]);
  });

  it("propagates the same services into onError fallback steps", async () => {
    const seen: string[] = [];
    const failing = defineStep({
      type: "boom",
      input: z.object({}),
      output: z.any(),
      async run(_cfg, ctx) {
        seen.push(`primary:${(ctx.services as { tag: string }).tag}`);
        throw new Error("nope");
      },
    });
    const fallback = defineStep({
      type: "rescue",
      input: z.object({}),
      output: z.string(),
      async run(_cfg, ctx) {
        const tag = (ctx.services as { tag: string }).tag;
        seen.push(`fallback:${tag}`);
        return tag;
      },
    });

    const wf = flow("fallback", {
      input: z.object({}),
      steps: [
        {
          id: "x",
          type: "boom",
          config: {},
          options: { onError: step("y", "rescue", {}) },
        },
      ],
    });

    const result = await runWorkflow<{ tag: string }>(
      wf,
      {},
      { boom: failing, rescue: fallback } as StepRegistry,
      { services: { tag: "test-env" } },
    );

    assert.equal(result.status, "success");
    assert.equal(result.output, "test-env");
    assert.deepEqual(seen, ["primary:test-env", "fallback:test-env"]);
  });
});
