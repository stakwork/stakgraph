import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { WorkspaceManager, buildRegistry, defineStep, runWorkflow } from "strut";
import describeCode, { listContained } from "../steps/describe-code.js";
import { seedConceptSteps, seedConceptWorkflows } from "../seed.js";
import type { ContainedNode } from "../services.js";

type Node = {
  ref_id: string;
  labels: string[];
  name: string;
  file: string;
  docs: string;
  body: string;
  description?: string;
  summary?: string;
};

function graph(n: number): Node[] {
  return Array.from({ length: n }, (_, i) => ({
    ref_id: `r${i}`,
    labels: ["Data_Bank", i % 2 ? "Class" : "Function"],
    name: `node${i}`,
    file: `o/r/src/node${i}.ts`,
    docs: "",
    body: `function node${i}() {}`,
  }));
}

/** The services + registry the runner would hand the step: an in-memory
 *  graph and an `llm` step whose answer the test decides. */
function harness(
  nodes: Node[],
  answer: (prompt: string) => { text?: string; cost?: number } | Error,
  children: Record<string, ContainedNode[]> = {},
) {
  const prompts: string[] = [];
  const reads: Array<{ types: string[]; field: string }> = [];
  const containedReads: Array<{ refIds: string[]; types: string[]; max: number }> = [];
  const writes: string[] = [];
  const codeNodes = {
    async withoutDescription(limit: number, _repos: string[], _files: string[], opts: { types: string[]; field: string }) {
      reads.push(opts);
      const field = opts.field as "description" | "summary";
      return nodes.filter((x) => x.labels.some((l) => opts.types.includes(l)) && !x[field]).slice(0, limit);
    },
    async contained(refIds: string[], types: string[], max: number) {
      containedReads.push({ refIds, types, max });
      const out: Record<string, { total: number; nodes: ContainedNode[] }> = {};
      for (const id of refIds) {
        const all = (children[id] ?? []).filter((c) => types.includes(c.type));
        if (all.length) out[id] = { total: all.length, nodes: all.slice(0, max) };
      }
      return out;
    },
    async writeDescriptions(batch: Array<{ ref_id: string; description: string }>, field: string) {
      writes.push(field);
      for (const { ref_id, description } of batch) nodes.find((x) => x.ref_id === ref_id)![field as "description" | "summary"] = description;
    },
  };
  const ctx = {
    services: { codeNodes },
    registry: {
      llm: {
        input: z.object({ prompt: z.string(), model: z.string().optional() }),
        async run(cfg: { prompt: string }) {
          prompts.push(cfg.prompt);
          const out = answer(cfg.prompt);
          if (out instanceof Error) throw out;
          return out;
        },
      },
    },
  };
  return { ctx, prompts, reads, containedReads, writes, codeNodes };
}

const cfg = (over: Record<string, unknown> = {}) =>
  describeCode.input.parse({ model: "m", prompt: "{type} {name} {docs}|{code}", batchSize: 3, concurrency: 2, ...over });

test("describes every node, filling the prompt, until none is left", async () => {
  const nodes = graph(5);
  const { ctx, prompts, reads, containedReads, writes } = harness(nodes, (p) => ({ text: ` about ${p.split(" ")[1]} `, cost: 0.01 }));
  const out = await describeCode.run(cfg(), ctx as never);
  assert.equal(out.processed, 5);
  assert.equal(out.stopped, "done");
  assert.equal(nodes[1]!.description, "about node1");
  assert.ok(prompts.includes("Class node1 |function node1() {}"));
  assert.ok(Math.abs(out.cost - 0.05) < 1e-9);
  // the defaults: the code types, into `description`, and no {contains} read
  assert.deepEqual(reads[0], {
    types: ["Class", "Endpoint", "Request", "Function", "Datamodel", "Page", "Trait", "Var"],
    field: "description",
  });
  assert.ok(writes.every((f) => f === "description"));
  assert.equal(containedReads.length, 0);
});

test("with types: [File] and field: summary, summarizes Files from their path and what they contain", async () => {
  const nodes: Node[] = [
    ...graph(2).map((n) => ({ ...n, description: "already" })),
    { ref_id: "f1", labels: ["Data_Bank", "File"], name: "route.ts", file: "o/r/src/app/api/route.ts", docs: "", body: "export async function GET() {}" },
    { ref_id: "f2", labels: ["Data_Bank", "File"], name: "empty.ts", file: "o/r/src/empty.ts", docs: "", body: "// nothing" },
  ];
  const children: Record<string, ContainedNode[]> = {
    f1: [
      { type: "Function", name: "GET", description: "Lists   the\nworkspaces." },
      { type: "Var", name: "runtime", description: "" },
      { type: "Import", name: "imports", description: "" },
    ],
  };
  const { ctx, prompts, reads, containedReads, writes } = harness(nodes, (p) => ({ text: `summary of ${p.split("\n")[0]}`, cost: 0 }), children);
  const out = await describeCode.run(
    cfg({ types: ["File"], field: "summary", prompt: "{type} {path}\n{contains}\n{code}" }),
    ctx as never,
  );
  assert.equal(out.processed, 2);
  assert.equal(out.stopped, "done");
  assert.deepEqual(reads[0], { types: ["File"], field: "summary" });
  assert.ok(writes.every((f) => f === "summary"));
  assert.deepEqual(containedReads[0]!.refIds, ["f1", "f2"]);
  assert.equal(containedReads[0]!.max, 40);
  assert.ok(!containedReads[0]!.types.includes("Import"));
  assert.ok(prompts.includes("File o/r/src/app/api/route.ts\n- Function GET: Lists the workspaces.\n- Var runtime\nexport async function GET() {}"));
  assert.ok(prompts.includes("File o/r/src/empty.ts\n(nothing)\n// nothing"));
  assert.equal(nodes[2]!.summary, "summary of File o/r/src/app/api/route.ts");
  assert.equal(nodes[2]!.description, undefined);
  assert.equal(nodes[0]!.summary, undefined);
});

test("{contains} lists what fits in its budget, then how many more there are", () => {
  const nodes = Array.from({ length: 5 }, (_, i) => ({ type: "Function", name: `f${i}`, description: "x".repeat(10) }));
  // each line is "- Function fN: xxxxxxxxxx" — 25 chars, 26 with its newline
  assert.equal(listContained({ total: 5, nodes }, 1000), nodes.map((n) => `- Function ${n.name}: ${n.description}`).join("\n"));
  assert.equal(listContained({ total: 5, nodes }, 60), "- Function f0: xxxxxxxxxx\n- Function f1: xxxxxxxxxx\n- … and 3 more");
  assert.equal(listContained({ total: 90, nodes }, 1000).split("\n").at(-1), "- … and 85 more");
  assert.equal(listContained(undefined, 1000), "(nothing)");
});

test("refuses a field other than description or summary", () => {
  assert.throws(() => cfg({ field: "name" }), /description|summary/);
  assert.throws(() => cfg({ field: "body" }));
  assert.throws(() => cfg({ types: [] }));
  assert.throws(() => cfg({ types: ["File) OR (1=1"] }), /a node label/);
});

test("stops instead of paying again for a node whose write did not take", async () => {
  const nodes = graph(3);
  const { ctx, prompts, codeNodes } = harness(nodes, () => ({ text: "ok", cost: 0 }));
  codeNodes.writeDescriptions = async () => {};
  const out = await describeCode.run(cfg({ batchSize: 3 }), ctx as never);
  assert.equal(out.stopped, "no_progress");
  assert.equal(prompts.length, 3);
});

test("reads past a node whose call failed instead of stopping on it", async () => {
  const nodes = graph(5);
  const { ctx } = harness(nodes, (p) => (p.includes("node0 ") ? new Error("boom") : { text: "ok", cost: 0 }));
  const out = await describeCode.run(cfg(), ctx as never);
  assert.equal(out.processed, 4);
  assert.equal(out.failed, 1);
  assert.equal(out.stopped, "done");
  assert.equal(nodes[0]!.description, undefined);
});

test("stops when a whole batch fails", async () => {
  const { ctx } = harness(graph(5), () => ({ text: "  " }));
  const out = await describeCode.run(cfg(), ctx as never);
  assert.equal(out.processed, 0);
  assert.equal(out.stopped, "no_progress");
});

test("starts no call once the cost limit is spent", async () => {
  const nodes = graph(10);
  const { ctx, prompts } = harness(nodes, () => ({ text: "ok", cost: 0.2 }));
  const out = await describeCode.run(cfg({ costLimit: 0.5, concurrency: 1 }), ctx as never);
  assert.equal(out.stopped, "cost_limit");
  assert.equal(prompts.length, 3);
  assert.equal(out.processed, 3);
});

test("a key or quota error fails the step after writing what it has", async () => {
  const nodes = graph(3);
  let calls = 0;
  const { ctx } = harness(nodes, () => (++calls === 3 ? new Error("401 Unauthorized") : { text: "ok", cost: 0 }));
  await assert.rejects(describeCode.run(cfg({ concurrency: 1 }), ctx as never), /stopped after 2 description\(s\): 401/);
  assert.equal(nodes.filter((n) => n.description).length, 2);
});

test("a fatal error says what the provider answered, not just the SDK's summary", async () => {
  const nodes = graph(2);
  // The shape of the AI SDK's "Invalid JSON response" (an APICallError):
  // the parse failure as `cause`, the provider's status and body on it.
  const sdk = Object.assign(new Error("Invalid JSON response", { cause: new Error("JSON parsing failed: Text: Insufficient credits") }), {
    statusCode: 200,
    responseBody: "Insufficient credits",
  });
  const { ctx } = harness(nodes, () => sdk);
  await assert.rejects(
    describeCode.run(cfg({ concurrency: 1 }), ctx as never),
    (e: Error) =>
      e.message ===
      "concepts/describe-code: stopped after 0 description(s): Invalid JSON response: JSON parsing failed: Text: Insufficient credits (HTTP 200: Insufficient credits)",
  );
});

// The mcp dir, like the real lab-workspace: a seeded step's bare `strut`
// import resolves via mcp/node_modules from there.
const MCP_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "..");

test("the seeded workflow describes the code nodes, then summarizes the Files, each under its own cap", async () => {
  const root = await mkdtemp(join(MCP_DIR, ".lab-describe-code-"));
  try {
    const ws = new WorkspaceManager(root);
    await seedConceptWorkflows(ws);
    await seedConceptSteps(ws);
    const { registry } = await buildRegistry(await ws.materializeCustomSteps());
    const prompts: string[] = [];
    const llm = defineStep({
      type: "llm",
      input: z.object({ prompt: z.string(), model: z.string().optional() }),
      output: z.any(),
      async run(cfg) {
        prompts.push(cfg.prompt);
        return { text: "ok", cost: 0.2 };
      },
    });
    const run = async (input: Record<string, unknown>) => {
      const nodes: Node[] = [
        ...graph(4),
        ...graph(4).map((n) => ({ ...n, ref_id: `f${n.ref_id}`, labels: ["Data_Bank", "File"] })),
      ];
      const h = harness(nodes, () => ({ text: "ok" }));
      // the File prompt's {contains} carries what the first step wrote
      h.codeNodes.contained = async (refIds: string[]) =>
        Object.fromEntries(
          refIds.map((id) => [id, { total: 1, nodes: [{ type: "Function", name: "f", description: String(nodes[0]!.description) }] }]),
        );
      const res = await runWorkflow(await ws.getWorkflow("describe-code"), input, { ...registry, llm } as typeof registry, {
        services: { codeNodes: h.codeNodes },
        params: { concurrency: 1 },
      });
      assert.equal(res.status, "success", JSON.stringify(res.error));
      return { out: res.output as any, nodes, reads: h.reads };
    };

    // file_cost_limit omitted: the Files get cost_limit too
    const a = await run({ cost_limit: 0.5 });
    assert.deepEqual(a.reads.map((r) => r.field), ["description", "summary"]);
    assert.ok(prompts[3]!.includes("Path: o/r/src/node0.ts\n\nContains:\n- Function f: ok\n"), prompts[3]);
    assert.equal(a.out.code.processed, 3);
    assert.equal(a.out.files.processed, 3);
    assert.equal(a.out.processed, 6);
    assert.equal(a.out.stopped, "cost_limit");
    assert.ok(Math.abs(a.out.cost - 1.2) < 1e-9);
    assert.equal(a.nodes.filter((n) => n.labels.includes("File") && n.description).length, 0);

    const b = await run({ cost_limit: 0.5, file_cost_limit: 0.1 });
    assert.equal(b.out.code.processed, 3);
    assert.equal(b.out.files.processed, 1);
    assert.equal(b.out.files.stopped, "cost_limit");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
