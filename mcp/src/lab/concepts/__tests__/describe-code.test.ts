import { test } from "node:test";
import assert from "node:assert/strict";
import { z } from "zod";
import describeCode from "../steps/describe-code.js";

type Node = { ref_id: string; labels: string[]; name: string; docs: string; body: string; description?: string };

function graph(n: number): Node[] {
  return Array.from({ length: n }, (_, i) => ({
    ref_id: `r${i}`,
    labels: ["Data_Bank", i % 2 ? "Class" : "Function"],
    name: `node${i}`,
    docs: "",
    body: `function node${i}() {}`,
  }));
}

/** The services + registry the runner would hand the step: an in-memory
 *  graph and an `llm` step whose answer the test decides. */
function harness(nodes: Node[], answer: (prompt: string) => { text?: string; cost?: number } | Error) {
  const prompts: string[] = [];
  const ctx = {
    services: {
      codeNodes: {
        async withoutDescription(limit: number) {
          return nodes.filter((x) => !x.description).slice(0, limit);
        },
        async writeDescriptions(batch: Array<{ ref_id: string; description: string }>) {
          for (const { ref_id, description } of batch) nodes.find((x) => x.ref_id === ref_id)!.description = description;
        },
      },
    },
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
  return { ctx, prompts };
}

const cfg = (over: Record<string, unknown> = {}) =>
  describeCode.input.parse({ model: "m", prompt: "{type} {name} {docs}|{code}", batchSize: 3, concurrency: 2, ...over });

test("describes every node, filling the prompt, until none is left", async () => {
  const nodes = graph(5);
  const { ctx, prompts } = harness(nodes, (p) => ({ text: ` about ${p.split(" ")[1]} `, cost: 0.01 }));
  const out = await describeCode.run(cfg(), ctx as never);
  assert.equal(out.processed, 5);
  assert.equal(out.stopped, "done");
  assert.equal(nodes[1]!.description, "about node1");
  assert.ok(prompts.includes("Class node1 |function node1() {}"));
  assert.ok(Math.abs(out.cost - 0.05) < 1e-9);
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
