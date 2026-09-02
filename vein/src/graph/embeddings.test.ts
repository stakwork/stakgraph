import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Bolt } from "./bolt.js";
import { seedVeinDomain } from "./schema-seed.js";
import { NodeWriter, type Embedder } from "./node-writer.js";
import { EMBEDDING_DIM, MiniLMEmbedder, backfillEmbeddings, cosine, meanPoolNormalize } from "./embeddings.js";
import { testGraphConfig, wipeGraph } from "./test-util.js";

const cfg = testGraphConfig();
const runModel = process.env["VEIN_TEST_EMBEDDINGS"] === "1";

interface Golden {
  model: string;
  max_seq_length: number;
  vectors: Record<string, number[]>;
  texts: Record<string, string>;
}
const golden = JSON.parse(readFileSync(new URL("./fixtures/minilm-golden.json", import.meta.url), "utf8")) as Golden;

describe("meanPoolNormalize (pure)", () => {
  it("averages masked positions and L2-normalizes", () => {
    // batch=1, seq=3, dim=2; third position masked out.
    const hidden = new Float32Array([1, 0, 0, 1, 100, 100]);
    const [v] = meanPoolNormalize(hidden, [1, 3, 2], [1, 1, 0]);
    assert.ok(Math.abs(v![0]! - Math.SQRT1_2) < 1e-6);
    assert.ok(Math.abs(v![1]! - Math.SQRT1_2) < 1e-6);
  });
  it("golden fixture is the model jarvis uses, at 256 tokens", () => {
    assert.equal(golden.model, "sentence-transformers/all-MiniLM-L6-v2");
    assert.equal(golden.max_seq_length, 256);
    assert.equal(golden.vectors["short"]!.length, EMBEDDING_DIM);
  });
});

describe("MiniLMEmbedder parity vs sentence-transformers", { skip: runModel ? false : "VEIN_TEST_EMBEDDINGS=1 not set (downloads the model)" }, () => {
  let embedder: MiniLMEmbedder;
  before(async () => {
    embedder = await MiniLMEmbedder.load();
  });

  it("matches the Python golden vectors (cosine ≥ 0.999), including the >256-token truncation case", async () => {
    const keys = Object.keys(golden.texts);
    const vectors = await embedder.embed(keys.map((k) => golden.texts[k]!));
    keys.forEach((k, i) => {
      const c = cosine(vectors[i]!, golden.vectors[k]!);
      assert.ok(c >= 0.999, `${k}: cosine ${c}`);
      assert.equal(vectors[i]!.length, EMBEDDING_DIM);
      const norm = Math.sqrt(vectors[i]!.reduce((s, x) => s + x * x, 0));
      assert.ok(Math.abs(norm - 1) < 1e-4, `${k}: not unit length (${norm})`);
    });
  });

  it("truncation actually happens at 256 tokens (a 512-cap vector would differ)", async () => {
    const long = golden.texts["long_truncated"]!;
    const [truncated] = await embedder.embed([long]);
    // The same text with its tail removed past token 256 must embed
    // identically; cut generously short and it must not.
    const words = long.split(" ");
    const [head] = await embedder.embed([words.slice(0, 40).join(" ")]);
    assert.ok(cosine(truncated!, head!) < 0.999, "40-word prefix should differ from the 256-token vector");
    assert.ok(cosine(truncated!, golden.vectors["long_truncated"]!) >= 0.999);
  });
});

describe("backfillEmbeddings (live Neo4j)", { skip: cfg ? false : "VEIN_TEST_NEO4J_URI not set" }, () => {
  let bolt: Bolt;
  before(async () => {
    bolt = new Bolt(cfg!);
    await bolt.verify();
  });
  after(async () => {
    await bolt?.close();
  });
  beforeEach(async () => {
    await wipeGraph(bolt);
    await seedVeinDomain(bolt);
  });

  const fake: Embedder = {
    async embed(texts) {
      return texts.map((t) => Array.from({ length: EMBEDDING_DIM }, (_, i) => (i === 0 ? t.length : 0)));
    },
  };

  it("heals NULL text_embeddings and per-stem vectors, then is a no-op", async () => {
    // Written WITHOUT an embedder → vectors NULL (the crash case).
    const w = new NodeWriter(bolt);
    await w.writeMany([
      { type: "VeinRun", data: { run_id: "1", workflow_name: "wf", status: "ok", started_at: 1 } },
      { type: "VeinRun", data: { run_id: "2", workflow_name: "wf", status: "ok", started_at: 1 } },
      { type: "VeinStep", data: { step_type: "s", input_schema: "{in}", output_schema: "   " } },
      { type: "VeinTurn", data: { chat_id: "c", turn: 0 } }, // no index text → kitchen-sink Data_Bank "c\n0"
    ]);
    const r1 = await backfillEmbeddings(bolt, fake, 1);
    assert.equal(r1.text_embeddings, 4);
    assert.deepEqual(r1.vector_fields, { input_embeddings: 1 });
    const rows = await bolt.run(
      `MATCH (n:Domain_vein) RETURN n.node_key AS k, n.text_embeddings[0] AS t0, n.input_embeddings[0] AS i0, n.output_embeddings AS o ORDER BY k`,
    );
    assert.deepEqual(rows, [
      { k: "veinrun-1", t0: "wf\nok".length, i0: null, o: null },
      { k: "veinrun-2", t0: "wf\nok".length, i0: null, o: null },
      { k: "veinstep-s", t0: "s".length, i0: "Input:\n{in}".length, o: null },
      { k: "veinturn-c-0", t0: "c\n0".length, i0: null, o: null },
    ]);
    const r2 = await backfillEmbeddings(bolt, fake);
    assert.deepEqual(r2, { text_embeddings: 0, vector_fields: {} });
  });
});
