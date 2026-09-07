/**
 * Local MiniLM embeddings (`plans/jarvis-graph-compat.md` §2).
 *
 * jarvis encodes `Data_Bank` with
 * `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` — 384
 * dims, mean pooling over the attention mask, L2-normalized, truncated at
 * **256 tokens** (sentence-transformers' `max_seq_length`, not the
 * tokenizer's 512). We run the same graph through transformers.js /
 * onnxruntime (identical on ARM and Intel), driving tokenizer + model
 * directly so the 256 cap is explicit. Parity is proven by
 * `embeddings.test.ts` against a Python-produced golden vector.
 *
 * No Redis queue: vectors are computed in-process before the node write.
 * jarvis's "pending" state is purely `text_embeddings IS NULL`, so a crash
 * between MERGE and vector write is healed by `backfillEmbeddings` at boot —
 * the same NULL-scan idiom jarvis's own migration uses (`migration.py:857`).
 */
import { modelDirFromEnv } from "../model-dir.js";
import { join } from "node:path";
import { int, type Bolt } from "./bolt.js";
import type { Embedder } from "./node-writer.js";
import { renderVectorField } from "./node-writer.js";
import { STRUT_DOMAIN_LABEL, embeddingColumn, vectorIndexedPairs } from "./strut-schemas.js";

export const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
export const EMBEDDING_DIM = 384;
export const EMBEDDING_MAX_TOKENS = 256;

export interface MiniLMOptions {
  /** HF repo id of an ONNX export of all-MiniLM-L6-v2. */
  model?: string;
  /** Where model files are cached. See src/model-dir.ts for the default
   *  (`STRUT_MODEL_DIR`, `STRUT_MODEL_CACHE`, or `<cache root>/strut/models`). */
  cacheDir?: string;
  /** Texts per forward pass. */
  batchSize?: number;
}

type Transformers = typeof import("@huggingface/transformers");

export class MiniLMEmbedder implements Embedder {
  private constructor(
    private readonly tf: Transformers,
    private readonly tokenizer: import("@huggingface/transformers").PreTrainedTokenizer,
    private readonly model: import("@huggingface/transformers").PreTrainedModel,
    private readonly cls: number,
    private readonly sep: number,
    private readonly pad: number,
    private readonly batchSize: number,
  ) {}

  /** Load (downloading on first use) and self-check the output dimension. */
  static async load(opts: MiniLMOptions = {}): Promise<MiniLMEmbedder> {
    let tf: Transformers;
    try {
      tf = await import("@huggingface/transformers");
    } catch (e) {
      // The desktop package (scripts/package-desktop.mjs) omits the
      // embeddings stack: MiniLM only serves graph search, which needs a
      // Neo4j the desktop build doesn't ship. Say so instead of ENOENT.
      if ((e as { code?: string }).code === "ERR_MODULE_NOT_FOUND") {
        throw new Error(
          "embeddings are not available in this strut build (@huggingface/transformers is not installed); " +
            "graph search needs a server build or `package:desktop --embeddings`",
        );
      }
      throw e;
    }
    tf.env.cacheDir = opts.cacheDir ?? modelDirFromEnv();
    tf.env.allowLocalModels = false;
    const model = opts.model ?? EMBEDDING_MODEL;
    const [tokenizer, net] = await Promise.all([
      tf.AutoTokenizer.from_pretrained(model),
      // fp32 explicitly: quantized weights would drift from jarvis's encoder.
      tf.AutoModel.from_pretrained(model, { dtype: "fp32" }),
    ]);
    // Special-token ids via the public API: an empty input encodes to
    // exactly [CLS, SEP].
    const specials = tokenizer.encode("", { add_special_tokens: true });
    if (specials.length !== 2) throw new Error(`unexpected special-token layout: ${specials.join(",")}`);
    const e = new MiniLMEmbedder(tf, tokenizer, net, specials[0]!, specials[1]!, tokenizer.pad_token_id ?? 0, opts.batchSize ?? 32);
    const probe = await e.embed(["dimension probe"]);
    if (probe[0]!.length !== EMBEDDING_DIM) {
      throw new Error(`embedding model produced ${probe[0]!.length} dims, expected ${EMBEDDING_DIM}`);
    }
    return e;
  }

  async embed(texts: string[]): Promise<number[][]> {
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += this.batchSize) {
      out.push(...(await this.forward(texts.slice(i, i + this.batchSize))));
    }
    return out;
  }

  /**
   * Tokenize like sentence-transformers: the 256 cap INCLUDES [CLS]/[SEP],
   * so the body is cut to 254 and the specials are always present.
   * (transformers.js's own `truncation: true` truncates after adding
   * specials and chops [SEP] off long inputs — measurably different
   * vectors.) Right-padded with the pad id; mask 1 on real tokens.
   */
  private encodeBatch(texts: string[]) {
    const seqs = texts.map((t) => {
      const body = this.tokenizer.encode(t, { add_special_tokens: false }).slice(0, EMBEDDING_MAX_TOKENS - 2);
      return [this.cls, ...body, this.sep];
    });
    const batch = seqs.length;
    const seq = Math.max(...seqs.map((s) => s.length));
    const ids = new BigInt64Array(batch * seq).fill(BigInt(this.pad));
    const mask = new BigInt64Array(batch * seq);
    const types = new BigInt64Array(batch * seq);
    seqs.forEach((s, b) => {
      s.forEach((id, i) => {
        ids[b * seq + i] = BigInt(id);
        mask[b * seq + i] = 1n;
      });
    });
    const { Tensor } = this.tf;
    return {
      input_ids: new Tensor("int64", ids, [batch, seq]),
      attention_mask: new Tensor("int64", mask, [batch, seq]),
      token_type_ids: new Tensor("int64", types, [batch, seq]),
      maskData: mask,
    };
  }

  private async forward(texts: string[]): Promise<number[][]> {
    const { input_ids, attention_mask, token_type_ids, maskData } = this.encodeBatch(texts);
    const output = await this.model({ input_ids, attention_mask, token_type_ids });
    const hidden = output["last_hidden_state"] as { data: Float32Array; dims: number[] };
    return meanPoolNormalize(hidden.data, hidden.dims, maskData);
  }
}

/** Mean pooling over the attention mask, then L2 normalization — exactly
 *  sentence-transformers' `Pooling(mean)` + `Normalize`. */
export function meanPoolNormalize(
  hidden: Float32Array,
  dims: number[],
  mask: ArrayLike<number | bigint>,
): number[][] {
  const [batch, seq, dim] = dims as [number, number, number];
  const out: number[][] = [];
  for (let b = 0; b < batch; b++) {
    const vec = new Float64Array(dim);
    let count = 0;
    for (let s = 0; s < seq; s++) {
      if (Number(mask[b * seq + s]) === 0) continue;
      count++;
      const base = (b * seq + s) * dim;
      for (let d = 0; d < dim; d++) vec[d]! += hidden[base + d]!;
    }
    let norm = 0;
    for (let d = 0; d < dim; d++) {
      vec[d]! /= Math.max(count, 1e-9);
      norm += vec[d]! * vec[d]!;
    }
    norm = Math.sqrt(norm) || 1;
    out.push(Array.from(vec, (x) => x / norm));
  }
  return out;
}

export function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

export interface BackfillReport {
  /** Nodes whose `text_embeddings` was filled. */
  text_embeddings: number;
  /** Per `{stem}_embeddings` column, nodes filled. */
  vector_fields: Record<string, number>;
}

/**
 * Crash-safe sweep: embed every Strut node whose search text exists but
 * whose vector is NULL, in batches, until none remain. Idempotent and cheap
 * when clean. Run at every boot of the graph backend. Also covers the
 * per-property `{stem}_embeddings` of the `vector_index` types.
 */
export async function backfillEmbeddings(bolt: Bolt, embedder: Embedder, batchSize = 100): Promise<BackfillReport> {
  const report: BackfillReport = { text_embeddings: 0, vector_fields: {} };

  for (;;) {
    const rows = await bolt.run(
      `MATCH (n:\`${STRUT_DOMAIN_LABEL}\`)
       WHERE n.Data_Bank IS NOT NULL AND n.text_embeddings IS NULL
       RETURN n.ref_id AS ref_id, n.Data_Bank AS text LIMIT $limit`,
      { limit: int(batchSize) },
    );
    if (rows.length === 0) break;
    const vectors = await embedder.embed(rows.map((r) => String(r["text"])));
    await bolt.run(
      `UNWIND $rows AS row
       MATCH (n:Data_Bank {ref_id: row.ref_id})
       SET n.text_embeddings = row.vector`,
      { rows: rows.map((r, i) => ({ ref_id: r["ref_id"], vector: vectors[i] })) },
    );
    report.text_embeddings += rows.length;
  }

  for (const { type, prop } of vectorIndexedPairs()) {
    const column = embeddingColumn(prop);
    for (;;) {
      const rows = await bolt.run(
        `MATCH (n:\`${type}\`)
         WHERE n.\`${prop}\` IS NOT NULL AND trim(n.\`${prop}\`) <> '' AND n.\`${column}\` IS NULL
         RETURN n.ref_id AS ref_id, n.\`${prop}\` AS text LIMIT $limit`,
        { limit: int(batchSize) },
      );
      if (rows.length === 0) break;
      const vectors = await embedder.embed(rows.map((r) => renderVectorField(prop, String(r["text"]))!));
      await bolt.run(
        `UNWIND $rows AS row
         MATCH (n:Data_Bank {ref_id: row.ref_id})
         SET n.\`${column}\` = row.vector`,
        { rows: rows.map((r, i) => ({ ref_id: r["ref_id"], vector: vectors[i] })) },
      );
      report.vector_fields[column] = (report.vector_fields[column] ?? 0) + rows.length;
    }
  }
  return report;
}
