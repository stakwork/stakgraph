/**
 * The vein graph backend as one object: a bolt connection plus the writers,
 * the reader, and (optionally) the local embedder — opened once per
 * config and cached, with the boot-time obligations run on open:
 *
 *   1. `seedVeinDomain` — schema meta-graph, constraints, indexes (§4);
 *   2. `backfillEmbeddings` — heal any NULL vectors left by a crash (§2).
 *
 * Consumers (the `graph/*` lab steps, a future `Neo4jWorkspaceStore` and
 * run projector) call `openGraphBackend(cfg)` and share the instance.
 */
import { Bolt, graphConfigFromEnv, type GraphConfig } from "./bolt.js";
import { EdgeWriter } from "./edge-writer.js";
import { MiniLMEmbedder, backfillEmbeddings, type BackfillReport } from "./embeddings.js";
import { NodeWriter, type Embedder } from "./node-writer.js";
import { seedVeinDomain, type SeedReport } from "./schema-seed.js";
import { GraphReader } from "./search.js";

export interface GraphBackendOptions {
  /** `false` disables embeddings entirely (vectors stay NULL, search is
   *  fulltext-only). Pass an `Embedder` to inject one (tests). Default:
   *  load the local MiniLM model. */
  embeddings?: boolean | Embedder;
  /** Skip the boot-time seed + backfill (tests that manage the DB). */
  skipBoot?: boolean;
}

export interface GraphBackend {
  readonly cfg: GraphConfig;
  readonly bolt: Bolt;
  readonly nodes: NodeWriter;
  readonly edges: EdgeWriter;
  readonly reader: GraphReader;
  readonly embedder: Embedder | undefined;
  /** What the boot-time seed did (undefined when skipped). */
  readonly seed: SeedReport | undefined;
  readonly backfill: BackfillReport | undefined;
  close(): Promise<void>;
}

const cache = new Map<string, Promise<GraphBackend>>();

function keyOf(cfg: GraphConfig, opts: GraphBackendOptions): string {
  const emb = opts.embeddings === false ? "off" : typeof opts.embeddings === "object" ? "custom" : "minilm";
  return [cfg.uri, cfg.user, cfg.database ?? "", cfg.namespace, emb].join("|");
}

/**
 * Open (or reuse) the backend for `cfg`. The first call per config pays for
 * connectivity, seeding, embedder load, and the backfill sweep; later calls
 * get the same instance. A failed open is not cached.
 */
export function openGraphBackend(cfg: GraphConfig, opts: GraphBackendOptions = {}): Promise<GraphBackend> {
  const key = keyOf(cfg, opts);
  let p = cache.get(key);
  if (!p) {
    p = open(cfg, opts).catch((e) => {
      cache.delete(key);
      throw e;
    });
    cache.set(key, p);
  }
  return p;
}

/** `openGraphBackend` from `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD`/
 *  `VEIN_GRAPH_NAMESPACE`; null when `NEO4J_URI` is unset. */
export function openGraphBackendFromEnv(
  env: Record<string, string | undefined> = process.env,
  opts: GraphBackendOptions = {},
): Promise<GraphBackend> | null {
  const cfg = graphConfigFromEnv(env);
  if (!cfg) return null;
  const emb = env["VEIN_GRAPH_EMBEDDINGS"];
  return openGraphBackend(cfg, { ...opts, embeddings: opts.embeddings ?? (emb === "off" || emb === "0" || emb === "false" ? false : true) });
}

/** Drop every cached backend and close its driver. */
export async function closeGraphBackends(): Promise<void> {
  const all = [...cache.values()];
  cache.clear();
  await Promise.all(all.map((p) => p.then((b) => b.bolt.close()).catch(() => undefined)));
}

async function open(cfg: GraphConfig, opts: GraphBackendOptions): Promise<GraphBackend> {
  const bolt = new Bolt(cfg);
  try {
    await bolt.verify();
    const embedder: Embedder | undefined =
      opts.embeddings === false ? undefined : typeof opts.embeddings === "object" ? opts.embeddings : await MiniLMEmbedder.load();
    let seed: SeedReport | undefined;
    let backfill: BackfillReport | undefined;
    if (!opts.skipBoot) {
      seed = await seedVeinDomain(bolt);
      if (embedder) backfill = await backfillEmbeddings(bolt, embedder);
    }
    const backend: GraphBackend = {
      cfg,
      bolt,
      nodes: new NodeWriter(bolt, { embedder }),
      edges: new EdgeWriter(bolt),
      reader: new GraphReader(bolt, { embedder }),
      embedder,
      seed,
      backfill,
      async close() {
        for (const [k, p] of cache) if ((await p.catch(() => null)) === backend) cache.delete(k);
        await bolt.close();
      },
    };
    return backend;
  } catch (e) {
    await bolt.close().catch(() => undefined);
    throw e;
  }
}
