import { createClient, RedisClientType } from "redis";

/**
 * Jarvis interop for Concept nodes.
 *
 * Jarvis fetches Concepts through domain-scoped vector indexes
 * (`FOR (n:Domain_general) ON n.text_embeddings`), while this pipeline writes
 * `:Data_Bank:Concept` nodes over bolt with its own `embeddings` property.
 * Two things bridge the gap:
 *
 * 1. The `Domain_general` label (applied in saveConcept / backfilled at init)
 *    puts Concepts inside jarvis's `domain_general_vector_index`.
 * 2. Jarvis populates `text_embeddings` only via its Redis embedding queue
 *    (`jarvis:embedding_queue`, drained by run_embedding_worker), so we push
 *    a job there after each Concept write. The worker matches nodes by
 *    `(n:Data_Bank {ref_id})` and writes `n.text_embeddings` back.
 *
 * `embeddings` (BGE, used by list_concepts / local semantic search) is
 * untouched — the two embedding spaces coexist on the node.
 */

export const JARVIS_EMBEDDING_QUEUE = "jarvis:embedding_queue";

// Same shape jarvis's push_embedding_job produces; kind "data_bank" makes the
// worker write back to n.text_embeddings.
interface JarvisEmbeddingJob {
  ref_id: string;
  text: string;
  kind: "data_bank";
}

let clientPromise: Promise<RedisClientType | null> | null = null;

export function jarvisRedisEnabled(): boolean {
  return !!process.env.REDIS_URL;
}

async function getClient(): Promise<RedisClientType | null> {
  if (!jarvisRedisEnabled()) return null;
  if (!clientPromise) {
    clientPromise = (async () => {
      const client: RedisClientType = createClient({
        url: process.env.REDIS_URL,
      });
      // Without a listener, a dropped connection emits an unhandled 'error'
      // event and crashes the process; reconnects are automatic.
      client.on("error", (err) =>
        console.error("[jarvis] Redis client error:", err.message || err)
      );
      await client.connect();
      return client;
    })().catch((error) => {
      clientPromise = null; // allow a retry on the next push
      throw error;
    });
  }
  return clientPromise;
}

/**
 * Queue text_embeddings generation for a Concept in jarvis's embedding
 * worker. Fire-and-forget semantics: never throws, no-ops when REDIS_URL is
 * unset, logs and moves on when Redis is unreachable.
 */
export async function pushJarvisEmbeddingJob(
  refId: string | null | undefined,
  text: string
): Promise<boolean> {
  if (!jarvisRedisEnabled()) return false;
  if (!refId || !text.trim()) return false;
  try {
    const client = await getClient();
    if (!client) return false;
    const job: JarvisEmbeddingJob = {
      ref_id: refId,
      text,
      kind: "data_bank",
    };
    await client.rPush(JARVIS_EMBEDDING_QUEUE, JSON.stringify(job));
    return true;
  } catch (error: any) {
    console.error(
      `[jarvis] Failed to queue embedding job for ref_id=${refId}:`,
      error.message || error
    );
    return false;
  }
}
