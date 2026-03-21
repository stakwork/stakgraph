import { chunkCode, weightedPooling } from "./utils.js";
import { EmbeddingModel, FlagEmbedding } from "./interop.js";

export const DIMENSIONS = 384;
export const MODEL = EmbeddingModel.BGESmallENV15;

// Initialize the embedding model once and reuse it
let flagEmbeddingInstance: Awaited<ReturnType<typeof FlagEmbedding.init>> | null = null;
async function getFlagEmbedding() {
  if (!flagEmbeddingInstance) {
    flagEmbeddingInstance = await FlagEmbedding.init({
      model: MODEL,
      maxLength: 512,
    });
  }
  return flagEmbeddingInstance;
}

export async function vectorizeQuery(query: string): Promise<number[]> {
  const flagEmbedding = await getFlagEmbedding();
  return await flagEmbedding.queryEmbed(query);
}

export async function vectorizeBatch(texts: string[]): Promise<number[][]> {
  const results: number[][] = new Array(texts.length);
  const shortIndices: number[] = [];
  const shortTexts: string[] = [];
  const longIndices: number[] = [];

  for (let i = 0; i < texts.length; i++) {
    if (texts[i].length < 400) {
      shortIndices.push(i);
      shortTexts.push(texts[i]);
    } else {
      longIndices.push(i);
    }
  }

  const flagEmbedding = await getFlagEmbedding();

  if (shortTexts.length > 0) {
    const gen = flagEmbedding.passageEmbed(shortTexts);
    let idx = 0;
    for await (const batch of gen) {
      for (const vec of batch) {
        results[shortIndices[idx++]] = vec;
      }
    }
  }

  for (const i of longIndices) {
    results[i] = await vectorizeCodeDocument(texts[i]);
  }

  return results;
}

export async function vectorizeCodeDocument(
  codeString: string,
): Promise<number[]> {
  const flagEmbedding = await getFlagEmbedding();

  // Use overlapping chunks for better context capture
  const chunks = chunkCode(codeString, 400);

  // Generate embeddings for all chunks
  const embeddingsGenerator = flagEmbedding.passageEmbed(chunks);
  let allEmbeddings: number[][] = [];

  for await (const embeddings of embeddingsGenerator) {
    allEmbeddings = [...allEmbeddings, ...embeddings];
  }

  // First chunk has the function signature, so give it more weight
  const weights = new Array(allEmbeddings.length).fill(1);
  weights[0] = 1.2;

  // Compute weighted pooling
  let pooledEmbedding = weightedPooling(allEmbeddings, weights);

  // Normalize the final vector
  const magnitude = Math.sqrt(
    pooledEmbedding.reduce((sum, val) => sum + val * val, 0),
  );

  return pooledEmbedding.map((val) => val / magnitude);
}
