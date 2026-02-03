import { chunkCode, weightedPooling } from "./utils.js";
import { EmbeddingModel, FlagEmbedding } from "./interop.js";

export const DIMENSIONS = 384;
export const MODEL = EmbeddingModel.BGESmallENV15;

export async function vectorizeQuery(query: string): Promise<number[]> {
  const flagEmbedding = await FlagEmbedding.init({
    model: MODEL,
    maxLength: 512,
  });
  return await flagEmbedding.queryEmbed(query);
}

export async function vectorizeCodeDocument(
  codeString: string,
): Promise<number[]> {
  const flagEmbedding = await FlagEmbedding.init({
    model: MODEL,
    maxLength: 512,
  });

  if (codeString.length < 400) {
    const embeddingsGenerator = flagEmbedding.embed([codeString]);
    let embedding: number[] = [];

    for await (const embeddings of embeddingsGenerator) {
      embedding = embeddings[0];
    }

    // Normalize to ensure consistent L2 norm = 1.0
    const magnitude = Math.sqrt(
      embedding.reduce((sum, val) => sum + val * val, 0),
    );

    return embedding.map((val) => val / magnitude);
  }

  const chunks = chunkCode(codeString, 400);

  const embeddingsGenerator = flagEmbedding.embed(chunks);
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
