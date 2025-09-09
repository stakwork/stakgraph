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
  codeString: string
): Promise<number[]> {
  const flagEmbedding = await FlagEmbedding.init({
    model: MODEL,
    maxLength: 512,
  });

  // For smaller documents, use directly
  if (codeString.length < 400) {
    const embedding = await flagEmbedding.queryEmbed(codeString);
    return embedding; // Ensure we return a single vector
  }

  // Use overlapping chunks for better context capture
  const chunks = chunkCode(codeString, 400);

  // Generate embeddings for all chunks
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
    pooledEmbedding.reduce((sum, val) => sum + val * val, 0)
  );

  return pooledEmbedding.map((val) => val / magnitude);
}
