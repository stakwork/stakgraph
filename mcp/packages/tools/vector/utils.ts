// Weighted pooling - giving more weight to certain chunks
export function weightedPooling(embeddings: number[][], weights: number[]) {
  const dimensions = embeddings[0].length;
  const result = new Array(dimensions).fill(0);
  let totalWeight = 0;
  for (let i = 0; i < embeddings.length; i++) {
    const weight = weights[i];
    totalWeight += weight;
    for (let j = 0; j < dimensions; j++) {
      result[j] += embeddings[i][j] * weight;
    }
  }
  // Normalize by total weight
  for (let j = 0; j < dimensions; j++) {
    result[j] /= totalWeight;
  }
  return result;
}

// Better chunking for code - respect semantic boundaries
export function chunkCode(codeString: string, chunkSize = 400) {
  // Split by potential code boundaries
  const lines = codeString.split("\n");
  const chunks = [];
  let currentChunk = [];
  let currentLength = 0;
  for (const line of lines) {
    // If adding this line would exceed our limit
    if (currentLength + line.length > chunkSize) {
      // If we already have content, finish this chunk
      if (currentLength > 0) {
        chunks.push(currentChunk.join("\n"));
        currentChunk = [];
        currentLength = 0;
      }
      // If the line itself is too long, split it
      if (line.length > chunkSize) {
        for (let i = 0; i < line.length; i += chunkSize) {
          chunks.push(line.substring(i, i + chunkSize));
        }
      } else {
        currentChunk.push(line);
        currentLength = line.length;
      }
    } else {
      currentChunk.push(line);
      currentLength += line.length;
    }
  }
  // Don't forget the last chunk
  if (currentChunk.length > 0) {
    chunks.push(currentChunk.join("\n"));
  }
  return chunks;
}

// not used
export function createOverlappingChunks(
  codeString: string,
  chunkSize = 400,
  overlap = 100
) {
  const chunks = [];
  for (let i = 0; i < codeString.length; i += chunkSize - overlap) {
    chunks.push(codeString.substring(i, i + chunkSize));
    // Stop if we've captured the entire document
    if (i + chunkSize >= codeString.length) break;
  }
  return chunks;
}
