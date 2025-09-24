export function cleanSegments(segments: (string | object)[]): string[] {
  const seen = new Set<string>();
  return segments
    .map((s) => (typeof s === "string" ? s.trim() : JSON.stringify(s).trim()))
    .filter((s) => s.length > 0 && !seen.has(s) && seen.add(s));
}

export function chunkSegments(segments: string[], maxChunkSize: number): string[][] {
  const chunks: string[][] = [];
  for (let i = 0; i < segments.length; i += maxChunkSize) {
    chunks.push(segments.slice(i, i + maxChunkSize));
  }
  return chunks;
}

export function countTokens(text: string): number {
  return text.split(/\s+/).length;
}
