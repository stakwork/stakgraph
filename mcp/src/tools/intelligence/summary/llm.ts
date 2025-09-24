import { extractiveSummary } from "./extractive.js";

export async function llmSummary(
  text: string,
  opts: { maxSentences?: number; keywords?: string[]; llm?: (input: string) => Promise<string> } = {}
): Promise<string> {
  if (opts.llm) {
    try {
      const result = await opts.llm(text);
      if (result && result.trim().length > 0) return result.trim();
    } catch (e) {}
  }
  return extractiveSummary(text, opts.maxSentences, opts.keywords);
}
