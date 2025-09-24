export function splitSentences(text: string): string[] {
  return text
    .replace(/([.!?])\s+/g, "$1\n")
    .split("\n")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

export function scoreSentences(sentences: string[], keywords: string[] = []): number[] {
  return sentences.map((s) => {
    let score = 0;
    for (const k of keywords) {
      if (s.toLowerCase().includes(k.toLowerCase())) score++;
    }
    score += s.length / 100;
    return score;
  });
}

export function extractiveSummary(text: string, maxSentences: number = 5, keywords: string[] = []): string {
  const sentences = splitSentences(text);
  const scores = scoreSentences(sentences, keywords);
  const sorted = sentences
    .map((s, i) => ({ s, score: scores[i], idx: i }))
    .sort((a, b) => b.score - a.score || a.idx - b.idx)
    .slice(0, maxSentences)
    .sort((a, b) => a.idx - b.idx)
    .map((x) => x.s);
  return sorted.join(" ");
}