/**
 * Simple scoring: did the agent produce the expected files?
 *
 * 0   = missing files or empty
 * 0.5 = files produced but content differs significantly
 * 1   = files produced and content is close
 *
 * The real evaluation is Opus reading the diff during reflection.
 * This score just tracks progress across generations.
 */

import type { ScoreResult, TrainingExample } from "./types.js";

export function score(
  generatedFiles: Record<string, string>,
  example: TrainingExample
): ScoreResult {
  const expectedNames = Object.keys(example.expected_files);
  const reasons: string[] = [];

  if (expectedNames.length === 0) {
    return { total: 1, reason: "No expected files to compare." };
  }

  let fileScores: number[] = [];

  for (const name of expectedNames) {
    const expected = example.expected_files[name]?.trim() || "";
    const generated = generatedFiles[name]?.trim() || "";

    if (!generated) {
      reasons.push(`${name}: MISSING`);
      fileScores.push(0);
      continue;
    }

    if (generated.length < 10) {
      reasons.push(`${name}: nearly empty (${generated.length} chars)`);
      fileScores.push(0);
      continue;
    }

    // Simple similarity: what fraction of expected lines appear in generated?
    const expectedLines = new Set(
      expected
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l.length > 0 && !l.startsWith("#") && !l.startsWith("//"))
    );
    const generatedLines = new Set(
      generated
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l.length > 0)
    );

    if (expectedLines.size === 0) {
      reasons.push(`${name}: produced (no meaningful expected lines to compare)`);
      fileScores.push(0.5);
      continue;
    }

    let matches = 0;
    for (const line of expectedLines) {
      if (generatedLines.has(line)) {
        matches++;
      }
    }
    const lineRecall = matches / expectedLines.size;

    if (lineRecall > 0.8) {
      reasons.push(`${name}: PASS (${(lineRecall * 100).toFixed(0)}% line recall)`);
      fileScores.push(1);
    } else if (lineRecall > 0.3) {
      reasons.push(`${name}: PARTIAL (${(lineRecall * 100).toFixed(0)}% line recall)`);
      fileScores.push(0.5);
    } else {
      reasons.push(`${name}: FAIL (${(lineRecall * 100).toFixed(0)}% line recall)`);
      fileScores.push(0);
    }
  }

  const total =
    fileScores.length > 0
      ? fileScores.reduce((a, b) => a + b, 0) / fileScores.length
      : 0;

  return {
    total,
    reason: reasons.join("\n"),
  };
}

/** One-line summary */
export function formatScore(s: ScoreResult): string {
  return `${(s.total * 100).toFixed(0)}% — ${s.reason}`;
}
