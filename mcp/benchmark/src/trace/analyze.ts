import { stringify, previewStr } from "../utils";
import type { CallEntry, ResultEntry, IssueKind, AnalyzedStep, TraceAnalysis } from "./types";

function normaliseText(s: string): string {
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}

function payloadMetrics(value: unknown): {
  text: string;
  size: number;
  lines: number;
} {
  const text = stringify(value);
  return { text, size: text.length, lines: text.split(/\r?\n/).length };
}

function hasNamedConcept(prompt: string): boolean {
  const p = prompt.trim();
  if (!p) return false;
  return /`[^`]+`|\b[a-z]+[A-Z][A-Za-z0-9]*\b|\b[A-Za-z_][A-Za-z0-9_]*\(|\b[A-Za-z0-9_-]+\/[A-Za-z0-9_.-]+\b|\b[a-z_]{3,}[a-z0-9_]*_[a-z0-9_]+\b/.test(
    p,
  );
}

function dedupeFlags(flags: IssueKind[]): IssueKind[] {
  return [...new Set(flags)];
}

export function analyzeTrace(
  parsed: { calls: CallEntry[]; results: ResultEntry[] },
  prompt: string,
): TraceAnalysis {
  const resultById = new Map(parsed.results.map((r) => [r.id, r]));
  const steps: AnalyzedStep[] = parsed.calls.map((call) => {
    const result = resultById.get(call.id);
    const output = result?.output;
    const metrics = payloadMetrics(output);
    return {
      id: call.id,
      index: call.index,
      toolName: call.toolName,
      input: call.input,
      output,
      outputSize: metrics.size,
      outputLines: metrics.lines,
      flags: [],
    };
  });

  const promptHasConcept = hasNamedConcept(prompt);
  for (let i = 0; i < steps.length; i += 1) {
    const step = steps[i];
    const prev = steps[i - 1];
    const prev2 = steps[i - 2];
    const outputText = normaliseText(stringify(step.output));
    const inputText = normaliseText(previewStr(step.input));
    const flags: IssueKind[] = [];

    if (
      !outputText ||
      outputText === "[]" ||
      outputText === "{}" ||
      outputText === "null" ||
      outputText === "\u2014" ||
      /not found|no results|no matches|empty/i.test(outputText) ||
      step.outputSize < 16
    ) {
      flags.push("empty");
    }
    if (step.outputSize > 4000 || step.outputLines > 80)
      flags.push("oversized");
    if (
      (prev &&
        prev.toolName === step.toolName &&
        normaliseText(previewStr(prev.input)) === inputText) ||
      (prev &&
        prev2 &&
        prev.toolName === step.toolName &&
        prev2.toolName === step.toolName)
    ) {
      flags.push("repeat");
    }
    if (
      prev &&
      prev.toolName === step.toolName &&
      normaliseText(stringify(prev.output)) === outputText &&
      step.outputSize > 0
    ) {
      flags.push("duplicate");
    }
    if (
      (step.toolName === "bash" || step.toolName === "fulltext_search") &&
      prev &&
      (prev.toolName.startsWith("stakgraph") ||
        prev.toolName === "repo_overview") &&
      !prev.flags.includes("empty")
    ) {
      flags.push("fallback");
    }
    if (step.toolName === "repo_overview" && promptHasConcept) {
      flags.push("noisy-overview");
    }
    step.flags = dedupeFlags(flags);
  }

  const counts: Record<IssueKind, number> = {
    empty: 0,
    oversized: 0,
    repeat: 0,
    fallback: 0,
    duplicate: 0,
    "noisy-overview": 0,
  };
  for (const step of steps) {
    for (const flag of step.flags) counts[flag] += 1;
  }
  return { steps, counts };
}
