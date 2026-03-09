/**
 * Scoring via Opus-as-judge.
 *
 * Shows Opus the expected output and the generated output,
 * asks it to score 0 / 0.5 / 1 and explain why.
 *
 * Catches things regex can't: internal consistency (passwords match
 * between docker-compose and pm2 env), semantic equivalence
 * (different port number but correctly wired), etc.
 */

import { generateText } from "ai";
import {
  getModel,
  type Provider,
  type ModelName,
} from "../../../aieo/src/index.js";
import type { ScoreResult, TrainingExample } from "./types.js";

const JUDGE_SYSTEM = `You are evaluating whether a generated configuration would actually work for running a project.

You will see EXPECTED output (known-good) and GENERATED output (what the agent produced).

Score the generated output:
- 1   = Would work. Services, ports, env vars, and commands are correct. Minor cosmetic differences are fine. Passwords/secrets don't need to match the expected output, but they MUST be internally consistent (e.g. same password in docker-compose and DATABASE_URL).
- 0.5 = Partially correct. Structure is right but has issues that would cause failures (wrong port, missing env var, inconsistent credentials, missing service).
- 0   = Wrong or missing. Would not work at all.

You MUST respond in exactly this format:
SCORE: <0 or 0.5 or 1>
REASON: <1-3 sentences explaining why>`;

let _judgeModel: ReturnType<typeof getModel> | null = null;

function getJudgeModel(): ReturnType<typeof getModel> {
  if (!_judgeModel) {
    _judgeModel = getModel("anthropic", { modelName: "opus" });
  }
  return _judgeModel;
}

/**
 * Score a single example using Opus as judge.
 * Falls back to a basic "files present" check if the LLM call fails.
 */
export async function score(
  generatedFiles: Record<string, string>,
  example: TrainingExample
): Promise<ScoreResult> {
  const expectedNames = Object.keys(example.expected_files);

  if (expectedNames.length === 0) {
    return { total: 1, reason: "No expected files to compare." };
  }

  // Quick check: are any files completely missing?
  const missing = expectedNames.filter(
    (name) => !generatedFiles[name]?.trim()
  );
  if (missing.length === expectedNames.length) {
    return { total: 0, reason: `All files missing: ${missing.join(", ")}` };
  }

  // Build the comparison prompt
  const sections: string[] = [];
  for (const name of expectedNames) {
    const expected = example.expected_files[name]?.trim() || "(empty)";
    const generated = generatedFiles[name]?.trim() || "(not produced)";
    sections.push(`## ${name}\n`);
    sections.push(`### Expected:\n\`\`\`\n${expected}\n\`\`\`\n`);
    sections.push(`### Generated:\n\`\`\`\n${generated}\n\`\`\`\n`);
  }

  const prompt = sections.join("\n");

  try {
    const judge = getJudgeModel();
    const { text } = await generateText({
      model: judge,
      system: JUDGE_SYSTEM,
      prompt,
    });

    return parseJudgeResponse(text);
  } catch (error) {
    console.warn("Judge LLM failed, falling back to basic check:", error);
    // Fallback: at least some files were produced
    const produced = expectedNames.filter(
      (name) => (generatedFiles[name]?.trim().length || 0) > 10
    );
    const ratio = produced.length / expectedNames.length;
    return {
      total: ratio > 0.5 ? 0.5 : 0,
      reason: `Judge failed. ${produced.length}/${expectedNames.length} files produced.`,
    };
  }
}

function parseJudgeResponse(text: string): ScoreResult {
  const scoreMatch = text.match(/SCORE:\s*(0\.5|0|1)/);
  const reasonMatch = text.match(/REASON:\s*(.+)/s);

  const total = scoreMatch ? parseFloat(scoreMatch[1]) : 0;
  const reason = reasonMatch
    ? reasonMatch[1].trim().split("\n")[0] // first line only
    : "Could not parse judge response";

  return { total, reason };
}

/** One-line summary */
export function formatScore(s: ScoreResult): string {
  const label =
    s.total === 1 ? "PASS" : s.total === 0.5 ? "PARTIAL" : "FAIL";
  return `${label} (${s.total}) — ${s.reason}`;
}
