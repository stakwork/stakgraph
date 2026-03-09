/**
 * GEPA adapter for the services agent.
 *
 * This wraps `gitsee_context` so that GEPA can:
 * 1. Inject evolved EXPLORER + FINAL_ANSWER prompts
 * 2. Run the agent against a repo
 * 3. Score the output against gold standards
 *
 * The adapter handles cloning, running, parsing, and scoring.
 */

import { gitsee_context } from "../explore.js";
import { RepoCloner } from "../repo-cloner.js";
import { parse_files_contents } from "../utils.js";
import { score, formatScore } from "./scoring.js";
import type {
  TrainingExample,
  CandidatePrompts,
  EvalResult,
  ScoreBreakdown,
} from "./types.js";

// ---------------------------------------------------------------------------
// The standard prompt sent to the agent (same as repo/services.ts)
// ---------------------------------------------------------------------------
const SERVICES_PROMPT =
  "How do I set up this repo? I want to run the project on my remote code-server environment. Please prioritize web services that I will be able to run there (so ignore fancy stuff like web extension, desktop app using electron, etc). Lets just focus on bare-bones setup to install, build, and run a web frontend, and supporting services like the backend.";

// ---------------------------------------------------------------------------
// Evaluate a single example with a candidate prompt set
// ---------------------------------------------------------------------------

export async function evaluateExample(
  example: TrainingExample,
  candidate: CandidatePrompts,
  verbose = false
): Promise<EvalResult> {
  const startTime = Date.now();

  // 1. Ensure repo is cloned
  await RepoCloner.waitForClone(example.owner, example.repo);
  const cloneResult = await RepoCloner.getCloneResult(
    example.owner,
    example.repo
  );
  if (!cloneResult?.success) {
    throw new Error(
      `Failed to clone ${example.owner}/${example.repo}: ${cloneResult}`
    );
  }
  const repoPath = cloneResult.localPath;
  const repoName = example.repo;

  // 2. Prepare the final_answer with repo name substituted
  const final_answer = candidate.final_answer.replaceAll(
    "MY_REPO_NAME",
    repoName
  );

  // 3. Run the agent with overridden prompts
  //    We pass the evolved system prompt and final_answer via overrides.
  //    The prompt itself stays the same (it's the user's question).
  const prompt = SERVICES_PROMPT + "\n" + final_answer;
  const raw_output = await gitsee_context(prompt, repoPath, "services", {
    system_prompt: candidate.explorer,
    final_answer_description: final_answer,
  });

  // 4. Parse the output into files
  const files = parse_files_contents(raw_output);
  const generated_pm2 = files["pm2.config.js"] || "";
  const generated_docker_compose = files["docker-compose.yml"] || "";

  // 5. Score against gold standard
  const scoreResult = score(generated_pm2, generated_docker_compose, example);

  const duration_ms = Date.now() - startTime;

  if (verbose) {
    console.log(`\n--- ${example.owner}/${example.repo} ---`);
    console.log(formatScore(scoreResult));
    console.log(`  duration: ${(duration_ms / 1000).toFixed(1)}s`);
  }

  return {
    example,
    generated_pm2,
    generated_docker_compose,
    score: scoreResult,
    raw_output,
    duration_ms,
  };
}

// ---------------------------------------------------------------------------
// Evaluate a candidate across an entire dataset
// ---------------------------------------------------------------------------

export async function evaluateBatch(
  examples: TrainingExample[],
  candidate: CandidatePrompts,
  verbose = false
): Promise<{ results: EvalResult[]; aggregate: number }> {
  const results: EvalResult[] = [];

  for (const example of examples) {
    try {
      const result = await evaluateExample(example, candidate, verbose);
      results.push(result);
    } catch (error) {
      console.error(
        `Error evaluating ${example.owner}/${example.repo}:`,
        error
      );
      // Push a zero-score result on failure
      results.push({
        example,
        generated_pm2: "",
        generated_docker_compose: "",
        score: {
          format_score: 0,
          pm2_structure_score: 0,
          docker_structure_score: 0,
          env_vars_score: 0,
          app_service_score: 0,
          cwd_score: 0,
          frontend_naming_score: 0,
          total: 0,
        },
        raw_output: `ERROR: ${error}`,
        duration_ms: 0,
      });
    }
  }

  const aggregate =
    results.length > 0
      ? results.reduce((sum, r) => sum + r.score.total, 0) / results.length
      : 0;

  if (verbose) {
    console.log(`\n========== Aggregate score: ${(aggregate * 100).toFixed(1)}% ==========\n`);
  }

  return { results, aggregate };
}

// ---------------------------------------------------------------------------
// Build the reflection dataset for GEPA
// ---------------------------------------------------------------------------

/** Summary of a previous attempt, for Opus's memory */
export interface PastAttempt {
  generation: number;
  aggregate: number;
  /** Short description of what changed in the explorer prompt */
  explorer_preview: string;
}

/**
 * Creates a structured dataset that the reflection LLM uses to understand
 * what went wrong and generate improved prompts.
 *
 * Includes history of all previous attempts so Opus doesn't repeat itself.
 */
export function makeReflectionDataset(
  results: EvalResult[],
  candidate: CandidatePrompts,
  pastAttempts: PastAttempt[] = []
): string {
  const sections: string[] = [];

  // -- History of past attempts (so Opus doesn't go in circles) --
  if (pastAttempts.length > 0) {
    sections.push(`# Previous Attempts (DO NOT repeat these)\n`);
    for (const attempt of pastAttempts) {
      sections.push(
        `- Gen ${attempt.generation}: scored ${(attempt.aggregate * 100).toFixed(1)}% — explorer started with: "${attempt.explorer_preview}"`
      );
    }
    sections.push("");
    sections.push(
      `You have already tried ${pastAttempts.length} variations. ` +
        `Do NOT regenerate a prompt similar to any of the above. ` +
        `Try a meaningfully different approach.\n`
    );
  }

  // -- Current prompts --
  sections.push(`# Current Prompts Being Evaluated\n`);
  sections.push(`## EXPLORER (system prompt):\n\`\`\`\n${candidate.explorer}\n\`\`\`\n`);
  sections.push(
    `## FINAL_ANSWER (tool description):\n\`\`\`\n${candidate.final_answer.substring(0, 2000)}...\n\`\`\`\n`
  );

  // -- Eval results --
  sections.push(`# Evaluation Results\n`);

  for (const result of results) {
    const ex = result.example;
    const s = result.score;
    sections.push(`## ${ex.owner}/${ex.repo}`);
    if (ex.notes) sections.push(`Notes: ${ex.notes}`);
    sections.push(`Score: ${(s.total * 100).toFixed(1)}%`);
    sections.push(`Breakdown:`);
    sections.push(`  format: ${(s.format_score * 100).toFixed(0)}%`);
    sections.push(`  pm2_structure: ${(s.pm2_structure_score * 100).toFixed(0)}%`);
    sections.push(`  docker_structure: ${(s.docker_structure_score * 100).toFixed(0)}%`);
    sections.push(`  env_vars: ${(s.env_vars_score * 100).toFixed(0)}%`);
    sections.push(`  app_service: ${(s.app_service_score * 100).toFixed(0)}%`);
    sections.push(`  cwd: ${(s.cwd_score * 100).toFixed(0)}%`);
    sections.push(`  frontend_naming: ${(s.frontend_naming_score * 100).toFixed(0)}%`);

    // Show what was generated vs expected (truncated)
    if (s.total < 0.9) {
      sections.push(`\n### Generated pm2.config.js (first 500 chars):`);
      sections.push(
        `\`\`\`\n${result.generated_pm2.substring(0, 500)}\n\`\`\``
      );
      sections.push(`### Expected pm2.config.js (first 500 chars):`);
      sections.push(
        `\`\`\`\n${ex.gold_pm2.substring(0, 500)}\n\`\`\``
      );
      sections.push(`\n### Generated docker-compose.yml (first 500 chars):`);
      sections.push(
        `\`\`\`\n${result.generated_docker_compose.substring(0, 500)}\n\`\`\``
      );
      sections.push(`### Expected docker-compose.yml (first 500 chars):`);
      sections.push(
        `\`\`\`\n${ex.gold_docker_compose.substring(0, 500)}\n\`\`\``
      );
    }
    sections.push("");
  }

  return sections.join("\n");
}
