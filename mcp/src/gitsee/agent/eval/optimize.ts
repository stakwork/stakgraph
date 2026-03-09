/**
 * GEPA optimization loop for the services agent prompts.
 *
 * Models used:
 *   - Reflection + merge: Opus  (smart, generates better prompts)
 *   - Agent evaluation:   Sonnet (cheaper, runs once per repo per generation)
 *
 * The agent model is whatever gitsee_context uses (LLM_PROVIDER env var).
 * The reflection model defaults to Opus but can be overridden.
 */

import { generateText } from "ai";
import {
  getModel,
  getApiKeyForProvider,
  type Provider,
  type ModelName,
} from "../../../aieo/src/index.js";
import {
  evaluateBatch,
  makeReflectionDataset,
  type PastAttempt,
} from "./adapter.js";
import { loadPrompts } from "./train-data.js";
import type {
  TrainingExample,
  CandidatePrompts,
  OptimizationResult,
} from "./types.js";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export interface OptimizeConfig {
  /** Training examples to optimize against */
  trainset: TrainingExample[];
  /** Validation examples (defaults to trainset) */
  valset?: TrainingExample[];
  /** Max total evaluation calls (each = one full agent run) */
  maxEvalCalls: number;
  /** Stop early if score >= this (default 0.95) */
  perfectScore?: number;
  /** Enable merge/crossover of candidates (default false) */
  useMerge?: boolean;
  /** Verbose logging (default true) */
  verbose?: boolean;
  /**
   * Model for reflection + merge. Default: "opus"
   * This is where the smart thinking happens (analyzing failures,
   * generating better prompts). Opus is worth it here.
   */
  reflectionModel?: ModelName | string;
  /** Seed prompt overrides (defaults to eval/prompts/ files) */
  seedCandidate?: CandidatePrompts;
  /** Called after each generation with the current best. Use for incremental saving. */
  onGeneration?: (state: {
    generation: number;
    bestCandidate: CandidatePrompts;
    bestScore: number;
    history: OptimizationResult["history"];
    /** Eval results from this generation (includes insights) */
    evalResults: import("./types.js").EvalResult[];
  }) => void;
}

// ---------------------------------------------------------------------------
// Reflection prompt
// ---------------------------------------------------------------------------

const REFLECTION_SYSTEM = `You are an expert prompt engineer optimizing prompts for a code exploration agent.

The agent's job is to analyze GitHub repositories and generate:
1. A pm2.config.js file (process manager config for running the project)
2. A docker-compose.yml file (auxiliary services like databases)

You will be shown the current prompts (EXPLORER system prompt and FINAL_ANSWER tool description) along with evaluation results showing where the agent succeeded and failed.

Your task is to generate IMPROVED versions of these prompts that will produce higher scores. Focus on:
- Fixing specific failures shown in the evaluation (low sub-scores)
- Being more specific about output format requirements
- Adding clearer instructions for edge cases
- Keeping instructions that already work well

IMPORTANT CONSTRAINTS:
- The EXPLORER prompt should focus on WHAT to look for during codebase exploration
- The FINAL_ANSWER prompt should focus on HOW to format the output
- Keep the MY_REPO_NAME placeholder (it gets replaced at runtime)
- The prompts must remain general enough to work across different repo types
- Do NOT add repo-specific instructions
- The FINAL_ANSWER prompt MUST keep this output format instruction (the parser depends on it):
  For each file, put "FILENAME: " followed by the filename, then the content in backticks.
  Example: FILENAME: pm2.config.js followed by a fenced code block.
  If you remove this format, the output will fail to parse!

Return your answer in this exact format:

EXPLORER:
\`\`\`
<your improved explorer prompt>
\`\`\`

FINAL_ANSWER:
\`\`\`
<your improved final_answer prompt>
\`\`\``;

// ---------------------------------------------------------------------------
// Parse reflection output
// ---------------------------------------------------------------------------

function parseReflectionOutput(output: string): CandidatePrompts | null {
  const explorerMatch = output.match(
    /EXPLORER:\s*```\s*\n?([\s\S]*?)```/
  );
  const finalAnswerMatch = output.match(
    /FINAL_ANSWER:\s*```\s*\n?([\s\S]*?)```/
  );

  if (!explorerMatch || !finalAnswerMatch) {
    console.warn("Failed to parse reflection output");
    return null;
  }

  return {
    explorer: explorerMatch[1].trim(),
    final_answer: finalAnswerMatch[1].trim(),
  };
}

// ---------------------------------------------------------------------------
// Merge two candidates
// ---------------------------------------------------------------------------

async function mergeCandidates(
  a: CandidatePrompts,
  b: CandidatePrompts,
  reflectionModel: ReturnType<typeof getModel>
): Promise<CandidatePrompts | null> {
  const mergePrompt = `You are merging two prompt variants that both performed well for a code exploration agent.

VARIANT A - EXPLORER:
\`\`\`
${a.explorer}
\`\`\`

VARIANT A - FINAL_ANSWER:
\`\`\`
${a.final_answer.substring(0, 2000)}
\`\`\`

VARIANT B - EXPLORER:
\`\`\`
${b.explorer}
\`\`\`

VARIANT B - FINAL_ANSWER:
\`\`\`
${b.final_answer.substring(0, 2000)}
\`\`\`

Create a MERGED version that combines the strengths of both. Return in this exact format:

EXPLORER:
\`\`\`
<merged explorer prompt>
\`\`\`

FINAL_ANSWER:
\`\`\`
<merged final_answer prompt>
\`\`\``;

  try {
    const { text } = await generateText({
      model: reflectionModel,
      prompt: mergePrompt,
      system: "You are an expert prompt engineer. Merge the two variants, keeping the best parts of each.",
    });
    return parseReflectionOutput(text);
  } catch (error) {
    console.warn("Merge failed:", error);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Main optimization loop
// ---------------------------------------------------------------------------

export async function optimize(
  config: OptimizeConfig
): Promise<OptimizationResult> {
  const {
    trainset,
    valset = trainset,
    maxEvalCalls,
    perfectScore = 0.95,
    useMerge = false,
    verbose = true,
    reflectionModel: reflectionModelName = "opus",
  } = config;

  const startTime = Date.now();
  let totalEvalCalls = 0;

  // Reflection + merge model: Opus by default (the smart one)
  const reflectionModel = getModel("anthropic", {
    modelName: reflectionModelName,
  });

  // The agent itself uses Sonnet (via LLM_PROVIDER / default in gitsee_context)
  // No config needed here — gitsee_context handles its own model

  // Seed candidate (current prompts from eval/prompts/ files)
  const seed: CandidatePrompts = config.seedCandidate || loadPrompts();

  // History tracking
  const history: OptimizationResult["history"] = [];
  let bestCandidate = seed;
  let bestScore = -1;
  let generation = 0;

  // Track top-k candidates for merging
  const topCandidates: Array<{ candidate: CandidatePrompts; score: number }> =
    [];

  // Accumulated memory of all past attempts (so Opus doesn't repeat itself)
  const pastAttempts: PastAttempt[] = [];

  if (verbose) {
    console.log("=== GEPA Optimization for Services Agent ===");
    console.log(`Reflection model: ${reflectionModelName} (generates better prompts)`);
    console.log(`Agent model:      Sonnet (runs the actual exploration)`);
    console.log(`Training set:     ${trainset.length} examples`);
    console.log(`Validation set:   ${valset.length} examples`);
    console.log(`Max eval calls:   ${maxEvalCalls}`);
    console.log(`Perfect score:    ${perfectScore}`);
    console.log(`Merge enabled:    ${useMerge}`);
    console.log("");
  }

  // ------- Initial evaluation -------
  if (verbose) console.log(`\n>>> Generation 0: Evaluating seed candidate...`);
  const seedResult = await evaluateBatch(trainset, seed, verbose);
  totalEvalCalls += trainset.length;

  // Store full EvalResults alongside history for reflection (avoids re-running)
  let lastEvalResults = seedResult.results;

  history.push({
    candidate: seed,
    scores: seedResult.results.map((r) => r.score.total),
    aggregate: seedResult.aggregate,
    generation: 0,
  });
  bestScore = seedResult.aggregate;
  topCandidates.push({ candidate: seed, score: seedResult.aggregate });
  pastAttempts.push({
    generation: 0,
    aggregate: seedResult.aggregate,
    explorer_preview: seed.explorer.substring(0, 150),
  });

  if (verbose) {
    console.log(`Seed score: ${(bestScore * 100).toFixed(1)}%`);
  }

  config.onGeneration?.({ generation: 0, bestCandidate: bestCandidate, bestScore, history, evalResults: seedResult.results });

  // ------- Evolution loop -------
  while (totalEvalCalls < maxEvalCalls && bestScore < perfectScore) {
    generation++;
    if (verbose)
      console.log(
        `\n>>> Generation ${generation}: Reflecting with Opus...`
      );

    // 1. Build reflection dataset from the LAST evaluation (no re-run!)
    const latestCandidate = history[history.length - 1].candidate;
    const reflectionDataset = makeReflectionDataset(
      lastEvalResults,
      latestCandidate,
      pastAttempts
    );

    // 2. Ask Opus to generate improved prompts
    let newCandidate: CandidatePrompts | null = null;
    try {
      const { text } = await generateText({
        model: reflectionModel,
        system: REFLECTION_SYSTEM,
        prompt: reflectionDataset,
      });
      if (verbose) {
        console.log(`Reflection output (first 200 chars): ${text.substring(0, 200)}`);
      }
      newCandidate = parseReflectionOutput(text);
    } catch (error) {
      console.error("Reflection failed:", error);
    }

    if (!newCandidate) {
      if (verbose) console.log("Reflection produced no valid candidate, retrying...");
      continue;
    }

    // Check if Opus actually changed anything
    if (
      newCandidate.explorer === latestCandidate.explorer &&
      newCandidate.final_answer === latestCandidate.final_answer
    ) {
      if (verbose) console.log("Opus returned identical prompts, skipping...");
      continue;
    }

    if (verbose) {
      // Show what changed
      const explorerChanged = newCandidate.explorer !== latestCandidate.explorer;
      const fadChanged = newCandidate.final_answer !== latestCandidate.final_answer;
      console.log(`  Explorer changed: ${explorerChanged}, Final answer changed: ${fadChanged}`);
    }

    // 3. Evaluate the new candidate (Sonnet runs the agent)
    if (verbose) console.log(`Evaluating reflected candidate with Sonnet...`);
    const newResult = await evaluateBatch(trainset, newCandidate, verbose);
    totalEvalCalls += trainset.length;
    lastEvalResults = newResult.results;
    history.push({
      candidate: newCandidate,
      scores: newResult.results.map((r) => r.score.total),
      aggregate: newResult.aggregate,
      generation,
    });

    // Remember this attempt so Opus doesn't repeat it
    pastAttempts.push({
      generation,
      aggregate: newResult.aggregate,
      explorer_preview: newCandidate.explorer.substring(0, 150),
    });

    // Track top candidates
    topCandidates.push({
      candidate: newCandidate,
      score: newResult.aggregate,
    });
    topCandidates.sort((a, b) => b.score - a.score);
    if (topCandidates.length > 5) topCandidates.pop();

    // Update best
    if (newResult.aggregate > bestScore) {
      bestScore = newResult.aggregate;
      bestCandidate = newCandidate;
      if (verbose)
        console.log(
          `New best! Score: ${(bestScore * 100).toFixed(1)}% (gen ${generation})`
        );
    } else {
      if (verbose)
        console.log(
          `No improvement: ${(newResult.aggregate * 100).toFixed(1)}% vs best ${(bestScore * 100).toFixed(1)}%`
        );
    }

    config.onGeneration?.({ generation, bestCandidate, bestScore, history, evalResults: newResult.results });

    // 4. Optional merge step (also uses Opus)
    if (useMerge && topCandidates.length >= 2 && totalEvalCalls < maxEvalCalls) {
      if (verbose) console.log("Merging top 2 candidates with Opus...");
      const merged = await mergeCandidates(
        topCandidates[0].candidate,
        topCandidates[1].candidate,
        reflectionModel
      );
      if (merged) {
        const mergeResult = await evaluateBatch(trainset, merged, verbose);
        totalEvalCalls += trainset.length;
        history.push({
          candidate: merged,
          scores: mergeResult.results.map((r) => r.score.total),
          aggregate: mergeResult.aggregate,
          generation,
        });
        if (mergeResult.aggregate > bestScore) {
          bestScore = mergeResult.aggregate;
          bestCandidate = merged;
          topCandidates.push({
            candidate: merged,
            score: mergeResult.aggregate,
          });
          topCandidates.sort((a, b) => b.score - a.score);
          if (verbose)
            console.log(
              `Merge improved! Score: ${(bestScore * 100).toFixed(1)}%`
            );
        }
      }
    }
  }

  // ------- Final validation -------
  if (verbose) {
    console.log(`\n>>> Final validation on val set...`);
  }
  const valResult = await evaluateBatch(valset, bestCandidate, verbose);
  const finalScore = valResult.aggregate;

  const totalDuration = Date.now() - startTime;

  if (verbose) {
    console.log(`\n=== Optimization Complete ===`);
    console.log(`Generations: ${generation}`);
    console.log(`Total eval calls: ${totalEvalCalls}`);
    console.log(`Best training score: ${(bestScore * 100).toFixed(1)}%`);
    console.log(`Final validation score: ${(finalScore * 100).toFixed(1)}%`);
    console.log(`Total time: ${(totalDuration / 1000).toFixed(0)}s`);
    console.log(`\n--- Best EXPLORER prompt ---`);
    console.log(bestCandidate.explorer);
    console.log(`\n--- Best FINAL_ANSWER prompt (first 500 chars) ---`);
    console.log(bestCandidate.final_answer.substring(0, 500));
  }

  return {
    best_candidate: bestCandidate,
    best_score: finalScore,
    history,
    total_eval_calls: totalEvalCalls,
    total_duration_ms: totalDuration,
  };
}
