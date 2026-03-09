#!/usr/bin/env tsx
/**
 * CLI for GEPA optimization of the services agent.
 *
 * Usage:
 *   npx tsx eval/run.ts --eval
 *   npx tsx eval/run.ts --optimize
 *   npx tsx eval/run.ts --optimize --max-evals 30 --merge
 *   npx tsx eval/run.ts --apply eval/runs/20260309-143022
 */

import "dotenv/config";
import { optimize } from "./optimize.js";
import { evaluateBatch } from "./adapter.js";
import { TRAIN_SET, VAL_SET, loadPrompts } from "./train-data.js";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const RUNS_DIR = path.join(__dirname, "runs");
const SERVICES_TS = path.join(__dirname, "..", "prompts", "services.ts");

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const args = process.argv.slice(2);
const flags = new Set(args.filter((a) => a.startsWith("--")));
const getArg = (name: string): string | undefined => {
  const idx = args.indexOf(name);
  return idx !== -1 && idx + 1 < args.length ? args[idx + 1] : undefined;
};

const MODE = flags.has("--optimize")
  ? "optimize"
  : flags.has("--eval")
  ? "eval"
  : flags.has("--apply")
  ? "apply"
  : flags.has("--list")
  ? "list"
  : "help";

// ---------------------------------------------------------------------------
// Save a run to eval/runs/{timestamp}/
// ---------------------------------------------------------------------------

function saveRun(
  result: Awaited<ReturnType<typeof optimize>>,
  trainset: typeof TRAIN_SET,
  runDir?: string
) {
  const ts = new Date()
    .toISOString()
    .replace(/[T:]/g, "-")
    .replace(/\..+/, "");
  if (!runDir) {
    runDir = path.join(RUNS_DIR, ts);
  }
  fs.mkdirSync(runDir, { recursive: true });

  // Save prompts as readable files
  fs.writeFileSync(
    path.join(runDir, "explorer.md"),
    result.best_candidate.explorer
  );
  fs.writeFileSync(
    path.join(runDir, "final_answer.md"),
    result.best_candidate.final_answer
  );

  // Save score summary
  const summary = [
    `Score: ${(result.best_score * 100).toFixed(1)}%`,
    `Eval calls: ${result.total_eval_calls}`,
    `Duration: ${(result.total_duration_ms / 1000).toFixed(0)}s`,
    `Generations: ${result.history.length}`,
    "",
    "Per-generation scores:",
    ...result.history.map(
      (h) => `  gen ${h.generation}: ${(h.aggregate * 100).toFixed(1)}%`
    ),
    "",
    "Training repos:",
    ...trainset.map((t) => `  ${t.owner}/${t.repo}`),
  ].join("\n");
  fs.writeFileSync(path.join(runDir, "summary.txt"), summary);

  // Save full history as JSON
  fs.writeFileSync(
    path.join(runDir, "history.json"),
    JSON.stringify(
      {
        best_score: result.best_score,
        total_eval_calls: result.total_eval_calls,
        total_duration_ms: result.total_duration_ms,
        history: result.history.map((h) => ({
          generation: h.generation,
          aggregate: h.aggregate,
          scores: h.scores,
        })),
      },
      null,
      2
    )
  );

  return { runDir, ts };
}

// ---------------------------------------------------------------------------
// Apply a run's prompts to prompts/services.ts
// ---------------------------------------------------------------------------

function applyRun(runDir: string) {
  const explorerPath = path.join(runDir, "explorer.md");
  const finalAnswerPath = path.join(runDir, "final_answer.md");

  if (!fs.existsSync(explorerPath) || !fs.existsSync(finalAnswerPath)) {
    console.error(`Missing explorer.md or final_answer.md in ${runDir}`);
    process.exit(1);
  }

  const explorer = fs.readFileSync(explorerPath, "utf-8").trim();
  const finalAnswer = fs.readFileSync(finalAnswerPath, "utf-8").trim();

  // Read current services.ts
  const current = fs.readFileSync(SERVICES_TS, "utf-8");

  // Replace the EXPLORER and FINAL_ANSWER exports
  // Use backtick template literals, escaping any backticks in the content
  const escapedExplorer = explorer.replace(/`/g, "\\`").replace(/\$/g, "\\$");
  const escapedFinalAnswer = finalAnswer.replace(/`/g, "\\`").replace(/\$/g, "\\$");

  const newContent = `export const FILE_LINES = 100;

export const EXPLORER = \`
${escapedExplorer}
\`;

export const FINAL_ANSWER = \`
${escapedFinalAnswer}
\`;
`;

  fs.writeFileSync(SERVICES_TS, newContent);
  console.log(`Applied prompts from ${runDir} to prompts/services.ts`);
}

// ---------------------------------------------------------------------------
// List all runs
// ---------------------------------------------------------------------------

function listRuns() {
  if (!fs.existsSync(RUNS_DIR)) {
    console.log("No runs yet.");
    return;
  }
  const entries = fs.readdirSync(RUNS_DIR, { withFileTypes: true });
  const runs = entries.filter((e) => e.isDirectory()).sort();

  if (runs.length === 0) {
    console.log("No runs yet.");
    return;
  }

  console.log("Optimization runs:\n");
  for (const run of runs) {
    const summaryPath = path.join(RUNS_DIR, run.name, "summary.txt");
    if (fs.existsSync(summaryPath)) {
      const summary = fs.readFileSync(summaryPath, "utf-8");
      const scoreLine = summary.split("\n")[0];
      console.log(`  ${run.name}  ${scoreLine}`);
    } else {
      console.log(`  ${run.name}`);
    }
  }
  console.log(`\nApply one with: npx tsx eval/run.ts --apply eval/runs/{name}`);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  if (MODE === "help") {
    console.log(`
GEPA Services Agent Optimizer

Commands:
  --eval                    Score current prompts against training set
  --optimize                Run GEPA evolution loop
  --apply <run-dir>         Apply a run's prompts to prompts/services.ts
  --list                    List all optimization runs

Options:
  --max-evals <n>           Max eval calls (default: 20)
  --merge                   Enable candidate merging
  --reflection-model <m>    Model for reflection/merge (default: opus)
  --quiet                   Less output

Models:
  Reflection + merge:  Opus   (analyzes failures, generates better prompts)
  Agent evaluation:    Sonnet (runs the actual repo exploration)
    `);
    return;
  }

  if (MODE === "list") {
    listRuns();
    return;
  }

  if (MODE === "apply") {
    const runDir = getArg("--apply");
    if (!runDir) {
      console.error("Usage: --apply <run-dir>");
      process.exit(1);
    }
    const resolved = path.resolve(runDir);
    applyRun(resolved);
    return;
  }

  // Load training data
  const trainset = TRAIN_SET;
  if (trainset.length === 0) {
    console.error(
      "No training examples found.\n" +
        "Add directories to eval/train/{owner}--{repo}/ with pm2.config.js and docker-compose.yml"
    );
    process.exit(1);
  }
  const valset = VAL_SET.length > 0 ? VAL_SET : trainset;

  console.log(`Training: ${trainset.length} repos`);
  for (const t of trainset) {
    console.log(`  ${t.owner}/${t.repo}${t.notes ? ` (${t.notes})` : ""}`);
  }
  console.log("");

  const prompts = loadPrompts();
  const verbose = !flags.has("--quiet");

  if (MODE === "eval") {
    console.log("=== Evaluating current prompts ===\n");
    const { results, aggregate } = await evaluateBatch(
      trainset,
      prompts,
      verbose
    );

    console.log("\n=== Summary ===");
    for (const r of results) {
      console.log(
        `${r.example.owner}/${r.example.repo}: ${(r.score.total * 100).toFixed(1)}% (${(r.duration_ms / 1000).toFixed(1)}s)`
      );
    }
    console.log(`\nAggregate: ${(aggregate * 100).toFixed(1)}%`);
  }

  if (MODE === "optimize") {
    const maxEvals = parseInt(getArg("--max-evals") || "20", 10);
    const useMerge = flags.has("--merge");
    const reflectionModel = getArg("--reflection-model") || "opus";

    // Create run dir upfront so we can save incrementally
    const ts = new Date()
      .toISOString()
      .replace(/[T:]/g, "-")
      .replace(/\..+/, "");
    const runDir = path.join(RUNS_DIR, ts);
    fs.mkdirSync(runDir, { recursive: true });
    console.log(`Saving to: eval/runs/${ts}/\n`);

    const result = await optimize({
      trainset,
      valset,
      maxEvalCalls: maxEvals,
      useMerge,
      verbose,
      seedCandidate: prompts,
      reflectionModel,
      onGeneration: ({ generation, bestCandidate, bestScore, history }) => {
        // Save best prompts after every generation
        fs.writeFileSync(
          path.join(runDir, "explorer.md"),
          bestCandidate.explorer
        );
        fs.writeFileSync(
          path.join(runDir, "final_answer.md"),
          bestCandidate.final_answer
        );
        fs.writeFileSync(
          path.join(runDir, "summary.txt"),
          [
            `Score: ${(bestScore * 100).toFixed(1)}%`,
            `Generation: ${generation}`,
            `Training repos:`,
            ...trainset.map((t) => `  ${t.owner}/${t.repo}`),
            "",
            "Per-generation scores:",
            ...history.map(
              (h) => `  gen ${h.generation}: ${(h.aggregate * 100).toFixed(1)}%`
            ),
          ].join("\n")
        );
        if (verbose) {
          console.log(`  [saved gen ${generation} to eval/runs/${ts}/]`);
        }
      },
    });

    // Final save with complete stats
    saveRun(result, trainset, runDir);

    console.log(`\nRun saved to: eval/runs/${ts}/`);
    console.log(`  explorer.md       - evolved system prompt`);
    console.log(`  final_answer.md   - evolved tool description`);
    console.log(`  summary.txt       - score + metadata`);
    console.log(`  history.json      - full generation history`);
    console.log(
      `\nTo apply: npx tsx eval/run.ts --apply ${runDir}`
    );
  }
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
