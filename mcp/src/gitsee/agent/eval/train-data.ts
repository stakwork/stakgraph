/**
 * Loads training data and prompts from the filesystem.
 *
 * Training examples:
 *   eval/train/{owner}--{repo}/
 *     pm2.config.js           (or any expected output files)
 *     docker-compose.yml
 *     notes.txt               (optional)
 *
 * Prompts (what GEPA evolves):
 *   eval/prompts/
 *     explorer.md             system prompt
 *     final_answer.md         tool description
 *
 * Any file in the repo dir that isn't notes.txt is treated
 * as an expected output file.
 */

import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import type { TrainingExample, CandidatePrompts } from "./types.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const IGNORED_FILES = new Set(["notes.txt"]);

// ---------------------------------------------------------------------------
// Load training/validation examples
// ---------------------------------------------------------------------------

function loadExamplesFromDir(dir: string): TrainingExample[] {
  if (!fs.existsSync(dir)) return [];

  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const examples: TrainingExample[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;

    const parts = entry.name.split("--");
    if (parts.length !== 2) {
      console.warn(`Skipping ${entry.name}: expected {owner}--{repo} format`);
      continue;
    }

    const [owner, repo] = parts;
    const exDir = path.join(dir, entry.name);

    // Load all files in the directory as expected outputs
    const files = fs.readdirSync(exDir).filter(
      (f) => !IGNORED_FILES.has(f) && fs.statSync(path.join(exDir, f)).isFile()
    );

    if (files.length === 0) {
      console.warn(`Skipping ${entry.name}: no expected output files`);
      continue;
    }

    const expected_files: Record<string, string> = {};
    for (const file of files) {
      expected_files[file] = fs.readFileSync(path.join(exDir, file), "utf-8");
    }

    const notesPath = path.join(exDir, "notes.txt");
    const notes = fs.existsSync(notesPath)
      ? fs.readFileSync(notesPath, "utf-8").trim()
      : undefined;

    examples.push({ owner, repo, expected_files, notes });
  }

  return examples;
}

// ---------------------------------------------------------------------------
// Load prompts from files
// ---------------------------------------------------------------------------

const PROMPTS_DIR = path.join(__dirname, "prompts");

function loadPrompt(filename: string): string {
  const p = path.join(PROMPTS_DIR, filename);
  if (!fs.existsSync(p)) {
    throw new Error(`Missing prompt file: ${p}`);
  }
  return fs.readFileSync(p, "utf-8").trim();
}

export function loadPrompts(): CandidatePrompts {
  return {
    explorer: loadPrompt("explorer.md"),
    final_answer: loadPrompt("final_answer.md"),
  };
}

export function savePrompts(candidate: CandidatePrompts): void {
  fs.writeFileSync(
    path.join(PROMPTS_DIR, "explorer.md"),
    candidate.explorer
  );
  fs.writeFileSync(
    path.join(PROMPTS_DIR, "final_answer.md"),
    candidate.final_answer
  );
  console.log(`Prompts saved to ${PROMPTS_DIR}/`);
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

export const TRAIN_SET = loadExamplesFromDir(path.join(__dirname, "train"));
export const VAL_SET = loadExamplesFromDir(path.join(__dirname, "val"));
