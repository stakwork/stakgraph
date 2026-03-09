/**
 * Loads training data and prompts from the filesystem.
 *
 * Training examples:
 *   eval/train/{owner}--{repo}/
 *     pm2.config.js
 *     docker-compose.yml
 *     notes.txt             (optional)
 *
 * Prompts (what GEPA evolves):
 *   eval/prompts/
 *     explorer.md           system prompt
 *     final_answer.md       tool description
 */

import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import type { TrainingExample, CandidatePrompts } from "./types.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

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

    const pm2Path = path.join(exDir, "pm2.config.js");
    const dockerPath = path.join(exDir, "docker-compose.yml");

    if (!fs.existsSync(pm2Path) || !fs.existsSync(dockerPath)) {
      console.warn(
        `Skipping ${entry.name}: missing pm2.config.js or docker-compose.yml`
      );
      continue;
    }

    const notesPath = path.join(exDir, "notes.txt");
    const notes = fs.existsSync(notesPath)
      ? fs.readFileSync(notesPath, "utf-8").trim()
      : undefined;

    examples.push({
      owner,
      repo,
      gold_pm2: fs.readFileSync(pm2Path, "utf-8"),
      gold_docker_compose: fs.readFileSync(dockerPath, "utf-8"),
      notes,
    });
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
