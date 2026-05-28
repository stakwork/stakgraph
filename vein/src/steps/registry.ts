import { readdir } from "node:fs/promises";
import { dirname, join, relative } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { AnyStepDef, StepRegistry } from "../core.js";

// ── Built-in core steps (always available) ─────────────────────────────────

import http from "./core/http.js";
import ifStep from "./core/if.js";
import loop from "./core/loop.js";
import subflow from "./core/subflow.js";
import log from "./core/log.js";
import llm from "./core/llm.js";
import wait from "./core/wait.js";

const CORE_STEPS: StepRegistry = {
  http,
  if: ifStep,
  loop,
  subflow,
  log,
  llm,
  wait,
};

export const CORE_STEP_TYPES = Object.freeze(Object.keys(CORE_STEPS));

/** Directory containing built-in lib steps, resolved relative to this file. */
const LIB_DIR = join(dirname(fileURLToPath(import.meta.url)), "lib");

// ── Auto-discovery ─────────────────────────────────────────────────────────

/**
 * Recursively find all .ts/.js files in a directory.
 */
async function findStepFiles(dir: string): Promise<string[]> {
  const files: string[] = [];

  let entries: string[];
  try {
    entries = await readdir(dir);
  } catch {
    return files; // directory doesn't exist
  }

  const { stat } = await import("node:fs/promises");

  for (const name of entries) {
    const fullPath = join(dir, name);
    const st = await stat(fullPath);
    if (st.isDirectory()) {
      const nested = await findStepFiles(fullPath);
      files.push(...nested);
    } else if (
      st.isFile() &&
      (name.endsWith(".ts") || name.endsWith(".js")) &&
      !name.startsWith("_") &&
      !name.endsWith(".test.ts") &&
      !name.endsWith(".spec.ts")
    ) {
      files.push(fullPath);
    }
  }

  return files;
}

/**
 * Derive a step type name from a file path relative to its base directory.
 *
 * - `lib/github/fetch-prs.ts` → `"github/fetch-prs"`
 * - `custom/my-scorer.ts` → `"my-scorer"`
 * - `custom/utils/parse-diff.ts` → `"utils/parse-diff"`
 */
function stepNameFromPath(filePath: string, baseDir: string): string {
  const rel = relative(baseDir, filePath);
  // Remove extension
  const withoutExt = rel.replace(/\.(ts|js)$/, "");
  // Normalize separators
  return withoutExt.split(/[/\\]/).join("/");
}

/**
 * Dynamically import a step definition file.
 */
async function loadStepFile(filePath: string): Promise<AnyStepDef | null> {
  try {
    const url = pathToFileURL(filePath).href;
    const mod = await import(url);
    const def = mod.default ?? mod;
    if (def && typeof def === "object" && "type" in def && "run" in def) {
      return def as AnyStepDef;
    }
    return null;
  } catch (err) {
    console.warn(`Warning: failed to load step from ${filePath}:`, err);
    return null;
  }
}

/**
 * Build the complete step registry by merging core steps (statically
 * imported) with lib steps (dynamically imported from `src/steps/lib/`)
 * and custom steps (dynamically imported from `<workspace>/steps/custom/`).
 *
 * Resolution order: core/ → lib/ → custom/. Higher tiers cannot shadow
 * lower ones — a name collision is skipped with a warning.
 *
 * Lib and custom steps are loaded with dynamic `import()` so their
 * dependencies are only resolved when actually used.
 */
export async function buildRegistry(workspacePath?: string): Promise<StepRegistry> {
  const registry: StepRegistry = { ...CORE_STEPS };

  await loadStepsFrom(LIB_DIR, registry, "lib");

  if (workspacePath) {
    await loadStepsFrom(join(workspacePath, "steps", "custom"), registry, "custom");
  }

  return registry;
}

/**
 * Discover and dynamically import all step files in `baseDir`, adding them
 * to `registry`. Names that collide with already-registered steps are
 * skipped with a warning.
 */
async function loadStepsFrom(
  baseDir: string,
  registry: StepRegistry,
  tier: "lib" | "custom",
): Promise<void> {
  const files = await findStepFiles(baseDir);
  for (const file of files) {
    const name = stepNameFromPath(file, baseDir);
    if (name in registry) {
      console.warn(`Warning: ${tier} step "${name}" conflicts with existing step, skipping`);
      continue;
    }
    const def = await loadStepFile(file);
    if (def) registry[name] = def;
  }
}

/**
 * Get the core-only registry (no workspace steps). Useful for testing.
 */
export function coreRegistry(): StepRegistry {
  return { ...CORE_STEPS };
}
