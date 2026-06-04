import { readdir, readFile, stat } from "node:fs/promises";
import { dirname, join, relative } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { AnyStepDef, StepRegistry } from "../core.js";

/** Where a registered step type came from. */
export type StepSource = "core" | "lib" | "custom";

/** Map of step type name → its source tier. */
export type StepSources = Record<string, StepSource>;

/** Result of building the registry: the registry itself plus a parallel
 *  map recording where each step was loaded from. */
export interface RegistryBundle {
  registry: StepRegistry;
  sources: StepSources;
}

// ── Built-in core steps (always available) ─────────────────────────────────

import http from "./core/http.js";
import ifStep from "./core/if.js";
import loop from "./core/loop.js";
import foreach from "./core/foreach.js";
import subflow from "./core/subflow.js";
import log from "./core/log.js";
import llm from "./core/llm.js";
import agent from "./core/agent.js";
import wait from "./core/wait.js";

const CORE_STEPS: StepRegistry = {
  http,
  if: ifStep,
  loop,
  foreach,
  subflow,
  log,
  llm,
  agent,
  wait,
};

export const CORE_STEP_TYPES = Object.freeze(Object.keys(CORE_STEPS));

/** Directory containing built-in lib steps, resolved relative to this file. */
export const LIB_DIR = join(dirname(fileURLToPath(import.meta.url)), "lib");

/** Directory containing built-in core steps, resolved relative to this file. */
export const CORE_DIR = join(dirname(fileURLToPath(import.meta.url)), "core");

/**
 * Read a step's source code from disk. Resolves core, built-in lib, and
 * workspace custom steps (trying both `.ts` and `.js` so it works whether
 * vein runs from source via tsx or from a compiled build). Returns the
 * code plus which tier it came from, or `null` when no file is found.
 *
 * In-code steps injected via `createRegistry([...])` have no on-disk file —
 * those carry their source on the step def itself (`AnyStepDef.source`) and
 * are handled by the caller before falling back to this.
 */
export async function readStepSourceFromDisk(
  type: string,
  workspacePath: string,
): Promise<{ code: string; origin: StepSource } | null> {
  const parts = type.split("/");
  const leaf = parts.at(-1)!;
  const nested = parts.slice(0, -1);

  const candidates: Array<{ base: string; origin: StepSource }> = [];
  if (CORE_STEP_TYPES.includes(type)) {
    candidates.push({ base: join(CORE_DIR, type), origin: "core" });
  }
  candidates.push({ base: join(LIB_DIR, ...nested, leaf), origin: "lib" });
  candidates.push({
    base: join(workspacePath, "steps", "custom", ...nested, leaf),
    origin: "custom",
  });

  for (const { base, origin } of candidates) {
    for (const ext of [".ts", ".js"]) {
      try {
        return { code: await readFile(base + ext, "utf-8"), origin };
      } catch {
        // try next
      }
    }
  }
  return null;
}

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
      !name.endsWith(".d.ts") &&
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
    // Cache-bust the ESM module cache with the file's mtime so that
    // re-publishing a step (overwriting `custom/<name>.ts`) and rebuilding
    // the registry actually re-imports the new source instead of serving the
    // stale cached module. Unchanged files keep a stable URL (no churn).
    let suffix = "";
    try {
      const { mtimeMs } = await stat(filePath);
      suffix = `?v=${mtimeMs}`;
    } catch {
      // stat failed — fall back to no cache-bust
    }
    const url = pathToFileURL(filePath).href + suffix;
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
 *
 * Returns both the registry and a parallel `sources` map so callers can
 * report which tier each step came from without guessing from the name.
 */
export async function buildRegistry(workspacePath?: string): Promise<RegistryBundle> {
  const registry: StepRegistry = { ...CORE_STEPS };
  const sources: StepSources = {};

  for (const name of Object.keys(CORE_STEPS)) {
    sources[name] = "core";
  }

  await loadStepsFrom(LIB_DIR, registry, sources, "lib");

  if (workspacePath) {
    await loadStepsFrom(
      join(workspacePath, "steps", "custom"),
      registry,
      sources,
      "custom",
    );
  }

  return { registry, sources };
}

/**
 * Discover and dynamically import all step files in `baseDir`, adding them
 * to `registry` and recording the load tier in `sources`. Names that
 * collide with already-registered steps are skipped with a warning so a
 * lower-priority tier can never shadow a higher one.
 */
async function loadStepsFrom(
  baseDir: string,
  registry: StepRegistry,
  sources: StepSources,
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
    if (def) {
      registry[name] = def;
      sources[name] = tier;
    }
  }
}

/**
 * Get the core-only registry (no workspace steps). Useful for testing.
 */
export function coreRegistry(): StepRegistry {
  return { ...CORE_STEPS };
}

/**
 * Build a registry from in-code step definitions, layered on top of the
 * engine-shipped **core** and **lib** steps. For consumers using vein
 * as a library who prefer registering steps in code rather than via
 * filesystem discovery.
 *
 * The resulting registry contains:
 *   - **core/** — 8 built-in steps (http, log, if, loop, foreach,
 *     subflow, llm, wait)
 *   - **lib/** — engine-shipped domain integrations (e.g.
 *     `github/fetch-pr`). These are dynamically imported, so their
 *     deps are only resolved at registry-build time.
 *   - whatever you pass in `steps`
 *
 * Workspace **custom/** steps live on disk and are loaded via
 * `buildRegistry(workspacePath)` instead — they're never included
 * here.
 *
 * Each step is keyed by its `type` field. Duplicates among `steps`
 * throw. A user step whose `type` collides with a core or lib step
 * shadows it (with a warning), so callers can deliberately override
 * e.g. the built-in `http` step.
 *
 * ```ts
 * const registry = await createRegistry([myStep, anotherStep]);
 * await runWorkflow(flow, input, registry, { services });
 * ```
 */
export async function createRegistry(
  steps: AnyStepDef[],
): Promise<StepRegistry> {
  const registry: StepRegistry = { ...CORE_STEPS };

  // Load lib/ steps (dynamic imports).
  const sources: StepSources = {};
  for (const name of Object.keys(CORE_STEPS)) sources[name] = "core";
  await loadStepsFrom(LIB_DIR, registry, sources, "lib");

  // Layer user-supplied steps on top (shadowing allowed, with warnings).
  const seen = new Set<string>();
  for (const def of steps) {
    if (!def || typeof def !== "object" || !def.type || !def.run) {
      throw new Error(
        `createRegistry: invalid step definition (missing "type" or "run")`,
      );
    }
    if (seen.has(def.type)) {
      throw new Error(`createRegistry: duplicate step type "${def.type}"`);
    }
    seen.add(def.type);
    if (def.type in registry) {
      const tier = def.type in CORE_STEPS ? "core" : "lib";
      console.warn(
        `[vein] createRegistry: user step "${def.type}" shadows built-in ${tier} step`,
      );
    }
    registry[def.type] = def;
  }
  return registry;
}
