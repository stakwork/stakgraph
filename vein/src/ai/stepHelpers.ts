
import { readdir, readFile, stat } from "node:fs/promises";
import { join } from "node:path";
import { CORE_STEP_TYPES, LIB_DIR } from "../steps/registry.js";
import { AiDeps } from "./prompts.js";

// ── Step explorer helpers ──────────────────────────────────────────────────

/** Normalize a path like "/steps", "steps/", "steps" → "steps". */
function normalizeStepsPath(path: string): string[] {
  const trimmed = path.replace(/^\/+|\/+$/g, "");
  const parts = trimmed.split("/").filter(Boolean);
  if (parts[0] !== "steps") return ["__invalid__"];
  return parts;
}

export async function lsSteps(path: string, deps: AiDeps) {
  const parts = normalizeStepsPath(path);
  if (parts[0] === "__invalid__") {
    return { error: `Invalid path "${path}". Paths must start with "steps".` };
  }

  // ls steps  →  show the three roots
  if (parts.length === 1) {
    return { entries: ["core/", "lib/", "custom/"] };
  }

  const root = parts[1];

  // ls steps/core
  if (root === "core" && parts.length === 2) {
    return { entries: [...CORE_STEP_TYPES] };
  }

  // ls steps/lib  → list built-in lib namespaces (subdirectories of LIB_DIR)
  if (root === "lib" && parts.length === 2) {
    const namespaces = await listSubdirs(LIB_DIR);
    return { entries: namespaces.map((n) => `${n}/`) };
  }

  // ls steps/lib/<namespace>  → list step files in that namespace
  if (root === "lib" && parts.length >= 3) {
    const subPath = parts.slice(2).join("/");
    const dir = join(LIB_DIR, subPath);
    const entries = await listStepDir(dir);
    if (!entries) return { error: `Path "steps/lib/${subPath}" not found.` };
    return { entries };
  }

  // ls steps/custom  → list workspace custom steps (flat)
  if (root === "custom" && parts.length === 2) {
    const custom = await deps.workspace.listSteps();
    return {
      entries: custom.map((s) =>
        s.description ? `${s.type}  — ${s.description}` : s.type,
      ),
    };
  }

  return { error: `Unknown path "${path}". Try "steps", "steps/core", "steps/lib", "steps/custom".` };
}

/** List immediate subdirectory names in a directory. Returns [] if missing. */
async function listSubdirs(dir: string): Promise<string[]> {
  try {
    const entries = await readdir(dir);
    const subs: string[] = [];
    for (const name of entries) {
      const st = await stat(join(dir, name));
      if (st.isDirectory()) subs.push(name);
    }
    return subs.sort();
  } catch {
    return [];
  }
}

/** List step files (without extension) and subdirectories in a lib namespace. */
async function listStepDir(dir: string): Promise<string[] | null> {
  let names: string[];
  try {
    names = await readdir(dir);
  } catch {
    return null;
  }
  const entries: string[] = [];
  for (const name of names) {
    if (name.startsWith("_") || name.startsWith(".")) continue;
    const st = await stat(join(dir, name));
    if (st.isDirectory()) {
      entries.push(`${name}/`);
    } else if (
      st.isFile() &&
      (name.endsWith(".ts") || name.endsWith(".js")) &&
      !name.endsWith(".test.ts") &&
      !name.endsWith(".spec.ts")
    ) {
      entries.push(name.replace(/\.(ts|js)$/, ""));
    }
  }
  return entries.sort();
}

export async function searchSteps(query: string, deps: AiDeps) {
  const q = query.trim().toLowerCase();
  if (!q) return { matches: [] };
  const terms = q.split(/\s+/).filter(Boolean);

  const results: { type: string; description?: string; score: number }[] = [];
  for (const [type, def] of Object.entries(deps.registry)) {
    const desc = (def as any).description as string | undefined;
    const haystack = `${type} ${desc ?? ""}`.toLowerCase();
    let score = 0;
    for (const t of terms) {
      if (type.toLowerCase().includes(t)) score += 2;
      else if (haystack.includes(t)) score += 1;
    }
    if (score > 0) results.push({ type, description: desc, score });
  }

  results.sort((a, b) => b.score - a.score || a.type.localeCompare(b.type));
  return {
    matches: results.slice(0, 20).map(({ score: _s, ...rest }) => rest),
  };
}

/**
 * Read the source code for a step. Looks in:
 *   - src/steps/lib/<...>.ts  (built-in lib steps)
 *   - <workspace>/steps/custom/<name>.ts  (user custom steps)
 * Returns undefined for core steps or anything not found.
 */
export async function readStepSource(
  type: string,
  deps: AiDeps,
): Promise<string | undefined> {
  if (CORE_STEP_TYPES.includes(type)) return undefined;

  const parts = type.split("/");
  const candidates: string[] = [];

  if (parts.length > 1) {
    // Likely a lib step: src/steps/lib/<...>.ts
    candidates.push(join(LIB_DIR, ...parts.slice(0, -1), `${parts.at(-1)}.ts`));
    // Or a nested custom step: <workspace>/steps/custom/<...>.ts
    candidates.push(
      join(deps.workspace.path, "steps", "custom", ...parts.slice(0, -1), `${parts.at(-1)}.ts`),
    );
  } else {
    // Flat custom step
    candidates.push(join(deps.workspace.path, "steps", "custom", `${type}.ts`));
  }

  for (const file of candidates) {
    try {
      return await readFile(file, "utf-8");
    } catch {
      // try next
    }
  }
  return undefined;
}
