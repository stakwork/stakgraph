import { readFile, writeFile, readdir, mkdir, stat, unlink, rmdir, rm } from "node:fs/promises";
import { dirname, join, relative, sep } from "node:path";
import yaml from "js-yaml";
import { z } from "zod";
import type { Flow } from "./core.js";
import { contentHash, nextVersionLabel } from "./version.js";
import { evaluateExpr } from "./expr.js";

// Match a `{{ params.<path> }}` reference (and ONLY a params reference) so we can
// resolve param-to-param references at load time without touching `{{ input.* }}`
// or step-output references (which don't exist until run time).
const PARAM_SELF_REF = /\{\{\s*(params(?:\.[\w$]+|\[[^\]]+\])+)\s*\}\}/g;

/**
 * Resolve `{{ params.* }}` references that appear INSIDE other param values, so a
 * workflow can factor a shared value into one param and reuse it (e.g. a base
 * `podDomain` referenced by a big prompt param). Walks the params deeply; for
 * each string it substitutes only `params.*` templates (evaluated against the
 * params themselves), leaving every other template intact for run-time
 * resolution. One pass over the ORIGINAL params — chained references aren't
 * re-expanded, and the substitution uses the param DEFAULTS (a per-run override
 * of a referenced param won't retro-edit a value that embedded it). Unknown/erroring
 * refs are left verbatim rather than throwing.
 */
function resolveParamSelfReferences(params: Record<string, unknown>): Record<string, unknown> {
  const resolveStr = (s: string): string =>
    s.replace(PARAM_SELF_REF, (whole, expr: string) => {
      try {
        const val = evaluateExpr(expr, { params });
        if (val === null || val === undefined) return "";
        return typeof val === "object" ? JSON.stringify(val) : String(val);
      } catch {
        return whole; // leave unresolved on any error
      }
    });
  const walk = (v: unknown): unknown => {
    if (typeof v === "string") return v.includes("{{") ? resolveStr(v) : v;
    if (Array.isArray(v)) return v.map(walk);
    if (v && typeof v === "object") {
      const out: Record<string, unknown> = {};
      for (const [k, val] of Object.entries(v as Record<string, unknown>)) out[k] = walk(val);
      return out;
    }
    return v;
  };
  return walk(params) as Record<string, unknown>;
}

// ── Types ──────────────────────────────────────────────────────────────────

export interface WorkflowVersionInfo {
  createdAt: string;
  description?: string;
  /** Content hash of this version's source — internal dedup key for
   *  content-hash publishing. Not the user-facing version id. */
  hash?: string;
}

export interface WorkflowMetadata {
  active: string;
  versions: Record<string, WorkflowVersionInfo>;
}

export interface StepVersionInfo {
  createdAt: string;
  description?: string;
  /** Content hash of this version's source — internal dedup key. */
  hash?: string;
}

export interface StepInfo {
  /** Active version id (a content hash, e.g. "c-1a2b3c4d5e"). The active
   *  version's source is materialized at `custom/<name>.ts` for the registry
   *  loader; every version (incl. active) is archived under
   *  `steps/_history/<name>/<vid>.ts`. */
  active: string;
  /** All known versions, keyed by content-hash version id. */
  versions: Record<string, StepVersionInfo>;
  /** Optional identifier of the service that published this step.
   *  Used by `deleteStepsByPublisher` for bulk lifecycle ops. */
  publisher?: string;
}

export interface StepDirMetadata {
  /** Keys are full step names with optional slashes (e.g. "gitree/save-feature"). */
  steps: Record<string, StepInfo>;
}

export interface StepVersionsResult {
  active: string;
  versions: string[];
}

export interface WorkflowListEntry {
  name: string;
  activeVersion: string;
  versions: string[];
  description?: string;
}

export interface StepListEntry {
  type: string;
  description?: string;
  createdAt?: string;
  publisher?: string;
}

// ── Workspace Manager ──────────────────────────────────────────────────────

export class WorkspaceManager {
  private root: string;

  constructor(root?: string) {
    this.root = root ?? process.env["VEIN_WORKSPACE"] ?? "./workspace";
  }

  get path(): string {
    return this.root;
  }

  // ── Workflows ──────────────────────────────────────────────────────────

  async listWorkflows(): Promise<WorkflowListEntry[]> {
    const workflowsDir = join(this.root, "workflows");
    const entries = await safeReaddir(workflowsDir);
    const results: WorkflowListEntry[] = [];

    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const meta = await this.readWorkflowMetadata(entry.name);
      if (meta) {
        const activeDesc = meta.versions[meta.active]?.description;
        results.push({
          name: entry.name,
          activeVersion: meta.active,
          versions: Object.keys(meta.versions),
          description: activeDesc,
        });
      }
    }

    return results;
  }

  /** Load the active version of a workflow. */
  async getWorkflow(name: string): Promise<Flow> {
    const meta = await this.readWorkflowMetadata(name);
    if (!meta) {
      throw new Error(`Workflow "${name}" not found`);
    }
    return this.loadFlowYaml(name, meta.active);
  }

  /** Load a specific version of a workflow. */
  async getWorkflowVersion(name: string, version: string): Promise<Flow> {
    return this.loadFlowYaml(name, version);
  }

  /** Get the raw YAML source for a workflow version. */
  async getWorkflowSource(name: string, version: string): Promise<string> {
    const dir = join(this.root, "workflows", name);
    return readFile(join(dir, `${version}.yaml`), "utf-8");
  }

  private async loadFlowYaml(name: string, version: string): Promise<Flow> {
    const dir = join(this.root, "workflows", name);
    const raw = await readFile(join(dir, `${version}.yaml`), "utf-8");
    const data = yaml.load(raw) as any;

    if (!data || !data.steps) {
      throw new Error(
        `Invalid workflow YAML for "${name}" version "${version}"`,
      );
    }

    return {
      name: data.name ?? name,
      input: z.any(),
      steps: data.steps,
      // Resolve param-to-param references (`{{ params.* }}` nested inside another
      // param) once at load, so a shared value can be factored into one knob.
      ...(data.params != null ? { params: resolveParamSelfReferences(data.params) } : {}),
    };
  }

  /**
   * Create a brand-new workflow at v1. If `<workspace>/workflows/<name>/`
   * already exists, a numeric suffix is appended (`<name>-2`, `<name>-3`, ...)
   * until a free slot is found. Returns the actual name written.
   *
   * Use this for "create" intents (UI's Create Workflow dialog, AI's
   * `create_workflow` tool). For adding a new version to an existing workflow,
   * call `publishWorkflow` directly with the existing name and the next version.
   */
  async createWorkflow(
    name: string,
    content: { steps: any[]; params?: Record<string, unknown> } | string,
    description?: string,
  ): Promise<{ name: string; version: string }> {
    const workflowsDir = join(this.root, "workflows");
    let finalName = name;
    let n = 2;
    while (await pathExists(join(workflowsDir, finalName))) {
      finalName = `${name}-${n++}`;
    }

    // If the YAML embeds a `name:` field, rewrite it so the on-disk name
    // matches the directory — runner.ts keys run storage off `workflow.name`.
    let resolvedContent = content;
    if (finalName !== name && typeof content === "string") {
      const parsed = yaml.load(content) as any;
      if (parsed && typeof parsed === "object") {
        parsed.name = finalName;
        resolvedContent = yaml.dump(parsed, { lineWidth: 120, noRefs: true });
      }
    }

    await this.publishWorkflow(finalName, "v1", resolvedContent, description);
    return { name: finalName, version: "v1" };
  }

  /** Publish a new workflow version. Accepts steps array or raw YAML string. */
  async publishWorkflow(
    name: string,
    version: string,
    content: { steps: any[]; params?: Record<string, unknown> } | string,
    description?: string,
  ): Promise<void> {
    const dir = join(this.root, "workflows", name);
    await mkdir(dir, { recursive: true });

    // Write YAML
    const yamlStr =
      typeof content === "string"
        ? content
        : yaml.dump(
            {
              name,
              steps: content.steps,
              ...(content.params != null ? { params: content.params } : {}),
            },
            { lineWidth: 120, noRefs: true },
          );

    await writeFile(join(dir, `${version}.yaml`), yamlStr, "utf-8");

    // Update metadata
    const meta = (await this.readWorkflowMetadata(name)) ?? {
      active: version,
      versions: {},
    };

    meta.versions[version] = {
      createdAt: new Date().toISOString(),
      description,
      hash: contentHash(yamlStr),
    };
    meta.active = version;

    await writeFile(
      join(dir, "_metadata.json"),
      JSON.stringify(meta, null, 2),
      "utf-8",
    );
  }

  async setActiveVersion(name: string, version: string): Promise<void> {
    const meta = await this.readWorkflowMetadata(name);
    if (!meta) {
      throw new Error(`Workflow "${name}" not found`);
    }
    if (!(version in meta.versions)) {
      throw new Error(
        `Version "${version}" not found for workflow "${name}". Available: ${Object.keys(meta.versions).join(", ")}`,
      );
    }

    meta.active = version;
    const dir = join(this.root, "workflows", name);
    await writeFile(
      join(dir, "_metadata.json"),
      JSON.stringify(meta, null, 2),
      "utf-8",
    );
  }

  /**
   * Publish a workflow keyed by content hash but labeled with a friendly,
   * sequential version id (`v1`, `v2`, …). The hash is an internal dedup key;
   * the version id is what the UI shows. Idempotent: identical content that's
   * already active is a no-op; identical content of an older version re-points
   * active at it (no new version); changed content publishes the next `vN` and
   * activates it, retaining prior versions. This is the content-hash seeder's
   * primitive. Returns the version id and whether anything changed.
   */
  async publishWorkflowByContent(
    name: string,
    yamlStr: string,
    description?: string,
  ): Promise<{ version: string; changed: boolean }> {
    const hash = contentHash(yamlStr);
    const meta = await this.readWorkflowMetadata(name);

    if (meta) {
      const match = Object.entries(meta.versions).find(
        ([, info]) => info.hash === hash,
      );
      if (match) {
        const [vid] = match;
        if (meta.active === vid) return { version: vid, changed: false };
        await this.setActiveVersion(name, vid);
        return { version: vid, changed: true };
      }
    }

    const next = nextVersionLabel(meta ? Object.keys(meta.versions) : []);
    await this.publishWorkflow(name, next, yamlStr, description);
    return { version: next, changed: true };
  }

  // ── Steps ──────────────────────────────────────────────────────────────

  /**
   * List user-authored custom steps from `<workspace>/steps/custom/`,
   * recursively. Files starting with `_` are treated as helpers (importable
   * by sibling steps but not registered as their own step type) and are
   * omitted from the result.
   *
   * Lib steps live in the engine source tree and are not listed here.
   *
   * Pass `filter.publisher` to limit results to a specific publisher.
   */
  async listSteps(filter?: { publisher?: string }): Promise<StepListEntry[]> {
    const customDir = join(this.root, "steps", "custom");
    const meta = await this.readStepMetadata(customDir);
    const results: StepListEntry[] = [];

    const files = await findStepFilesRecursive(customDir);
    for (const file of files) {
      const stepName = stepNameFromFile(file, customDir);
      const info = meta?.steps[stepName];
      if (filter?.publisher && info?.publisher !== filter.publisher) continue;
      const activeVer = info ? info.versions[info.active] : undefined;
      results.push({
        type: stepName,
        description: activeVer?.description,
        createdAt: activeVer?.createdAt,
        publisher: info?.publisher,
      });
    }

    return results;
  }

  /**
   * Publish a custom step, keyed by content hash but labeled with a friendly,
   * sequential version id (`v1`, `v2`, …). Writes the active source to
   * `<workspace>/steps/custom/<name>.ts` (what the registry loads) and
   * archives every version under `steps/_history/<name>/<vid>.ts`.
   *
   * `name` may contain slashes to nest the file under subdirectories
   * (e.g. `"gitree/save-feature"` writes to `custom/gitree/save-feature.ts`).
   * Names starting with `_` (or with any path segment starting with `_`)
   * are treated as helper files: they're saved and importable by sibling
   * steps but are skipped by registry discovery.
   *
   * Idempotent by content hash: republishing identical content that's already
   * active is a no-op; identical content of an older version re-activates it
   * (no new version); changed content publishes the next `vN` and activates it
   * while prior versions are retained for rollback.
   *
   * Returns the version id and whether anything changed.
   *
   * Lib steps cannot be published at runtime — they ship with the engine.
   */
  async publishStep(
    name: string,
    code: string,
    description?: string,
    publisher?: string,
  ): Promise<{ version: string; changed: boolean }> {
    validateStepName(name);

    const customDir = join(this.root, "steps", "custom");
    const filePath = join(customDir, `${name}.ts`);
    const hash = contentHash(code);

    const meta = (await this.readStepMetadata(customDir)) ?? { steps: {} };
    const existing = meta.steps[name];

    // Content already known → no-op (if active) or re-activate that version.
    if (existing) {
      const match = Object.entries(existing.versions).find(
        ([, info]) => info.hash === hash,
      );
      if (match) {
        const [vid] = match;
        let changed = false;
        if (existing.active !== vid) {
          const archived = await readFile(this.stepVersionPath(name, vid), "utf-8");
          await mkdir(dirname(filePath), { recursive: true });
          await writeFile(filePath, archived, "utf-8");
          existing.active = vid;
          changed = true;
        }
        if (publisher !== undefined && existing.publisher !== publisher) {
          existing.publisher = publisher;
          await this.writeStepMetadata(customDir, meta);
        } else if (changed) {
          await this.writeStepMetadata(customDir, meta);
        }
        return { version: vid, changed };
      }
    }

    // New content → next sequential version. Materialize active source for the
    // registry loader + archive it.
    const vid = nextVersionLabel(existing ? Object.keys(existing.versions) : []);
    await mkdir(dirname(filePath), { recursive: true });
    await writeFile(filePath, code, "utf-8");
    const archivePath = this.stepVersionPath(name, vid);
    await mkdir(dirname(archivePath), { recursive: true });
    await writeFile(archivePath, code, "utf-8");

    meta.steps[name] = {
      active: vid,
      versions: {
        ...(existing?.versions ?? {}),
        [vid]: {
          createdAt: new Date().toISOString(),
          hash,
          ...(description !== undefined ? { description } : {}),
        },
      },
      ...(publisher !== undefined
        ? { publisher }
        : existing?.publisher !== undefined
          ? { publisher: existing.publisher }
          : {}),
    };

    await this.writeStepMetadata(customDir, meta);
    return { version: vid, changed: true };
  }

  /** Absolute path to an archived step version source file. */
  private stepVersionPath(name: string, version: string): string {
    return join(this.root, "steps", "_history", name, `${version}.ts`);
  }

  private async writeStepMetadata(
    customDir: string,
    meta: StepDirMetadata,
  ): Promise<void> {
    await writeFile(
      join(customDir, "_metadata.json"),
      JSON.stringify(meta, null, 2),
      "utf-8",
    );
  }

  /** List a step's versions and its active version id. */
  async listStepVersions(name: string): Promise<StepVersionsResult> {
    validateStepName(name);
    const customDir = join(this.root, "steps", "custom");
    const meta = await this.readStepMetadata(customDir);
    const info = meta?.steps[name];
    if (!info) throw new Error(`Step "${name}" not found`);
    return { active: info.active, versions: Object.keys(info.versions) };
  }

  /** Get the archived source for a specific step version. */
  async getStepVersionSource(name: string, version: string): Promise<string> {
    validateStepName(name);
    return readFile(this.stepVersionPath(name, version), "utf-8");
  }

  /**
   * Switch a step's active version. Copies the archived version's source
   * into the flat `custom/<name>.ts` the registry loads, and updates the
   * active pointer in metadata.
   */
  async setActiveStepVersion(name: string, version: string): Promise<void> {
    validateStepName(name);
    const customDir = join(this.root, "steps", "custom");
    const meta = await this.readStepMetadata(customDir);
    const info = meta?.steps[name];
    if (!meta || !info) throw new Error(`Step "${name}" not found`);
    if (!(version in info.versions)) {
      throw new Error(
        `Version "${version}" not found for step "${name}". Available: ${Object.keys(info.versions).join(", ")}`,
      );
    }
    const code = await readFile(this.stepVersionPath(name, version), "utf-8");
    const filePath = join(customDir, `${name}.ts`);
    await mkdir(dirname(filePath), { recursive: true });
    await writeFile(filePath, code, "utf-8");
    info.active = version;
    await this.writeStepMetadata(customDir, meta);
  }

  /**
   * Delete a single custom step by name. Removes the source file, its
   * version archive, and its metadata entry, then cleans up any empty parent
   * directories within `steps/custom/` so namespace directories disappear
   * once their last step is removed.
   *
   * No-ops silently if the step does not exist.
   */
  async deleteStep(name: string): Promise<boolean> {
    validateStepName(name);

    const customDir = join(this.root, "steps", "custom");
    const filePath = join(customDir, `${name}.ts`);

    let removed = false;
    try {
      await unlink(filePath);
      removed = true;
    } catch (err: any) {
      if (err?.code !== "ENOENT") throw err;
    }

    // Remove the version archive dir for this step (best-effort).
    await rm(join(this.root, "steps", "_history", name), {
      recursive: true,
      force: true,
    });

    const meta = await this.readStepMetadata(customDir);
    if (meta && meta.steps[name]) {
      delete meta.steps[name];
      await writeFile(
        join(customDir, "_metadata.json"),
        JSON.stringify(meta, null, 2),
        "utf-8",
      );
      removed = true;
    }

    if (removed) {
      await pruneEmptyDirs(dirname(filePath), customDir);
    }

    return removed;
  }

  /**
   * Bulk delete all custom steps published by `publisher`. Returns the list
   * of step names that were removed. Useful for service shutdown:
   * `await ws.deleteStepsByPublisher("mcp-gitree")` on SIGTERM tears down
   * everything a service registered in one call.
   */
  async deleteStepsByPublisher(publisher: string): Promise<string[]> {
    const customDir = join(this.root, "steps", "custom");
    const meta = await this.readStepMetadata(customDir);
    if (!meta) return [];

    const toDelete = Object.entries(meta.steps)
      .filter(([, info]) => info.publisher === publisher)
      .map(([name]) => name);

    for (const name of toDelete) {
      await this.deleteStep(name);
    }

    return toDelete;
  }

  // ── Private helpers ────────────────────────────────────────────────────

  private async readWorkflowMetadata(
    name: string,
  ): Promise<WorkflowMetadata | null> {
    try {
      const raw = await readFile(
        join(this.root, "workflows", name, "_metadata.json"),
        "utf-8",
      );
      return JSON.parse(raw) as WorkflowMetadata;
    } catch {
      return null;
    }
  }

  private async readStepMetadata(
    dir: string,
  ): Promise<StepDirMetadata | null> {
    try {
      const raw = await readFile(join(dir, "_metadata.json"), "utf-8");
      return JSON.parse(raw) as StepDirMetadata;
    } catch {
      return null;
    }
  }
}

// ── Utilities ──────────────────────────────────────────────────────────────

async function safeReaddir(dir: string) {
  try {
    return await readdir(dir, { withFileTypes: true });
  } catch {
    return [];
  }
}

async function pathExists(p: string): Promise<boolean> {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

/**
 * Validate a custom step name. Allows nested names with slashes
 * (e.g. `gitree/save-feature`) and helper names with leading underscores
 * (e.g. `gitree/_shared`). Rejects path traversal and absolute paths.
 */
function validateStepName(name: string): void {
  if (!name) {
    throw new Error("Step name cannot be empty");
  }
  if (name.startsWith("/") || name.includes("\\")) {
    throw new Error(`Invalid step name "${name}": must not contain absolute or back-slash paths`);
  }
  if (name.includes("//") || name.endsWith("/") || name.startsWith("/")) {
    throw new Error(`Invalid step name "${name}": malformed path`);
  }
  const segments = name.split("/");
  for (const seg of segments) {
    if (!seg || seg === "." || seg === "..") {
      throw new Error(`Invalid step name "${name}": path traversal not allowed`);
    }
    if (!/^[a-zA-Z_][a-zA-Z0-9_-]*$/.test(seg)) {
      throw new Error(
        `Invalid step name "${name}": each segment must match [a-zA-Z_][a-zA-Z0-9_-]*`,
      );
    }
  }
}

/**
 * Recursively find step files (`.ts` / `.js`) under `dir`, skipping helper
 * files (`_*`), hidden files, and test files. Returns absolute paths.
 *
 * This mirrors the registry's discovery rules so `listSteps` shows exactly
 * what the registry will load.
 */
async function findStepFilesRecursive(dir: string): Promise<string[]> {
  const results: string[] = [];

  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return results;
  }

  for (const e of entries) {
    const full = join(dir, e.name);
    if (e.isDirectory()) {
      if (e.name.startsWith("_") || e.name.startsWith(".")) continue;
      const nested = await findStepFilesRecursive(full);
      results.push(...nested);
    } else if (e.isFile()) {
      if (
        e.name.startsWith("_") ||
        e.name.startsWith(".") ||
        !(e.name.endsWith(".ts") || e.name.endsWith(".js")) ||
        e.name.endsWith(".test.ts") ||
        e.name.endsWith(".spec.ts")
      ) continue;
      results.push(full);
    }
  }

  return results;
}

/**
 * Convert an absolute file path under `baseDir` to a slash-separated step
 * name (extension stripped). E.g. `<base>/gitree/save-feature.ts` → `"gitree/save-feature"`.
 */
function stepNameFromFile(filePath: string, baseDir: string): string {
  const rel = relative(baseDir, filePath).replace(/\.(ts|js)$/, "");
  return rel.split(sep).join("/");
}

/**
 * Walk upward from `startDir` removing empty directories, stopping when
 * `stopDir` is reached (inclusive boundary — we never remove `stopDir`).
 * Silently ignores non-empty dirs and any errors.
 */
async function pruneEmptyDirs(startDir: string, stopDir: string): Promise<void> {
  let dir = startDir;
  while (dir.startsWith(stopDir) && dir !== stopDir) {
    try {
      await rmdir(dir);
    } catch {
      return; // not empty, or doesn't exist
    }
    dir = dirname(dir);
  }
}
