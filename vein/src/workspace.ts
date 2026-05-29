import { readFile, writeFile, readdir, mkdir, stat, unlink, rmdir } from "node:fs/promises";
import { dirname, join, relative, sep } from "node:path";
import yaml from "js-yaml";
import { z } from "zod";
import type { Flow } from "./core.js";

// ── Types ──────────────────────────────────────────────────────────────────

export interface WorkflowVersionInfo {
  createdAt: string;
  description?: string;
}

export interface WorkflowMetadata {
  active: string;
  versions: Record<string, WorkflowVersionInfo>;
}

export interface StepInfo {
  createdAt: string;
  description?: string;
  /** Optional identifier of the service that published this step.
   *  Used by `deleteStepsByPublisher` for bulk lifecycle ops. */
  publisher?: string;
}

export interface StepDirMetadata {
  /** Keys are full step names with optional slashes (e.g. "gitree/save-feature"). */
  steps: Record<string, StepInfo>;
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
    content: { steps: any[] } | string,
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
    content: { steps: any[] } | string,
    description?: string,
  ): Promise<void> {
    const dir = join(this.root, "workflows", name);
    await mkdir(dir, { recursive: true });

    // Write YAML
    const yamlStr =
      typeof content === "string"
        ? content
        : yaml.dump({ name, steps: content.steps }, { lineWidth: 120, noRefs: true });

    await writeFile(join(dir, `${version}.yaml`), yamlStr, "utf-8");

    // Update metadata
    const meta = (await this.readWorkflowMetadata(name)) ?? {
      active: version,
      versions: {},
    };

    meta.versions[version] = {
      createdAt: new Date().toISOString(),
      description,
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
      results.push({
        type: stepName,
        description: info?.description,
        createdAt: info?.createdAt,
        publisher: info?.publisher,
      });
    }

    return results;
  }

  /**
   * Write a user-authored step to `<workspace>/steps/custom/<name>.ts`.
   *
   * `name` may contain slashes to nest the file under subdirectories
   * (e.g. `"gitree/save-feature"` writes to `custom/gitree/save-feature.ts`).
   * Names starting with `_` (or with any path segment starting with `_`)
   * are treated as helper files: they're saved and importable by sibling
   * steps but are skipped by registry discovery.
   *
   * Lib steps cannot be published at runtime — they ship with the engine.
   */
  async publishStep(
    name: string,
    code: string,
    description?: string,
    publisher?: string,
  ): Promise<void> {
    validateStepName(name);

    const customDir = join(this.root, "steps", "custom");
    const filePath = join(customDir, `${name}.ts`);

    await mkdir(dirname(filePath), { recursive: true });
    await writeFile(filePath, code, "utf-8");

    const meta = (await this.readStepMetadata(customDir)) ?? { steps: {} };
    meta.steps[name] = {
      createdAt: new Date().toISOString(),
      ...(description !== undefined ? { description } : {}),
      ...(publisher !== undefined ? { publisher } : {}),
    };

    await writeFile(
      join(customDir, "_metadata.json"),
      JSON.stringify(meta, null, 2),
      "utf-8",
    );
  }

  /**
   * Delete a single custom step by name. Removes the source file and its
   * metadata entry, then cleans up any empty parent directories within
   * `steps/custom/` so namespace directories disappear once their last step
   * is removed.
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
