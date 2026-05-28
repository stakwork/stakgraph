import { readFile, writeFile, readdir, mkdir } from "node:fs/promises";
import { join } from "node:path";
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
}

export interface StepDirMetadata {
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
   * List user-authored custom steps from `<workspace>/steps/custom/`.
   * Lib steps live in the engine source tree and are not listed here.
   */
  async listSteps(): Promise<StepListEntry[]> {
    const results: StepListEntry[] = [];

    const customDir = join(this.root, "steps", "custom");
    const meta = await this.readStepMetadata(customDir);
    const customFiles = await safeReaddir(customDir);
    for (const f of customFiles) {
      if (!f.isFile() || !f.name.endsWith(".ts") || f.name.startsWith("_"))
        continue;
      const stepName = f.name.replace(/\.ts$/, "");
      results.push({
        type: stepName,
        description: meta?.steps[stepName]?.description,
        createdAt: meta?.steps[stepName]?.createdAt,
      });
    }

    return results;
  }

  /**
   * Write a user-authored step to `<workspace>/steps/custom/<name>.ts`.
   * Lib steps cannot be published at runtime — they ship with the engine.
   */
  async publishStep(
    name: string,
    code: string,
    description?: string,
  ): Promise<void> {
    const dir = join(this.root, "steps", "custom");

    await mkdir(dir, { recursive: true });
    await writeFile(join(dir, `${name}.ts`), code, "utf-8");

    const meta = (await this.readStepMetadata(dir)) ?? { steps: {} };
    meta.steps[name] = {
      createdAt: new Date().toISOString(),
      description,
    };

    await writeFile(
      join(dir, "_metadata.json"),
      JSON.stringify(meta, null, 2),
      "utf-8",
    );
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
