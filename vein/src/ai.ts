import { z } from "zod";
import { tool } from "ai";
import type { StepRegistry } from "./core.js";
import type { WorkspaceManager } from "./workspace.js";

// ── Types ──────────────────────────────────────────────────────────────────

export interface AiDeps {
  workspace: WorkspaceManager;
  registry: StepRegistry;
  getRegistry: () => Promise<StepRegistry>;
}

// ── System prompt ──────────────────────────────────────────────────────────

export const SYSTEM = `You are a workflow builder. Users describe what they want and you create workflows.

A workflow is YAML with this shape:

name: my-workflow
steps:
  - id: fetch
    type: http
    config:
      url: "https://httpbin.org/json"
  - id: done
    type: log
    config:
      message: "result: {{ fetch.body }}"

Rules:
- Step ids must be unique, alphanumeric + underscores only.
- Steps run sequentially by default (each depends on the previous).
- Use "depends" to control ordering. depends: [] means run immediately (parallel).
- Use {{ }} templates to reference previous step outputs or input, e.g. {{ fetch.body.name }} or {{ input.url }}.
- Use === for equality in expressions, not ==.

Branching (if):
- The "if" step is a GATE. It evaluates "cond" and returns a boolean.
- Downstream steps branch using "depends: <if-id>" plus "when: true" or "when: false".
- Each branch can be a chain of multiple steps — they all flow from the gate via "depends".
- A step that fans in (depends on both branches) runs as long as at least one branch ran.
- Example:
    - id: check
      type: if
      config:
        cond: "{{ input.fast }}"
    - id: quick
      type: log
      config: { message: "fast path" }
      depends: check
      when: true
    - id: slow
      type: log
      config: { message: "slow path" }
      depends: check
      when: false

Subflows:
- The "subflow" step calls a PUBLISHED workflow by name (and optional version).
- Config: { workflow: "<name>", version?: "<version>", input: { ... } }.
- The referenced workflow must already exist in the workspace.

Workflow:
1. Call list_steps to see what's available.
2. Call get_step for EVERY step type you will use. Each has a description with the exact YAML config format — you MUST read it before writing. Do not guess config fields.
3. If you see lib namespaces, explore the relevant ones with list_steps.
4. Call create_workflow with the final YAML.

Be concise. Don't over-explain.`;

// ── Tools ──────────────────────────────────────────────────────────────────

export function buildTools(deps: AiDeps) {
  return {
    list_steps: tool({
      description:
        "List available step types. No args = top-level view (core types + lib namespaces + custom steps). Pass a namespace like 'lib/github' to see steps inside it.",
      inputSchema: z.object({
        namespace: z
          .string()
          .optional()
          .describe("e.g. 'lib/github' to list steps in that namespace"),
      }),
      execute: async ({ namespace }) => {
        if (namespace) {
          // Drill into a namespace — list steps inside it
          const prefix = namespace.replace(/^lib\//, "");
          const all = await deps.workspace.listSteps();
          const matches = all.filter((s) => s.type.startsWith(prefix + "/"));
          return matches.map((s) => ({
            type: s.type,
            description: s.description,
          }));
        }

        // Top-level: core types + lib namespaces + custom steps
        const core = [
          "http",
          "log",
          "if",
          "loop",
          "subflow",
          "llm",
          "wait",
        ];
        const wsSteps = await deps.workspace.listSteps();

        const namespaces = new Set<string>();
        const custom: { type: string; description?: string }[] = [];
        for (const s of wsSteps) {
          if (s.type.includes("/")) {
            namespaces.add(s.type.split("/")[0]!);
          } else {
            custom.push({ type: s.type, description: s.description });
          }
        }

        return {
          core,
          lib: [...namespaces],
          custom,
        };
      },
    }),

    get_step: tool({
      description:
        "Get details for a specific step type: its input schema fields, and source code for lib/custom steps.",
      inputSchema: z.object({
        type: z.string().describe("Step type, e.g. 'http' or 'github/fetch-prs'"),
      }),
      execute: async ({ type }) => {
        const registry = deps.registry;
        const def = registry[type];
        if (!def) {
          return { error: `Step type "${type}" not found` };
        }

        const fields = zodToFields(def.input);

        // Try to read source for lib/custom steps
        let source: string | undefined;
        if (!["http", "if", "loop", "subflow", "log", "llm", "wait"].includes(type)) {
          try {
            const { readFile } = await import("node:fs/promises");
            const { join } = await import("node:path");
            const base = deps.workspace.path;
            const parts = type.split("/");
            const filePath =
              parts.length > 1
                ? join(base, "steps", "lib", ...parts.slice(0, -1), `${parts.at(-1)}.ts`)
                : join(base, "steps", "custom", `${type}.ts`);
            source = await readFile(filePath, "utf-8");
          } catch {
            // no source available
          }
        }

        return { type, description: def.description, fields, source };
      },
    }),

    create_workflow: tool({
      description: "Create and publish a new workflow from YAML.",
      inputSchema: z.object({
        name: z.string().describe("Workflow name (kebab-case)"),
        yaml: z.string().describe("Full workflow YAML"),
        description: z.string().optional(),
      }),
      execute: async ({ name, yaml, description }) => {
        await deps.workspace.publishWorkflow(name, "v1", yaml, description);
        // Rebuild registry in case the workflow references new patterns
        deps.registry = await deps.getRegistry();
        return { ok: true, name, version: "v1" };
      },
    }),
  };
}

// ── Schema helpers (reused from server.ts pattern) ─────────────────────────

interface FieldDesc {
  name: string;
  kind: "string" | "number" | "boolean" | "enum" | "json";
  required: boolean;
  default?: unknown;
  enumValues?: string[];
}

function zodToFields(schema: z.ZodTypeAny): FieldDesc[] {
  const shape = getObjectShape(schema);
  if (!shape) return [];
  return Object.entries(shape).map(([name, s]) =>
    describeField(name, s as z.ZodTypeAny),
  );
}

function getObjectShape(s: z.ZodTypeAny): Record<string, z.ZodTypeAny> | null {
  const def = s._def;
  if (def.typeName === "ZodObject") return (def as any).shape();
  if (def.typeName === "ZodEffects") return getObjectShape(def.schema);
  return null;
}

function describeField(name: string, s: z.ZodTypeAny): FieldDesc {
  let required = true;
  let defaultVal: unknown;
  let inner = s;
  for (;;) {
    const def = inner._def;
    if (def.typeName === "ZodOptional") {
      required = false;
      inner = def.innerType;
    } else if (def.typeName === "ZodDefault") {
      required = false;
      defaultVal = def.defaultValue();
      inner = def.innerType;
    } else if (def.typeName === "ZodNullable") {
      required = false;
      inner = def.innerType;
    } else break;
  }
  const typeName = inner._def.typeName as string;
  if (typeName === "ZodEnum")
    return { name, kind: "enum", required, default: defaultVal, enumValues: inner._def.values };
  if (typeName === "ZodString") return { name, kind: "string", required, default: defaultVal };
  if (typeName === "ZodNumber") return { name, kind: "number", required, default: defaultVal };
  if (typeName === "ZodBoolean") return { name, kind: "boolean", required, default: defaultVal };
  return { name, kind: "json", required, default: defaultVal };
}
