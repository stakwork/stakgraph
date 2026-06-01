import { z } from "zod";
import { tool } from "ai";
import { runWorkflow } from "../runner.js";
import { AiDeps } from "./prompts.js";
import { lsSteps, searchSteps, readStepSource } from "./stepHelpers.js";
import { zodToFields } from "./schemaHelpers.js";

// ── Tools ──────────────────────────────────────────────────────────────────

export function buildTools(deps: AiDeps) {
  return {
    list_steps: tool({
      description:
        "List contents of a step path, like a filesystem. Valid paths: 'steps' (shows core/, lib/, custom/), 'steps/core', 'steps/lib', 'steps/lib/<namespace>', 'steps/custom'.",
      inputSchema: z.object({
        path: z
          .string()
          .default("steps")
          .describe(
            "Path to list. Defaults to 'steps' (the root). Use 'steps/lib' to see lib namespaces, 'steps/lib/github' to see steps in a namespace, etc.",
          ),
      }),
      execute: async ({ path }) => lsSteps(path, deps),
    }),

    search_steps: tool({
      description:
        "Search for step types by keyword. Matches against the step type name and its description across core, lib, and custom steps. Returns ranked matches.",
      inputSchema: z.object({
        query: z
          .string()
          .describe("Search keywords, e.g. 'github pr' or 'http request'"),
      }),
      execute: async ({ query }) => searchSteps(query, deps),
    }),

    get_step: tool({
      description:
        "Get details for a specific step type: its input schema fields, and source code for lib/custom steps.",
      inputSchema: z.object({
        type: z.string().describe("Step type, e.g. 'http' or 'github/fetch-pr'"),
      }),
      execute: async ({ type }) => {
        const registry = deps.registry;
        const def = registry[type];
        if (!def) {
          return { error: `Step type "${type}" not found` };
        }

        const fields = zodToFields(def.input);
        const source = await readStepSource(type, deps);

        return { type, description: def.description, fields, source };
      },
    }),

    create_step: tool({
      description:
        "Author a NEW custom step type from TypeScript source. The code MUST be a self-contained vein step: the ONLY runtime import is `import { z, defineStep } from \"vein\"`, it `export default defineStep({ type, input, output, async run(cfg, ctx) {...} })`, and it reaches every external capability (db, http clients, llm, git, …) through `ctx.services` — never by importing other files. Use this only for step types that don't exist yet; use edit_step to change an existing one. Publishing as a new step creates version v1.",
      inputSchema: z.object({
        name: z
          .string()
          .describe(
            "Step type name. Slashes nest it (e.g. 'concepts/my-fetcher') and become the registry type.",
          ),
        code: z
          .string()
          .describe(
            "Full TypeScript source. Shape: import { z, defineStep } from \"vein\"; export default defineStep({ type: \"<name>\", input: z.object({...}), output: z.any(), async run(cfg, ctx) { /* use ctx.services for capabilities */ } });",
          ),
        description: z.string().optional(),
      }),
      execute: async ({ name, code, description }) => {
        if (deps.publishingEnabled === false) {
          return { error: "Step publishing is disabled (the registry was injected at construction)." };
        }
        const customs = await deps.workspace.listSteps();
        if (customs.some((s) => s.type === name)) {
          return { error: `Step "${name}" already exists. Use edit_step to publish a new version.` };
        }
        if (deps.registry[name]) {
          return { error: `"${name}" conflicts with a built-in (core/lib) step. Choose another name.` };
        }
        let result;
        try {
          result = await deps.workspace.publishStep(name, code, description, "ai");
        } catch (err) {
          return { error: err instanceof Error ? err.message : String(err) };
        }
        deps.registry = await deps.getRegistry();
        const loaded = Boolean(deps.registry[name]);
        return {
          ok: true,
          type: name,
          version: result.version,
          loaded,
          ...(loaded
            ? {}
            : {
                warning:
                  "Published but failed to load into the registry — check the source imports only 'vein' and has a valid defineStep default export.",
              }),
        };
      },
    }),

    edit_step: tool({
      description:
        "Publish a NEW VERSION of an EXISTING custom step (e.g. tweak its prompt, logic, or config schema). Same self-contained rules as create_step. Call get_step first to read the current source. Identical content is a no-op; a change increments the version (v1 → v2 → …) and prior versions are kept for rollback. Built-in core/lib steps cannot be edited.",
      inputSchema: z.object({
        type: z.string().describe("Existing custom step type to edit, e.g. 'concepts/decide'."),
        code: z
          .string()
          .describe("Full updated TypeScript source (same self-contained shape as create_step)."),
        description: z.string().optional(),
      }),
      execute: async ({ type, code, description }) => {
        if (deps.publishingEnabled === false) {
          return { error: "Step publishing is disabled (the registry was injected at construction)." };
        }
        const customs = await deps.workspace.listSteps();
        if (!customs.some((s) => s.type === type)) {
          return {
            error: deps.registry[type]
              ? `"${type}" is a built-in step and can't be edited. Use create_step with a new name.`
              : `Step "${type}" not found. Use create_step to author a new step.`,
          };
        }
        let result;
        try {
          result = await deps.workspace.publishStep(type, code, description);
        } catch (err) {
          return { error: err instanceof Error ? err.message : String(err) };
        }
        deps.registry = await deps.getRegistry();
        return {
          ok: true,
          type,
          version: result.version,
          changed: result.changed,
          loaded: Boolean(deps.registry[type]),
        };
      },
    }),

    create_workflow: tool({
      description:
        "Create and publish a new workflow from YAML. If the name already " +
        "exists, a numeric suffix is appended (e.g. `send-email-2`). The " +
        "response includes the final name used. To publish a new version of " +
        "an existing workflow, use `edit_workflow` (coming soon) instead.",
      inputSchema: z.object({
        name: z.string().describe("Workflow name (kebab-case)"),
        yaml: z.string().describe("Full workflow YAML"),
        description: z.string().optional(),
      }),
      execute: async ({ name, yaml, description }) => {
        const { name: finalName, version } = await deps.workspace.createWorkflow(
          name,
          yaml,
          description,
        );
        // Rebuild registry in case the workflow references new patterns
        deps.registry = await deps.getRegistry();
        return {
          ok: true,
          name: finalName,
          version,
          renamed: finalName !== name,
          requested: name,
        };
      },
    }),

    run_workflow: tool({
      description:
        "Run a published workflow with a given input and return the result. Use this to test workflows you just created. Returns status (success/error), output (on success), error details (on failure), and the runId.",
      inputSchema: z.object({
        name: z.string().describe("Workflow name to run"),
        input: z
          .any()
          .optional()
          .describe(
            "Input object passed to the workflow. Shape depends on the workflow's input schema; use {} if none.",
          ),
        version: z
          .string()
          .optional()
          .describe("Optional specific version. Defaults to the active version."),
      }),
      execute: async ({ name, input, version }) => {
        let flow;
        try {
          flow = version
            ? await deps.workspace.getWorkflowVersion(name, version)
            : await deps.workspace.getWorkflow(name);
        } catch (err) {
          return {
            ok: false,
            error: `Workflow not found: ${err instanceof Error ? err.message : String(err)}`,
          };
        }

        const result = await runWorkflow(flow, input ?? {}, deps.registry, {
          store: deps.store,
          workspace: deps.workspace,
        });

        return result;
      },
    }),
  };
}
