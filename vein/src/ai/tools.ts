import { z } from "zod";
import { tool } from "ai";
import { runWorkflow } from "../runner.js";
import type { RunEvent, RunSummary } from "../core.js";
import { AiDeps } from "./prompts.js";
import { lsSteps, searchSteps, readStepSource } from "./stepHelpers.js";
import { zodToFields } from "./schemaHelpers.js";
import { runSingleStep, cassettePath } from "../run-step.js";

// The run-history read methods live on `FileRunStore`, not the base `RunStore`
// interface (which is write-only: append/finalize). `MemoryRunStore` lacks them.
// Feature-detect so the run-history tools degrade gracefully on stores that
// can't read back (they return an error rather than throwing).
interface RunReadStore {
  listRuns(workflow: string): Promise<string[]>;
  getRunSummary(workflow: string, runId: string): Promise<RunSummary | null>;
  getRunEvents(workflow: string, runId: string): Promise<RunEvent[]>;
}

function asReadStore(store: unknown): RunReadStore | null {
  const s = store as Partial<RunReadStore> | undefined;
  return s &&
    typeof s.listRuns === "function" &&
    typeof s.getRunSummary === "function" &&
    typeof s.getRunEvents === "function"
    ? (s as RunReadStore)
    : null;
}

/** Drop bulky input/output payloads from an event so a run's event list stays
 *  token-cheap; the agent can re-fetch a specific run's full events if needed. */
function slimEvent(e: RunEvent) {
  return {
    type: e.type,
    path: e.path,
    ...(e.stepType ? { stepType: e.stepType } : {}),
    ...(e.durationMs != null ? { durationMs: e.durationMs } : {}),
    ...(e.iteration != null ? { iteration: e.iteration } : {}),
    ...(e.error ? { error: e.error } : {}),
  };
}

// ── Tools ──────────────────────────────────────────────────────────────────

/** LLMs sometimes pass an object-valued tool arg as a JSON *string* (e.g.
 *  run_workflow's `input`). The template engine then sees a string, so
 *  `{{ input.owner }}` resolves to undefined and every field fails validation.
 *  Defensively parse a JSON string back into the object/array it represents;
 *  leave anything else untouched. */
function coerceJsonArg(v: unknown): unknown {
  if (typeof v !== "string") return v;
  const t = v.trim();
  if (!(t.startsWith("{") || t.startsWith("["))) return v;
  try {
    return JSON.parse(t);
  } catch {
    return v;
  }
}

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

    list_secrets: tool({
      description:
        "List the NAMES of credentials available in the deployment's secret store (e.g. GITHUB_TOKEN, GOOGLE_SERVICE_ACCOUNT_JSON). Returns names + metadata ONLY — never the secret values. Use this before authoring a step that needs auth: reference an existing name in ctx.services.secrets.get(\"NAME\"), and if the credential you need isn't listed, tell the user to add it via the Secrets dialog (the value is never visible to you).",
      inputSchema: z.object({}),
      execute: async () => {
        if (!deps.secrets) {
          return { error: "Secret store is not available in this deployment." };
        }
        const secrets = await deps.secrets.list();
        return { secrets: secrets.map((s) => ({ name: s.name, updatedAt: s.updatedAt })) };
      },
    }),

    create_step: tool({
      description:
        "Author a NEW custom step type from TypeScript source. The code is a self-contained vein step: `import { z, defineStep } from \"vein\"` and `export default defineStep({ type, input, output, async run(cfg, ctx) {...} })`. Reach external capabilities through `ctx.services` — for network calls use `ctx.services.http(url, opts)` and for credentials `ctx.services.secrets.get(name)` (NOT the global fetch / process.env), so the step is recordable/replayable by run_step's cassette and secrets are scrubbed from fixtures. Call get_step(\"http\") to read the canonical ctx.services.http example. Prefer raw REST over vendor SDKs; only import a package other than \"vein\" if the deployment has pre-installed it. Use this only for step types that don't exist yet; use edit_step to change an existing one. Publishing as a new step creates version v1.",
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
        "Create and publish a NEW workflow from YAML. If the name already " +
        "exists, a numeric suffix is appended (e.g. `send-email-2`). The " +
        "response includes the final name used. To publish a new version of " +
        "an EXISTING workflow, use `edit_workflow` instead.",
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

    edit_workflow: tool({
      description:
        "Publish a NEW VERSION of an EXISTING workflow from YAML. Call " +
        "get_workflow first to read the current source. Identical content is " +
        "a no-op; a change increments the version (v1 → v2 → …) and activates " +
        "it, retaining prior versions for rollback. Use this for STRUCTURAL " +
        "changes (adding/removing steps, rewiring `depends`, or promoting a " +
        "winning `params` default). To merely try a different prompt or " +
        "threshold value, do NOT publish a version — pass `params` to " +
        "run_workflow instead (those are runs, not versions).",
      inputSchema: z.object({
        name: z.string().describe("Existing workflow name to edit"),
        yaml: z.string().describe("Full updated workflow YAML"),
        description: z.string().optional(),
      }),
      execute: async ({ name, yaml, description }) => {
        const exists = (await deps.workspace.listWorkflows()).some(
          (w) => w.name === name,
        );
        if (!exists) {
          return {
            error: `Workflow "${name}" not found. Use create_workflow to author a new one.`,
          };
        }
        let result;
        try {
          result = await deps.workspace.publishWorkflowByContent(
            name,
            yaml,
            description,
          );
        } catch (err) {
          return { error: err instanceof Error ? err.message : String(err) };
        }
        deps.registry = await deps.getRegistry();
        return {
          ok: true,
          name,
          version: result.version,
          changed: result.changed,
        };
      },
    }),

    list_workflows: tool({
      description:
        "List all published workflows in the workspace, with each one's active version, all versions, and description. Use this to discover what workflows already exist before creating a new one or referencing one in a subflow.",
      inputSchema: z.object({}),
      execute: async () => {
        const workflows = await deps.workspace.listWorkflows();
        return { workflows };
      },
    }),

    get_workflow: tool({
      description:
        "Get a published workflow's full YAML source plus its version metadata. Defaults to the active version; pass `version` for a specific one. Use this to read an existing workflow before editing, referencing it in a subflow, or running it.",
      inputSchema: z.object({
        name: z.string().describe("Workflow name"),
        version: z
          .string()
          .optional()
          .describe("Optional specific version. Defaults to the active version."),
      }),
      execute: async ({ name, version }) => {
        const entry = (await deps.workspace.listWorkflows()).find(
          (w) => w.name === name,
        );
        if (!entry) {
          return { error: `Workflow "${name}" not found` };
        }
        const resolved = version ?? entry.activeVersion;
        let yaml;
        try {
          yaml = await deps.workspace.getWorkflowSource(name, resolved);
        } catch (err) {
          return {
            error: `Version "${resolved}" not found for "${name}". Available: ${entry.versions.join(", ")}`,
          };
        }
        return {
          name,
          version: resolved,
          activeVersion: entry.activeVersion,
          versions: entry.versions,
          description: entry.description,
          yaml,
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
            "Input passed to the workflow as a JSON OBJECT (not a string), referenced in step configs via {{ input.* }} — the run subject, e.g. { owner, repo, pull_number }. Use {} if none.",
          ),
        params: z
          .record(z.any())
          .optional()
          .describe(
            "Optional overrides for the workflow's `params` knobs (prompts, thresholds, sample sizes). Shallow-merged over the workflow's `params` defaults — set just the knobs you want to vary for this trial. Referenced in step configs via {{ params.* }}.",
          ),
        version: z
          .string()
          .optional()
          .describe("Optional specific version. Defaults to the active version."),
      }),
      execute: async ({ name, input, params, version }) => {
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

        const result = await runWorkflow(flow, coerceJsonArg(input) ?? {}, deps.registry, {
          store: deps.store,
          workspace: deps.workspace,
          services: deps.services,
          params: coerceJsonArg(params) as Record<string, unknown> | undefined,
        });

        return result;
      },
    }),

    run_step: tool({
      description:
        "Run a SINGLE step in isolation with a given config + input, and return its output + events — WITHOUT wiring it into a workflow. This is the inner loop for authoring an adapter: create_step → run_step → edit_step → run_step until the output is right. " +
        "Set cassette:'record' to run live AND capture the step's external service calls (http, etc.) to a reusable fixture (secrets are scrubbed); then cassette:'replay' to iterate OFFLINE against that fixture — deterministic, no rate limits, no cost, no side effects (so you don't, e.g., create a real charge on every test). " +
        "Returns { status, output?, error?, events, recorded? }.",
      inputSchema: z.object({
        type: z.string().describe("Step type to run, e.g. 'stripe/list-charges' or 'http'."),
        config: z
          .record(z.any())
          .optional()
          .describe("The step's config (same shape as in a workflow). Templates like {{ input.* }} / {{ params.* }} are resolved."),
        input: z
          .any()
          .optional()
          .describe("Workflow input object, referenced in config via {{ input.* }}."),
        params: z
          .record(z.any())
          .optional()
          .describe("Params knobs, referenced via {{ params.* }}."),
        cassette: z
          .enum(["record", "replay"])
          .optional()
          .describe("record: run live + capture external calls to a fixture. replay: serve them from the fixture (offline). Omit for a plain live run."),
        cassetteName: z
          .string()
          .optional()
          .describe("Fixture name (defaults to the step type). Use distinct names to keep multiple scenarios per step."),
      }),
      execute: async ({ type, config, input, params, cassette, cassetteName }) => {
        const registry = deps.registry;
        if (!registry[type]) return { error: `Step type "${type}" not found` };
        return runSingleStep(type, registry, deps.services, {
          config: coerceJsonArg(config) as Record<string, unknown> | undefined,
          input: coerceJsonArg(input),
          params: coerceJsonArg(params) as Record<string, unknown> | undefined,
          workspace: deps.workspace,
          ...(cassette
            ? { cassette: { mode: cassette, path: cassettePath(deps.workspace.path, cassetteName ?? type) } }
            : {}),
        });
      },
    }),

    list_runs: tool({
      description:
        "List past runs of a workflow (newest first), each with its status, duration, and timestamps. Use this to inspect a workflow's run history — e.g. to compare experiment runs or find a failing run. Then call get_run for a specific run's full input/output/events.",
      inputSchema: z.object({
        name: z.string().describe("Workflow name whose runs to list"),
        limit: z
          .number()
          .int()
          .positive()
          .default(20)
          .describe("Max number of recent runs to return (default 20)."),
      }),
      execute: async ({ name, limit }) => {
        const store = asReadStore(deps.store);
        if (!store) {
          return {
            error:
              "Run history is unavailable (the run store does not support reading back runs).",
          };
        }
        const ids = (await store.listRuns(name)).slice(0, limit);
        const runs = await Promise.all(
          ids.map(async (runId) => {
            const s = await store.getRunSummary(name, runId);
            return {
              runId,
              status: s?.status,
              startedAt: s?.startedAt,
              durationMs: s?.durationMs,
              ...(s?.error ? { error: s.error } : {}),
            };
          }),
        );
        return { workflow: name, runs };
      },
    }),

    get_run: tool({
      description:
        "Get a single run's details: its summary (input, output, status, error, duration) and its event log. By default the event log is slimmed (type/path/duration/error per step, no payloads) to stay token-cheap; set fullEvents:true to include each step's input/output. Use this to debug why a run failed or to read what each step produced.",
      inputSchema: z.object({
        name: z.string().describe("Workflow name"),
        runId: z.string().describe("Run id (a millisecond timestamp, from list_runs)"),
        fullEvents: z
          .boolean()
          .default(false)
          .describe("Include full per-step input/output payloads in events (default false: slimmed)."),
      }),
      execute: async ({ name, runId, fullEvents }) => {
        const store = asReadStore(deps.store);
        if (!store) {
          return {
            error:
              "Run history is unavailable (the run store does not support reading back runs).",
          };
        }
        const [summary, rawEvents] = await Promise.all([
          store.getRunSummary(name, runId),
          store.getRunEvents(name, runId),
        ]);
        if (!summary && rawEvents.length === 0) {
          return { error: `Run "${runId}" not found for workflow "${name}".` };
        }
        return {
          workflow: name,
          runId,
          summary,
          events: fullEvents ? rawEvents : rawEvents.map(slimEvent),
        };
      },
    }),
  };
}
