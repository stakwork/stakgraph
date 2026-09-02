import { z } from "zod";
import { tool } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { runWorkflow } from "../runner.js";
import { AiDeps } from "./prompts.js";
import { lsSteps, searchSteps, readStepSource } from "./stepHelpers.js";
import { zodToFields } from "./schemaHelpers.js";
import { runSingleStep, cassettePath } from "../run-step.js";
import { generateRunId } from "../store.js";
// The shared authoring core — the same mechanism the meta/* steps' capability
// sits on (see authoring.ts): publish checks + strict load-verification, and
// the run-history reads. The chat tools layer their own policy on top (no
// ownership gating — this surface is human-supervised).
import {
  AI_PUBLISHER,
  coerceJsonArg,
  listRunSummaries,
  publishNewStep,
  publishStepVersion,
  readRun,
  searchRunEvents,
} from "../authoring.js";

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
        const result = await publishNewStep(deps, name, code, description, AI_PUBLISHER);
        deps.registry = await deps.getRegistry();
        if (result.ok && result.loaded === false) {
          const { loadError, ...rest } = result;
          return { ...rest, warning: `Published but failed to load into the registry: ${loadError}` };
        }
        return result;
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
        const result = await publishStepVersion(deps, type, code, description);
        deps.registry = await deps.getRegistry();
        if (result.ok && result.loaded === false) {
          const { loadError, ...rest } = result;
          return { ...rest, warning: `Published but failed to load into the registry: ${loadError}` };
        }
        return result;
      },
    }),

    create_workflow: tool({
      description:
        "Create and publish a NEW workflow from YAML. If the name already " +
        "exists, a numeric suffix is appended (e.g. `send-email-2`). The " +
        "response includes the final name used. To publish a new version of " +
        "an EXISTING workflow, use `edit_workflow` instead. Pass `category` " +
        "to group the workflow in the UI sidebar (e.g. an experiment or " +
        "project name) — set it when the user asks for one or when the " +
        "workflow clearly belongs to an existing category (see " +
        "list_workflows for categories already in use).",
      inputSchema: z.object({
        name: z.string().describe("Workflow name (kebab-case)"),
        yaml: z.string().describe("Full workflow YAML"),
        description: z.string().optional(),
        category: z
          .string()
          .optional()
          .describe(
            "Optional sidebar grouping label (kebab-case, e.g. an experiment name). Omit to leave uncategorized.",
          ),
      }),
      execute: async ({ name, yaml, description, category }) => {
        const { name: finalName, version } = await deps.workspace.createWorkflow(
          name,
          yaml,
          description,
          category,
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
        category: z
          .string()
          .optional()
          .describe(
            "Optional sidebar grouping label. Only pass to CHANGE the category (to merely re-categorize without editing YAML, use set_workflow_category).",
          ),
      }),
      execute: async ({ name, yaml, description, category }) => {
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
            category,
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

    set_workflow_category: tool({
      description:
        "Set or clear an existing workflow's sidebar category (the grouping " +
        "label in the UI). Metadata-only: no new version is published and the " +
        "workflow YAML is untouched. Use when the user asks to categorize, " +
        "re-categorize, or group workflows. Check list_workflows first to " +
        "reuse an existing category name where one fits.",
      inputSchema: z.object({
        name: z.string().describe("Existing workflow name"),
        category: z
          .string()
          .nullable()
          .describe("New category label, or null to clear it"),
      }),
      execute: async ({ name, category }) => {
        try {
          await deps.workspace.setWorkflowCategory(name, category);
          return { ok: true, name, category };
        } catch (err) {
          return { error: err instanceof Error ? err.message : String(err) };
        }
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
        "Run a published workflow with a given input and return the result. Use this to test workflows you just created. Returns status (success/error), output (on success), error details (on failure), and the runId. " +
        "Long runs AUTO-DETACH: if the run is still executing after the wait window, this returns { status: 'running', detached: true, runId } and the run continues in the background — when it finishes, a [run-notification] user message starts your next turn with the outcome. Do NOT poll get_run in a loop while waiting; finish your turn normally.",
      inputSchema: z.object({
        name: z.string().describe("Workflow name to run"),
        input: z
          .any()
          .optional()
          .describe(
            "Input passed to the workflow as a JSON OBJECT (not a string), referenced in step configs via {{ input.* }} — the run subject, e.g. { owner, repo, pull_number }. Use {} if none.",
          ),
        params: z
          .record(z.string(), z.any())
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

        // Generate the runId here (not in the runner) so the detached stub
        // can report it before the run finishes.
        const runId = generateRunId();
        const startedAt = Date.now();
        // Register with the host's controller registry (when wired) so the
        // run is cancellable/pausable and lists as live from launch.
        const tracked = deps.trackRun?.(name, runId);
        const promise = runWorkflow(flow, coerceJsonArg(input) ?? {}, deps.registry, {
          runId,
          store: deps.store,
          workspace: deps.workspace,
          services: deps.services,
          params: coerceJsonArg(params) as Record<string, unknown> | undefined,
          controller: tracked?.controller,
          workflowHash:
            (await deps.workspace.getWorkflowHash(name, version)) ?? undefined,
        }).finally(() => tracked?.untrack());

        // No detach seam (tests / non-chat embedders) → await as before.
        const detach = deps.detach;
        if (!detach) return promise;

        // Dispatch mode: race the run against the wait window. Fast runs
        // return synchronously (the quick inner-loop path); a run that
        // outlives the window converts to detached — the host takes the
        // pending promise and wakes the chat when it settles.
        const pending = Symbol("pending");
        let timer: ReturnType<typeof setTimeout> | undefined;
        const winner = await Promise.race([
          promise,
          new Promise<typeof pending>((res) => {
            timer = setTimeout(() => res(pending), detach.waitMs);
          }),
        ]).finally(() => clearTimeout(timer));
        if (winner !== pending) return winner;

        detach.onDetach({ workflow: name, runId, startedAt, promise });
        return {
          status: "running",
          detached: true,
          runId,
          workflow: name,
          note:
            `Run still executing after ${Math.round(detach.waitMs / 1000)}s — it continues detached in the background. ` +
            "When it finishes, a [run-notification] message will start your next turn with the result. " +
            "Do NOT poll get_run in a loop; finish this turn normally (note anything you'll need when the result arrives).",
        };
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
          .record(z.string(), z.any())
          .optional()
          .describe("The step's config (same shape as in a workflow). Templates like {{ input.* }} / {{ params.* }} are resolved."),
        input: z
          .any()
          .optional()
          .describe("Workflow input object, referenced in config via {{ input.* }}."),
        params: z
          .record(z.string(), z.any())
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
        if (cassette && !deps.dataDir) {
          return { error: "Cassette record/replay is unavailable (no local data dir configured)." };
        }
        return runSingleStep(type, registry, deps.services, {
          config: coerceJsonArg(config) as Record<string, unknown> | undefined,
          input: coerceJsonArg(input),
          params: coerceJsonArg(params) as Record<string, unknown> | undefined,
          workspace: deps.workspace,
          ...(cassette
            ? { cassette: { mode: cassette, path: cassettePath(deps.dataDir!, cassetteName ?? type) } }
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
        return { workflow: name, runs: await listRunSummaries(deps.store, name, limit) };
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
        return readRun(deps.store, name, runId, fullEvents);
      },
    }),

    search_runs: tool({
      description:
        "Grep across a workflow's recent runs: match a regex against every event's JSON (inputs, outputs, errors) and get back (runId, event path, snippet) tuples plus a per-run frequency summary. The cross-run complement to get_run — use it to answer 'which runs hit this, and how often?' (e.g. a recurring error signature across a batch), then get_run to investigate one run. Note: tool outputs are truncated in the event log (~1500 chars), so a signature deep in long output can be missed.",
      inputSchema: z.object({
        name: z.string().describe("Workflow name whose runs to search"),
        pattern: z
          .string()
          .describe(
            "JavaScript regular expression matched against each event's JSON line, e.g. \"command not found|ModuleNotFoundError\".",
          ),
        runIds: z
          .array(z.string())
          .optional()
          .describe("Explicit run ids to search. Default: the newest runLimit runs."),
        runLimit: z
          .number()
          .int()
          .positive()
          .default(20)
          .describe("How many recent runs to scan when runIds is absent (default 20)."),
        maxMatches: z
          .number()
          .int()
          .positive()
          .default(50)
          .describe(
            "Cap on returned matches; scanning stops once reached (truncated: true). Narrow the pattern or run window rather than raising this (default 50).",
          ),
        ignoreCase: z.boolean().default(true).describe("Case-insensitive matching (default true)."),
      }),
      execute: async ({ name, pattern, runIds, runLimit, maxMatches, ignoreCase }) => {
        return searchRunEvents(deps.store, name, pattern, { runIds, runLimit, maxMatches, ignoreCase });
      },
    }),

    // Build-time shell — only offered when the host wires `deps.shell` (the
    // standard server does; embedders/tests without it get no bash tool).
    ...(deps.shell
      ? {
          bash: tool({
            description:
              "Execute a bash command in the server's data directory — for BUILD-TIME exploration while authoring: curl an API to see its real response shape before writing a step, clone a repo into scratch/ to inspect a data format, check a CLI exists, read a run's file outputs under artifacts/<runId>/. Commands run under a scrubbed environment (no server API keys — use placeholder values when probing an authed API, or author the step and run_step it with the real secret). This tool is NOT how production workflows reach the outside world: steps you author must still use ctx.services.http + ctx.services.secrets so runs stay recordable and secrets scrubbed. Output is captured with a cap; default timeout 30s (raise timeoutMs up to 10 min for clones/installs).",
            inputSchema: z.object({
              command: z.string().describe("The bash command to execute"),
              timeoutMs: z
                .number()
                .int()
                .positive()
                .max(600_000)
                .default(30_000)
                .describe("Kill the command after this long (default 30s, max 10 min)."),
            }),
            execute: async ({ command, timeoutMs }) => {
              const { cwd } = deps.shell!;
              try {
                // Lazy so embedders that never call bash don't load the module.
                const { runShell } = await import("../shell.js");
                const { mkdir } = await import("node:fs/promises");
                const { join } = await import("node:path");
                // scratch/ always exists — the advertised home for clones and
                // experiments, so they never land among workspace internals.
                await mkdir(join(cwd, "scratch"), { recursive: true });
                return { output: await runShell(command, cwd, timeoutMs, 20_000) };
              } catch (e) {
                return { error: e instanceof Error ? e.message : String(e) };
              }
            },
          }),
        }
      : {}),

    // Provider-executed web search (same tool the agent step ships) — for
    // reading API docs while authoring adapters. Anthropic-only; the host
    // opts in (the standard chat server does).
    // eslint-disable-next-line @typescript-eslint/no-explicit-any — provider
    // tool's inferred generics don't satisfy ToolSet's index signature (same
    // workaround as the agent step).
    ...(deps.webSearch ? { web_search: anthropic.tools.webSearch_20260209({ maxUses: 5 }) as any } : {}),
  };
}
