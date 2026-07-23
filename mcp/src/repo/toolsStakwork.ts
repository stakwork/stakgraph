import { tool, Tool } from "ai";
import { z } from "zod";
import axios from "axios";

/**
 * Stakwork run-research tools — read-only lookups against the Stakwork API
 * (jobs.stakwork.com) for ground-truth workflow execution data: which
 * workflows actually run, their success/error states, and the real params
 * and outputs each step sent and received.
 *
 * Registered only when the caller supplies a Stakwork API key on the request
 * body (`stakworkApiKey`) — the key is plumbed server-to-server and is never
 * an LLM-visible parameter. All endpoints are GETs; runs/stats are scoped by
 * Stakwork to the customer that owns the API key.
 *
 * Exception: `stakwork_run_step` EXECUTES a single step of a published
 * workflow (a real, billable run) via POST /workflows/:id/run_step_from_template,
 * with ancestor-keyed inputs seeded as literals into a synthesized set_var
 * step. It is double-gated — the API key must be present AND the caller must
 * opt in with a truthy `toolsConfig.stakwork_run_step`.
 */

const DEFAULT_STAKWORK_API_URL = "https://jobs.stakwork.com/api/v1";

/** Cap for params/output previews in the per-step list view. */
const STEP_FIELD_PREVIEW_CHARS = 1500;
/** Cap for each side (inputs/outputs) of a single-step IO drill-down. */
const STEP_IO_FIELD_CHARS = 12000;
/** Cap for each curated skill example's input/output. */
const EXAMPLE_FIELD_CHARS = 1500;

export interface StakworkToolsOptions {
  apiKey: string;
  baseUrl?: string;
  /** Opt-in for the execute tool stakwork_run_step (launches real runs). */
  runStep?: boolean;
}

/** Poll interval and wait bounds for stakwork_run_step. */
const RUN_STEP_POLL_MS = 5000;
const RUN_STEP_DEFAULT_WAIT_S = 120;
const RUN_STEP_MAX_WAIT_S = 600;
/** Project states after which a run will not progress further. */
const TERMINAL_RUN_STATES = new Set(["completed", "error", "halted", "stopped", "failed"]);

async function stakworkGet(
  url: string,
  apiKey: string,
): Promise<{ ok: boolean; status: number; body: any }> {
  const resp = await axios.get(url, {
    headers: { Authorization: `Token token=${apiKey}` },
    validateStatus: () => true,
    responseType: "text",
    timeout: 60_000,
  });
  const text: string =
    typeof resp.data === "string" ? resp.data : JSON.stringify(resp.data);
  let body: any = text;
  try {
    body = JSON.parse(text);
  } catch (_) {
    // leave as raw text
  }
  return { ok: resp.status >= 200 && resp.status < 300, status: resp.status, body };
}

async function stakworkPost(
  url: string,
  apiKey: string,
  payload: unknown,
): Promise<{ ok: boolean; status: number; body: any }> {
  const resp = await axios.post(url, payload, {
    headers: {
      Authorization: `Token token=${apiKey}`,
      "Content-Type": "application/json",
    },
    validateStatus: () => true,
    responseType: "text",
    timeout: 60_000,
  });
  const text: string =
    typeof resp.data === "string" ? resp.data : JSON.stringify(resp.data);
  let body: any = text;
  try {
    body = JSON.parse(text);
  } catch (_) {
    // leave as raw text
  }
  return { ok: resp.status >= 200 && resp.status < 300, status: resp.status, body };
}

/** Stringify a value and truncate with an explicit marker so the agent knows data was elided. */
export function truncateJson(value: unknown, maxChars: number): string {
  if (value === null || value === undefined) return "null";
  const str = typeof value === "string" ? value : JSON.stringify(value);
  if (str.length <= maxChars) return str;
  return `${str.slice(0, maxChars)}…[truncated, ${str.length} chars total]`;
}

function errorResult(label: string, status: number, body: any): string {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return `${label} failed: HTTP ${status}: ${truncateJson(text, 500)}`;
}

export function registerStakworkTools(
  allTools: Record<string, Tool<any, any>>,
  options: StakworkToolsOptions,
): void {
  const baseUrl = (options.baseUrl || process.env.STAKWORK_API_URL || DEFAULT_STAKWORK_API_URL).replace(/\/$/, "");
  const apiKey = options.apiKey;

  allTools.stakwork_skill_usage = tool({
    description:
      "Look up real-world usage of a Stakwork skill by NAME: total + last-30-days use counts, " +
      "and the breakdown of which workflows actually invoke it ({workflow_id, use_count}), computed from run telemetry. " +
      "Optionally includes curated input/output examples for the skill. " +
      "Use this as the entry point for 'how is skill X actually used' — then follow the top workflow_id " +
      "into stakwork_workflow_runs → stakwork_run_steps to see real params from a live run.",
    inputSchema: z.object({
      skill_name: z
        .string()
        .describe("Exact skill name, e.g. 'AzureOCR' (as it appears in workflow transitions)."),
      include_examples: z
        .boolean()
        .optional()
        .default(true)
        .describe("Also fetch curated input/output examples for the skill (first page)."),
    }),
    execute: async ({
      skill_name,
      include_examples = true,
    }: {
      skill_name: string;
      include_examples?: boolean;
    }) => {
      console.log(`[stakwork_skill_usage] skill_name=${skill_name} examples=${include_examples}`);
      try {
        const statsUrl = `${baseUrl}/skills/${encodeURIComponent(skill_name)}/stats`;
        const stats = await stakworkGet(statsUrl, apiKey);
        if (!stats.ok) return errorResult("stakwork_skill_usage", stats.status, stats.body);
        const data = stats.body?.data ?? stats.body;

        let examples: unknown[] | undefined;
        if (include_examples && data?.id !== undefined) {
          const exResp = await stakworkGet(
            `${baseUrl}/skills/${encodeURIComponent(String(data.id))}/examples?limit=5`,
            apiKey,
          );
          if (exResp.ok) {
            const exData = exResp.body?.data ?? exResp.body;
            examples = (exData?.results ?? []).map((ex: any) => ({
              id: ex.id,
              description: ex.description,
              input: truncateJson(ex.input, EXAMPLE_FIELD_CHARS),
              output: truncateJson(ex.output, EXAMPLE_FIELD_CHARS),
            }));
          }
        }

        return JSON.stringify({ ...data, ...(examples ? { examples } : {}) });
      } catch (err: any) {
        return `stakwork_skill_usage failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  allTools.stakwork_workflow_runs = tool({
    description:
      "List recent execution runs of a Stakwork workflow by workflow_id (the same id cited from graph results). " +
      "Each run has an id (project_id), workflow_state (completed | error | halted | stopped | in-flight states), " +
      "started_at and duration. Use it to verify a workflow actually runs successfully and recently, " +
      "then feed a run's id into stakwork_run_steps to inspect the real data it processed. " +
      "Only runs owned by this customer are visible.",
    inputSchema: z.object({
      workflow_id: z.number().describe("Stakwork workflow id, e.g. 55639."),
      limit: z
        .number()
        .optional()
        .default(10)
        .describe("Max runs to return, newest first (1-100)."),
    }),
    execute: async ({ workflow_id, limit = 10 }: { workflow_id: number; limit?: number }) => {
      console.log(`[stakwork_workflow_runs] workflow_id=${workflow_id} limit=${limit}`);
      try {
        const url = `${baseUrl}/workflows/${encodeURIComponent(String(workflow_id))}/runs?limit=${encodeURIComponent(String(limit))}`;
        const resp = await stakworkGet(url, apiKey);
        if (!resp.ok) return errorResult("stakwork_workflow_runs", resp.status, resp.body);
        return JSON.stringify(resp.body?.data ?? resp.body);
      } catch (err: any) {
        return `stakwork_workflow_runs failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  allTools.stakwork_run_steps = tool({
    description:
      "Inspect the executed steps of one Stakwork run (project): per step the skill invoked (skill_name) and the " +
      "ACTUAL params sent and output produced — ground truth for how a skill is configured in practice " +
      "(exact URL formats, variable interpolations like [#(step).output.var], attribute values). " +
      "Params/outputs are previews truncated to ~1.5KB each; pass step_name to drill into ONE step's full inputs/outputs. " +
      "Filter with skill_name to keep only steps that invoked that skill. " +
      "Only runs owned by this customer are visible.",
    inputSchema: z.object({
      project_id: z.number().describe("Run id from stakwork_workflow_runs (a Stakwork project id)."),
      skill_name: z
        .string()
        .optional()
        .describe("Only return steps whose skill_name matches (case-insensitive)."),
      step_name: z
        .string()
        .optional()
        .describe(
          "A step's `name` from a prior stakwork_run_steps call. When set, returns that single step's FULL inputs/outputs instead of the step list."
        ),
      limit: z
        .number()
        .optional()
        .default(50)
        .describe("Max steps to return in list mode."),
    }),
    execute: async ({
      project_id,
      skill_name,
      step_name,
      limit = 50,
    }: {
      project_id: number;
      skill_name?: string;
      step_name?: string;
      limit?: number;
    }) => {
      console.log(
        `[stakwork_run_steps] project_id=${project_id} skill_name=${skill_name ?? "*"} step_name=${step_name ?? "-"}`,
      );
      try {
        // Drill-down: one step's full IO (still capped to protect context).
        if (step_name) {
          const url = `${baseUrl}/projects/${encodeURIComponent(String(project_id))}/steps/${encodeURIComponent(step_name)}/io`;
          const resp = await stakworkGet(url, apiKey);
          if (!resp.ok) return errorResult("stakwork_run_steps", resp.status, resp.body);
          const io = resp.body?.data ?? resp.body;
          return JSON.stringify({
            step_name,
            inputs: truncateJson(io?.inputs, STEP_IO_FIELD_CHARS),
            outputs: truncateJson(io?.outputs, STEP_IO_FIELD_CHARS),
            prompt_resolutions: truncateJson(io?.prompt_resolutions, STEP_IO_FIELD_CHARS),
          });
        }

        const url = `${baseUrl}/projects/${encodeURIComponent(String(project_id))}/steps?limit=${encodeURIComponent(String(limit))}`;
        const resp = await stakworkGet(url, apiKey);
        if (!resp.ok) return errorResult("stakwork_run_steps", resp.status, resp.body);
        const data = resp.body?.data ?? resp.body;
        let steps: any[] = data?.steps ?? [];
        if (skill_name) {
          const want = skill_name.toLowerCase();
          steps = steps.filter((s) => String(s.skill_name ?? "").toLowerCase() === want);
        }
        return JSON.stringify({
          pagination: data?.pagination,
          ...(skill_name ? { filtered_by_skill: skill_name } : {}),
          steps: steps.map((s) => ({
            name: s.name,
            skill_name: s.skill_name,
            has_output: s.has_output,
            created_at: s.created_at,
            params: truncateJson(s.params, STEP_FIELD_PREVIEW_CHARS),
            output: truncateJson(s.output, STEP_FIELD_PREVIEW_CHARS),
          })),
        });
      } catch (err: any) {
        return `stakwork_run_steps failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  if (options.runStep) {
    allTools.stakwork_run_step = tool({
      description:
        "EXECUTE one step of a PUBLISHED Stakwork workflow with your own inputs and return its output. " +
        "This launches a REAL run that consumes execution resources — use it deliberately for testing a specific step, never for browsing. " +
        "params are ANCESTOR-KEYED: { \"<ancestor_step_id>\": { \"<output.path>\": value } } — one value for each " +
        "[$(ancestor).output.path] reference the step consumes (flat \"ancestor_id.output.path\" keys also accepted). " +
        "Stakwork seeds them as literals into a synthesized set_var step, so NO prior run or ancestor execution is needed. " +
        "DISCOVERY: call with workflow_id + step_id and NO params — the response lists every required ancestor key, which is " +
        "exactly the input shape to fill in. Or read a real run first via stakwork_run_steps to copy actual values. " +
        "{{SECRET_NAME}} aliases resolve server-side at execution (pass them through unchanged, never inline real secrets); " +
        "set mock_mode: true to use stored mock step outputs instead of live execution. " +
        "The tool polls until the run reaches a terminal state or wait_seconds elapses; on timeout it returns status in_progress — " +
        "call it again with the returned project_id (plus step_id) to continue waiting without launching a new run.",
      inputSchema: z.object({
        step_id: z
          .string()
          .describe("The step's id in the workflow spec (its transition id, e.g. 'call_swarm_agent')."),
        workflow_id: z
          .number()
          .optional()
          .describe("Source workflow (must have a published version) that defines the step. Required to LAUNCH; omit when resuming with project_id."),
        params: z
          .record(z.string(), z.any())
          .optional()
          .describe(
            "Ancestor-keyed inputs: { \"<ancestor_step_id>\": { \"<output.path>\": value } }. Omit to DISCOVER the required keys without running anything.",
          ),
        mock_mode: z
          .boolean()
          .optional()
          .describe("Run with stored mock step outputs instead of live execution (cheap dry test)."),
        project_id: z
          .number()
          .optional()
          .describe("A project_id returned by a previous stakwork_run_step call: resume waiting on that run instead of launching a new one."),
        wait_seconds: z
          .number()
          .optional()
          .default(RUN_STEP_DEFAULT_WAIT_S)
          .describe(`How long to wait for the run to finish before returning in_progress (max ${RUN_STEP_MAX_WAIT_S}).`),
      }),
      execute: async ({
        step_id,
        workflow_id,
        params,
        mock_mode,
        project_id,
        wait_seconds = RUN_STEP_DEFAULT_WAIT_S,
      }: {
        step_id: string;
        workflow_id?: number;
        params?: Record<string, unknown>;
        mock_mode?: boolean;
        project_id?: number;
        wait_seconds?: number;
      }) => {
        console.log(
          `[stakwork_run_step] workflow_id=${workflow_id ?? "-"} step_id=${step_id} project_id=${project_id ?? "-"} mock=${mock_mode ?? false}`,
        );
        try {
          let runId = project_id;

          if (runId === undefined) {
            if (workflow_id === undefined) {
              return "stakwork_run_step failed: pass workflow_id + step_id to launch a run, or project_id to resume waiting on one.";
            }
            const probing = !params || Object.keys(params).length === 0;
            // The endpoint rejects blank params before validating coverage, so a
            // structurally-empty probe object triggers the missing-keys check,
            // whose error message enumerates every required ancestor input.
            const launch = await stakworkPost(
              `${baseUrl}/workflows/${encodeURIComponent(String(workflow_id))}/run_step_from_template`,
              apiKey,
              {
                step_id,
                params: probing ? { _probe: { _: "_" } } : params,
                ...(mock_mode !== undefined ? { mock_mode } : {}),
              },
            );
            if (!launch.ok) return errorResult("stakwork_run_step", launch.status, launch.body);
            if (launch.body?.success === false) {
              const errors = launch.body?.errors ?? launch.body?.error;
              return JSON.stringify({
                launched: false,
                ...(probing ? { discovery: true } : {}),
                errors: truncateJson(errors, 2000),
                hint: "'Missing required ancestor keys: a.b, c.d' enumerates the step's full input shape — relaunch with params: { \"a\": { \"b\": <value> }, \"c\": { \"d\": <value> } }.",
              });
            }
            runId = (launch.body?.data ?? launch.body)?.project_id;
            if (runId === undefined) {
              return `stakwork_run_step failed: launch response had no project_id: ${truncateJson(launch.body, 500)}`;
            }
          }

          const waitMs = Math.min(Math.max(wait_seconds, 0), RUN_STEP_MAX_WAIT_S) * 1000;
          const deadline = Date.now() + waitMs;
          let status = "unknown";

          for (;;) {
            const st = await stakworkGet(
              `${baseUrl}/projects/${encodeURIComponent(String(runId))}/status`,
              apiKey,
            );
            if (st.ok) status = String((st.body?.data ?? st.body)?.status ?? "unknown");
            if (TERMINAL_RUN_STATES.has(status) || Date.now() >= deadline) break;
            await new Promise((r) => setTimeout(r, RUN_STEP_POLL_MS));
          }

          if (!TERMINAL_RUN_STATES.has(status)) {
            return JSON.stringify({
              project_id: runId,
              step_id,
              status,
              note: "Run still in progress — call stakwork_run_step again with this project_id (and step_id) to continue waiting.",
            });
          }

          const io = await stakworkGet(
            `${baseUrl}/projects/${encodeURIComponent(String(runId))}/steps/${encodeURIComponent(step_id)}/io`,
            apiKey,
          );
          const ioData = io.ok ? (io.body?.data ?? io.body) : undefined;

          return JSON.stringify({
            project_id: runId,
            step_id,
            status,
            inputs: truncateJson(ioData?.inputs, STEP_IO_FIELD_CHARS),
            outputs: truncateJson(ioData?.outputs, STEP_IO_FIELD_CHARS),
            ...(io.ok ? {} : { io_error: errorResult("step io read", io.status, io.body) }),
          });
        } catch (err: any) {
          return `stakwork_run_step failed: ${err?.message ?? String(err)}`;
        }
      },
    });
  }

  console.log(
    `===> registered stakwork run tools: stakwork_skill_usage, stakwork_workflow_runs, stakwork_run_steps${options.runStep ? ", stakwork_run_step" : ""}`,
  );
}
