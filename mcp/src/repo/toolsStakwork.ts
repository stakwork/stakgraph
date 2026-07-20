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
}

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

  console.log(
    "===> registered stakwork run tools: stakwork_skill_usage, stakwork_workflow_runs, stakwork_run_steps",
  );
}
