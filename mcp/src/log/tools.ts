import { tool, Tool } from "ai";
import { z } from "zod";
import { redactSecrets } from "./redact.js";
import * as fs from "fs";
import * as path from "path";
import {
  fetchCloudwatchLogsInsights,
  listCloudwatchLogGroups,
} from "./cloudwatch.js";
import { fetchWorkflowRunLogs } from "./stakwork.js";
import { fetchAgentLog } from "./hive.js";
import { searchLogs } from "../handler/logs.js";
import { executeBashCommand } from "../repo/bash.js";
import { StakworkRunSummary } from "./types.js";

export interface LogToolsOptions {
  logsDir: string;
  stakworkApiKey?: string;
  stakworkRuns?: StakworkRunSummary[];
  abortSignal?: AbortSignal;
  /** Wall-clock start time (ms) used to report elapsed time back to the agent. */
  startTime?: number;
  /** Total time budget (ms) the agent should stay within. Default 8 min. */
  budgetMs?: number;
}

// ── What `/logs/agent` and the lab's `logs` workflow share ─────────────────
//
// Three tools' descriptions, inputs and bodies, exported so the lab's logs/*
// steps (lab/logs, through `services.logs`) say and do exactly what these
// tools do. The one difference is how a fetch result tells the agent to read
// the file it saved (`searchHint`): bash here; logs/search and logs/read
// there. The fetches are injectable for tests.

/** How a fetch result says to read the file it saved, for an agent with bash. */
export const BASH_SEARCH_HINT = "Use bash to search through it (e.g. rg, grep, head, tail, awk).";

export const FETCH_CLOUDWATCH_DESCRIPTION =
  'Fetch logs from AWS CloudWatch using Logs Insights and save them to a local file for searching. Results are returned **newest-first**. Plain string filter patterns are supported (e.g. "ERROR"); JSON-style patterns (e.g. "{ $.statusCode = 500 }") are **not** supported — use a plain string instead.';
export const fetchCloudwatchInput = z.object({
  log_group: z
    .string()
    .describe(
      "CloudWatch log group name (e.g. /swarms/38)"
    ),
  log_stream_names: z
    .array(z.string())
    .optional()
    .describe(
      'Filter to specific log streams (e.g. ["stakgraph.sphinx", "boltwall.sphinx"])'
    ),
  filter_pattern: z
    .string()
    .optional()
    .describe(
      'Plain string filter pattern (e.g. "ERROR", "147426952"). JSON-style patterns (e.g. "{ $.statusCode = 500 }") are not supported.'
    ),
  minutes: z
    .number()
    .optional()
    .describe("How many minutes back to fetch (default 30)"),
  limit: z
    .number()
    .optional()
    .describe("Max number of log events to fetch (default 10000)"),
});

export async function fetchCloudwatchText(
  {
    log_group,
    log_stream_names,
    filter_pattern,
    minutes,
    limit,
  }: z.infer<typeof fetchCloudwatchInput>,
  {
    logsDir,
    abortSignal,
    searchHint = BASH_SEARCH_HINT,
    fetch = fetchCloudwatchLogsInsights,
  }: { logsDir: string; abortSignal?: AbortSignal; searchHint?: string; fetch?: typeof fetchCloudwatchLogsInsights }
): Promise<string> {
  try {
    const result = await fetch({
      logGroupName: log_group,
      logStreamNames: log_stream_names,
      filterPattern: filter_pattern,
      minutes,
      limit,
      logsDir,
      abortSignal,
    });
    const truncNote = result.truncated
      ? " NOTE: results may be incomplete (fetch hit the wall-clock cap or the query timed out on AWS) — narrow the filter_pattern, log_stream_names, or minutes to get complete results."
      : "";
    const hydrateNote = result.hydratedLines
      ? ` ${result.hydratedLines} long line(s) were recovered to full length via GetLogRecord (Logs Insights truncates long messages in query results).`
      : "";
    return `Fetched ${result.lineCount} log lines from ${result.logGroup} (${result.timeRange.startTime} to ${result.timeRange.endTime}). Saved to file: ${result.file}. ${searchHint}${hydrateNote}${truncNote}`;
  } catch (e: any) {
    return `Failed to fetch CloudWatch logs: ${e.message}`;
  }
}

export const LIST_CLOUDWATCH_GROUPS_DESCRIPTION =
  "List available CloudWatch log groups. Use this to discover which log groups exist before fetching logs.";
export const listCloudwatchGroupsInput = z.object({
  prefix: z
    .string()
    .optional()
    .describe(
      "Optional prefix to filter log groups (e.g. /ecs/ or /aws/lambda/)"
    ),
});

export async function listCloudwatchGroupsText(
  { prefix }: z.infer<typeof listCloudwatchGroupsInput>,
  { list = listCloudwatchLogGroups }: { list?: typeof listCloudwatchLogGroups } = {}
): Promise<string> {
  try {
    const result = await list(prefix);
    if (result.logGroups.length === 0) {
      return "No log groups found" + (prefix ? ` with prefix "${prefix}"` : "");
    }
    return result.logGroups
      .map(
        (g) =>
          `${g.name} (${g.storedBytes ? Math.round(g.storedBytes / 1024 / 1024) + "MB" : "?"}${g.retentionDays ? ", " + g.retentionDays + "d retention" : ""})`
      )
      .join("\n");
  } catch (e: any) {
    return `Failed to list log groups: ${e.message}`;
  }
}

export const FETCH_QUICKWIT_DESCRIPTION =
  "Fetch endpoint logs from the user's actual application (such as vercel logs) and save them to a local file for searching. Supports Lucene query syntax.";
export const fetchQuickwitInput = z.object({
  query: z
    .string()
    .describe(
      'Lucene query string (e.g. "level:error AND service:payments")'
    ),
  max_hits: z
    .number()
    .optional()
    .describe("Maximum number of results (default 1000)"),
  start_timestamp: z
    .number()
    .optional()
    .describe("Start timestamp (Unix epoch seconds)"),
  end_timestamp: z
    .number()
    .optional()
    .describe("End timestamp (Unix epoch seconds)"),
});

export async function fetchQuickwitText(
  {
    query,
    max_hits,
    start_timestamp,
    end_timestamp,
  }: z.infer<typeof fetchQuickwitInput>,
  {
    logsDir,
    searchHint = BASH_SEARCH_HINT,
    search = searchLogs,
  }: { logsDir: string; searchHint?: string; search?: typeof searchLogs }
): Promise<string> {
  try {
    const result = await search({
      query,
      max_hits: max_hits || 1000,
      start_timestamp,
      end_timestamp,
    });

    // Write to file
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    const safeQuery = query.replace(/[^a-zA-Z0-9_-]/g, "_").substring(0, 40);
    const filename = `qw-${safeQuery}-${ts}.log`;
    const filepath = path.join(logsDir, filename);

    const lines = result.hits.map((h) => {
      const time = h.timestamp
        ? new Date(h.timestamp * 1000).toISOString()
        : "unknown";
      const level = h.level || "";
      const service = h.service || "";
      const msg = (h.message || "").trimEnd();
      return `[${time}] [${level}] [${service}] ${msg}`;
    });

    fs.writeFileSync(filepath, lines.join("\n"), "utf-8");

    return `Fetched ${result.num_hits} log hits from Quickwit (query: "${query}"). Saved ${lines.length} lines to file: ${filename}. ${searchHint}`;
  } catch (e: any) {
    return `Failed to fetch Quickwit logs: ${e.message}`;
  }
}

/** Appended to every tool result so the agent can actually see how much of
 *  its time budget it has used (it has no clock of its own between steps). */
export function timeBudgetNote(startTime: number, budgetMs: number): string {
  const elapsedMin = (Date.now() - startTime) / 60_000;
  const budgetMin = budgetMs / 60_000;
  const remainingMin = Math.max(0, budgetMin - elapsedMin);
  return `\n\n[time budget: ${elapsedMin.toFixed(1)} min elapsed of ${budgetMin.toFixed(0)} min; ~${remainingMin.toFixed(1)} min left${remainingMin <= 2 ? " — wrap up and write your final answer now" : ""}]`;
}

export function get_log_tools(
  opts: LogToolsOptions
): Record<string, Tool<any, any>> {
  const { logsDir, abortSignal } = opts;
  const redactOpts = { literals: [opts.stakworkApiKey].filter(Boolean) as string[] };
  const startTime = opts.startTime ?? Date.now();
  const budgetMs = opts.budgetMs ?? 8 * 60 * 1000;
  const elapsedNote = (): string => timeBudgetNote(startTime, budgetMs);
  const tools: Record<string, Tool<any, any>> = {
    fetch_cloudwatch: tool({
      description: FETCH_CLOUDWATCH_DESCRIPTION,
      inputSchema: fetchCloudwatchInput,
      execute: (args: z.infer<typeof fetchCloudwatchInput>) =>
        fetchCloudwatchText(args, { logsDir, abortSignal }),
    }),

    list_cloudwatch_groups: tool({
      description: LIST_CLOUDWATCH_GROUPS_DESCRIPTION,
      inputSchema: listCloudwatchGroupsInput,
      execute: (args: z.infer<typeof listCloudwatchGroupsInput>) =>
        listCloudwatchGroupsText(args),
    }),

    fetch_quickwit: tool({
      description: FETCH_QUICKWIT_DESCRIPTION,
      inputSchema: fetchQuickwitInput,
      execute: (args: z.infer<typeof fetchQuickwitInput>) =>
        fetchQuickwitText(args, { logsDir }),
    }),

    bash: tool({
      description:
        "Execute a bash command in the logs directory. Use this for any log analysis that the other tools don't cover — awk, sort, uniq, wc, jq, etc.",
      inputSchema: z.object({
        command: z.string().describe("The bash command to execute"),
      }),
      execute: async ({ command }: { command: string }) => {
        try {
          const output = await executeBashCommand(command, logsDir, 15000);
          return redactSecrets(output, redactOpts);
        } catch (e: any) {
          return `Command failed: ${e.message}`;
        }
      },
    }),
  };

  if (opts?.stakworkRuns && opts.stakworkRuns.length > 0) {
    const runs = opts.stakworkRuns;

    tools.fetch_agent_log = tool({
      description:
        "Fetch the full log content for an agent from a Stakwork workflow run. If the run has only one agent log, the agent parameter can be omitted. The agent names and project IDs are provided in the recent runs context.",
      inputSchema: z.object({
        project_id: z
          .number()
          .describe("The Stakwork project ID (from the recent runs list)"),
        agent: z
          .string()
          .optional()
          .describe(
            "The agent name to fetch logs for (e.g. 'architecture', 'code'). If omitted, uses the first (or only) agent log."
          ),
      }),
      execute: async ({
        project_id,
        agent,
      }: {
        project_id: number;
        agent?: string;
      }) => {
        try {
          const result = await fetchAgentLog({
            runs,
            projectId: project_id,
            agent,
            logsDir,
          });
          return `Fetched agent log for "${result.agent}" from run ${result.projectId} (${result.lineCount} lines). Saved to file: ${result.file}. Use bash to search through it.`;
        } catch (e: any) {
          return `Failed to fetch agent log: ${e.message}`;
        }
      },
    });
  }

  if (opts?.stakworkApiKey) {
    const apiKey = opts.stakworkApiKey;

    tools.fetch_workflow_run = tool({
      description:
        'Fetch logs for a Stakwork workflow run. Pick the projectId from the list of recent runs provided in the prompt. Returns logs in reverse chronological order. Supports filtering by step name, status, and pagination. NOTE: messages are truncated to 1000 chars per line (ending in "... (continued)") — the full untruncated lines live in the CloudWatch log group "/stakwork/production", searchable with fetch_cloudwatch using filter_pattern "project_id_<projectId>".',
      inputSchema: z.object({
        project_id: z
          .string()
          .describe("The Stakwork project ID (from the recent runs list)"),
        step: z
          .string()
          .optional()
          .describe("Filter logs by workflow step name"),
        status: z
          .enum(["success", "error", "warning", "failed"])
          .optional()
          .describe("Filter by log status"),
        include_children: z
          .boolean()
          .optional()
          .describe(
            "Include logs from child projects (sub-runs). Only applies to parent projects."
          ),
        limit: z
          .number()
          .optional()
          .describe("Number of logs per page (default 50, max 100)"),
        page: z
          .number()
          .optional()
          .describe("Page number for pagination (default 1)"),
      }),
      execute: async ({
        project_id,
        step,
        status,
        include_children,
        limit,
        page,
      }: {
        project_id: string;
        step?: string;
        status?: "success" | "error" | "warning" | "failed";
        include_children?: boolean;
        limit?: number;
        page?: number;
      }) => {
        try {
          const result = await fetchWorkflowRunLogs({
            apiKey,
            projectId: project_id,
            step,
            status,
            include_children,
            limit,
            page,
            logsDir,
          });
          const { pagination } = result;
          return `Fetched ${result.logCount} logs from Stakwork project ${project_id} (page ${pagination.page}/${pagination.pages}, total ${pagination.count}). Saved to file: ${result.file}. Use bash to search through it.`;
        } catch (e: any) {
          return `Failed to fetch Stakwork workflow logs: ${e.message}`;
        }
      },
    });
  }

  // Wrap every tool so its (string) result carries the current time-budget
  // status. This is the only way the agent gets a sense of elapsed wall-clock
  // time, since it has no clock between steps.
  for (const key of Object.keys(tools)) {
    const t = tools[key];
    const orig = t.execute;
    if (!orig) continue;
    t.execute = (async (args: any, options: any) => {
      const out = await orig(args, options);
      return typeof out === "string" ? out + elapsedNote() : out;
    }) as typeof t.execute;
  }

  return tools;
}
