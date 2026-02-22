import { tool, Tool } from "ai";
import { z } from "zod";
import * as fs from "fs";
import * as path from "path";
import {
  fetchCloudwatchLogs,
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
}

export function get_log_tools(
  opts: LogToolsOptions
): Record<string, Tool<any, any>> {
  const { logsDir } = opts;
  const tools: Record<string, Tool<any, any>> = {
    fetch_cloudwatch: tool({
      description:
        "Fetch logs from AWS CloudWatch and save them to a local file for searching. Call this first to pull logs, then use bash to search through them (e.g. rg, grep, awk). Supports CloudWatch filter patterns.",
      inputSchema: z.object({
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
            'CloudWatch filter pattern (e.g. "ERROR" or "{ $.statusCode = 500 }")'
          ),
        minutes: z
          .number()
          .optional()
          .describe("How many minutes back to fetch (default 30)"),
        limit: z
          .number()
          .optional()
          .describe("Max number of log events to fetch (default 10000)"),
      }),
      execute: async ({
        log_group,
        log_stream_names,
        filter_pattern,
        minutes,
        limit,
      }: {
        log_group: string;
        log_stream_names?: string[];
        filter_pattern?: string;
        minutes?: number;
        limit?: number;
      }) => {
        try {
          const result = await fetchCloudwatchLogs({
            logGroupName: log_group,
            logStreamNames: log_stream_names,
            filterPattern: filter_pattern,
            minutes,
            limit,
            logsDir,
          });
          return `Fetched ${result.lineCount} log lines from ${result.logGroup} (${result.timeRange.startTime} to ${result.timeRange.endTime}). Saved to file: ${result.file}. Use bash to search through it (e.g. rg, grep, head, tail, awk).`;
        } catch (e: any) {
          return `Failed to fetch CloudWatch logs: ${e.message}`;
        }
      },
    }),

    list_cloudwatch_groups: tool({
      description:
        "List available CloudWatch log groups. Use this to discover which log groups exist before fetching logs.",
      inputSchema: z.object({
        prefix: z
          .string()
          .optional()
          .describe(
            "Optional prefix to filter log groups (e.g. /ecs/ or /aws/lambda/)"
          ),
      }),
      execute: async ({ prefix }: { prefix?: string }) => {
        try {
          const result = await listCloudwatchLogGroups(prefix);
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
      },
    }),

    fetch_quickwit: tool({
      description:
        "Fetch logs from Quickwit and save them to a local file for searching. Supports Lucene query syntax.",
      inputSchema: z.object({
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
      }),
      execute: async ({
        query,
        max_hits,
        start_timestamp,
        end_timestamp,
      }: {
        query: string;
        max_hits?: number;
        start_timestamp?: number;
        end_timestamp?: number;
      }) => {
        try {
          const result = await searchLogs({
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

          return `Fetched ${result.num_hits} log hits from Quickwit (query: "${query}"). Saved ${lines.length} lines to file: ${filename}. Use bash to search through it (e.g. rg, grep, head, tail, awk).`;
        } catch (e: any) {
          return `Failed to fetch Quickwit logs: ${e.message}`;
        }
      },
    }),

    bash: tool({
      description:
        "Execute a bash command in the logs directory. Use this for any log analysis that the other tools don't cover — awk, sort, uniq, wc, jq, etc.",
      inputSchema: z.object({
        command: z.string().describe("The bash command to execute"),
      }),
      execute: async ({ command }: { command: string }) => {
        try {
          return await executeBashCommand(command, logsDir, 15000);
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
        "Fetch logs for a Stakwork workflow run. Pick the projectId from the list of recent runs provided in the prompt. Returns logs in reverse chronological order. Supports filtering by step name, status, and pagination.",
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

  return tools;
}
