import { listCloudwatchLogStreams as listStreams } from "./cloudwatch.js";
import { getCurrentDateSnippet as dateSnippet } from "../repo/utils.js";
import { StakworkRunSummary } from "./types.js";

/** Convert a UI swarm name like "swarm38" or "swarmHDYF7D" to a CloudWatch log group like "/swarms/38" or "/swarms/HDYF7D" */
export function swarmNameToLogGroup(swarmName: string): string | null {
  const match = swarmName.match(/^swarm(.+)$/);
  if (!match) return null;
  return `/swarms/${match[1]}`;
}

/** What the logs agent is told about where to look, beside the question. */
export interface LogsPromptOptions {
  swarmName?: string;
  stakworkRuns?: StakworkRunSummary[];
  workspaceSlug?: string;
  logGroups?: string[];
  poolName?: string;
}

/** The tools the context names, as the agent reading it has them: the
 *  `/logs/agent` tools by default; the lab's `logs` workflow passes its
 *  steps' tool names. */
export interface LogsPromptTools {
  fetchCloudwatch: string;
  fetchQuickwit: string;
  search: string;
}

const AGENT_TOOLS: LogsPromptTools = { fetchCloudwatch: "fetch_cloudwatch", fetchQuickwit: "fetch_quickwit", search: "bash" };

/** The two calls the assembly makes, injectable for tests. */
export interface LogsPromptDeps {
  listCloudwatchLogStreams: typeof listStreams;
  getCurrentDateSnippet: typeof dateSnippet;
}

/**
 * The logs agent's prompt: the question with the context that tells it
 * where to look — the date, the swarm's log group and its streams, recent
 * Stakwork runs, the app's Quickwit logs, Stakwork production, extra log
 * groups, the pool's workspace pods. Moved out of the `/logs/agent` handler
 * (index.ts) unchanged, so the lab's `logs` workflow (lab/logs) builds the
 * same prompt. Each block is PREPENDED, so the last one added reads first.
 */
export async function buildLogsPrompt(
  prompt: string,
  { swarmName, stakworkRuns, workspaceSlug, logGroups, poolName }: LogsPromptOptions,
  { listCloudwatchLogStreams = listStreams, getCurrentDateSnippet = dateSnippet }: Partial<LogsPromptDeps> = {},
  tools: LogsPromptTools = AGENT_TOOLS,
): Promise<string> {
  let finalPrompt = prompt;
  finalPrompt = `${getCurrentDateSnippet()}\n\n${finalPrompt}`;
  if (swarmName) {
    const logGroup = swarmNameToLogGroup(swarmName);
    if (logGroup) {
      const context: string[] = [];
      context.push(`The CloudWatch log group for this swarm is "${logGroup}". Use this log group when fetching logs.`);
      try {
        const streams = await listCloudwatchLogStreams(logGroup);
        if (streams.length > 0) {
          context.push(`\nAvailable log streams (services) in this log group:`);
          for (const s of streams) {
            const serviceName = s.name.replace(/\.sphinx$/, "");
            context.push(`  - "${s.name}" (service: ${serviceName}${s.lastEventTime ? ", last event: " + s.lastEventTime : ""})`);
          }
          context.push(`\nIf the user mentions a specific service, use the log_stream_names parameter to filter to that stream. If no specific service is mentioned, fetch all streams.`);
        }
      } catch (e) {
        console.warn("Failed to list log streams:", e);
      }
      finalPrompt = context.join("\n") + `\n\n${prompt}`;
    }
  }

  if (stakworkRuns && stakworkRuns.length > 0) {
    const runsJson = JSON.stringify(stakworkRuns, null, 2);
    const runsContext = [
      `\nRecent Stakwork workflow runs (use projectId with fetch_workflow_run to get logs):`,
      runsJson,
      `\nPick the most relevant run based on the user's question, if its about a recent workflow (like a feature architecture, hive task, etc.) Use the projectId to fetch logs. If a run has agentLogs, you can use fetch_agent_log to read the full log content for a specific agent.`,
      `\nIMPORTANT: fetch_workflow_run messages are truncated to 1000 characters per line (truncated lines end with "... (continued)"). The full untruncated lines for a run are in the CloudWatch log group "/stakwork/production" — if you hit a truncated line you need in full, use fetch_cloudwatch with log_group "/stakwork/production", filter_pattern "project_id_<projectId>", and minutes covering the run's createdAt time.`,
    ].join("\n");
    finalPrompt = runsContext + `\n\n${finalPrompt}`;
  }

  if (workspaceSlug) {
    finalPrompt = `If the user mentions their actual application name (e.g. "${workspaceSlug}"), use the ${tools.fetchQuickwit} tool to fetch logs from the actual application.\n\n${finalPrompt}`;
  }

  if (workspaceSlug === "stakwork") {
    const prodLogGroup = "/stakwork/production";
    const context: string[] = [];
    context.push(`The CloudWatch log group for Stakwork production is "${prodLogGroup}". Use this log group when fetching production logs.`);
    try {
      const streams = await listCloudwatchLogStreams(prodLogGroup);
      if (streams.length > 0) {
        context.push(`\nAvailable log streams (services) in this log group:`);
        for (const s of streams) {
          const serviceName = s.name.replace(/\.sphinx$/, "");
          context.push(`  - "${s.name}" (service: ${serviceName}${s.lastEventTime ? ", last event: " + s.lastEventTime : ""})`);
        }
        context.push(`\nIf the user mentions a specific service, use the log_stream_names parameter to filter to that stream. If no specific service is mentioned, fetch all streams.`);
      }
    } catch (e) {
      console.warn("Failed to list log streams for /stakwork/production:", e);
    }
    finalPrompt = context.join("\n") + `\n\n${finalPrompt}`;
  }

  if (logGroups && logGroups.length > 0) {
    for (const logGroup of logGroups) {
      const context: string[] = [];
      context.push(`The CloudWatch log group "${logGroup}" is available. Use this log group when fetching logs.`);
      try {
        const streams = await listCloudwatchLogStreams(logGroup);
        if (streams.length > 0) {
          context.push(`\nAvailable log streams (services) in this log group:`);
          for (const s of streams) {
            const serviceName = s.name.replace(/\.sphinx$/, "");
            context.push(`  - "${s.name}" (service: ${serviceName}${s.lastEventTime ? ", last event: " + s.lastEventTime : ""})`);
          }
          context.push(`\nIf the user mentions a specific service, use the log_stream_names parameter to filter to that stream. If no specific service is mentioned, fetch all streams.`);
        }
      } catch (e) {
        console.warn(`Failed to list log streams for ${logGroup}:`, e);
      }
      finalPrompt = context.join("\n") + `\n\n${finalPrompt}`;
    }
  }

  if (poolName) {
    const poolLogGroup = `/workspaces/workspace-cluster/pools/${poolName}/workspaces`;
    const context: string[] = [];
    context.push(
      `If the user asks about "pod", "pool", or "sandbox" logs, they mean the workspace pod logs in the CloudWatch log group "${poolLogGroup}". Use the ${tools.fetchCloudwatch} tool with this log group, then use ${tools.search} to search the saved file.`
    );
    context.push(
      `\nEach workspace pod has its own set of log streams named "{workspace-id}/{container}". The main workspace container is "code-server" — for most questions about a running workspace/sandbox, fetch the code-server stream first. Other containers: port-detector (port detection sidecar), build-base-image / build-wrapper-image (init image builders), init-workspace (init: workspace setup), docker-auth (init: registry auth).`
    );
    try {
      const streams = await listCloudwatchLogStreams(poolLogGroup);
      if (streams.length > 0) {
        context.push(`\nAvailable log streams in this log group:`);
        for (const s of streams) {
          context.push(`  - "${s.name}"${s.lastEventTime ? " (last event: " + s.lastEventTime + ")" : ""}`);
        }
        context.push(
          `\nUse the log_stream_names parameter to filter to a specific workspace/container (e.g. the "{workspace-id}/code-server" stream).`
        );
      }
    } catch (e) {
      console.warn(`Failed to list log streams for ${poolLogGroup}:`, e);
    }
    finalPrompt = context.join("\n") + `\n\n${finalPrompt}`;
  }

  return finalPrompt;
}
