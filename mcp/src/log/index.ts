import { Request, Response } from "express";
import { log_agent_context } from "./agent.js";
import * as asyncReqs from "../graph/reqs.js";
import { startTracking, endTracking } from "../busy.js";
import { ModelName } from "../aieo/src/index.js";
import { SessionConfig } from "../repo/session.js";
import { listCloudwatchLogStreams } from "./cloudwatch.js";
import { createRunLogsDir, cleanupRunLogsDir } from "./utils.js";
import { StakworkRunSummary } from "./types.js";

export type { AgentLogSummary, StakworkRunSummary } from "./types.js";

/** Convert a UI swarm name like "swarm38" or "swarmHDYF7D" to a CloudWatch log group like "/swarms/38" or "/swarms/HDYF7D" */
function swarmNameToLogGroup(swarmName: string): string | null {
  const match = swarmName.match(/^swarm(.+)$/);
  if (!match) return null;
  return `/swarms/${match[1]}`;
}

export async function logs_agent(req: Request, res: Response) {
  console.log("===> logs_agent", req.body);
  const request_id = asyncReqs.startReq();

  const prompt = req.body.prompt as string;
  const modelName = req.body.model as ModelName | undefined;
  const logs = req.body.logs as boolean | undefined;
  const swarmName = req.body.swarmName as string | undefined;
  const sessionId = req.body.sessionId as string | undefined;
  const sessionConfig = req.body.sessionConfig as SessionConfig | undefined;
  const stakworkApiKey = req.body.stakworkApiKey as string | undefined;
  const stakworkRuns = req.body.stakworkRuns as StakworkRunSummary[] | undefined;
  const printAgentProgress = req.body.printAgentProgress as boolean | undefined;
  const workspaceSlug = req.body.workspaceSlug as string | undefined;

  if (!prompt) {
    res.status(400).json({ error: "Missing prompt" });
    return;
  }

  let finalPrompt = prompt;
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
    ].join("\n");
    finalPrompt = runsContext + `\n\n${finalPrompt}`;
  }

  if (workspaceSlug) {
    finalPrompt = `If the user mentions their actual application name (e.g. "${workspaceSlug}"), use the fetch_quickwit tool to fetch logs from the actual application.`;
  }

  // Per-run logs directory: use sessionId if present (persists across turns),
  // otherwise generate a random one (cleaned up after the agent finishes).
  const logsDir = createRunLogsDir(sessionId);

  const opId = startTracking("logs_agent");

  try {
    log_agent_context(finalPrompt, { modelName, logs, sessionId, sessionConfig, stakworkApiKey, stakworkRuns, logsDir, printAgentProgress })
      .then((result) => {
        asyncReqs.finishReq(request_id, {
          success: true,
          final_answer: result.final,
          tool_use: result.tool_use,
          content: result.content,
          usage: result.usage,
          logs: result.logs,
          sessionId: result.sessionId,
        });
      })
      .catch((error) => {
        console.error("[logs_agent] Background work failed:", error);
        asyncReqs.failReq(request_id, error.message || error.toString());
      })
      .finally(() => {
        endTracking(opId);
        // Clean up logs directory for non-session runs
        if (!sessionId) {
          cleanupRunLogsDir(logsDir);
        }
      });

    res.json({ request_id, status: "pending" });
  } catch (error: any) {
    console.error("Error in logs_agent", error);
    asyncReqs.failReq(request_id, error);
    res.status(500).json({ error: "Internal server error" });
    endTracking(opId);
  }
}

/*
curl -X POST -H "Content-Type: application/json" -d '{
  "prompt": "How long did latest stakgraph ingest take? Only read stakgraph logs.",
  "swarmName": "swarm38",
  "model": "haiku",
  "printAgentProgress": true
}' "http://localhost:3355/logs/agent"

curl "http://localhost:3355/progress?request_id=d3ffa3f7-6fd3-4565-b6bd-6c5c1a456c5b"
*/
