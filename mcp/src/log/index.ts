import { Request, Response } from "express";
import { log_agent_context } from "./agent.js";
import * as asyncReqs from "../graph/reqs.js";
import { startTracking, endTracking } from "../busy.js";
import {
  registerAbortController,
  unregisterAbortController,
} from "../repo/events.js";
import { ModelName } from "../aieo/src/index.js";
import { randomUUID } from "crypto";
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

/**
 * Coerce a request-body `headers` value into a clean Record<string, string>.
 * Accepts a plain object whose values are strings/numbers/booleans; drops
 * non-string values and ignores anything else. Returns undefined when empty.
 */
function normalizeHeaders(input: unknown): Record<string, string> | undefined {
  if (!input || typeof input !== "object" || Array.isArray(input)) return undefined;
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(input as Record<string, unknown>)) {
    if (typeof k !== "string" || !k) continue;
    if (typeof v === "string") out[k] = v;
    else if (typeof v === "number" || typeof v === "boolean") out[k] = String(v);
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

export async function logs_agent(req: Request, res: Response) {
  console.log("===> logs_agent", req.method, req.path, {
    hasPrompt: Boolean(req.body?.prompt),
    hasStakworkApiKey: Boolean(req.body?.stakworkApiKey),
    hasSessionId: Boolean(req.body?.sessionId),
  });
  const request_id = asyncReqs.startReq();

  const prompt = req.body.prompt as string;
  const modelName = req.body.model as ModelName | undefined;
  const apiKey = req.body.apiKey as string | undefined;
  const baseUrl = req.body.baseUrl as string | undefined;
  const logs = req.body.logs as boolean | undefined;
  const swarmName = req.body.swarmName as string | undefined;
  const sessionId = (req.body.sessionId as string | undefined) || randomUUID();
  const sessionConfig = req.body.sessionConfig as SessionConfig | undefined;
  const stakworkApiKey = req.body.stakworkApiKey as string | undefined;
  const stakworkRuns = req.body.stakworkRuns as StakworkRunSummary[] | undefined;
  const printAgentProgress = req.body.printAgentProgress as boolean | undefined;
  const workspaceSlug = req.body.workspaceSlug as string | undefined;
  const logGroups = req.body.logGroups as string[] | undefined;
  const poolName = req.body.poolName as string | undefined;
  const headers = normalizeHeaders(req.body.headers);

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
    finalPrompt = `If the user mentions their actual application name (e.g. "${workspaceSlug}"), use the fetch_quickwit tool to fetch logs from the actual application.\n\n${finalPrompt}`;
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
      `If the user asks about "pod", "pool", or "sandbox" logs, they mean the workspace pod logs in the CloudWatch log group "${poolLogGroup}". Use the fetch_cloudwatch tool with this log group, then use bash to search the saved file.`
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

  // Per-run logs directory: use sessionId if present (persists across turns),
  // otherwise generate a random one (cleaned up after the agent finishes).
  const logsDir = createRunLogsDir(sessionId);

  // Register abort controller keyed by request_id (and mirrored under sessionId)
  // so the 30-min safety timeout in busy.ts can actually cancel the run.
  const abortController = registerAbortController(request_id);
  if (sessionId && sessionId !== request_id) {
    registerAbortController(sessionId, abortController);
  }
  const opId = startTracking("logs_agent", abortController);

  try {
    log_agent_context(finalPrompt, { modelName, apiKey, baseUrl, logs, sessionId, sessionConfig, stakworkApiKey, stakworkRuns, logsDir, printAgentProgress, source: "logs_agent", headers, abortSignal: abortController.signal })
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
        if (abortController.signal.aborted) {
          console.log(`[logs_agent] Run aborted: ${request_id}`);
          asyncReqs.failReq(request_id, "aborted");
        } else {
          console.error("[logs_agent] Background work failed:", error);
          asyncReqs.failReq(request_id, error.message || error.toString());
        }
      })
      .finally(() => {
        unregisterAbortController(request_id);
        if (sessionId && sessionId !== request_id) {
          unregisterAbortController(sessionId);
        }
        endTracking(opId);
        // Clean up logs directory for non-session runs
        if (!sessionId) {
          cleanupRunLogsDir(logsDir);
        }
      });

    res.json({ request_id, status: "pending", sessionId });
  } catch (error: any) {
    console.error("Error in logs_agent", error);
    asyncReqs.failReq(request_id, error);
    unregisterAbortController(request_id);
    if (sessionId && sessionId !== request_id) {
      unregisterAbortController(sessionId);
    }
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

curl -X POST -H "Content-Type: application/json" -d '{
  "prompt": "Why did my sandbox fail to start? Check the pod logs.",
  "poolName": "cmdx1snwz0001l204jdb1ah1n",
  "model": "haiku",
  "printAgentProgress": true
}' "http://localhost:3355/logs/agent"

curl "http://localhost:3355/progress?request_id=d3ffa3f7-6fd3-4565-b6bd-6c5c1a456c5b"
*/
