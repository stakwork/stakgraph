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
import { buildLogsPrompt } from "./context.js";
import { createRunLogsDir, cleanupRunLogsDir } from "./utils.js";
import { StakworkRunSummary } from "./types.js";

export type { AgentLogSummary, StakworkRunSummary } from "./types.js";

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
  const _metadataRaw = req.body._metadata;
  let _metadata: unknown = undefined;
  if (_metadataRaw !== undefined) {
    const serialized = JSON.stringify(_metadataRaw);
    const METADATA_SIZE_CAP = 64 * 1024; // 64KB
    if (serialized.length > METADATA_SIZE_CAP) {
      res.status(400).json({ error: "_metadata exceeds maximum allowed size of 64KB" });
      return;
    }
    _metadata = _metadataRaw;
  }

  if (!prompt) {
    res.status(400).json({ error: "Missing prompt" });
    return;
  }

  const finalPrompt = await buildLogsPrompt(prompt, { swarmName, stakworkRuns, workspaceSlug, logGroups, poolName });

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
    log_agent_context(finalPrompt, { modelName, apiKey, baseUrl, logs, sessionId, sessionConfig, stakworkApiKey, stakworkRuns, logsDir, printAgentProgress, source: "logs_agent", headers, abortSignal: abortController.signal, _metadata })
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
          unregisterAbortController(sessionId, abortController);
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
      unregisterAbortController(sessionId, abortController);
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
