import { Request, Response } from "express";
import { log_agent_context } from "./agent.js";
import * as asyncReqs from "../graph/reqs.js";
import { startTracking, endTracking } from "../busy.js";
import { ModelName } from "../aieo/src/index.js";
import { listCloudwatchLogStreams } from "./cloudwatch.js";

/** Convert a UI swarm name like "swarm38" to a CloudWatch log group like "/swarms/38" */
function swarmNameToLogGroup(swarmName: string): string | null {
  const match = swarmName.match(/^swarm(\d+)$/);
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

  const opId = startTracking("logs_agent");

  try {
    log_agent_context(finalPrompt, { modelName, logs })
      .then((result) => {
        asyncReqs.finishReq(request_id, {
          success: true,
          final_answer: result.final,
          tool_use: result.tool_use,
          content: result.content,
          usage: result.usage,
          logs: result.logs,
        });
      })
      .catch((error) => {
        console.error("[logs_agent] Background work failed:", error);
        asyncReqs.failReq(request_id, error.message || error.toString());
      })
      .finally(() => {
        endTracking(opId);
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
  "prompt": "Why is stakgraph erroring?",
  "swarmName": "swarm38",
  "model": "sonnet"
}' "http://localhost:3355/logs/agent"

curl "http://localhost:3355/progress?request_id=<request_id>"
*/
