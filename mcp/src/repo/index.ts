import { cloneOrUpdateRepo } from "./clone.js";
import { get_context } from "./agent.js";
import { ToolsConfig, getDefaultToolDescriptions } from "./tools.js";
import { Request, Response } from "express";
import { gitleaksDetect, gitleaksProtect } from "./gitleaks.js";
import * as asyncReqs from "../graph/reqs.js";
import { startTracking, endTracking } from "../busy.js";
import { services_agent } from "./services.js";
import { mocks_agent } from "./mocks.js";
import { ModelName } from "../aieo/src/index.js";
import { SessionConfig, loadSession, sessionExists } from "./session.js";
import { McpServer } from "./mcpServers.js";
import {
  repoAgentBodySchema,
  getAgentSessionQuerySchema,
  getLeaksQuerySchema,
} from "./schemas/routes.js";
import { sendValidationError } from "../validation.js";

import { describe_nodes_agent } from "./descriptions.js";
export { services_agent, mocks_agent, describe_nodes_agent };

function prependRepoInfo(prompt: any, repoList: string[]): any {
  if (repoList.length <= 1) return prompt;
  const repoInfo = `You are exploring the following repositories: ${repoList.join(", ")}. Files are located under /tmp/{owner}/{repo}.\n\n`;
  if (typeof prompt === "string") {
    return repoInfo + prompt;
  }
  if (Array.isArray(prompt)) {
    return prompt.map((msg: any, i: number) => {
      if (i === 0 && msg.role === "user") {
        return { ...msg, content: repoInfo + msg.content };
      }
      return msg;
    });
  }
  return prompt;
}

// modelName can be a shortcut like "kimi" or a full model name like "anthropic/claude-sonnet-4-5" or "openrouter/moonshotai/kimi-k2.5"
export async function repo_agent(req: Request, res: Response) {
  // curl -X POST -H "Content-Type: application/json" -d '{"repo_url": "https://github.com/stakwork/hive", "prompt": "how does auth work in the repo"}' "http://localhost:3355/repo/agent"
  // curl -X POST -H "Content-Type: application/json" -d '{"repo_url": "https://github.com/stakwork/hive,https://github.com/stakwork/stakgraph", "prompt": "how do these two repos relate to each other?"}' "http://localhost:3355/repo/agent"
  // curl "http://localhost:3355/progress?request_id=5c501254-cc7b-44f4-9537-6fa18f642b5c"
  console.log("===> repo_agent", req.method, req.path, {
    hasPat: Boolean(req.body?.pat),
    hasUsername: Boolean(req.body?.username),
    hasRepoUrl: Boolean(req.body?.repo_url),
    hasPrompt: Boolean(req.body?.prompt),
  });

  const parsed = repoAgentBodySchema.safeParse(req.body);
  if (!parsed.success) {
    sendValidationError(res, "body", parsed.error);
    return;
  }

  const request_id = asyncReqs.startReq();

  const repoUrl = parsed.data.repo_url;
  const username = parsed.data.username;
  const pat = parsed.data.pat;
  const commit = parsed.data.commit;
  const branch = parsed.data.branch;
  const prompt = parsed.data.prompt;
  const toolsConfig = parsed.data.toolsConfig as ToolsConfig | undefined;
  const schema = parsed.data.jsonSchema as { [key: string]: any } | undefined;
  const modelName = parsed.data.model as ModelName | undefined;
  const logs = parsed.data.logs;
  const sessionId = parsed.data.sessionId;
  const sessionConfig = parsed.data.sessionConfig as SessionConfig | undefined;
  const mcpServers = parsed.data.mcpServers as McpServer[] | undefined;


  const opId = startTracking("repo_agent");

  try {
    // Extract owner/repo from each URL for multi-repo filtering
    const repoList = repoUrl
      .split(",")
      .map((url) => url.trim())
      .filter((url) => url.length > 0)
      .map((url) => {
        const parts = url.replace(/\.git$/, "").split("/");
        const repoName = parts.pop() || "";
        const owner = parts.pop() || "";
        return `${owner}/${repoName}`;
      });

    cloneOrUpdateRepo(repoUrl, username, pat, commit)
      .then((repoDir) => {
        console.log(`===> POST /repo/agent ${repoDir}`);
        return get_context(prependRepoInfo(prompt, repoList), repoDir, {
          pat,
          toolsConfig,
          schema,
          modelName,
          logs,
          sessionId,
          sessionConfig,
          mcpServers,
          repos: repoList.length > 1 ? repoList : undefined,
        });
      })
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
        console.error("[repo_agent] Background work failed with error:", error);
        asyncReqs.failReq(request_id, error.message || error.toString());
      })
      .finally(() => {
        endTracking(opId);
      });
    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error");
    asyncReqs.failReq(request_id, error);
    console.error("Error in repo_agent", error);
    res.status(500).json({ error: "Internal server error" });
    endTracking(opId);
  }
}

export async function get_agent_tools(req: Request, res: Response) {
  console.log("===> GET /repo/agent/tools");
  try {
    const toolsConfig = getDefaultToolDescriptions();
    res.json({ success: true, toolsConfig });
  } catch (e) {
    console.error("Error in get_agent_tools", e);
    res.status(500).json({ error: "Internal server error" });
  }
}

export async function get_agent_session(req: Request, res: Response) {
  const parsed = getAgentSessionQuerySchema.safeParse(req.query);
  if (!parsed.success) {
    sendValidationError(res, "query", parsed.error);
    return;
  }

  const sessionId = parsed.data.session_id || parsed.data.sessionId || "";
  console.log("===> GET /repo/agent/session", { hasSessionId: Boolean(sessionId) });

  if (!sessionExists(sessionId)) {
    res.status(404).json({ error: "Session not found" });
    return;
  }

  try {
    const messages = loadSession(sessionId);
    res.json({ sessionId, messages });
  } catch (e) {
    console.error("Error in get_agent_session", e);
    res.status(500).json({ error: "Internal server error" });
  }
}

export async function validate_agent_session(req: Request, res: Response) {
  const sessionId = req.query.session_id as string || req.query.sessionId as string;
  console.log("===> GET /repo/agent/validate_session", { hasSessionId: Boolean(sessionId) });

  if (!sessionId) {
    res.status(400).json({ error: "Missing session_id" });
    return;
  }

  if (!sessionExists(sessionId)) {
    res.status(404).json({ error: "Session not found" });
    return;
  }

  res.json({ exists: true, valid: true });
}

export async function get_leaks(req: Request, res: Response) {
  const parsed = getLeaksQuerySchema.safeParse(req.query);
  if (!parsed.success) {
    sendValidationError(res, "query", parsed.error);
    return;
  }

  const repoUrl = parsed.data.repo_url;
  const username = parsed.data.username;
  const pat = parsed.data.pat;
  const commit = parsed.data.commit;
  const ignore = parsed.data.ignore;

  const repoDir = await cloneOrUpdateRepo(repoUrl, username, pat, commit);

  console.log(`===> GET /leaks ${repoDir}`);
  try {
    const ignoreList = ignore?.split(",").map((dir) => dir.trim()) || [];
    const detect = gitleaksDetect(repoDir, ignoreList);
    const protect = gitleaksProtect(repoDir, ignoreList);
    res.json({ success: true, detect, protect });
  } catch (e) {
    console.error("Error running gitleaks:", e);
    res.status(500).json({ error: "Error running gitleaks" });
  }
}
