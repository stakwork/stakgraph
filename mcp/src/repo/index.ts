import { cloneOrUpdateRepo } from "./clone.js";
import { get_context, stream_context, CLI_SYSTEM } from "./agent.js";
import { ToolsConfig, SkillsConfig, GgnnConfig, getDefaultToolDescriptions, normalizeToolsConfig } from "./tools.js";
import { type SubAgent, normalizeSubAgent } from "./subagent.js";
import { Request, Response } from "express";
import { gitleaksDetect, gitleaksProtect } from "./gitleaks.js";
import * as asyncReqs from "../graph/reqs.js";
import { startTracking, endTracking } from "../busy.js";
import { services_agent } from "./services.js";
import { mocks_agent } from "./mocks.js";
import { ModelName } from "../aieo/src/index.js";
import { SessionConfig, loadSession, sessionExists } from "./session.js";
import { McpServer } from "./mcpServers.js";
import { existsSync } from "fs";
import path from "path";
import { createBus, filterStepContent, signEventsToken } from "./events.js";
import { db } from "../graph/neo4j.js";

import { describe_nodes_agent, embed_nodes_agent } from "./descriptions.js";
export { services_agent, mocks_agent, describe_nodes_agent, embed_nodes_agent };

function normalizeRepoRef(input?: string): string {
  if (!input) return "";
  const trimmed = input.trim().replace(/\.git$/, "");
  if (!trimmed) return "";
  if (/^https?:\/\//.test(trimmed)) {
    const parts = trimmed.split("/").filter(Boolean);
    const repo = parts.pop() || "";
    const owner = parts.pop() || "";
    return owner && repo ? `${owner}/${repo}` : repo;
  }
  return trimmed;
}

async function getGraphRepoList(): Promise<string[]> {
  try {
    const repos = await db.get_repositories();
    return [...new Set(
      repos
        .map((repo) => {
          const sourceLink = String(repo.properties?.source_link || "");
          const name = String(repo.properties?.name || "");
          return normalizeRepoRef(sourceLink || name);
        })
        .filter((repo) => repo.length > 0)
    )];
  } catch (error) {
    console.error("[repo_agent] Failed to load graph repositories:", error);
    return [];
  }
}

function resolveRepoDir(repoList: string[]): string {
  if (repoList.length === 1) {
    const [owner, repo] = repoList[0].split("/");
    if (owner && repo) {
      const repoDir = path.join("/tmp", owner, repo);
      if (existsSync(repoDir)) return repoDir;
    }
  }
  return "/tmp";
}

function prependRepoInfo(prompt: any, clonedRepos: string[], graphRepos: string[]): any {
  const uniqueGraphRepos = [...new Set(graphRepos.filter(Boolean))];
  const uniqueClonedRepos = [...new Set(clonedRepos.filter(Boolean))];
  const clonedOnlyRepos = uniqueClonedRepos.filter((repo) => !uniqueGraphRepos.includes(repo));

  const lines: string[] = ["Repository availability for this session:"];

  if (uniqueGraphRepos.length > 0) {
    lines.push(
      "Graph-backed repos (prefer repo_overview, stakgraph_search, stakgraph_map, and stakgraph_code for these):",
      ...uniqueGraphRepos.map((repo) => `- ${repo}`)
    );
  }

  if (clonedOnlyRepos.length > 0) {
    lines.push(
      "Additional repos cloned locally under /tmp/{owner}/{repo} but not ingested in the graph (use bash/fulltext_search for these):",
      ...clonedOnlyRepos.map((repo) => `- ${repo}`)
    );
  }

  if (uniqueClonedRepos.length === 0 && uniqueGraphRepos.length > 0) {
    lines.push("No repo_url was supplied. Use the ingested repos above; they are also available locally under /tmp/{owner}/{repo}.");
  }

  if (uniqueGraphRepos.length === 0 && uniqueClonedRepos.length > 0) {
    lines.push("No ingested Repository nodes were found for the requested repos, so rely on bash/fulltext_search over the cloned repos.");
  }

  const repoInfo = `${lines.join("\n")}\n\n`;
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
/** Parse shared request body params for repo_agent. */
function parseAgentBody(req: Request) {
  const repoUrl = req.body.repo_url as string | undefined;
  const username = req.body.username as string | undefined;
  const pat = req.body.pat as string | undefined;
  const commit = req.body.commit as string | undefined;
  const prompt = req.body.prompt as any;
  const toolsConfig = normalizeToolsConfig(req.body.toolsConfig);
  const schema = req.body.jsonSchema as { [key: string]: any } | undefined;
  const modelName = req.body.model as ModelName | undefined;
  const apiKey = req.body.apiKey as string | undefined;
  const logs = req.body.logs as boolean | undefined;
  const sessionId = req.body.sessionId as string | undefined;
  const sessionConfig = req.body.sessionConfig as SessionConfig | undefined;
  const mcpServers = req.body.mcpServers as McpServer[] | undefined;
  const systemOverride = req.body.systemOverride as string | undefined;
  const cliMode = Boolean(req.body.cliMode ?? req.body.climode);
  const skills = req.body.skills as SkillsConfig | undefined;
  const subAgents = (req.body.subAgents as Record<string, unknown>[] | undefined)
    ?.map(normalizeSubAgent) as SubAgent[] | undefined;
  const ggnn = req.body.ggnn as GgnnConfig | undefined;
  const stream = req.body.stream as boolean | undefined;

  const repoList = (repoUrl || "")
    .split(",")
    .map((url: string) => normalizeRepoRef(url))
    .filter((url: string) => url.length > 0);

  return {
    repoUrl, username, pat, commit, prompt, toolsConfig, schema,
    modelName, apiKey, logs, sessionId, sessionConfig, mcpServers,
    systemOverride, cliMode, skills, subAgents, ggnn, stream, repoList,
  };
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
    stream: Boolean(req.body?.stream),
    cliMode: Boolean(req.body?.cliMode ?? req.body?.climode),
    hasApiKey: Boolean(req.body?.apiKey),
    apiKeyPrefix: req.body?.apiKey ? String(req.body.apiKey).slice(0, 12) + "..." : "(none)",
    modelName: req.body?.model || "(none)",
  });

  const body = parseAgentBody(req);

  if (!body.prompt) {
    res.status(400).json({ error: "Missing prompt" });
    return;
  }

  const graphRepos = await getGraphRepoList();
  const effectiveRepos = body.repoList.length > 0 ? body.repoList : graphRepos;
  // Only prepend repo info on the first message of a session (or when there's no session)
  const isExistingSession = body.sessionId && sessionExists(body.sessionId);
  const promptWithRepoInfo = isExistingSession
    ? body.prompt
    : prependRepoInfo(body.prompt, body.repoList, graphRepos);
  const repoDirPromise = body.repoUrl
    ? cloneOrUpdateRepo(body.repoUrl, body.username, body.pat, body.commit)
    : Promise.resolve(resolveRepoDir(effectiveRepos));

  // ── Streaming path: direct SSE response ──────────────────────────────
  if (body.stream) {
    const opId = startTracking("repo_agent_stream");
    try {
      const repoDir = await repoDirPromise;
      console.log(`===> POST /repo/agent (stream) ${repoDir}`);

      const streamResult = await stream_context(
        promptWithRepoInfo,
        repoDir,
        {
          pat: body.pat,
          toolsConfig: body.toolsConfig,
          schema: body.schema,
          modelName: body.modelName,
          apiKey: body.apiKey,
          logs: body.logs,
          sessionId: body.sessionId,
          sessionConfig: body.sessionConfig,
          mcpServers: body.mcpServers,
          repos: effectiveRepos.length > 1 ? effectiveRepos : undefined,
          systemOverride: body.systemOverride ?? (body.cliMode ? CLI_SYSTEM : undefined),
          cliMode: body.cliMode,
          skills: body.skills,
          subAgents: body.subAgents,
          ggnn: body.ggnn,
        },
      );

      const streamResponse = streamResult.toUIMessageStreamResponse();

      // Forward status + headers
      res.status(streamResponse.status);
      streamResponse.headers.forEach((value: string, key: string) => {
        res.setHeader(key, value);
      });

      // Pipe the body
      const reader = streamResponse.body?.getReader();
      if (!reader) {
        res.status(500).json({ error: "No stream body" });
        endTracking(opId);
        return;
      }

      const pump = async () => {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          res.write(value);
        }
        res.end();
      };

      pump()
        .catch((err) => {
          console.error("[repo_agent] Stream error:", err);
          if (!res.headersSent) {
            res.status(500).json({ error: "Stream error" });
          } else {
            res.end();
          }
        })
        .finally(() => endTracking(opId));

      return;
    } catch (error: any) {
      console.error("[repo_agent] Stream setup error:", error);
      endTracking(opId);
      if (!res.headersSent) {
        res.status(500).json({ error: error.message || "Internal server error" });
      }
      return;
    }
  }

  // ── Non-streaming path: async job with event bus ─────────────────────
  const request_id = asyncReqs.startReq();
  const opId = startTracking("repo_agent");

  // Create an event bus for real-time SSE streaming of this request
  const bus = createBus(request_id);

  // Generate a short-lived JWT scoped to this request_id (only if API_TOKEN is set)
  let events_token: string | undefined;
  try {
    events_token = signEventsToken(request_id);
  } catch (_) {
    // API_TOKEN not set — SSE auth via JWT won't be available
  }

  try {
    repoDirPromise
      .then((repoDir) => {
        console.log(`===> POST /repo/agent ${repoDir}`);
        return get_context(promptWithRepoInfo, repoDir, {
          pat: body.pat,
          toolsConfig: body.toolsConfig,
          schema: body.schema,
          modelName: body.modelName,
          apiKey: body.apiKey,
          logs: body.logs,
          sessionId: body.sessionId,
          sessionConfig: body.sessionConfig,
          mcpServers: body.mcpServers,
          repos: effectiveRepos.length > 1 ? effectiveRepos : undefined,
          systemOverride: body.systemOverride ?? (body.cliMode ? CLI_SYSTEM : undefined),
          cliMode: body.cliMode,
          skills: body.skills,
          subAgents: body.subAgents,
          ggnn: body.ggnn,
          onStepEvent: (content) => {
            const events = filterStepContent(content);
            for (const ev of events) bus.emit(ev);
          },
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
        bus.emit({
          type: "done",
          result: { final_answer: result.final, usage: result.usage },
          timestamp: new Date().toISOString(),
        });
      })
      .catch((error) => {
        console.error("[repo_agent] Background work failed with error:", error);
        asyncReqs.failReq(request_id, error.message || error.toString());
        bus.emit({
          type: "error",
          error: error.message || error.toString(),
          timestamp: new Date().toISOString(),
        });
      })
      .finally(() => {
        endTracking(opId);
      });
    res.json({ request_id, status: "pending", ...(events_token && { events_token }) });
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
  const sessionId = req.query.session_id as string || req.query.sessionId as string;
  console.log("===> GET /repo/agent/session", { hasSessionId: Boolean(sessionId) });

  if (!sessionId) {
    res.status(400).json({ error: "Missing session_id" });
    return;
  }

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

  const exists = sessionExists(sessionId);
  res.json({ exists, valid: exists });
}

export async function get_leaks(req: Request, res: Response) {
  const repoUrl = req.query.repo_url as string;
  const username = req.query.username as string | undefined;
  const pat = req.query.pat as string | undefined;
  const commit = req.query.commit as string | undefined;
  const ignore = req.query.ignore as string | undefined;

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

export async function get_agent_file(req: Request, res: Response) {
  const filePath = req.query.path as string;
  console.log("===> GET /repo/agent/file", { filePath });

  if (!filePath) {
    res.status(400).json({ error: "Missing path" });
    return;
  }

  if (!existsSync(filePath)) {
    res.status(404).json({ error: "File not found" });
    return;
  }

  res.sendFile(path.resolve(filePath));
}
