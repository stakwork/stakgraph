import { cloneOrUpdateRepo } from "./clone.js";
import { get_context, stream_context } from "./agent.js";
import { ToolsConfig, SkillsConfig, GgnnConfig, getDefaultToolDescriptions, normalizeToolsConfig, editorRoots } from "./tools.js";
import { resolveInCwd } from "./textEdit.js";
import { type SubAgent, normalizeSubAgent } from "./subagent.js";
import { Request, Response } from "express";
import { ModelMessage } from "ai";
import { randomUUID } from "crypto";
import { gitleaksDetect, gitleaksProtect } from "./gitleaks.js";
import * as asyncReqs from "../graph/reqs.js";
import { startTracking, endTracking } from "../busy.js";
import { services_agent } from "./services.js";
import { mocks_agent } from "./mocks.js";
import { ModelName } from "../aieo/src/index.js";
import { SessionConfig, loadSession, loadSessionConfig, loadSessionMetadata, sessionExists } from "./session.js";
import { McpServer } from "./mcpServers.js";
import { existsSync } from "fs";
import path from "path";
import {
  createBus,
  filterStepContent,
  signEventsToken,
  registerAbortController,
  unregisterAbortController,
  abortRequest,
} from "./events.js";
import { db } from "../graph/neo4j.js";

import { describe_nodes_agent, embed_nodes_agent } from "./descriptions.js";
export { services_agent, mocks_agent, describe_nodes_agent, embed_nodes_agent };

export function parseCommitList(commit?: string): string[] {
  return (commit || "").split(",").map(c => c.trim()).filter(c => c.length > 0);
}

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

// modelName can be a shortcut like "kimi" or a full model name like "anthropic/claude-sonnet-4-5" or "openrouter/moonshotai/kimi-k2.6"
/** Parse shared request body params for repo_agent. */
function parseAgentBody(req: Request) {
  const repoUrl = req.body.repo_url as string | undefined;
  const username = req.body.username as string | undefined;
  const pat = req.body.pat as string | undefined;
  const commitList = parseCommitList(req.body.commit as string | undefined);
  const prompt = req.body.prompt as any;
  // Transparent replay: a full ai-sdk transcript (including the system turn) to
  // run verbatim. When present, the agent skips all prompt/instruction
  // enrichment, session history, attachments, and persistence.
  const messages = Array.isArray(req.body.messages)
    ? (req.body.messages as ModelMessage[])
    : undefined;
  const toolsConfig = normalizeToolsConfig(req.body.toolsConfig);
  const schema = req.body.jsonSchema as { [key: string]: any } | undefined;
  const modelName = req.body.model as ModelName | undefined;
  const apiKey = req.body.apiKey as string | undefined;
  const baseUrl = req.body.baseUrl as string | undefined;
  const logs = req.body.logs as boolean | undefined;
  const sessionId = (req.body.sessionId as string | undefined) || randomUUID();
  const sessionConfig = req.body.sessionConfig as SessionConfig | undefined;
  const mcpServers = req.body.mcpServers as McpServer[] | undefined;
  const systemOverride = req.body.systemOverride as string | undefined;
  const mode = req.body.mode === "graph" ? "graph" as const : undefined;
  const skills = req.body.skills as SkillsConfig | undefined;
  const subAgents = (req.body.subAgents as Record<string, unknown>[] | undefined)
    ?.map(normalizeSubAgent) as SubAgent[] | undefined;
  const ggnn = req.body.ggnn as GgnnConfig | undefined;
  const stream = req.body.stream as boolean | undefined;
  const maxTurns = typeof req.body.maxTurns === "number" ? req.body.maxTurns : undefined;
  const headers = normalizeHeaders(req.body.headers);
  const ignoreRepoInfo = req.body.ignoreRepoInfo as boolean | undefined;
  const attachments = Array.isArray(req.body.attachments)
    ? (req.body.attachments as unknown[]).filter(
        (a): a is string => typeof a === "string" && a.trim().length > 0,
      )
    : undefined;
  const _metadata = req.body._metadata as unknown;

  const repoList = (repoUrl || "")
    .split(",")
    .map((url: string) => normalizeRepoRef(url))
    .filter((url: string) => url.length > 0);

  return {
    repoUrl, username, pat, commitList, prompt, messages, toolsConfig, schema,
    modelName, apiKey, baseUrl, logs, sessionId, sessionConfig, mcpServers,
    systemOverride, mode, skills, subAgents, ggnn, stream, repoList, maxTurns, headers,
    ignoreRepoInfo, attachments, _metadata,
  };
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

// modelName can be a shortcut like "kimi" or a full model name like "anthropic/claude-sonnet-4-5" or "openrouter/moonshotai/kimi-k2.6"
export async function repo_agent(req: Request, res: Response) {
  // curl -X POST -H "Content-Type: application/json" -d '{"repo_url": "https://github.com/stakwork/hive", "prompt": "how does auth work in the repo"}' "http://localhost:3355/repo/agent"
  // curl -X POST -H "Content-Type: application/json" -d '{"repo_url": "https://github.com/stakwork/hive,https://github.com/stakwork/stakgraph", "prompt": "how do these two repos relate to each other?"}' "http://localhost:3355/repo/agent"
  // curl "http://localhost:3355/progress?request_id=5c501254-cc7b-44f4-9537-6fa18f642b5c"
  const subAgentNames = Array.isArray(req.body?.subAgents)
    ? req.body.subAgents.map((a: any) => a?.name).filter(Boolean)
    : [];
  const repoListPreview = (req.body?.repo_url || "")
    .split(",")
    .map((url: string) => normalizeRepoRef(url))
    .filter((url: string) => url.length > 0);
  const toolsConfigKeys =
    req.body?.toolsConfig && typeof req.body.toolsConfig === "object"
      ? Object.keys(req.body.toolsConfig)
      : [];
  const mcpServersCount = Array.isArray(req.body?.mcpServers) ? req.body.mcpServers.length : 0;
  console.log("===> repo_agent", req.method, req.path, {
    sessionId: req.body?.sessionId || "(new)",
    hasPat: Boolean(req.body?.pat),
    hasUsername: Boolean(req.body?.username),
    hasRepoUrl: Boolean(req.body?.repo_url),
    repoList: repoListPreview.length > 0 ? repoListPreview : "(none)",
    hasPrompt: Boolean(req.body?.prompt),
    stream: Boolean(req.body?.stream),
    hasApiKey: Boolean(req.body?.apiKey),
    apiKeyPrefix: req.body?.apiKey ? String(req.body.apiKey).slice(0, 12) + "..." : "(none)",
    baseUrl: req.body?.baseUrl || "(none)",
    modelName: req.body?.model || "(none)",
    subAgents: subAgentNames.length > 0 ? subAgentNames : "(none)",
    toolsConfig: toolsConfigKeys.length > 0 ? toolsConfigKeys : "(default)",
    hasSchema: Boolean(req.body?.jsonSchema),
    hasSystemOverride: Boolean(req.body?.systemOverride),
    mcpServers: mcpServersCount,
  });

  const body = parseAgentBody(req);

  // Transparent replay: `messages` is a full transcript run verbatim. It takes
  // the place of `prompt` and disables all enrichment/history/persistence.
  const transparent = Boolean(body.messages && body.messages.length > 0);

  if (!body.prompt && !transparent) {
    res.status(400).json({ error: "Missing prompt" });
    return;
  }

  const graphRepos = await getGraphRepoList();
  const effectiveRepos = body.repoList.length > 0 ? body.repoList : graphRepos;
  // Only prepend repo info on the first message of a session (or when there's no session).
  // In transparent mode the replayed messages are sent verbatim — no repo info.
  const isExistingSession = body.sessionId && sessionExists(body.sessionId);
  const promptInput: string | ModelMessage[] = transparent
    ? body.messages!
    : (isExistingSession || body.ignoreRepoInfo || body.mode === "graph")
      ? body.prompt
      : prependRepoInfo(body.prompt, body.repoList, graphRepos);
  // ── Streaming path: direct SSE response ──────────────────────────────
  if (body.stream) {
    // Register abort controller keyed by sessionId so a separate request can cancel it
    const abortController = registerAbortController(body.sessionId);
    const opId = startTracking("repo_agent_stream", abortController);
    const repoDirPromise = body.repoUrl
      ? cloneOrUpdateRepo(body.repoUrl, body.username, body.pat, body.commitList, abortController.signal)
      : Promise.resolve(resolveRepoDir(effectiveRepos));
    try {
      const repoDir = await repoDirPromise;
      console.log(`===> POST /repo/agent (stream) ${repoDir}`);

      const { streamResult, finalizeSession } = await stream_context(
        promptInput,
        repoDir,
        {
          transparent,
          pat: body.pat,
          toolsConfig: body.toolsConfig,
          schema: body.schema,
          modelName: body.modelName,
          apiKey: body.apiKey,
          baseUrl: body.baseUrl,
          logs: body.logs,
          sessionId: body.sessionId,
          sessionConfig: body.sessionConfig,
          mcpServers: body.mcpServers,
          repos: effectiveRepos,
          systemOverride: body.systemOverride,
          mode: body.mode,
          skills: body.skills,
          subAgents: body.subAgents,
          ggnn: body.ggnn,
          source: "repo_agent",
          abortSignal: abortController.signal,
          maxTurns: body.maxTurns,
          headers: body.headers,
          attachments: body.attachments,
          _metadata: body._metadata,
          commitList: body.commitList,
          ignoreRepoInfo: body.ignoreRepoInfo,
        },
      );

      const streamResponse = streamResult.toUIMessageStreamResponse();

      // Forward status + headers
      res.status(streamResponse.status);
      res.setHeader("X-Session-Id", body.sessionId);
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

      // If the client disconnects mid-stream, abort the in-flight agent run
      const onClientClose = () => {
        if (!abortController.signal.aborted) {
          console.log(`[repo_agent] Client disconnected; aborting session ${body.sessionId}`);
          try { abortController.abort(); } catch (_) {}
        }
      };
      res.on("close", onClientClose);

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
        .finally(async () => {
          res.off("close", onClientClose);
          await finalizeSession();
          unregisterAbortController(body.sessionId);
          endTracking(opId);
        });

      return;
    } catch (error: any) {
      console.error("[repo_agent] Stream setup error:", error);
      unregisterAbortController(body.sessionId);
      endTracking(opId);
      if (!res.headersSent) {
        res.status(500).json({ error: error.message || "Internal server error" });
      }
      return;
    }
  }

  // ── Non-streaming path: async job with event bus ─────────────────────
  const request_id = asyncReqs.startReq();

  // Create an event bus for real-time SSE streaming of this request
  const bus = createBus(request_id);

  // Register abort controller keyed by request_id (so /repo/agent/abort can cancel it).
  // Also mirror under sessionId for callers who only have that.
  const abortController = registerAbortController(request_id);
  if (body.sessionId && body.sessionId !== request_id) {
    registerAbortController(body.sessionId, abortController);
  }
  const opId = startTracking("repo_agent", abortController);

  const repoDirPromise = body.repoUrl
    ? cloneOrUpdateRepo(body.repoUrl, body.username, body.pat, body.commitList, abortController.signal)
    : Promise.resolve(resolveRepoDir(effectiveRepos));

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
        return get_context(promptInput, repoDir, {
          transparent,
          pat: body.pat,
          toolsConfig: body.toolsConfig,
          schema: body.schema,
          modelName: body.modelName,
          apiKey: body.apiKey,
          baseUrl: body.baseUrl,
          logs: body.logs,
          sessionId: body.sessionId,
          sessionConfig: body.sessionConfig,
          mcpServers: body.mcpServers,
          repos: effectiveRepos,
          systemOverride: body.systemOverride,
          mode: body.mode,
          skills: body.skills,
          subAgents: body.subAgents,
          ggnn: body.ggnn,
          source: "repo_agent",
          abortSignal: abortController.signal,
          maxTurns: body.maxTurns,
          headers: body.headers,
          attachments: body.attachments,
          _metadata: body._metadata,
          commitList: body.commitList,
          ignoreRepoInfo: body.ignoreRepoInfo,
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
        const aborted = abortController.signal.aborted;
        if (aborted) {
          console.log(`[repo_agent] Run aborted: ${request_id}`);
          asyncReqs.failReq(request_id, "aborted");
          bus.emit({
            type: "error",
            error: "aborted",
            timestamp: new Date().toISOString(),
          });
        } else {
          console.error("[repo_agent] Background work failed with error:", error);
          asyncReqs.failReq(request_id, error.message || error.toString());
          bus.emit({
            type: "error",
            error: error.message || error.toString(),
            timestamp: new Date().toISOString(),
          });
        }
      })
      .finally(() => {
        unregisterAbortController(request_id);
        if (body.sessionId && body.sessionId !== request_id) {
          unregisterAbortController(body.sessionId);
        }
        endTracking(opId);
      });
    res.json({ request_id, status: "pending", sessionId: body.sessionId, ...(events_token && { events_token }) });
  } catch (error) {
    console.log("===> error");
    asyncReqs.failReq(request_id, error);
    console.error("Error in repo_agent", error);
    unregisterAbortController(request_id);
    if (body.sessionId && body.sessionId !== request_id) {
      unregisterAbortController(body.sessionId);
    }
    res.status(500).json({ error: "Internal server error" });
    endTracking(opId);
  }
}

// curl -X POST -H "Content-Type: application/json" \
//   -d '{"request_id":"<id>"}' \
//   "http://localhost:3355/repo/agent/abort"
// curl -X POST -H "Content-Type: application/json" \
//   -d '{"sessionId":"<id>"}' \
//   "http://localhost:3355/repo/agent/abort"
export async function abort_agent(req: Request, res: Response) {
  const request_id =
    (req.body?.request_id as string | undefined) ||
    (req.query?.request_id as string | undefined);
  const sessionId =
    (req.body?.sessionId as string | undefined) ||
    (req.body?.session_id as string | undefined) ||
    (req.query?.sessionId as string | undefined) ||
    (req.query?.session_id as string | undefined);
  const key = request_id || sessionId;
  console.log("===> POST /repo/agent/abort", { request_id, sessionId });
  if (!key) {
    res.status(400).json({ error: "Provide request_id or sessionId" });
    return;
  }
  const aborted = abortRequest(key);
  if (!aborted) {
    res.status(404).json({ aborted: false, error: "No active run found for the given key" });
    return;
  }
  res.json({ aborted: true, key });
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
    const config = loadSessionConfig(sessionId);
    const _metadata = loadSessionMetadata(sessionId);
    res.json({ sessionId, messages, config, _metadata });
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

  const repoDir = await cloneOrUpdateRepo(repoUrl, username, pat, commit ? [commit] : undefined);

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

// curl -X DELETE "http://localhost:3355/repo?repo=stakwork/hive"
// curl -X DELETE "http://localhost:3355/repo?repo_url=https://github.com/stakwork/hive"
export async function delete_repo(req: Request, res: Response) {
  const repoParam =
    (req.query.repo as string | undefined) ||
    (req.query.repo_url as string | undefined) ||
    (req.body && (req.body.repo as string | undefined)) ||
    (req.body && (req.body.repo_url as string | undefined));

  const repo = normalizeRepoRef(repoParam || "");
  if (!repo || !repo.includes("/")) {
    res.status(400).json({
      error:
        "Missing or invalid repo. Provide ?repo=owner/repo or ?repo_url=https://github.com/owner/repo",
    });
    return;
  }

  console.log(`===> DELETE /repo ${repo}`);
  try {
    const counts = await db.delete_repo(repo);
    res.json({ success: true, repo, ...counts });
  } catch (e: any) {
    console.error("[delete_repo] Failed:", e);
    res.status(500).json({ error: e?.message || "Internal server error" });
  }
}

export async function get_agent_file(req: Request, res: Response) {
  const filePath = req.query.path as string;
  console.log("===> GET /repo/agent/file", { filePath });

  if (!filePath) {
    res.status(400).json({ error: "Missing path" });
    return;
  }

  // Confine reads to the same sandbox the editor tool writes into (cloned repos
  // under /tmp, the OS scratch dir, and the durable artifacts dir). Refuses
  // absolute paths and ../ traversal that escape those roots.
  let resolved: string;
  try {
    resolved = resolveInCwd(filePath, editorRoots("/tmp"));
  } catch {
    res.status(403).json({ error: "Path outside allowed roots" });
    return;
  }

  if (!existsSync(resolved)) {
    res.status(404).json({ error: "File not found" });
    return;
  }

  res.sendFile(resolved);
}
