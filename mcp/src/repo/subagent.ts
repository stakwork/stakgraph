import { type ToolsConfig } from "./tools.js";

export interface SubAgent {
  /** Tool name exposed to the LLM, e.g. "backend_agent" */
  name: string;
  /** Description shown to the LLM so it knows when to call this tool */
  description: string;
  /** Base URL of the remote server (e.g. "https://other:3355" or "https://other:3355/repo/agent") */
  url: string;
  /** Auth token for the remote server (sent as x-api-token header) */
  apiToken: string;
  /** Repo URL for the remote agent to operate on */
  repoUrl?: string;
  /** Model to use on the remote agent */
  model?: string;
  /** ToolsConfig to pass to the remote agent */
  toolsConfig?: ToolsConfig;
  /** Max seconds to wait for the remote agent to finish (default: 300) */
  timeoutSeconds?: number;
}

/** Resolve a SubAgent URL to the /repo/agent endpoint and the base origin for /progress */
function resolveSubAgentUrls(rawUrl: string): {
  agentUrl: string;
  progressBaseUrl: string;
} {
  // Normalize: strip trailing slashes
  let url = rawUrl.replace(/\/+$/, "");

  // If URL already ends with /repo/agent, use it directly
  if (url.endsWith("/repo/agent")) {
    const parsed = new URL(url);
    return {
      agentUrl: url,
      progressBaseUrl: parsed.origin,
    };
  }

  // Otherwise treat it as the base URL and append /repo/agent
  const parsed = new URL(url);
  return {
    agentUrl: `${parsed.origin}${parsed.pathname.replace(/\/+$/, "")}/repo/agent`,
    progressBaseUrl: parsed.origin,
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const SUB_AGENT_DEFAULT_TIMEOUT_S = 600;
const SUB_AGENT_POLL_INTERVAL_MS = 5000;

/** Call a remote /repo/agent endpoint and poll /progress until complete */
export async function callRemoteAgent(
  subAgent: SubAgent,
  prompt: string
): Promise<string> {
  const { agentUrl, progressBaseUrl } = resolveSubAgentUrls(subAgent.url);
  const timeoutSeconds =
    subAgent.timeoutSeconds ?? SUB_AGENT_DEFAULT_TIMEOUT_S;
  const deadlineMs = Date.now() + timeoutSeconds * 1000;

  // 1. POST to /repo/agent
  const body: Record<string, unknown> = { prompt };
  if (subAgent.repoUrl) body.repo_url = subAgent.repoUrl;
  if (subAgent.model) body.model = subAgent.model;
  if (subAgent.toolsConfig) body.toolsConfig = subAgent.toolsConfig;

  console.log(
    `[sub-agent:${subAgent.name}] POST ${agentUrl} (timeout: ${timeoutSeconds}s)`
  );

  let resp: Response;
  try {
    resp = await fetch(agentUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-token": subAgent.apiToken,
      },
      body: JSON.stringify(body),
    });
  } catch (err: any) {
    const msg = err?.message || String(err);
    console.error(`[sub-agent:${subAgent.name}] POST failed: ${msg}`);
    return `Sub-agent request failed: ${msg}`;
  }

  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    console.error(
      `[sub-agent:${subAgent.name}] POST returned ${resp.status}: ${text}`
    );
    return `Sub-agent returned HTTP ${resp.status}: ${text}`;
  }

  let startPayload: any;
  try {
    startPayload = await resp.json();
  } catch {
    return "Sub-agent returned invalid JSON on start";
  }

  const requestId = startPayload?.request_id;
  if (!requestId) {
    // If the response already contains a result (synchronous mode), return it
    if (startPayload?.content || startPayload?.final_answer) {
      return startPayload.final_answer || startPayload.content;
    }
    return "Sub-agent did not return a request_id";
  }

  // 2. Poll /progress?request_id=...
  const progressUrl = `${progressBaseUrl}/progress?request_id=${encodeURIComponent(requestId)}`;
  console.log(`[sub-agent:${subAgent.name}] Polling ${progressUrl}`);

  while (Date.now() < deadlineMs) {
    await sleep(SUB_AGENT_POLL_INTERVAL_MS);

    let pollResp: Response;
    try {
      pollResp = await fetch(progressUrl, {
        headers: { "x-api-token": subAgent.apiToken },
      });
    } catch (err: any) {
      console.warn(
        `[sub-agent:${subAgent.name}] Poll fetch error: ${err?.message}`
      );
      continue; // transient network error, retry
    }

    if (!pollResp.ok) {
      // 404 means the request hasn't been written to disk yet — retry
      if (pollResp.status === 404) continue;
      const text = await pollResp.text().catch(() => "");
      console.warn(
        `[sub-agent:${subAgent.name}] Poll returned ${pollResp.status}: ${text}`
      );
      continue;
    }

    let status: any;
    try {
      status = await pollResp.json();
    } catch {
      continue;
    }

    if (status.status === "completed") {
      const result = status.result;
      const answer =
        result?.final_answer || result?.content || JSON.stringify(result);
      console.log(
        `[sub-agent:${subAgent.name}] Completed (${((Date.now() - (deadlineMs - timeoutSeconds * 1000)) / 1000).toFixed(1)}s)`
      );
      return answer;
    }

    if (status.status === "failed") {
      const errMsg =
        typeof status.error === "string"
          ? status.error
          : status.error?.message || JSON.stringify(status.error);
      console.error(`[sub-agent:${subAgent.name}] Failed: ${errMsg}`);
      return `Sub-agent failed: ${errMsg}`;
    }

    // status === "pending" — keep polling
  }

  return `Sub-agent timed out after ${timeoutSeconds}s`;
}
