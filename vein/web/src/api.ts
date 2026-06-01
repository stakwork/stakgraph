// Mount-path-agnostic API base. Derived at runtime from the app's own
// script URL so the UI works whether vein is served at the root (`/`) or
// under any sub-path (`/lab`, `/foo/bar`, …) — no build-time base needed.
// In a production build the bundle loads from `<mount>/assets/...`, so the
// prefix is everything before `/assets/`. In dev (vite) there is no
// `/assets/` segment, so the base is empty and the dev proxy handles it.
function deriveBase(): string {
  try {
    const path = new URL(import.meta.url).pathname;
    const m = path.match(/^(.*)\/assets\//);
    return m ? m[1] : "";
  } catch {
    return "";
  }
}

const BASE = deriveBase();

export async function fetchJSON<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.error) msg = body.error;
    } catch {}
    throw new Error(`${path}: ${msg}`);
  }
  return res.json() as Promise<T>;
}

// ── Workflows ──────────────────────────────────────────────────────────────

export interface WorkflowEntry {
  name: string;
  activeVersion: string;
  versions: string[];
  description?: string;
}

export interface WorkflowMeta {
  active: string;
  versions: Record<string, { createdAt: string; description?: string }>;
}

export const listWorkflows = () => fetchJSON<WorkflowEntry[]>("/workflows");

export const getWorkflowMeta = (name: string) =>
  fetchJSON<WorkflowMeta>(`/workflows/${name}`);

export const getWorkflowCode = async (name: string, version: string) => {
  const res = await fetch(`${BASE}/workflows/${name}/${version}`);
  return res.text();
};

export interface FlowDef {
  name: string;
  steps: { id: string; type: string; config: Record<string, any>; options?: any }[];
}

export const getWorkflowFlow = (name: string) =>
  fetchJSON<FlowDef>(`/workflows/${name}/flow`);

export interface CreateWorkflowResponse {
  ok: true;
  /** Final workflow name (may differ from `requested` if auto-suffixed). */
  workflow: string;
  version: string;
  active: string;
  renamed: boolean;
  requested: string;
}

/** Create a brand-new workflow at v1. Auto-suffixes the name on collision. */
export const createWorkflowYaml = (
  name: string,
  yamlStr: string,
  description?: string,
) =>
  fetchJSON<CreateWorkflowResponse>(`/workflows`, {
    method: "POST",
    body: JSON.stringify({ name, yaml: yamlStr, description }),
  });

/** Publish a new version of an existing workflow. */
export const publishWorkflow = (
  name: string,
  version: string,
  steps: any[],
  description?: string,
) =>
  fetchJSON<any>(`/workflows/${name}`, {
    method: "POST",
    body: JSON.stringify({ version, steps, description }),
  });

/** Publish a new version of an existing workflow (raw YAML). */
export const publishWorkflowYaml = (
  name: string,
  version: string,
  yamlStr: string,
  description?: string,
) =>
  fetchJSON<any>(`/workflows/${name}`, {
    method: "POST",
    body: JSON.stringify({ version, yaml: yamlStr, description }),
  });

export const getWorkflowYaml = async (name: string, version: string) => {
  const res = await fetch(`${BASE}/workflows/${name}/${version}`);
  return res.text();
};

/**
 * Run a workflow via SSE. Calls `onEvent` for each event as it streams in.
 * Returns the final RunResult when the stream closes.
 */
export async function runWorkflow(
  name: string,
  input: unknown,
  onEvent?: (event: RunEvent) => void,
): Promise<any> {
  const res = await fetch(`${BASE}/workflows/${name}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input }),
  });

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";
  let result: any = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Parse SSE lines
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    let eventType = "message";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const data = JSON.parse(line.slice(6));
        if (eventType === "done") {
          result = data;
        } else {
          onEvent?.(data as RunEvent);
        }
        eventType = "message";
      }
    }
  }

  return result;
}

// ── Steps ──────────────────────────────────────────────────────────────────

export interface StepsResponse {
  core: { type: string; source: string }[];
  workspace: { type: string; description?: string }[];
}

export const listSteps = () => fetchJSON<StepsResponse>("/steps");

export interface FieldDesc {
  name: string;
  kind: "string" | "number" | "boolean" | "enum" | "json";
  required: boolean;
  default?: unknown;
  enumValues?: string[];
}

export interface StepSchemaResponse {
  type: string;
  fields: FieldDesc[];
}

export const getStepSchema = (type: string) =>
  fetchJSON<StepSchemaResponse>(`/steps/${encodeURIComponent(type)}/schema`);

export interface StepSourceResponse {
  type: string;
  source: string | null;
  origin: "registry" | "core" | "lib" | "custom" | null;
}

export const getStepSource = (type: string) =>
  fetchJSON<StepSourceResponse>(`/steps/${encodeURIComponent(type)}/source`);

// ── Runs ───────────────────────────────────────────────────────────────────

export interface RunSummary {
  runId: string;
  workflow: string;
  startedAt?: string;
  finishedAt?: string;
  durationMs?: number;
  status: string;
  input?: unknown;
  output?: unknown;
  error?: { message: string };
}

export interface RunEvent {
  ts: string;
  runId: string;
  path: string;
  type: string;
  stepType?: string;
  input?: unknown;
  output?: unknown;
  error?: { message: string };
  durationMs?: number;
  iteration?: number;
}

export const listRuns = (workflow: string) =>
  fetchJSON<RunSummary[]>(`/workflows/${workflow}/runs`);

export const getRun = (workflow: string, runId: string) =>
  fetchJSON<RunSummary>(`/workflows/${workflow}/runs/${runId}`);

export const getRunEvents = (workflow: string, runId: string) =>
  fetchJSON<RunEvent[]>(`/workflows/${workflow}/runs/${runId}/events`);

// ── Chat (AI workflow builder) ─────────────────────────────────────────────

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ToolCallInfo {
  name: string;
  input: any;
}

export interface ToolResultInfo {
  name: string;
  input: any;
  output: any;
}

export interface ChatCallbacks {
  onTextDelta: (delta: string) => void;
  onToolCall: (tc: ToolCallInfo) => void;
  onToolResult?: (tr: ToolResultInfo) => void;
  onStepFinish: () => void;
  onFinish: () => void;
}

/**
 * Stream a chat message to the AI workflow builder.
 * Uses the same UI message stream protocol as Vercel AI SDK.
 */
export async function chat(
  messages: ChatMessage[],
  callbacks: ChatCallbacks,
): Promise<void> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buf = "";
  const toolCalls: Record<string, { name: string; inputBuf: string; input?: any }> = {};

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";

    for (const line of lines) {
      const raw = line.trim();
      let data = "";
      if (raw.startsWith("data:")) data = raw.slice(5).trim();
      else if (raw.startsWith("{")) data = raw;
      else continue;
      if (data === "[DONE]") continue;

      try {
        const msg = JSON.parse(data);
        switch (msg.type) {
          case "text-delta":
            if (msg.delta) callbacks.onTextDelta(msg.delta);
            break;
          case "tool-input-start":
            toolCalls[msg.toolCallId] = { name: msg.toolName, inputBuf: "" };
            break;
          case "tool-input-delta":
            if (toolCalls[msg.toolCallId]) {
              toolCalls[msg.toolCallId].inputBuf += msg.inputTextDelta;
            }
            break;
          case "tool-input-available": {
            // Remember the call so we can correlate with output later.
            toolCalls[msg.toolCallId] = {
              name: msg.toolName,
              inputBuf: "",
              input: msg.input,
            };
            callbacks.onToolCall({ name: msg.toolName, input: msg.input });
            break;
          }
          case "tool-output-available": {
            const call = toolCalls[msg.toolCallId];
            if (call && callbacks.onToolResult) {
              callbacks.onToolResult({
                name: call.name,
                input: call.input,
                output: msg.output,
              });
            }
            break;
          }
          case "finish-step":
            callbacks.onStepFinish();
            break;
        }
      } catch {
        // skip unparseable lines
      }
    }
  }

  callbacks.onFinish();
}
