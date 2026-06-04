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
  /** Tunable default knobs (prompts, thresholds, …), overridable per run. */
  params?: Record<string, unknown>;
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
 * Run a workflow. The launch is **detached** (§8): `POST …/run` starts the
 * run server-side and returns `{ runId }` immediately, then we **reattach**
 * to its event log via `GET …/runs/:runId/stream` (SSE tail). `onEvent` fires
 * for each event (replayed history + live appends); the returned RunResult
 * arrives on the terminal `done` event. The two-step launch+reattach is
 * invisible to callers — same signature as a single streamed request.
 */
export async function runWorkflow(
  name: string,
  input: unknown,
  onEvent?: (event: RunEvent) => void,
  params?: Record<string, unknown>,
): Promise<any> {
  const { runId } = await launchWorkflow(name, input, params);
  return streamRun(name, runId, onEvent);
}

/**
 * Launch a run **detached** (§8) and return its `runId` immediately — the run
 * keeps executing server-side regardless of this connection. Callers that want
 * to surface the run *before* it finishes (e.g. show it as "running" in a list)
 * launch first, then `streamRun(name, runId)` to follow its events.
 */
export async function launchWorkflow(
  name: string,
  input: unknown,
  params?: Record<string, unknown>,
): Promise<{ runId: string }> {
  const res = await fetch(`${BASE}/workflows/${name}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input,
      ...(params && Object.keys(params).length > 0 ? { params } : {}),
    }),
  });
  return (await res.json()) as { runId: string };
}

/**
 * Reattach to a run (live or completed) and stream its events. Tails the
 * server's append-only log from the start, so callers see full history even
 * when they attach late. Resolves to the final RunResult on `done`.
 */
export async function streamRun(
  name: string,
  runId: string,
  onEvent?: (event: RunEvent) => void,
): Promise<any> {
  const res = await fetch(`${BASE}/workflows/${name}/runs/${runId}/stream`);
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
//
// A chat is a DETACHED background job: `POST /chat` launches a turn
// server-side and returns `{ chatId, turn }`; we reattach via
// `GET /chat/:id/stream` (SSE tail). Close the tab and the agent keeps
// running — reopen, load the transcript (`getChat`), and reattach to the
// live turn. Mirrors the workflow-run launch+reattach model above.

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
  onFinish: (status: string) => void;
}

export type ChatStatus = "live" | "done" | "error";

export interface ChatMeta {
  id: string;
  title?: string;
  status: ChatStatus;
  model?: string;
  createdAt: string;
  updatedAt: string;
  currentTurn: number;
}

/** A normalized fine-grained chat event (matches the server's `ChatEvent`). */
export interface ChatEvent {
  ts: string;
  chatId: string;
  turn: number;
  type: "text-delta" | "tool-input" | "tool-output" | "step.finish" | "chat.end" | "chat.error";
  delta?: string;
  toolName?: string;
  toolCallId?: string;
  input?: any;
  output?: any;
  error?: { message: string };
}

export interface ChatTranscript {
  meta: ChatMeta;
  /** Stored AI SDK ModelMessage objects (opaque — rendered by the flyout). */
  messages: { role: string; content: unknown }[];
}

/** Launch a chat turn (detached). Pass `chatId` to continue an existing
 *  session, or omit it to start a new one. Returns the ids to reattach with. */
export async function sendChat(
  message: string,
  chatId?: string,
): Promise<{ chatId: string; turn: number }> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, ...(chatId ? { chatId } : {}) }),
  });
  if (!res.ok) throw new Error(`chat: ${res.status} ${res.statusText}`);
  return (await res.json()) as { chatId: string; turn: number };
}

/** Load a chat's full transcript + meta (for reload / reattach). */
export const getChat = (chatId: string) =>
  fetchJSON<ChatTranscript>(`/chat/${chatId}`);

/** List chat sessions (newest first). */
export const listChats = () => fetchJSON<ChatMeta[]>("/chats");

/**
 * Reattach to a chat turn (live or completed) and stream its events. Tails
 * the server's append-only log from the start of the turn, so callers see
 * full history even when they attach late. Resolves once the turn ends.
 */
export async function streamChat(
  chatId: string,
  turn: number,
  callbacks: ChatCallbacks,
): Promise<void> {
  const res = await fetch(`${BASE}/chat/${chatId}/stream?turn=${turn}`);
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buf = "";
  let status = "done";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";

    let eventType = "message";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const data = line.slice(6);
        try {
          const msg = JSON.parse(data);
          if (eventType === "done") {
            status = msg.status ?? status;
          } else {
            dispatchChatEvent(msg as ChatEvent, callbacks);
          }
        } catch {
          // skip unparseable lines
        }
        eventType = "message";
      }
    }
  }

  callbacks.onFinish(status);
}

function dispatchChatEvent(e: ChatEvent, cb: ChatCallbacks): void {
  switch (e.type) {
    case "text-delta":
      if (e.delta) cb.onTextDelta(e.delta);
      break;
    case "tool-input":
      cb.onToolCall({ name: e.toolName ?? "", input: e.input });
      break;
    case "tool-output":
      cb.onToolResult?.({ name: e.toolName ?? "", input: e.input, output: e.output });
      break;
    case "step.finish":
      cb.onStepFinish();
      break;
    case "chat.error":
      cb.onTextDelta(`\n\n⚠️ ${e.error?.message ?? "chat error"}`);
      break;
    case "chat.end":
      break;
  }
}
