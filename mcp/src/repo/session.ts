import { ModelMessage } from "ai";
import {
  existsSync,
  mkdirSync,
  appendFileSync,
  readFileSync,
  readdirSync,
  statSync,
  unlinkSync,
} from "fs";
import { randomUUID } from "crypto";
import path from "path";
import { db } from "../graph/neo4j.js";
import { getProviderForModel } from "../aieo/src/provider.js";

const SESSIONS_DIR = process.env.SESSIONS_DIR || ".sessions";

const sessionMeta = new Map<string, { source: string; start_time: string }>();

export interface Session {
  id: string;
  messages: ModelMessage[];
}

export interface SessionConfig {
  truncateToolResults?: boolean; // Enable truncation (default: false)
  maxToolResultLines?: number; // Default: 50
  maxToolResultChars?: number; // Default: 2000
}

export interface StepMeta {
  step: number;
  turn: number;
  usage: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  };
  cumulativeInput: number;
  cumulativeOutput: number;
  toolCalls: string[];
  timestamp: string;
}

/**
 * Get the file path for a session
 */
function getSessionFile(sessionId: string): string {
  const sessionDir = path.isAbsolute(SESSIONS_DIR)
    ? SESSIONS_DIR
    : path.join(process.cwd(), SESSIONS_DIR);
  if (!existsSync(sessionDir)) {
    mkdirSync(sessionDir, { recursive: true });
  }
  return path.join(sessionDir, `${sessionId}.jsonl`);
}

/**
 * Create a new session and return its ID.
 * If an ID is provided, use it; otherwise generate a random UUID.
 * Optionally writes the system prompt as the first JSONL entry.
 */
export function createSession(
  id?: string,
  system?: string,
  source?: string,
): string {
  const sessionId = id || randomUUID();
  sessionMeta.set(sessionId, {
    source: source || "unknown",
    start_time: new Date().toISOString(),
  });
  const filePath = getSessionFile(sessionId);
  if (system) {
    const systemMsg: ModelMessage = { role: "system", content: system };
    appendFileSync(filePath, JSON.stringify(systemMsg) + "\n");
  } else {
    appendFileSync(filePath, "");
  }
  return sessionId;
}

/**
 * Append end-of-session metadata (timing, model, token usage).
 */
export async function appendSessionEnd(
  sessionId: string,
  opts: {
    end_time: string;
    model?: string;
    provider?: string;
    duration_ms?: number;
    token_usage?: { input: number; cache_read: number; cache_write: number; output: number; total: number };
    status?: "success" | "error";
    error_message?: string;
  }
): Promise<void> {
  const stored = sessionMeta.get(sessionId) ?? { source: "unknown", start_time: opts.end_time };
  const start_time = new Date(stored.start_time).getTime();
  const end_time = new Date(opts.end_time).getTime();
  const resolvedProvider = opts.provider || getProviderForModel(opts.model);
  await db?.upsert_agent_session({
    session_id: sessionId,
    source: stored.source,
    model: opts.model || "",
    provider: resolvedProvider,
    start_time,
    end_time,
    duration_ms: opts.duration_ms ?? (end_time - start_time),
    input_tokens: opts.token_usage?.input || 0,
    cache_read_tokens: opts.token_usage?.cache_read || 0,
    cache_write_tokens: opts.token_usage?.cache_write || 0,
    output_tokens: opts.token_usage?.output || 0,
    total_tokens: opts.token_usage?.total || 0,
    status: opts.status || "success",
    error_message: opts.error_message || "",
  }).catch((e) => console.error("[sessions] Neo4j upsert failed:", e));
}

/**
 * Load all messages from a session (including system message if present).
 */
export function loadSession(sessionId: string): ModelMessage[] {
  const filePath = getSessionFile(sessionId);

  if (!existsSync(filePath)) {
    return [];
  }

  const content = readFileSync(filePath, "utf-8");
  const lines = content.split("\n").filter((line) => line.trim());

  return lines.map((line) => JSON.parse(line) as ModelMessage);
}

/**
 * Load conversation messages from a session, excluding the system prompt.
 * The system prompt (first entry with role "system") is stored once at
 * session creation for observability, but stripped here because the
 * ToolLoopAgent already sends it via its `instructions` field.
 */
export function loadSessionMessages(sessionId: string): ModelMessage[] {
  const all = loadSession(sessionId);
  if (all.length > 0 && all[0].role === "system") {
    return all.slice(1);
  }
  return all;
}

/**
 * Append messages to a session
 */
export function appendMessages(
  sessionId: string,
  messages: ModelMessage[]
): void {
  const filePath = getSessionFile(sessionId);
  const content = messages.map((m) => JSON.stringify(m)).join("\n") + "\n";
  appendFileSync(filePath, content);
}

/**
 * Check if a session exists
 */
export function sessionExists(sessionId: string): boolean {
  return existsSync(getSessionFile(sessionId));
}

/**
 * Delete a session
 */
export function deleteSession(sessionId: string): void {
  const filePath = getSessionFile(sessionId);
  if (existsSync(filePath)) {
    unlinkSync(filePath);
  }
  const metaPath = getMetaFile(sessionId);
  if (existsSync(metaPath)) {
    unlinkSync(metaPath);
  }
  const provPath = getProvenanceFile(sessionId);
  if (existsSync(provPath)) {
    unlinkSync(provPath);
  }
}

/**
 * Get the file path for a session's per-step metadata sidecar.
 * This is separate from the conversation JSONL to avoid corrupting session replay.
 */
function getMetaFile(sessionId: string): string {
  const sessionDir = path.isAbsolute(SESSIONS_DIR)
    ? SESSIONS_DIR
    : path.join(process.cwd(), SESSIONS_DIR);
  if (!existsSync(sessionDir)) {
    mkdirSync(sessionDir, { recursive: true });
  }
  return path.join(sessionDir, `${sessionId}.meta.jsonl`);
}

/**
 * Append per-step usage metadata for a turn.
 * Written to a sidecar file — never to the conversation JSONL.
 */
export function appendStepMeta(sessionId: string, steps: StepMeta[]): void {
  if (steps.length === 0) return;
  const filePath = getMetaFile(sessionId);
  const content = steps.map((s) => JSON.stringify(s)).join("\n") + "\n";
  appendFileSync(filePath, content);
}

/**
 * Load all per-step metadata for a session.
 * Returns an empty array if no sidecar file exists (old sessions).
 */
export function loadStepMeta(sessionId: string): StepMeta[] {
  const filePath = getMetaFile(sessionId);
  if (!existsSync(filePath)) return [];
  try {
    const content = readFileSync(filePath, "utf-8");
    return content
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => JSON.parse(l) as StepMeta);
  } catch {
    return [];
  }
}

function getProvenanceFile(sessionId: string): string {
  const sessionDir = path.isAbsolute(SESSIONS_DIR)
    ? SESSIONS_DIR
    : path.join(process.cwd(), SESSIONS_DIR);
  if (!existsSync(sessionDir)) {
    mkdirSync(sessionDir, { recursive: true });
  }
  return path.join(sessionDir, `${sessionId}.provenance.jsonl`);
}

export interface SearchProvenanceEntry {
  tool_call_id?: string;
  tool_name: string;
  timestamp: string;
  provenance: import("../graph/graph.js").SearchProvenance;
}

export function appendSearchProvenance(
  sessionId: string,
  entries: SearchProvenanceEntry[],
): void {
  if (entries.length === 0) return;
  const filePath = getProvenanceFile(sessionId);
  const content = entries.map((e) => JSON.stringify(e)).join("\n") + "\n";
  appendFileSync(filePath, content);
}

export function loadSearchProvenance(
  sessionId: string,
): SearchProvenanceEntry[] {
  const filePath = getProvenanceFile(sessionId);
  if (!existsSync(filePath)) return [];
  try {
    const content = readFileSync(filePath, "utf-8");
    return content
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => JSON.parse(l) as SearchProvenanceEntry);
  } catch {
    return [];
  }
}

const SESSION_MAX_AGE_MS = parseInt(
  process.env.SESSION_MAX_AGE_MS || String(30 * 24 * 60 * 60 * 1000),
  10
); // default 30 days, configurable via SESSION_MAX_AGE_MS env var

/**
 * Delete session files older than SESSION_MAX_AGE_MS.
 * Call this on startup or periodically.
 */
export function pruneExpiredSessions(): number {
  const sessionDir = path.isAbsolute(SESSIONS_DIR)
    ? SESSIONS_DIR
    : path.join(process.cwd(), SESSIONS_DIR);
  if (!existsSync(sessionDir)) return 0;

  const now = Date.now();
  let pruned = 0;
  for (const file of readdirSync(sessionDir)) {
    if (!file.endsWith(".jsonl") || file.endsWith(".meta.jsonl") || file.endsWith(".provenance.jsonl")) continue;
    const filePath = path.join(sessionDir, file);
    try {
      const { mtimeMs } = statSync(filePath);
      if (now - mtimeMs > SESSION_MAX_AGE_MS) {
        unlinkSync(filePath);
        const metaPath = filePath.replace(/\.jsonl$/, ".meta.jsonl");
        if (existsSync(metaPath)) unlinkSync(metaPath);
        const provPath = filePath.replace(/\.jsonl$/, ".provenance.jsonl");
        if (existsSync(provPath)) unlinkSync(provPath);
        pruned++;
      }
    } catch {
      // ignore stat/unlink errors for individual files
    }
  }
  if (pruned > 0) {
    console.log(`[sessions] pruned ${pruned} expired session(s)`);
  }
  return pruned;
}

/**
 * Truncate a tool result for storage efficiency.
 * The model already processed this data, so we can store references instead of full content.
 */
export function truncateToolResult(
  toolName: string,
  result: string,
  config: SessionConfig
): string {
  if (!config.truncateToolResults) {
    return result;
  }

  const { maxToolResultLines = 50, maxToolResultChars = 2000 } = config;

  const lines = result.split("\n");
  const totalLines = lines.length;
  const totalChars = result.length;

  let truncated = result;
  let wasTruncated = false;

  // Truncate by lines first
  if (totalLines > maxToolResultLines) {
    truncated = lines.slice(0, maxToolResultLines).join("\n");
    wasTruncated = true;
  }

  // Then by characters
  if (truncated.length > maxToolResultChars) {
    truncated = truncated.slice(0, maxToolResultChars);
    wasTruncated = true;
  }

  if (wasTruncated) {
    // Add reference marker with hint
    truncated +=
      `\n\n[TRUNCATED: ${totalLines} lines, ${totalChars} chars` +
      ` - use ${getToolHint(toolName)} to see full content]`;
  }

  return truncated;
}

/**
 * Get a hint for how to retrieve the full content for a given tool
 */
function getToolHint(toolName: string): string {
  switch (toolName) {
    case "read_file":
    case "read_files":
      return "read_file";
    case "repo_overview":
    case "list_files":
      return "list_files or repo_overview";
    case "grep":
    case "search":
      return "grep";
    default:
      return toolName;
  }
}
