import { ModelMessage } from "ai";
import {
  existsSync,
  mkdirSync,
  appendFileSync,
  readFileSync,
  readdirSync,
  statSync,
  unlinkSync,
  writeFileSync,
  rmSync,
} from "fs";
import { randomUUID } from "crypto";
import path from "path";
import { db } from "../graph/neo4j.js";
import { getProviderForModel } from "../aieo/src/provider.js";
import { AiUsage, AiUsageWithLegacy } from "../aieo/src/usage.js";

const SESSIONS_DIR = process.env.SESSIONS_DIR || ".sessions";

const sessionMeta = new Map<string, { source: string; start_time: string; repo?: string }>();

export interface Session {
  id: string;
  messages: ModelMessage[];
}

export interface SessionConfig {
  truncateToolResults?: boolean; // Enable truncation (default: false)
  maxToolResultLines?: number; // Default: 50
  maxToolResultChars?: number; // Default: 2000
}

export interface SessionInitConfig {
  model?: string;
  provider?: string;
  systemOverride?: string;
  toolsConfig?: { [key: string]: any };
  schema?: { [key: string]: any };
  sessionConfig?: SessionConfig;
  maxTurns?: number;
  isolatedContext?: boolean;
  source?: string;
  repos?: string[];
  temperature: number;
  tools?: Record<string, string>;          // name → description for every resolved tool
  providerConfig?: { [key: string]: any }; // resolved getProviderOptions output
  baseUrl?: string;
  requestUrl?: string;
  mcpServers?: { [key: string]: any }[];    // secrets (token/headers) redacted
  subAgents?: { [key: string]: any }[];     // secrets (apiToken) redacted
  ggnn?: { [key: string]: any };
  skills?: { [key: string]: any };
  commitList?: string[];
  ignoreRepoInfo?: boolean;
}

export interface StepMeta {
  step: number;
  turn: number;
  label?: string;
  usage: AiUsageWithLegacy;
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
  repo?: string,
): string {
  const sessionId = id || randomUUID();
  sessionMeta.set(sessionId, {
    source: source || "unknown",
    start_time: new Date().toISOString(),
    repo,
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
    token_usage?: AiUsage;
    status?: "success" | "error" | "aborted";
    error_message?: string;
  },
): Promise<void> {
  console.log(`[session] end session_id=${sessionId} model=${opts.model ?? ""} status=${opts.status ?? "success"} tokens=${opts.token_usage?.total ?? 0} duration_ms=${opts.duration_ms ?? 0}`);

  const stored = sessionMeta.get(sessionId);
  if (!stored) {
    console.error(`[session] appendSessionEnd: session_id=${sessionId} was never registered via createSession() in this process — refusing to create AgentSession node`);
    return;
  }
  const start_time = new Date(stored.start_time).getTime();
  const end_time = new Date(opts.end_time).getTime();
  const resolvedProvider = opts.provider || getProviderForModel(opts.model);
  await db
    ?.upsert_agent_session({
      session_id: sessionId,
      source: stored.source,
      repo: stored.repo || "",
      model: opts.model || "",
      provider: resolvedProvider,
      start_time,
      end_time,
      duration_ms: opts.duration_ms ?? end_time - start_time,
      input_tokens: opts.token_usage?.input || 0,
      cache_read_tokens: opts.token_usage?.cache_read || 0,
      cache_write_tokens: opts.token_usage?.cache_write || 0,
      output_tokens: opts.token_usage?.output || 0,
      total_tokens: opts.token_usage?.total || 0,
      status: opts.status || "success",
      error_message: opts.error_message || "",
    })
    .catch((e) => console.error("[sessions] Neo4j upsert failed:", e));
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
  const annPath = getAnnotationsFile(sessionId);
  if (existsSync(annPath)) {
    unlinkSync(annPath);
  }
  const configPath = getConfigFile(sessionId);
  if (existsSync(configPath)) {
    unlinkSync(configPath);
  }
  const metadataPath = getMetadataFile(sessionId);
  if (existsSync(metadataPath)) {
    unlinkSync(metadataPath);
  }
  deleteAttachments(sessionId);
}

/**
 * Get the file path for a session's init config sidecar.
 */
function getConfigFile(sessionId: string): string {
  const sessionDir = path.isAbsolute(SESSIONS_DIR)
    ? SESSIONS_DIR
    : path.join(process.cwd(), SESSIONS_DIR);
  if (!existsSync(sessionDir)) {
    mkdirSync(sessionDir, { recursive: true });
  }
  return path.join(sessionDir, `${sessionId}.config.json`);
}

/**
 * Persist session init config to a sidecar file.
 */
export function saveSessionConfig(sessionId: string, config: SessionInitConfig): void {
  writeFileSync(getConfigFile(sessionId), JSON.stringify(config, null, 2));
}

/**
 * Load session init config from the sidecar file.
 * Returns null if no sidecar exists (pre-change sessions) or if parsing fails.
 */
export function loadSessionConfig(sessionId: string): SessionInitConfig | null {
  const filePath = getConfigFile(sessionId);
  if (!existsSync(filePath)) return null;
  try {
    return JSON.parse(readFileSync(filePath, "utf-8")) as SessionInitConfig;
  } catch {
    return null;
  }
}

/**
 * Get the file path for a session's arbitrary metadata sidecar.
 */
function getMetadataFile(sessionId: string): string {
  const sessionDir = path.isAbsolute(SESSIONS_DIR)
    ? SESSIONS_DIR
    : path.join(process.cwd(), SESSIONS_DIR);
  if (!existsSync(sessionDir)) {
    mkdirSync(sessionDir, { recursive: true });
  }
  return path.join(sessionDir, `${sessionId}.metadata.json`);
}

/**
 * Persist arbitrary caller-provided metadata for a session to a sidecar file.
 */
export function saveSessionMetadata(sessionId: string, metadata: unknown): void {
  writeFileSync(getMetadataFile(sessionId), JSON.stringify(metadata, null, 2));
}

/**
 * Load arbitrary session metadata from the sidecar file.
 * Returns null if no sidecar exists or if parsing fails.
 */
export function loadSessionMetadata(sessionId: string): unknown | null {
  const filePath = getMetadataFile(sessionId);
  if (!existsSync(filePath)) return null;
  try {
    return JSON.parse(readFileSync(filePath, "utf-8"));
  } catch {
    return null;
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

export type AnnotationMarker =
  | "inefficient"
  | "bad_search"
  | "good_result"
  | "loop"
  | "wrong_tool"
  | "wasted_tokens";

export interface Annotation {
  ts: string;
  author?: string;
  target: "session" | "tool_call";
  target_id?: string;
  marker: AnnotationMarker;
  note?: string;
}

function getAnnotationsFile(sessionId: string): string {
  const sessionDir = path.isAbsolute(SESSIONS_DIR)
    ? SESSIONS_DIR
    : path.join(process.cwd(), SESSIONS_DIR);
  if (!existsSync(sessionDir)) {
    mkdirSync(sessionDir, { recursive: true });
  }
  return path.join(sessionDir, `${sessionId}.annotations.jsonl`);
}

export function appendAnnotation(sessionId: string, annotation: Annotation): void {
  const filePath = getAnnotationsFile(sessionId);
  appendFileSync(filePath, JSON.stringify(annotation) + "\n");
}

export function loadAnnotations(sessionId: string): Annotation[] {
  const filePath = getAnnotationsFile(sessionId);
  if (!existsSync(filePath)) return [];
  try {
    const content = readFileSync(filePath, "utf-8");
    return content
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => JSON.parse(l) as Annotation);
  } catch {
    return [];
  }
}

// ── Attachments cache ────────────────────────────────────────────────
// Image attachments (e.g. uploaded screenshots) are downloaded once and
// cached on disk, keyed by session. The conversation JSONL stores only a
// tiny `attachment://<id>` placeholder; the bytes live in a sidecar dir and
// are rehydrated on each turn. Both are torn down with the session.

export interface AttachmentMeta {
  id: string;
  mediaType: string;
  bytes: number;
  originalUrl?: string;
  createdAt: string;
}

function attachmentsDirFor(sessionId: string): string {
  const sessionDir = path.isAbsolute(SESSIONS_DIR)
    ? SESSIONS_DIR
    : path.join(process.cwd(), SESSIONS_DIR);
  return path.join(sessionDir, `${sessionId}.attachments`);
}

function attachmentsMetaFile(sessionId: string): string {
  return `${attachmentsDirFor(sessionId)}.jsonl`;
}

/** Persist attachment bytes to the session-scoped cache dir. */
export function writeAttachment(
  sessionId: string,
  id: string,
  bytes: Uint8Array,
): void {
  const dir = attachmentsDirFor(sessionId);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
  writeFileSync(path.join(dir, id), bytes);
}

/** Read cached attachment bytes, or null if missing. */
export function readAttachment(sessionId: string, id: string): Uint8Array | null {
  const filePath = path.join(attachmentsDirFor(sessionId), id);
  if (!existsSync(filePath)) return null;
  try {
    return readFileSync(filePath);
  } catch {
    return null;
  }
}

/** Append audit metadata for cached attachments (one JSON line each). */
export function appendAttachmentMeta(
  sessionId: string,
  entries: AttachmentMeta[],
): void {
  if (entries.length === 0) return;
  const content = entries.map((e) => JSON.stringify(e)).join("\n") + "\n";
  appendFileSync(attachmentsMetaFile(sessionId), content);
}

/** Remove the attachment cache dir + meta sidecar for a session. */
function deleteAttachments(sessionId: string): void {
  const dir = attachmentsDirFor(sessionId);
  if (existsSync(dir)) {
    rmSync(dir, { recursive: true, force: true });
  }
  const metaPath = attachmentsMetaFile(sessionId);
  if (existsSync(metaPath)) {
    unlinkSync(metaPath);
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
    // Only consider primary conversation files; skip every sidecar variant.
    if (
      !file.endsWith(".jsonl") ||
      file.endsWith(".meta.jsonl") ||
      file.endsWith(".provenance.jsonl") ||
      file.endsWith(".annotations.jsonl") ||
      file.endsWith(".attachments.jsonl")
    )
      continue;
    const filePath = path.join(sessionDir, file);
    try {
      const { mtimeMs } = statSync(filePath);
      if (now - mtimeMs > SESSION_MAX_AGE_MS) {
        const sessionId = file.replace(/\.jsonl$/, "");
        unlinkSync(filePath);
        const metaPath = filePath.replace(/\.jsonl$/, ".meta.jsonl");
        if (existsSync(metaPath)) unlinkSync(metaPath);
        const provPath = filePath.replace(/\.jsonl$/, ".provenance.jsonl");
        if (existsSync(provPath)) unlinkSync(provPath);
        const annPath = filePath.replace(/\.jsonl$/, ".annotations.jsonl");
        if (existsSync(annPath)) unlinkSync(annPath);
        const configPath = filePath.replace(/\.jsonl$/, ".config.json");
        if (existsSync(configPath)) unlinkSync(configPath);
        deleteAttachments(sessionId);
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
