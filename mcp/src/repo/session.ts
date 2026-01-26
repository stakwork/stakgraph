import { ModelMessage } from "ai";
import {
  existsSync,
  mkdirSync,
  appendFileSync,
  readFileSync,
  unlinkSync,
} from "fs";
import { randomUUID } from "crypto";
import path from "path";

const SESSIONS_DIR = process.env.SESSIONS_DIR || ".sessions";

export interface Session {
  id: string;
  messages: ModelMessage[];
}

export interface SessionConfig {
  truncateToolResults?: boolean; // Enable truncation (default: false)
  maxToolResultLines?: number; // Default: 50
  maxToolResultChars?: number; // Default: 2000
}

/**
 * Get the file path for a session
 */
function getSessionFile(sessionId: string): string {
  const sessionDir = path.join(process.cwd(), SESSIONS_DIR);
  if (!existsSync(sessionDir)) {
    mkdirSync(sessionDir, { recursive: true });
  }
  return path.join(sessionDir, `${sessionId}.jsonl`);
}

/**
 * Create a new session and return its ID.
 * If an ID is provided, use it; otherwise generate a random UUID.
 */
export function createSession(id?: string): string {
  const sessionId = id || randomUUID();
  const filePath = getSessionFile(sessionId);
  appendFileSync(filePath, "");
  return sessionId;
}

/**
 * Load all messages from a session
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
