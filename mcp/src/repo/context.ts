import { existsSync, readFileSync, writeFileSync } from "fs";
import {
  generateText,
  jsonSchema,
  LanguageModel,
  ModelMessage,
  Output,
} from "ai";
import { getSessionSidecarFile } from "./session.js";
import {
  deepParseJsonStrings,
  ensureAdditionalPropertiesFalse,
  estimateMessagesTokens,
} from "./utils.js";

export type SessionContextRefKind =
  | "file"
  | "function"
  | "endpoint"
  | "env"
  | "command"
  | "url"
  | "ref_id"
  | "other";

export interface SessionContextRef {
  kind: SessionContextRefKind;
  value: string;
  reason: string;
}

export interface SessionContextState {
  summary: string;
  goals: string[];
  decisions: string[];
  importantRefs: SessionContextRef[];
  checked: string[];
  openQuestions: string[];
  nextSteps: string[];
  warnings: string[];
  updated_at: string;
}

export interface PreparedSessionMessages {
  messages: ModelMessage[];
  usedOverflowSummary: boolean;
}

const CONTEXT_SUFFIX = ".context.json";
const DEFAULT_OVERFLOW_TOKEN_THRESHOLD = 50_000;
const DEFAULT_RECENT_MESSAGES = 4;
const MAX_PART_CHARS = 4_000;
const MAX_TOOL_OUTPUT_CHARS = 2_000;

const SESSION_CONTEXT_SCHEMA = {
  type: "object",
  properties: {
    summary: { type: "string" },
    goals: { type: "array", items: { type: "string" } },
    decisions: { type: "array", items: { type: "string" } },
    importantRefs: {
      type: "array",
      items: {
        type: "object",
        properties: {
          kind: {
            type: "string",
            enum: ["file", "function", "endpoint", "env", "command", "url", "ref_id", "other"],
          },
          value: { type: "string" },
          reason: { type: "string" },
        },
        required: ["kind", "value", "reason"],
      },
    },
    checked: { type: "array", items: { type: "string" } },
    openQuestions: { type: "array", items: { type: "string" } },
    nextSteps: { type: "array", items: { type: "string" } },
    warnings: { type: "array", items: { type: "string" } },
    updated_at: { type: "string" },
  },
  required: [
    "summary",
    "goals",
    "decisions",
    "importantRefs",
    "checked",
    "openQuestions",
    "nextSteps",
    "warnings",
    "updated_at",
  ],
};

function parsePositiveInt(value: string | undefined, fallback: number): number {
  const parsed = Number.parseInt(value || "", 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function overflowTokenThreshold(): number {
  return parsePositiveInt(
    process.env.SESSION_CONTEXT_TOKEN_THRESHOLD,
    DEFAULT_OVERFLOW_TOKEN_THRESHOLD,
  );
}

function recentMessageCount(): number {
  return parsePositiveInt(
    process.env.SESSION_CONTEXT_RECENT_MESSAGES,
    DEFAULT_RECENT_MESSAGES,
  );
}

function emptySessionContextState(): SessionContextState {
  return {
    summary: "",
    goals: [],
    decisions: [],
    importantRefs: [],
    checked: [],
    openQuestions: [],
    nextSteps: [],
    warnings: [],
    updated_at: "",
  };
}

function compactText(text: string, maxChars: number = MAX_PART_CHARS): string {
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars)}\n[TRUNCATED ${text.length - maxChars} chars]`;
}

function stringifyCompact(value: unknown, maxChars: number = MAX_PART_CHARS): string {
  if (typeof value === "string") return compactText(value, maxChars);
  try {
    return compactText(JSON.stringify(value), maxChars);
  } catch {
    return compactText(String(value), maxChars);
  }
}

function normalizeStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    .map((item) => item.trim());
}

function normalizeImportantRefs(value: unknown): SessionContextRef[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item): SessionContextRef[] => {
    if (!item || typeof item !== "object") return [];
    const ref = item as Partial<SessionContextRef>;
    if (!ref.value || typeof ref.value !== "string") return [];
    const kind = ref.kind && ["file", "function", "endpoint", "env", "command", "url", "ref_id", "other"].includes(ref.kind)
      ? ref.kind
      : "other";
    return [{
      kind,
      value: ref.value.trim(),
      reason: typeof ref.reason === "string" ? ref.reason.trim() : "",
    }];
  });
}

function normalizeSessionContextState(value: unknown): SessionContextState {
  if (!value || typeof value !== "object") return emptySessionContextState();
  const state = value as Partial<SessionContextState>;
  return {
    summary: typeof state.summary === "string" ? state.summary.trim() : "",
    goals: normalizeStringArray(state.goals),
    decisions: normalizeStringArray(state.decisions),
    importantRefs: normalizeImportantRefs(state.importantRefs),
    checked: normalizeStringArray(state.checked),
    openQuestions: normalizeStringArray(state.openQuestions),
    nextSteps: normalizeStringArray(state.nextSteps),
    warnings: normalizeStringArray(state.warnings),
    updated_at: typeof state.updated_at === "string" ? state.updated_at : "",
  };
}

export function loadSessionContextState(sessionId: string): SessionContextState | undefined {
  const filePath = getSessionSidecarFile(sessionId, CONTEXT_SUFFIX);
  if (!existsSync(filePath)) return undefined;
  try {
    return normalizeSessionContextState(JSON.parse(readFileSync(filePath, "utf-8")));
  } catch {
    return undefined;
  }
}

function saveSessionContextState(sessionId: string, state: SessionContextState): void {
  const filePath = getSessionSidecarFile(sessionId, CONTEXT_SUFFIX);
  writeFileSync(filePath, JSON.stringify(normalizeSessionContextState(state), null, 2) + "\n");
}

function renderList(title: string, values: string[]): string[] {
  if (values.length === 0) return [];
  return [title, ...values.map((value) => `- ${value}`)];
}

function renderSessionContextState(state: SessionContextState): string {
  const normalized = normalizeSessionContextState(state);
  const lines: string[] = [];
  if (normalized.summary) {
    lines.push("Summary", normalized.summary);
  }
  lines.push(...renderList("Goals", normalized.goals));
  lines.push(...renderList("Decisions", normalized.decisions));
  if (normalized.importantRefs.length > 0) {
    lines.push("Important refs");
    for (const ref of normalized.importantRefs) {
      const reason = ref.reason ? ` - ${ref.reason}` : "";
      lines.push(`- ${ref.kind}: ${ref.value}${reason}`);
    }
  }
  lines.push(...renderList("Already checked", normalized.checked));
  lines.push(...renderList("Open questions", normalized.openQuestions));
  lines.push(...renderList("Next steps", normalized.nextSteps));
  lines.push(...renderList("Warnings", normalized.warnings));
  return lines.join("\n").trim();
}

function summarizeOutput(output: unknown): string {
  if (!output || typeof output !== "object") return stringifyCompact(output, MAX_TOOL_OUTPUT_CHARS);
  const item = output as Record<string, unknown>;
  if (item.type === "text") return stringifyCompact(item.value ?? "", MAX_TOOL_OUTPUT_CHARS);
  if (item.type === "json") return stringifyCompact(item.value ?? {}, MAX_TOOL_OUTPUT_CHARS);
  return stringifyCompact(output, MAX_TOOL_OUTPUT_CHARS);
}

function renderMessageForSummary(message: ModelMessage): string[] {
  const lines: string[] = [];
  const role = String(message.role).toUpperCase();
  const content = (message as { content?: unknown }).content;
  if (typeof content === "string") {
    const text = content.trim();
    if (text) lines.push(`${role}:\n${compactText(text)}`);
    return lines;
  }
  if (!Array.isArray(content)) return lines;
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const item = part as Record<string, unknown>;
    if (item.type === "text" && typeof item.text === "string" && item.text.trim()) {
      lines.push(`${role} TEXT:\n${compactText(item.text.trim())}`);
    } else if (item.type === "tool-call") {
      lines.push(`TOOL CALL ${String(item.toolName ?? "unknown")}: ${stringifyCompact(item.input ?? {})}`);
    } else if (item.type === "tool-result") {
      lines.push(`TOOL RESULT ${String(item.toolName ?? "unknown")}: ${summarizeOutput(item.output)}`);
    }
  }
  return lines;
}

function summarizeMessagesForPrompt(messages: ModelMessage[]): string {
  return messages
    .flatMap(renderMessageForSummary)
    .filter(Boolean)
    .join("\n\n")
    .trim();
}

function buildUpdatePrompt(existing: SessionContextState, newMessages: string): string {
  return `You maintain compact working context for a tool-using code agent.

Update durable working memory so the next agent call can continue without replaying the full session.

Preserve:
- The user's current goal and constraints
- Decisions already made
- Exact file paths, function names, class names, endpoint paths, env vars, commands, package names, ports, URLs, IDs, and ref_ids
- Tool results that changed the agent's understanding
- Things already checked, especially dead ends that should not be repeated
- Errors, blockers, failed assumptions, and unresolved questions
- The latest concrete output or conclusion

Drop:
- Raw tool output unless a short exact excerpt is necessary
- Repeated assistant wording
- Conversational filler
- Large code blocks
- Logs that do not affect future work

Rules:
- Prefer exact names over paraphrases.
- If unsure whether a reference matters, keep it briefly.
- Never invent facts.
- Keep the result compact.
- Update the existing memory; do not duplicate old items.

EXISTING CONTEXT STATE:
${JSON.stringify(existing, null, 2)}

NEW SESSION MESSAGES:
${newMessages}`;
}

export async function updateSessionOverflowContext(
  sessionId: string,
  messages: ModelMessage[],
  model: LanguageModel,
): Promise<SessionContextState | undefined> {
  const newMessages = summarizeMessagesForPrompt(messages);
  if (!newMessages) return loadSessionContextState(sessionId);
  try {
    const existing = loadSessionContextState(sessionId) ?? emptySessionContextState();
    const result = await generateText({
      model,
      prompt: buildUpdatePrompt(existing, newMessages),
      output: Output.object({ schema: jsonSchema(ensureAdditionalPropertiesFalse(SESSION_CONTEXT_SCHEMA)) }),
    });
    const nextState = {
      ...normalizeSessionContextState(deepParseJsonStrings(result.output ?? {})),
      updated_at: new Date().toISOString(),
    };
    saveSessionContextState(sessionId, nextState);
    return nextState;
  } catch (error) {
    console.error("[context] Failed to update session overflow context:", error);
    return undefined;
  }
}

export async function prepareSessionMessagesWithOverflowSummary(
  sessionId: string,
  messages: ModelMessage[],
  model: LanguageModel,
): Promise<PreparedSessionMessages> {
  const threshold = overflowTokenThreshold();
  const tokens = estimateMessagesTokens(messages);
  if (tokens < threshold) {
    return { messages, usedOverflowSummary: false };
  }

  const tailCount = recentMessageCount();
  const recentMessages = messages.slice(-tailCount);
  let contextState = loadSessionContextState(sessionId);
  if (!contextState) {
    const olderMessages = messages.slice(0, Math.max(messages.length - tailCount, 0));
    contextState = await updateSessionOverflowContext(sessionId, olderMessages, model);
  }

  if (!contextState) {
    return { messages, usedOverflowSummary: false };
  }

  const renderedContext = renderSessionContextState(contextState);
  if (!renderedContext) {
    return { messages, usedOverflowSummary: false };
  }

  return {
    messages: [
      {
        role: "user",
        content:
          "Earlier session context summarized after the transcript exceeded the replay threshold. Use it for continuity, but treat the current user request as highest priority.\n\n" +
          renderedContext,
      },
      ...recentMessages,
    ],
    usedOverflowSummary: true,
  };
}