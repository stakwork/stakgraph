import { existsSync, readFileSync, writeFileSync } from "fs";
import {
  generateText,
  jsonSchema,
  LanguageModel,
  ModelMessage,
  Output,
} from "ai";
import {
  appendContextTimelineEntry,
  getSessionSidecarFile,
  loadSession,
  normalizeTokenUsage,
  type TokenUsageSummary,
} from "./session.js";
import {
  deepParseJsonStrings,
  ensureAdditionalPropertiesFalse,
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

const CONTEXT_SUFFIX = ".context.json";
const RECENT_CONVERSATION_MESSAGES = 4;
const MAX_TOOL_OUTPUT_CHARS = 2000;
const MAX_PART_CHARS = 4000;

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
  return value.filter((item) => typeof item === "string" && item.trim()).map((item) => item.trim());
}

function normalizeImportantRefs(value: unknown): SessionContextRef[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
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

function loadSessionContextState(sessionId: string): SessionContextState {
  const filePath = getSessionSidecarFile(sessionId, CONTEXT_SUFFIX);
  if (!existsSync(filePath)) return emptySessionContextState();
  try {
    return normalizeSessionContextState(JSON.parse(readFileSync(filePath, "utf-8")));
  } catch {
    return emptySessionContextState();
  }
}

function saveSessionContextState(sessionId: string, state: SessionContextState): void {
  const filePath = getSessionSidecarFile(sessionId, CONTEXT_SUFFIX);
  writeFileSync(filePath, JSON.stringify(normalizeSessionContextState(state), null, 2) + "\n");
}

function textFromContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .flatMap((part) => {
      if (!part || typeof part !== "object") return [];
      const item = part as any;
      if (item.type === "text" && typeof item.text === "string") return [item.text];
      return [];
    })
    .join("\n");
}

function textOnlyMessage(message: ModelMessage): ModelMessage | null {
  if (message.role !== "user" && message.role !== "assistant") return null;
  const text = textFromContent(message.content).trim();
  if (!text) return null;
  return { role: message.role, content: text } as ModelMessage;
}

function loadRecentConversationMessages(
  sessionId: string,
  limit: number = RECENT_CONVERSATION_MESSAGES,
): ModelMessage[] {
  return loadSession(sessionId)
    .flatMap((message) => {
      const textMessage = textOnlyMessage(message);
      return textMessage ? [textMessage] : [];
    })
    .slice(-limit);
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

const CONTEXT_DIFF_FIELDS = [
  "goals",
  "decisions",
  "importantRefs",
  "checked",
  "openQuestions",
  "nextSteps",
  "warnings",
] as const;

function contextItemKey(value: unknown): string {
  if (typeof value === "string") return value;
  if (value && typeof value === "object") {
    const item = value as Record<string, unknown>;
    return [item.kind, item.value, item.reason]
      .filter((part) => part !== undefined && part !== null)
      .map(String)
      .join("\u0000");
  }
  return String(value);
}

function diffContextState(before: SessionContextState, after: SessionContextState) {
  const added: Record<string, unknown[]> = {};
  const removed: Record<string, unknown[]> = {};

  for (const field of CONTEXT_DIFF_FIELDS) {
    const beforeValues = before[field] as unknown[];
    const afterValues = after[field] as unknown[];
    const beforeKeys = new Set(beforeValues.map(contextItemKey));
    const afterKeys = new Set(afterValues.map(contextItemKey));
    added[field] = afterValues.filter((value) => !beforeKeys.has(contextItemKey(value)));
    removed[field] = beforeValues.filter((value) => !afterKeys.has(contextItemKey(value)));
  }

  return { added, removed };
}

export function loadCompiledSessionMessages(sessionId: string): ModelMessage[] {
  const renderedContext = renderSessionContextState(loadSessionContextState(sessionId));
  const messages: ModelMessage[] = [];
  if (renderedContext) {
    messages.push({
      role: "user",
      content:
        "Previous session context compiled from earlier turns. Use it for continuity, but treat the current user request as highest priority.\n\n" +
        renderedContext,
    });
  }
  messages.push(...loadRecentConversationMessages(sessionId));
  return messages;
}

function summarizeToolOutput(output: unknown): string {
  if (!output || typeof output !== "object") return stringifyCompact(output, MAX_TOOL_OUTPUT_CHARS);
  const item = output as any;
  if (item.type === "text") return stringifyCompact(item.value ?? "", MAX_TOOL_OUTPUT_CHARS);
  if (item.type === "json") return stringifyCompact(item.value ?? {}, MAX_TOOL_OUTPUT_CHARS);
  return stringifyCompact(output, MAX_TOOL_OUTPUT_CHARS);
}

function renderMessageForSummary(message: ModelMessage): string[] {
  const lines: string[] = [];
  const role = String(message.role).toUpperCase();
  if (typeof message.content === "string") {
    const text = message.content.trim();
    if (text) lines.push(`${role}:\n${compactText(text)}`);
    return lines;
  }
  if (!Array.isArray(message.content)) return lines;
  for (const part of message.content as any[]) {
    if (!part || typeof part !== "object") continue;
    if (part.type === "text" && typeof part.text === "string" && part.text.trim()) {
      lines.push(`${role} TEXT:\n${compactText(part.text.trim())}`);
    } else if (part.type === "tool-call") {
      lines.push(`TOOL CALL ${part.toolName ?? "unknown"}: ${stringifyCompact(part.input ?? {})}`);
    } else if (part.type === "tool-result") {
      lines.push(`TOOL RESULT ${part.toolName ?? "unknown"}: ${summarizeToolOutput(part.output)}`);
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

function buildUpdatePrompt(existing: SessionContextState, newTurn: string): string {
  return `You maintain compact working context for a tool-using code agent.

Your job is not to summarize prose. Your job is to update durable working memory so the next agent call can continue without replaying the full session.

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

NEW TURN MESSAGES:
${newTurn}`;
}

export async function updateSessionContext(
  sessionId: string,
  newMessages: ModelMessage[],
  model: LanguageModel,
  turn: number = 0,
): Promise<TokenUsageSummary | undefined> {
  const newTurn = summarizeMessagesForPrompt(newMessages);
  if (!newTurn) return undefined;
  try {
    const existing = loadSessionContextState(sessionId);
    const normalizedSchema = ensureAdditionalPropertiesFalse(SESSION_CONTEXT_SCHEMA);
    const result = await generateText({
      model,
      prompt: buildUpdatePrompt(existing, newTurn),
      output: Output.object({ schema: jsonSchema(normalizedSchema) }),
    });
    const parsed = deepParseJsonStrings(result.output ?? {});
    const timestamp = new Date().toISOString();
    const nextState = {
      ...normalizeSessionContextState(parsed),
      updated_at: timestamp,
    };
    const usage = normalizeTokenUsage(result.usage);
    saveSessionContextState(sessionId, nextState);
    appendContextTimelineEntry(sessionId, {
      turn,
      timestamp,
      before: existing,
      after: nextState,
      usage,
      diff: diffContextState(existing, nextState),
      changedSummary: existing.summary !== nextState.summary,
      newMessagesPreview: compactText(newTurn, 2000),
    });
    return usage;
  } catch (e) {
    console.error("[context] Failed to update session context:", e);
    return undefined;
  }
}