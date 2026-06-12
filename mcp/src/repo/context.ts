import { existsSync, readFileSync, writeFileSync } from "fs";
import {
  generateText,
  jsonSchema,
  LanguageModel,
  ModelMessage,
  Output,
} from "ai";
import { getProviderOptions } from "../aieo/src/index.js";
import { getSessionSidecarFile, loadSessionMessages } from "./session.js";
import {
  buildRestorableStub,
  buildToolInputMap,
  deepParseJsonStrings,
  ensureAdditionalPropertiesFalse,
  estimateMessagesTokens,
  getTokenizer,
  CLEARED_PREFIX,
} from "./utils.js";

// ── Thresholds ───────────────────────────────────────────────────────
// < STUB: do nothing. >= STUB: stub old tool results at load (free).
// >= COMPACT (post-turn): compact old turns into the sidecar (one cheap call).
// The 90% mid-flight truncation in utils.ts remains as the emergency brake.
// Env overrides (CONTEXT_STUB_PCT / CONTEXT_COMPACT_PCT) exist for benchmarking.
function pctEnv(name: string, fallback: number): number {
  const v = parseFloat(process.env[name] || "");
  return Number.isFinite(v) && v > 0 && v < 1 ? v : fallback;
}
const STUB_THRESHOLD_PCT = pctEnv("CONTEXT_STUB_PCT", 0.35);
const COMPACT_THRESHOLD_PCT = pctEnv("CONTEXT_COMPACT_PCT", 0.5);
const KEEP_RECENT_TURNS = 2;
const MAX_STATE_ITEMS = 15;
const MAX_PART_CHARS = 4_000;
const MAX_TOOL_OUTPUT_CHARS = 2_000;

const COMPACT_SUFFIX = ".compact.json";

export function isContextManagementEnabled(flag?: boolean): boolean {
  if (typeof flag === "boolean") return flag;
  const env = process.env.CONTEXT_MANAGEMENT;
  return env === "1" || env === "true";
}

// ── Compact state ────────────────────────────────────────────────────

export type ContextRefKind =
  | "file"
  | "function"
  | "endpoint"
  | "env"
  | "command"
  | "url"
  | "ref_id"
  | "other";

export interface ContextRef {
  kind: ContextRefKind;
  value: string;
  reason: string;
}

export interface SessionContextState {
  summary: string;
  goals: string[];
  decisions: string[];
  importantRefs: ContextRef[];
  checked: string[];
  openQuestions: string[];
  nextSteps: string[];
  warnings: string[];
}

export interface CompactState {
  /** Index into loadSessionMessages() — everything <= this is covered by `state`. */
  compactedThroughIndex: number;
  state: SessionContextState;
  tokensBefore: number;
  tokensAfter: number;
  updated_at: string;
}

const STATE_SCHEMA = {
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
  ],
};

function emptyState(): SessionContextState {
  return {
    summary: "",
    goals: [],
    decisions: [],
    importantRefs: [],
    checked: [],
    openQuestions: [],
    nextSteps: [],
    warnings: [],
  };
}

function capStrings(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((v): v is string => typeof v === "string" && v.trim().length > 0)
    .map((v) => v.trim())
    .slice(0, MAX_STATE_ITEMS);
}

function capRefs(value: unknown): ContextRef[] {
  if (!Array.isArray(value)) return [];
  const kinds = ["file", "function", "endpoint", "env", "command", "url", "ref_id", "other"];
  return value
    .flatMap((item): ContextRef[] => {
      if (!item || typeof item !== "object") return [];
      const ref = item as Partial<ContextRef>;
      if (!ref.value || typeof ref.value !== "string") return [];
      return [
        {
          kind: kinds.includes(ref.kind as string) ? (ref.kind as ContextRefKind) : "other",
          value: ref.value.trim(),
          reason: typeof ref.reason === "string" ? ref.reason.trim() : "",
        },
      ];
    })
    .slice(0, MAX_STATE_ITEMS * 2);
}

function normalizeState(value: unknown): SessionContextState {
  if (!value || typeof value !== "object") return emptyState();
  const s = value as Partial<SessionContextState>;
  return {
    summary: typeof s.summary === "string" ? s.summary.trim() : "",
    goals: capStrings(s.goals),
    decisions: capStrings(s.decisions),
    importantRefs: capRefs(s.importantRefs),
    checked: capStrings(s.checked),
    openQuestions: capStrings(s.openQuestions),
    nextSteps: capStrings(s.nextSteps),
    warnings: capStrings(s.warnings),
  };
}

export function loadCompactState(sessionId: string): CompactState | undefined {
  const filePath = getSessionSidecarFile(sessionId, COMPACT_SUFFIX);
  if (!existsSync(filePath)) return undefined;
  try {
    const raw = JSON.parse(readFileSync(filePath, "utf-8"));
    if (typeof raw?.compactedThroughIndex !== "number") return undefined;
    return {
      compactedThroughIndex: raw.compactedThroughIndex,
      state: normalizeState(raw.state),
      tokensBefore: typeof raw.tokensBefore === "number" ? raw.tokensBefore : 0,
      tokensAfter: typeof raw.tokensAfter === "number" ? raw.tokensAfter : 0,
      updated_at: typeof raw.updated_at === "string" ? raw.updated_at : "",
    };
  } catch {
    return undefined;
  }
}

function saveCompactState(sessionId: string, state: CompactState): void {
  const filePath = getSessionSidecarFile(sessionId, COMPACT_SUFFIX);
  writeFileSync(filePath, JSON.stringify(state, null, 2) + "\n");
}

// ── Layer 1: load-time stubbing (zero tokens) ────────────────────────

/** Indexes of user messages, which mark turn boundaries. */
function userMessageIndexes(messages: ModelMessage[]): number[] {
  const idxs: number[] = [];
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role === "user") idxs.push(i);
  }
  return idxs;
}

/**
 * First message index of the most recent `keepTurns` turns.
 * Messages before this index are eligible for stubbing/compaction.
 */
function recentBoundary(messages: ModelMessage[], keepTurns: number): number {
  const userIdxs = userMessageIndexes(messages);
  if (userIdxs.length <= keepTurns) return 0;
  return userIdxs[userIdxs.length - keepTurns];
}

function isErrorOutput(part: any): boolean {
  if (part?.output?.type === "error-text" || part?.output?.type === "error-json") return true;
  if (part?.output?.type === "text" && typeof part.output.value === "string") {
    return /^error[:\s]/i.test(part.output.value);
  }
  return false;
}

/**
 * Replace tool-result outputs in turns older than the last KEEP_RECENT_TURNS
 * with restorable stubs. Deterministic and structure-preserving: tool-call /
 * tool-result pairing is untouched, only output values change, so the result
 * is byte-stable across requests (cache-friendly).
 *
 * No-op when history is under STUB_THRESHOLD_PCT of the context limit.
 */
export function stubOldToolResults(
  messages: ModelMessage[],
  contextLimit: number
): ModelMessage[] {
  const estimated = estimateMessagesTokens(messages);
  if (estimated < contextLimit * STUB_THRESHOLD_PCT) return messages;

  const boundary = recentBoundary(messages, KEEP_RECENT_TURNS);
  if (boundary === 0) return messages;

  const inputMap = buildToolInputMap(messages);
  const result = [...messages];
  let stubbed = 0;

  for (let i = 0; i < boundary; i++) {
    const msg = messages[i] as any;
    if (msg.role !== "tool" || !Array.isArray(msg.content)) continue;
    let copied = false;
    for (let j = 0; j < msg.content.length; j++) {
      const part = msg.content[j];
      if (part.type !== "tool-result") continue;
      if (isErrorOutput(part)) continue; // keep the wrong stuff in
      const value = part.output?.type === "text" ? part.output.value : undefined;
      if (typeof value === "string" && value.startsWith(CLEARED_PREFIX)) continue;
      if (!copied) {
        result[i] = { ...msg, content: [...msg.content] } as ModelMessage;
        copied = true;
      }
      (result[i] as any).content[j] = {
        ...part,
        output: {
          type: "text" as const,
          value: buildRestorableStub(part.toolName ?? "unknown", inputMap.get(part.toolCallId)),
        },
      };
      stubbed++;
    }
  }

  if (stubbed > 0) {
    console.log(
      `[context] stubbed ${stubbed} old tool result(s) at load (estimated=${estimated}, threshold=${Math.round(contextLimit * STUB_THRESHOLD_PCT)})`
    );
  }
  return stubbed > 0 ? result : messages;
}

// ── Layer 2b: load-time assembly ─────────────────────────────────────

function renderState(state: SessionContextState): string {
  const lines: string[] = [];
  const list = (title: string, values: string[]) => {
    if (values.length > 0) lines.push(title, ...values.map((v) => `- ${v}`));
  };
  if (state.summary) lines.push("Summary", state.summary);
  list("Goals", state.goals);
  list("Decisions", state.decisions);
  if (state.importantRefs.length > 0) {
    lines.push("Important refs");
    for (const ref of state.importantRefs) {
      lines.push(`- ${ref.kind}: ${ref.value}${ref.reason ? ` — ${ref.reason}` : ""}`);
    }
  }
  list("Already checked", state.checked);
  list("Open questions", state.openQuestions);
  list("Next steps", state.nextSteps);
  list("Warnings", state.warnings);
  return lines.join("\n").trim();
}

/**
 * Skeleton of compacted history: user messages and assistant text survive
 * verbatim (users refer back to them); tool calls/results are dropped whole —
 * their distilled content lives in the compact state.
 */
function extractSkeleton(messages: ModelMessage[]): ModelMessage[] {
  const skeleton: ModelMessage[] = [];
  for (const msg of messages) {
    if (msg.role === "user") {
      skeleton.push(msg);
      continue;
    }
    if (msg.role === "assistant") {
      if (typeof msg.content === "string") {
        if (msg.content.trim()) skeleton.push(msg);
        continue;
      }
      if (Array.isArray(msg.content)) {
        const textParts = (msg.content as any[]).filter(
          (p) => p?.type === "text" && typeof p.text === "string" && p.text.trim()
        );
        if (textParts.length > 0) {
          skeleton.push({ ...msg, content: textParts } as ModelMessage);
        }
      }
    }
    // tool messages dropped
  }
  return skeleton;
}

/**
 * Load session messages with context management applied:
 * - If a compact state exists, replace the compacted segment with
 *   [session memory message + user/assistant-text skeleton].
 * - Stub old tool results in whatever remains (layer 1).
 *
 * The session JSONL on disk is never modified.
 */
export async function loadContextManagedMessages(
  sessionId: string,
  contextLimit: number
): Promise<ModelMessage[]> {
  await getTokenizer().catch(() => {});
  const all = loadSessionMessages(sessionId);
  const compact = loadCompactState(sessionId);

  if (!compact || compact.compactedThroughIndex < 0) {
    return stubOldToolResults(all, contextLimit);
  }

  const splitAt = Math.min(compact.compactedThroughIndex + 1, all.length);
  const old = all.slice(0, splitAt);
  const recent = all.slice(splitAt);

  const rendered = renderState(compact.state);
  const memoryMessage: ModelMessage | undefined = rendered
    ? {
        role: "user",
        content:
          "[Session memory — compacted history]\nEarlier tool activity in this session was compacted. Key facts:\n\n" +
          rendered +
          "\n\nUse graph/bash tools to re-fetch any referenced item if needed.",
      }
    : undefined;

  const assembled = [
    ...(memoryMessage ? [memoryMessage] : []),
    ...extractSkeleton(old),
    ...recent,
  ];
  console.log(
    `[context] compacted load: ${all.length} msgs -> ${assembled.length} (compactedThroughIndex=${compact.compactedThroughIndex})`
  );
  return stubOldToolResults(assembled, contextLimit);
}

// ── Layer 2: post-turn compaction (cheap model, off the hot path) ────

function compactText(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars)}\n[TRUNCATED ${text.length - maxChars} chars]`;
}

function stringifyCompact(value: unknown, maxChars: number): string {
  if (typeof value === "string") return compactText(value, maxChars);
  try {
    return compactText(JSON.stringify(value), maxChars);
  } catch {
    return compactText(String(value), maxChars);
  }
}

function renderMessageForSummary(message: ModelMessage): string[] {
  const lines: string[] = [];
  const role = String(message.role).toUpperCase();
  const content = (message as { content?: unknown }).content;
  if (typeof content === "string") {
    if (content.trim()) lines.push(`${role}:\n${compactText(content.trim(), MAX_PART_CHARS)}`);
    return lines;
  }
  if (!Array.isArray(content)) return lines;
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const item = part as Record<string, unknown>;
    if (item.type === "text" && typeof item.text === "string" && item.text.trim()) {
      lines.push(`${role} TEXT:\n${compactText(item.text.trim(), MAX_PART_CHARS)}`);
    } else if (item.type === "tool-call") {
      lines.push(
        `TOOL CALL ${String(item.toolName ?? "unknown")}: ${stringifyCompact(item.input ?? {}, MAX_PART_CHARS)}`
      );
    } else if (item.type === "tool-result") {
      const output = item.output as Record<string, unknown> | undefined;
      const value =
        output?.type === "text" || output?.type === "json" ? output.value : output;
      lines.push(
        `TOOL RESULT ${String(item.toolName ?? "unknown")}: ${stringifyCompact(value ?? "", MAX_TOOL_OUTPUT_CHARS)}`
      );
    }
  }
  return lines;
}

function buildCompactionPrompt(existing: SessionContextState, rendered: string): string {
  return `You maintain compact working memory for a tool-using code agent.

Update the working memory so the next agent call can continue without replaying the full session.

Preserve:
- The user's current goal and constraints
- Decisions already made
- Exact file paths, function names, class names, endpoint paths, env vars, commands, package names, ports, URLs, IDs, and ref_ids
- Tool results that changed the agent's understanding
- Things already checked, especially dead ends that should not be repeated
- Errors, blockers, failed assumptions, and unresolved questions

Drop:
- Raw tool output unless a short exact excerpt is necessary
- Conversational filler and repeated wording
- Large code blocks and logs that do not affect future work

Rules:
- Prefer exact names over paraphrases. Never invent facts.
- Update the existing memory in place; merge duplicates, evict stale items.
- Hard limit: at most ${MAX_STATE_ITEMS} items per list. Keep it compact.

EXISTING MEMORY:
${JSON.stringify(existing, null, 2)}

NEW SESSION MESSAGES TO FOLD IN:
${rendered}`;
}

/**
 * Compact the session's older turns into the sidecar state if the (stubbed)
 * history exceeds COMPACT_THRESHOLD_PCT of the context limit.
 *
 * Incremental: only messages after the previous compactedThroughIndex and
 * before the recent-turns boundary are summarized. Intended to be called
 * post-turn, fire-and-forget, with fast-tier provider options.
 */
export async function maybeCompactSession(
  sessionId: string,
  model: LanguageModel,
  provider: string,
  modelId: string,
  contextLimit: number
): Promise<void> {
  await getTokenizer().catch(() => {});
  const all = loadSessionMessages(sessionId);
  if (all.length === 0) return;

  const prior = loadCompactState(sessionId);
  const alreadyCompacted = prior ? prior.compactedThroughIndex + 1 : 0;

  // Measure what the next turn would actually send (after stubbing).
  const wouldSend = stubOldToolResults(all.slice(alreadyCompacted), contextLimit);
  const tokensBefore = estimateMessagesTokens(wouldSend);
  if (tokensBefore < contextLimit * COMPACT_THRESHOLD_PCT) return;

  const boundary = recentBoundary(all, KEEP_RECENT_TURNS);
  if (boundary <= alreadyCompacted) return; // nothing new to compact

  const toCompact = all.slice(alreadyCompacted, boundary);
  const rendered = toCompact.flatMap(renderMessageForSummary).filter(Boolean).join("\n\n").trim();
  if (!rendered) return;

  const existing = prior?.state ?? emptyState();
  const { output } = await generateText({
    model,
    prompt: buildCompactionPrompt(existing, rendered),
    providerOptions: getProviderOptions(provider as any, "fast", modelId) as any,
    output: Output.object({
      schema: jsonSchema(ensureAdditionalPropertiesFalse(STATE_SCHEMA)),
    }),
  });

  const nextState = normalizeState(deepParseJsonStrings(output ?? {}));
  const next: CompactState = {
    compactedThroughIndex: boundary - 1,
    state: nextState,
    tokensBefore,
    tokensAfter: estimateMessagesTokens([
      { role: "user", content: renderState(nextState) },
      ...extractSkeleton(toCompact),
      ...all.slice(boundary),
    ]),
    updated_at: new Date().toISOString(),
  };
  saveCompactState(sessionId, next);
  console.log(
    `[context] compacted session ${sessionId}: through index ${next.compactedThroughIndex}, tokens ${tokensBefore} -> ~${next.tokensAfter}`
  );
}
