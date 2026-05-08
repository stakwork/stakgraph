import { useState, useEffect, useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { api } from "../api";
import type {
  ProductionRun,
  StepMeta,
  SearchProvenanceEntry,
  TokenUsage,
} from "../types";
import { AnnotationBadge, AnnotationForm } from "./Annotations";
import type { Annotation, AnnotationMarker } from "./Annotations";
import {
  formatNumber,
  formatDuration,
  formatSourceLabel,
  getRangeStart,
} from "../utils";
import { shortId, CopyableBlock } from "./ui";
import {
  parseSearchToolResultRows,
  SearchProvenancePanel,
  FormattedSearchResults,
} from "./Provenance";

// ── helpers ──────────────────────────────────────────────────────────────────

function buildToolFrequency(
  sequence: string[],
): Array<{ toolName: string; count: number }> {
  const counts = new Map<string, number>();
  for (const t of sequence) counts.set(t, (counts.get(t) ?? 0) + 1);
  return [...counts.entries()]
    .map(([toolName, count]) => ({ toolName, count }))
    .sort((a, b) => b.count - a.count || a.toolName.localeCompare(b.toolName));
}

function stringify(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function previewStr(value: unknown): string {
  const s = stringify(value).replace(/\s+/g, " ").trim();
  return s.length > 160 ? s.slice(0, 160) + "\u2026" : s || "\u2014";
}

function usageOf(usage?: Partial<TokenUsage>): TokenUsage {
  const cacheRead = usage?.cache_read || 0;
  const cacheWrite = usage?.cache_write || 0;
  const input =
    usage?.input ??
    Math.max((usage?.inputTokens || 0) - cacheRead - cacheWrite, 0);
  const output = usage?.output ?? usage?.outputTokens ?? 0;
  const total =
    usage?.total ??
    usage?.totalTokens ??
    input + cacheRead + cacheWrite + output;
  return {
    input,
    cache_read: cacheRead,
    cache_write: cacheWrite,
    output,
    total,
    inputTokens: usage?.inputTokens ?? input + cacheRead + cacheWrite,
    outputTokens: usage?.outputTokens ?? output,
    totalTokens: usage?.totalTokens ?? total,
  };
}

function formatUsageParts(usage: Partial<TokenUsage>): string {
  const normalized = usageOf(usage);
  return [
    `base ${formatNumber(normalized.input)}`,
    `read ${formatNumber(normalized.cache_read)}`,
    `write ${formatNumber(normalized.cache_write)}`,
    `out ${formatNumber(normalized.output)}`,
  ].join(" / ");
}

function sourceTheme(source: string): {
  fg: string;
  bg: string;
  border: string;
} {
  const key = (source || "unknown").toLowerCase();
  if (key.includes("gitree")) {
    return { fg: "#f5d08a", bg: "rgba(120,53,15,0.28)", border: "#78350f" };
  }
  if (key.includes("ask") || key.includes("explore")) {
    return { fg: "#93c5fd", bg: "rgba(30,64,175,0.28)", border: "#1e40af" };
  }
  if (key.includes("log")) {
    return { fg: "#86efac", bg: "rgba(21,128,61,0.24)", border: "#166534" };
  }
  return { fg: "#d4d4d8", bg: "rgba(63,63,70,0.32)", border: "#3f3f46" };
}

function getContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((p: any) => p?.type === "text")
    .map((p: any) => String(p.text ?? ""))
    .join("\n")
    .trim();
}

function normalise(v: unknown): unknown {
  if (!v || typeof v !== "object" || Array.isArray(v)) return v;
  const r = v as Record<string, unknown>;
  if ((r.type === "text" || r.type === "json") && "value" in r) return r.value;
  return v;
}

type CallEntry = {
  id: string;
  toolName: string;
  input: unknown;
  index: number;
};
type ResultEntry = {
  id: string;
  toolName: string;
  output: unknown;
  index: number;
};
type TraceEventKind =
  | "system"
  | "user"
  | "assistant-text"
  | "reasoning"
  | "assistant"
  | "tool-call"
  | "tool-result"
  | "tool";
type TraceTurnKind = "setup" | "direct" | "tool";
type TraceEvent = {
  id: string;
  index: number;
  role: string;
  kind: TraceEventKind;
  text: string;
  payload: unknown;
  toolName?: string;
  toolCallId?: string;
};
type TraceTurn = {
  id: string;
  index: number;
  kind: TraceTurnKind;
  title: string;
  preview: string;
  outputPreview: string;
  toolNames: string[];
  eventCount: number;
  toolCount: number;
  events: TraceEvent[];
};
type IssueKind =
  | "empty"
  | "oversized"
  | "repeat"
  | "fallback"
  | "duplicate"
  | "noisy-overview";
type AnalyzedStep = {
  id: string;
  index: number;
  toolName: string;
  input: unknown;
  output: unknown;
  outputSize: number;
  outputLines: number;
  flags: IssueKind[];
};

function normaliseText(s: string): string {
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}

function payloadMetrics(value: unknown): {
  text: string;
  size: number;
  lines: number;
} {
  const text = stringify(value);
  return { text, size: text.length, lines: text.split(/\r?\n/).length };
}

function hasNamedConcept(prompt: string): boolean {
  const p = prompt.trim();
  if (!p) return false;
  return /`[^`]+`|\b[a-z]+[A-Z][A-Za-z0-9]*\b|\b[A-Za-z_][A-Za-z0-9_]*\(|\b[A-Za-z0-9_-]+\/[A-Za-z0-9_.-]+\b|\b[a-z_]{3,}[a-z0-9_]*_[a-z0-9_]+\b/.test(
    p,
  );
}

function dedupeFlags(flags: IssueKind[]): IssueKind[] {
  return [...new Set(flags)];
}

function analyzeTrace(
  parsed: { calls: CallEntry[]; results: ResultEntry[] },
  prompt: string,
): { steps: AnalyzedStep[]; counts: Record<IssueKind, number> } {
  const resultById = new Map(parsed.results.map((r) => [r.id, r]));
  const steps: AnalyzedStep[] = parsed.calls.map((call) => {
    const result = resultById.get(call.id);
    const output = result?.output;
    const metrics = payloadMetrics(output);
    return {
      id: call.id,
      index: call.index,
      toolName: call.toolName,
      input: call.input,
      output,
      outputSize: metrics.size,
      outputLines: metrics.lines,
      flags: [],
    };
  });

  const promptHasConcept = hasNamedConcept(prompt);
  for (let i = 0; i < steps.length; i += 1) {
    const step = steps[i];
    const prev = steps[i - 1];
    const prev2 = steps[i - 2];
    const outputText = normaliseText(stringify(step.output));
    const inputText = normaliseText(previewStr(step.input));
    const flags: IssueKind[] = [];

    if (
      !outputText ||
      outputText === "[]" ||
      outputText === "{}" ||
      outputText === "null" ||
      outputText === "\u2014" ||
      /not found|no results|no matches|empty/i.test(outputText) ||
      step.outputSize < 16
    ) {
      flags.push("empty");
    }
    if (step.outputSize > 4000 || step.outputLines > 80) flags.push("oversized");
    if (
      (prev &&
        prev.toolName === step.toolName &&
        normaliseText(previewStr(prev.input)) === inputText) ||
      (prev &&
        prev2 &&
        prev.toolName === step.toolName &&
        prev2.toolName === step.toolName)
    ) {
      flags.push("repeat");
    }
    if (
      prev &&
      prev.toolName === step.toolName &&
      normaliseText(stringify(prev.output)) === outputText &&
      step.outputSize > 0
    ) {
      flags.push("duplicate");
    }
    if (
      (step.toolName === "bash" || step.toolName === "fulltext_search") &&
      prev &&
      (prev.toolName.startsWith("stakgraph") || prev.toolName === "repo_overview") &&
      !prev.flags.includes("empty")
    ) {
      flags.push("fallback");
    }
    if (step.toolName === "repo_overview" && promptHasConcept) {
      flags.push("noisy-overview");
    }
    step.flags = dedupeFlags(flags);
  }

  const counts: Record<IssueKind, number> = {
    empty: 0,
    oversized: 0,
    repeat: 0,
    fallback: 0,
    duplicate: 0,
    "noisy-overview": 0,
  };
  for (const step of steps) {
    for (const flag of step.flags) counts[flag] += 1;
  }
  return { steps, counts };
}

function createTraceEvent(
  index: number,
  role: string,
  kind: TraceEventKind,
  payload: unknown,
  text = "",
  toolName?: string,
  toolCallId?: string,
): TraceEvent {
  return {
    id: `${kind}-${index}`,
    index,
    role,
    kind,
    text,
    payload,
    toolName,
    toolCallId,
  };
}

function uniqueStrings(values: Array<string | undefined>): string[] {
  return [...new Set(values.filter(Boolean) as string[])];
}

function findLastToolResult(events: TraceEvent[]): TraceEvent | undefined {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event.kind === "tool-result") return event;
  }
  return undefined;
}

function finalizeTurn(index: number, events: TraceEvent[]): TraceTurn {
  const toolNames = uniqueStrings(
    events
      .filter((event) => event.kind === "tool-call" || event.kind === "tool-result")
      .map((event) => event.toolName),
  );
  const firstUser = events.find((event) => event.kind === "user")?.text ?? "";
  const assistantTexts = events
    .filter((event) => event.kind === "assistant-text")
    .map((event) => event.text)
    .filter(Boolean);
  const lastAssistant = assistantTexts[assistantTexts.length - 1] ?? "";
  const firstSystem = events.find((event) => event.kind === "system")?.text ?? "";
  const kind: TraceTurnKind = firstSystem
    ? "setup"
    : toolNames.length > 0
      ? "tool"
      : "direct";
  const title =
    kind === "setup"
      ? "System instructions"
      : firstUser || lastAssistant || toolNames.join(", ") || `Turn ${index}`;
  const preview =
    kind === "setup"
      ? firstSystem
      : firstUser || assistantTexts[0] || toolNames.join(", ") || "No prompt";
  const outputPreview =
    lastAssistant ||
    previewStr(findLastToolResult(events)?.payload ?? "");

  return {
    id: `turn-${index}`,
    index,
    kind,
    title: previewStr(title),
    preview: previewStr(preview),
    outputPreview: outputPreview || "\u2014",
    toolNames,
    eventCount: events.length,
    toolCount: toolNames.length,
    events,
  };
}

function parseTrace(trace: unknown): {
  userPrompt: string;
  answer: string;
  calls: CallEntry[];
  results: ResultEntry[];
  events: TraceEvent[];
  turns: TraceTurn[];
} {
  if (!Array.isArray(trace))
    return {
      userPrompt: "",
      answer: "",
      calls: [],
      results: [],
      events: [],
      turns: [],
    };

  let userPrompt = "";
  let answer = "";
  const calls: CallEntry[] = [];
  const results: ResultEntry[] = [];
  const events: TraceEvent[] = [];
  let ci = 0,
    ri = 0,
    ei = 0;

  for (const msg of trace) {
    if (!msg || typeof msg !== "object") continue;
    const role = String((msg as any).role ?? "");
    const content = (msg as any).content;

    if (!userPrompt && role === "user") {
      userPrompt =
        getContent(content) || (typeof content === "string" ? content : "");
    }
    if (role === "assistant") {
      const t =
        getContent(content) || (typeof content === "string" ? content : "");
      if (t) answer = t;
    }

    if (role === "system" && typeof content === "string") {
      ei += 1;
      events.push(createTraceEvent(ei, role, "system", content, content));
    }
    if (role === "user") {
      const text = getContent(content) || (typeof content === "string" ? content : "");
      ei += 1;
      events.push(createTraceEvent(ei, role, "user", content, text));
    }
    if (role === "assistant" && typeof content === "string") {
      ei += 1;
      events.push(createTraceEvent(ei, role, "assistant-text", content, content));
    }
    if (role === "tool" && typeof content === "string") {
      ei += 1;
      events.push(createTraceEvent(ei, role, "tool", content, content));
    }

    if (!Array.isArray(content)) continue;
    for (const item of content) {
      if (!item || typeof item !== "object") continue;
      const e = item as Record<string, unknown>;
      if (role === "assistant" && e.type === "text") {
        const text = String(e.text ?? "");
        ei += 1;
        events.push(createTraceEvent(ei, role, "assistant-text", text, text));
      }
      if (role === "assistant" && e.type === "reasoning") {
        const text = String(e.text ?? e.reasoning ?? e.content ?? "");
        ei += 1;
        events.push(createTraceEvent(ei, role, "reasoning", text, text));
      }
      if (e.type === "tool-call") {
        ci++;
        calls.push({
          id: String(e.toolCallId ?? `c${ci}`),
          toolName: String(e.toolName ?? "?"),
          input: normalise(e.input),
          index: ci,
        });
        ei += 1;
        events.push(
          createTraceEvent(
            ei,
            role,
            "tool-call",
            normalise(e.input),
            String(e.toolName ?? "?"),
            String(e.toolName ?? "?"),
            String(e.toolCallId ?? `c${ci}`),
          ),
        );
      }
      if (e.type === "tool-result") {
        ri++;
        results.push({
          id: String(e.toolCallId ?? `r${ri}`),
          toolName: String(e.toolName ?? "?"),
          output: normalise(e.output ?? e.result ?? e.content),
          index: ri,
        });
        ei += 1;
        events.push(
          createTraceEvent(
            ei,
            role,
            "tool-result",
            normalise(e.output ?? e.result ?? e.content),
            String(e.toolName ?? "?"),
            String(e.toolName ?? "?"),
            String(e.toolCallId ?? `r${ri}`),
          ),
        );
      }
    }
  }

  const turns: TraceTurn[] = [];
  let currentTurnEvents: TraceEvent[] = [];

  const flushTurn = () => {
    if (currentTurnEvents.length === 0) return;
    turns.push(finalizeTurn(turns.length + 1, currentTurnEvents));
    currentTurnEvents = [];
  };

  for (const event of events) {
    if (event.kind === "system") {
      flushTurn();
      turns.push(finalizeTurn(turns.length + 1, [event]));
      continue;
    }
    if (event.kind === "user") {
      flushTurn();
      currentTurnEvents = [event];
      continue;
    }
    if (currentTurnEvents.length === 0) {
      currentTurnEvents = [event];
      continue;
    }
    currentTurnEvents.push(event);
  }
  flushTurn();

  return { userPrompt, answer, calls, results, events, turns };
}

// ── shared styles ─────────────────────────────────────────────────────────────

const card: React.CSSProperties = {
  border: "1px solid #27272a",
  borderRadius: "8px",
  backgroundColor: "#111113",
  overflow: "hidden",
};
const summaryBase: React.CSSProperties = {
  listStyle: "none",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "10px 14px",
  cursor: "pointer",
  userSelect: "none",
};
const labelStyle: React.CSSProperties = {
  fontSize: "12px",
  fontWeight: 600,
  color: "#ededed",
};
const muted: React.CSSProperties = {
  fontSize: "11px",
  color: "#71717a",
  margin: 0,
};

const pillBase: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: "6px",
  fontSize: "11px",
  padding: "4px 10px",
  borderRadius: "9999px",
  border: "1px solid #27272a",
  backgroundColor: "#18181b",
  color: "#ededed",
};

function MetaPill({ label, value }: { label: string; value: string }) {
  return (
    <span style={pillBase}>
      <span
        style={{
          color: "#71717a",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          fontSize: "10px",
        }}
      >
        {label}
      </span>
      <span style={{ color: "#ededed" }}>{value}</span>
    </span>
  );
}

function SourceBadge({ source }: { source: string }) {
  const theme = sourceTheme(source);
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        padding: "4px 10px",
        borderRadius: "9999px",
        border: `1px solid ${theme.border}`,
        backgroundColor: theme.bg,
        color: theme.fg,
        fontSize: "11px",
        fontWeight: 600,
        textTransform: "lowercase",
        whiteSpace: "nowrap",
        flexShrink: 0,
      }}
    >
      {formatSourceLabel(source)}
    </span>
  );
}

function StatTile({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div
      style={{
        border: "1px solid #27272a",
        borderRadius: "10px",
        backgroundColor: "#111113",
        padding: "12px 14px",
        minWidth: 0,
      }}
    >
      <p
        style={{
          margin: 0,
          fontSize: "10px",
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "#71717a",
        }}
      >
        {label}
      </p>
      <p
        style={{
          margin: "8px 0 0 0",
          fontSize: "18px",
          fontWeight: 700,
          color: "#f5f5f5",
          lineHeight: 1.1,
          wordBreak: "break-word",
        }}
      >
        {value}
      </p>
    </div>
  );
}

function Section({
  title,
  badge,
  defaultOpen = true,
  children,
}: {
  title: string;
  badge?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  return (
    <details open={defaultOpen} style={card}>
      <summary style={summaryBase}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={labelStyle}>{title}</span>
          {badge !== undefined && <span style={muted}>{badge}</span>}
        </div>
        <span style={{ fontSize: "11px", color: "#52525b" }}>{"\u25be"}</span>
      </summary>
      <div>{children}</div>
    </details>
  );
}

function IssueBadge({ flag }: { flag: IssueKind }) {
  const colors: Record<IssueKind, { fg: string; bg: string; border: string }> =
    {
      empty: { fg: "#fca5a5", bg: "rgba(127,29,29,0.35)", border: "#7f1d1d" },
      oversized: { fg: "#fdba74", bg: "rgba(124,45,18,0.35)", border: "#7c2d12" },
      repeat: { fg: "#fcd34d", bg: "rgba(120,53,15,0.35)", border: "#78350f" },
      fallback: { fg: "#f9a8d4", bg: "rgba(131,24,67,0.35)", border: "#831843" },
      duplicate: { fg: "#c4b5fd", bg: "rgba(76,29,149,0.35)", border: "#4c1d95" },
      "noisy-overview": { fg: "#93c5fd", bg: "rgba(30,64,175,0.35)", border: "#1e40af" },
    };
  const color = colors[flag];
  return (
    <span
      style={{
        fontSize: "10px",
        lineHeight: 1,
        padding: "4px 6px",
        borderRadius: "9999px",
        border: `1px solid ${color.border}`,
        color: color.fg,
        backgroundColor: color.bg,
        textTransform: "lowercase",
      }}
    >
      {flag}
    </span>
  );
}

function EntryRow({
  kind,
  index,
  toolName,
  payload,
  flags = [],
}: {
  kind: string;
  index: number;
  toolName: string;
  payload: unknown;
  flags?: IssueKind[];
}) {
  const kindColor = kind === "call" ? "#c4b5fd" : "#93c5fd";
  return (
    <details style={{ borderTop: "1px solid #1f1f22" }}>
      <summary
        style={{
          listStyle: "none",
          display: "flex",
          alignItems: "flex-start",
          gap: "10px",
          padding: "9px 14px",
          cursor: "pointer",
          userSelect: "none",
        }}
      >
        <span
          style={{
            display: "inline-block",
            width: "64px",
            flexShrink: 0,
            fontSize: "10px",
            fontWeight: 700,
            letterSpacing: "0.16em",
            textTransform: "uppercase",
            color: kindColor,
            paddingTop: "2px",
          }}
        >
          {kind} {index}
        </span>
        <span style={{ flex: 1, minWidth: 0 }}>
          <span
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              flexWrap: "wrap",
            }}
          >
            <span
              style={{
                fontSize: "12px",
                fontWeight: 600,
                fontFamily: "ui-monospace,monospace",
                color: "#ededed",
              }}
            >
              {toolName}
            </span>
            {flags.map((flag) => (
              <IssueBadge key={flag} flag={flag} />
            ))}
          </span>
          <span
            style={{
              display: "block",
              fontSize: "11px",
              color: "#71717a",
              marginTop: "2px",
              lineHeight: 1.5,
            }}
          >
            {previewStr(payload)}
          </span>
        </span>
      </summary>
      <CopyableBlock value={payload} />
    </details>
  );
}

function TurnBadge({ kind }: { kind: TraceTurnKind }) {
  const color =
    kind === "setup"
      ? { fg: "#d4d4d8", bg: "#18181b", border: "#3f3f46" }
      : kind === "tool"
        ? { fg: "#93c5fd", bg: "rgba(30,64,175,0.18)", border: "#1e40af" }
        : { fg: "#c4b5fd", bg: "rgba(76,29,149,0.18)", border: "#4c1d95" };

  return (
    <span
      style={{
        fontSize: "10px",
        lineHeight: 1,
        padding: "4px 8px",
        borderRadius: "9999px",
        border: `1px solid ${color.border}`,
        color: color.fg,
        backgroundColor: color.bg,
        textTransform: "lowercase",
      }}
    >
      {kind === "setup" ? "setup" : "turn"}
    </span>
  );
}

function EventBadge({ event }: { event: TraceEvent }) {
  const label = event.kind.replace(/-/g, " ");
  const color =
    event.kind === "system"
      ? { fg: "#d4d4d8", bg: "#18181b", border: "#3f3f46" }
      : event.kind === "user"
        ? { fg: "#86efac", bg: "rgba(21,128,61,0.18)", border: "#166534" }
        : event.kind === "reasoning"
          ? { fg: "#fbbf24", bg: "rgba(146,64,14,0.16)", border: "#92400e" }
        : event.kind === "assistant-text" || event.kind === "assistant"
          ? { fg: "#c4b5fd", bg: "rgba(76,29,149,0.18)", border: "#4c1d95" }
          : { fg: "#93c5fd", bg: "rgba(30,64,175,0.18)", border: "#1e40af" };

  return (
    <span
      style={{
        fontSize: "10px",
        lineHeight: 1,
        padding: "4px 8px",
        borderRadius: "9999px",
        border: `1px solid ${color.border}`,
        color: color.fg,
        backgroundColor: color.bg,
        textTransform: "lowercase",
      }}
    >
      {label}
    </span>
  );
}

// ── display units (render layer only) ────────────────────────────────────────

type StandaloneUnit = { kind: "standalone"; event: TraceEvent };
type PairedUnit = { kind: "paired"; call: TraceEvent; result: TraceEvent | undefined };
type DisplayUnit = StandaloneUnit | PairedUnit;

function groupEvents(events: TraceEvent[]): DisplayUnit[] {
  const resultsByCallId = new Map<string, TraceEvent>();
  for (const event of events) {
    if (event.kind === "tool-result" && event.toolCallId) {
      resultsByCallId.set(event.toolCallId, event);
    }
  }

  const units: DisplayUnit[] = [];
  const consumed = new Set<string>();

  for (const event of events) {
    if (consumed.has(event.id)) continue;

    if (event.kind === "tool-call") {
      const result = event.toolCallId ? resultsByCallId.get(event.toolCallId) : undefined;
      consumed.add(event.id);
      if (result) consumed.add(result.id);
      units.push({ kind: "paired", call: event, result });
      continue;
    }

    if (event.kind === "tool-result") continue; // already consumed by its call

    consumed.add(event.id);
    units.push({ kind: "standalone", event });
  }

  return units;
}

const unitIndexStyle: React.CSSProperties = {
  display: "inline-block",
  width: "28px",
  flexShrink: 0,
  fontSize: "10px",
  fontWeight: 700,
  color: "#52525b",
};

const subLabelStyle: React.CSSProperties = {
  fontSize: "10px",
  textTransform: "uppercase",
  letterSpacing: "0.1em",
  color: "#52525b",
  padding: "8px 14px 4px 14px",
};

const chunkInputMaxHeight = "14rem";
const chunkOutputMaxHeight = "20rem";
const chunkEventMaxHeight = "16rem";
const chunkSummaryMaxHeight = "18rem";
const chunkScrollStyle: React.CSSProperties = {
  maxHeight: chunkOutputMaxHeight,
  overflowY: "auto",
  minHeight: 0,
};

function DisplayUnitRow({
  unit,
  unitIndex,
  provenance,
  annotations,
  onAnnotate,
}: {
  unit: DisplayUnit;
  unitIndex: number;
  provenance?: SearchProvenanceEntry;
  annotations?: Annotation[];
  onAnnotate?: (
    marker: AnnotationMarker,
    note: string,
    toolCallId?: string,
  ) => void;
}) {
  const [showAnnotationForm, setShowAnnotationForm] = useState(false);

  if (unit.kind === "paired") {
    const isSearchTool = unit.call.toolName === "stakgraph_search";
    const formattedResults =
      isSearchTool && unit.result
        ? parseSearchToolResultRows(unit.result.payload)
        : null;
    const myAnnotations = (annotations ?? []).filter(
      (a) => a.target === "tool_call" && a.target_id === unit.call.toolCallId,
    );

    return (
      <div style={{ borderTop: "1px solid #1f1f22" }}>
        <details>
          <summary
            style={{
              listStyle: "none",
              display: "flex",
              alignItems: "center",
              gap: "10px",
              padding: "9px 14px",
              cursor: "pointer",
              userSelect: "none",
            }}
          >
            <span style={unitIndexStyle}>{unitIndex}</span>
            <span
              style={{
                fontSize: "10px",
                lineHeight: 1,
                padding: "4px 8px",
                borderRadius: "9999px",
                border: "1px solid #1e40af",
                color: "#93c5fd",
                backgroundColor: "rgba(30,64,175,0.18)",
                textTransform: "lowercase",
                flexShrink: 0,
              }}
            >
              tool
            </span>
            <span
              style={{
                fontSize: "12px",
                fontWeight: 600,
                fontFamily: "ui-monospace,monospace",
                color: "#ededed",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                flex: 1,
                minWidth: 0,
              }}
            >
              {unit.call.toolName ?? "?"}
            </span>
            {myAnnotations.map((a) => (
              <AnnotationBadge key={a.ts} marker={a.marker} note={a.note} />
            ))}
            {onAnnotate && (
              <button
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setShowAnnotationForm((f) => !f);
                }}
                title="Add annotation"
                style={{
                  fontSize: "11px",
                  lineHeight: 1,
                  padding: "3px 7px",
                  borderRadius: "9999px",
                  border: "1px solid #3f3f46",
                  backgroundColor: "transparent",
                  color: "#71717a",
                  cursor: "pointer",
                  flexShrink: 0,
                }}
              >
                +
              </button>
            )}
            <span style={muted}>{unit.result ? "→ result" : "pending"}</span>
          </summary>
          <div style={{ borderTop: "1px solid #1f1f22" }}>
            <p style={subLabelStyle}>Input</p>
            <CopyableBlock
              value={unit.call.payload}
              maxHeight={chunkInputMaxHeight}
            />
            {unit.result && (
              <>
                <p style={{ ...subLabelStyle, borderTop: "1px solid #1f1f22" }}>
                  Output
                </p>
                {formattedResults ? (
                  <div style={chunkScrollStyle}>
                    <FormattedSearchResults
                      results={formattedResults}
                      provenance={provenance}
                      rawValue={unit.result.payload}
                    />
                  </div>
                ) : (
                  <>
                    {provenance && (
                      <SearchProvenancePanel provenance={provenance} />
                    )}
                    <CopyableBlock
                      value={unit.result.payload}
                      maxHeight={chunkOutputMaxHeight}
                    />
                  </>
                )}
              </>
            )}
          </div>
        </details>
        {showAnnotationForm && (
          <AnnotationForm
            onSubmit={(marker, note) => {
              onAnnotate?.(marker, note, unit.call.toolCallId);
              setShowAnnotationForm(false);
            }}
            onCancel={() => setShowAnnotationForm(false)}
          />
        )}
      </div>
    );
  }

  const { event } = unit;
  const name =
    event.kind === "reasoning" ? "reasoning" : event.toolName || event.role;

  return (
    <details style={{ borderTop: "1px solid #1f1f22" }}>
      <summary
        style={{
          listStyle: "none",
          display: "flex",
          alignItems: "center",
          gap: "10px",
          padding: "9px 14px",
          cursor: "pointer",
          userSelect: "none",
        }}
      >
        <span style={unitIndexStyle}>{unitIndex}</span>
        <EventBadge event={event} />
        <span
          style={{
            fontSize: "12px",
            fontWeight: 600,
            fontFamily: "ui-monospace,monospace",
            color: "#ededed",
            flex: 1,
            minWidth: 0,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {name}
        </span>
        <span
          style={{
            ...muted,
            maxWidth: "260px",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {previewStr(event.text || event.payload)}
        </span>
      </summary>
      <CopyableBlock value={event.payload} maxHeight={chunkEventMaxHeight} />
    </details>
  );
}

function StepDivider({ meta }: { meta: StepMeta }) {
  const label = meta.label;
  const isFinal = meta.toolCalls.length === 0 && !label;
  const usage = usageOf(meta.usage);
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "10px",
        padding: "4px 14px",
        fontSize: "10px",
        fontFamily: "ui-monospace,monospace",
        color: "#71717a",
        background: "rgba(163,230,53,0.04)",
        borderTop: "1px solid #2a2a2e",
      }}
    >
      <span style={{ color: "#52525b", fontWeight: 700, minWidth: 20 }}>
        S{meta.step}
      </span>
      {label ? (
        <span
          title={label}
          style={{
            color: "#a3e635",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {label}
        </span>
      ) : null}
      {isFinal ? <span style={{ color: "#a3e635" }}>final answer</span> : null}
      <span style={{ color: "#a3e635" }}>{formatUsageParts(usage)}</span>
      <span style={{ color: "#71717a" }}>
        total {formatNumber(usage.total)}
      </span>
    </div>
  );
}

const SEARCH_TOOL_NAMES = new Set(["stakgraph_search", "fulltext_search"]);

type ProvenanceMatchState = {
  cursor: number;
  consumed: Set<number>;
};

function matchProvenance(
  unit: DisplayUnit,
  provenanceEntries: SearchProvenanceEntry[],
  matchState: ProvenanceMatchState,
): SearchProvenanceEntry | undefined {
  if (unit.kind !== "paired") return undefined;
  const toolName = unit.call.toolName;
  if (!toolName || !SEARCH_TOOL_NAMES.has(toolName)) return undefined;

  if (unit.call.toolCallId) {
    const exactIdx = provenanceEntries.findIndex(
      (entry, idx) =>
        !matchState.consumed.has(idx) &&
        entry.tool_call_id === unit.call.toolCallId,
    );
    if (exactIdx >= 0) {
      matchState.consumed.add(exactIdx);
      return provenanceEntries[exactIdx];
    }
  }

  for (let idx = matchState.cursor; idx < provenanceEntries.length; idx++) {
    if (matchState.consumed.has(idx)) continue;
    const entry = provenanceEntries[idx];
    if (entry.tool_name && entry.tool_name !== toolName) continue;
    matchState.consumed.add(idx);
    matchState.cursor = idx + 1;
    return entry;
  }

  return undefined;
}

function interleaveUnitsWithSteps(
  units: DisplayUnit[],
  metas: StepMeta[],
  provenanceEntries: SearchProvenanceEntry[] = [],
  annotations: Annotation[] = [],
  onAnnotate?: (
    marker: AnnotationMarker,
    note: string,
    toolCallId?: string,
  ) => void,
): React.ReactNode[] {
  const matchState: ProvenanceMatchState = {
    cursor: 0,
    consumed: new Set<number>(),
  };

  if (metas.length === 0) {
    return units.map((unit, i) => (
      <DisplayUnitRow
        key={unit.kind === "paired" ? unit.call.id : unit.event.id}
        unit={unit}
        unitIndex={i + 1}
        provenance={matchProvenance(unit, provenanceEntries, matchState)}
        annotations={annotations}
        onAnnotate={onAnnotate}
      />
    ));
  }

  const nodes: React.ReactNode[] = [];
  let metaIdx = 0;
  let toolPtr = 0;
  let unitNum = 0;

  for (const unit of units) {
    const isPaired = unit.kind === "paired";
    if (isPaired && metaIdx < metas.length) {
      const currentMeta = metas[metaIdx];
      if (toolPtr === 0) {
        nodes.push(
          <StepDivider key={`step-${currentMeta.step}`} meta={currentMeta} />,
        );
      }
      toolPtr++;
      if (toolPtr >= currentMeta.toolCalls.length) {
        metaIdx++;
        toolPtr = 0;
      }
    }
    unitNum++;
    nodes.push(
      <DisplayUnitRow
        key={unit.kind === "paired" ? unit.call.id : unit.event.id}
        unit={unit}
        unitIndex={unitNum}
        provenance={matchProvenance(unit, provenanceEntries, matchState)}
        annotations={annotations}
        onAnnotate={onAnnotate}
      />,
    );
  }

  // Final step with no tool calls (the text response)
  if (metaIdx < metas.length) {
    const finalMeta = metas[metaIdx];
    if (finalMeta.toolCalls.length === 0) {
      nodes.push(
        <StepDivider key={`step-${finalMeta.step}`} meta={finalMeta} />,
      );
    }
  }

  return nodes;
}

function TurnCard({
  turn,
  stepMetas,
  provenanceEntries,
  annotations,
  onAnnotate,
  isOpen,
  onToggle,
}: {
  turn: TraceTurn;
  stepMetas?: StepMeta[];
  provenanceEntries?: SearchProvenanceEntry[];
  annotations?: Annotation[];
  onAnnotate?: (
    marker: AnnotationMarker,
    note: string,
    toolCallId?: string,
  ) => void;
  isOpen: boolean;
  onToggle: (turnId: string) => void;
}) {
  const units = groupEvents(turn.events);
  const metas = stepMetas ?? [];
  const turnUsage = metas.reduce(
    (sum, meta) => {
      const usage = usageOf(meta.usage);
      return {
        input: sum.input + usage.input,
        cache_read: sum.cache_read + usage.cache_read,
        cache_write: sum.cache_write + usage.cache_write,
        output: sum.output + usage.output,
        total: sum.total + usage.total,
      };
    },
    { input: 0, cache_read: 0, cache_write: 0, output: 0, total: 0 },
  );
  return (
    <details open={isOpen} style={{ ...card, flexShrink: 0 }}>
      <summary
        onClick={(e) => {
          e.preventDefault();
          onToggle(turn.id);
        }}
        style={{
          listStyle: "none",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "12px",
          padding: "10px 14px",
          cursor: "pointer",
          userSelect: "none",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            flex: 1,
            minWidth: 0,
          }}
        >
          <span
            style={{
              fontSize: "10px",
              color: "#52525b",
              textTransform: "uppercase",
              letterSpacing: "0.14em",
              flexShrink: 0,
            }}
          >
            {turn.index}
          </span>
          <TurnBadge kind={turn.kind} />
          <span
            style={{
              fontSize: "12px",
              fontWeight: 600,
              color: "#ededed",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {turn.title}
          </span>
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            flexShrink: 0,
          }}
        >
          {turn.toolCount > 0 && (
            <span style={muted}>{turn.toolCount} tools</span>
          )}
          <span style={muted}>{units.length} events</span>
          {metas.length > 0 && (
            <span
              style={{
                fontSize: "10px",
                padding: "2px 7px",
                borderRadius: "9999px",
                border: "1px solid #3f3f46",
                color: "#a3e635",
                backgroundColor: "rgba(163,230,53,0.08)",
                fontFamily: "ui-monospace,monospace",
                flexShrink: 0,
              }}
            >
              {formatUsageParts(turnUsage)}
            </span>
          )}
        </div>
      </summary>
      <div style={{ borderTop: "1px solid #1f1f22" }}>
        {interleaveUnitsWithSteps(
          units,
          metas,
          provenanceEntries,
          annotations,
          onAnnotate,
        )}
      </div>
    </details>
  );
}

// ── component ─────────────────────────────────────────────────────────────────

export function Sessions() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [runs, setRuns] = useState<ProductionRun[]>([]);
  const [selected, setSelected] = useState<ProductionRun | null>(null);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [showSessionAnnotationForm, setShowSessionAnnotationForm] =
    useState(false);
  const [loading, setLoading] = useState(true);
  const [repoSearch, setRepoSearch] = useState(searchParams.get("repo") || "");
  const [sourceFilter, setSourceFilter] = useState(searchParams.get("source") || "all");
  const [rangeFilter, setRangeFilter] = useState<"24h" | "7d" | "30d" | "all">(
    (searchParams.get("range") as "24h" | "7d" | "30d" | "all") || "all",
  );
  const [dayFilter, setDayFilter] = useState(searchParams.get("day") || "");
  const [openTurnId, setOpenTurnId] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const sessions = await api.sessions.list();
      setRuns(sessions);
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const loadDetail = async (run: ProductionRun) => {
    try {
      const detail = await api.sessions.get(run.id);
      setSelected(detail);
      setAnnotations(detail.annotations ?? []);
      setShowSessionAnnotationForm(false);
    } catch {
      setSelected(run);
      setAnnotations(run.annotations ?? []);
      setShowSessionAnnotationForm(false);
    }
  };

  const handleAnnotate = useCallback(
    async (marker: AnnotationMarker, note: string, toolCallId?: string) => {
      if (!selected) return;
      try {
        const ann = await api.sessions.annotate(selected.id, {
          target: toolCallId ? "tool_call" : "session",
          target_id: toolCallId,
          marker,
          note: note || undefined,
        });
        setAnnotations((prev) => [...prev, ann]);
        toast.success("Annotation saved");
      } catch (e: any) {
        toast.error(e.message);
      }
    },
    [selected],
  );

  const handleTurnToggle = useCallback((turnId: string) => {
    setOpenTurnId((prev) => (prev === turnId ? null : turnId));
  }, []);

  const freq = selected ? buildToolFrequency(selected.tool_sequence) : [];
  const parsed = useMemo(
    () =>
      selected
        ? parseTrace(selected.trace)
        : {
            userPrompt: "",
            answer: "",
            calls: [],
            results: [],
            events: [],
            turns: [],
          },
    [selected],
  );
  const selectedUsage = selected ? usageOf(selected.token_usage) : null;
  const prompt =
    parsed.userPrompt || selected?.user_prompt_preview || "No prompt preview";
  const answer =
    parsed.answer || selected?.answer_preview || "No answer preview";
  const diagnostics = useMemo(
    () => analyzeTrace(parsed, prompt),
    [parsed, prompt],
  );
  const flagsById = useMemo(
    () => new Map(diagnostics.steps.map((s) => [s.id, s.flags])),
    [diagnostics.steps],
  );

  const repoOptions = useMemo(
    () => [...new Set(runs.map((r) => r.repo))].sort(),
    [runs],
  );

  const sourceOptions = useMemo(
    () => [...new Set(runs.map((r) => r.source || "unknown"))].sort(),
    [runs],
  );

  useEffect(() => {
    setRepoSearch(searchParams.get("repo") || "");
    setSourceFilter(searchParams.get("source") || "all");
    setRangeFilter(
      (searchParams.get("range") as "24h" | "7d" | "30d" | "all") || "all",
    );
    setDayFilter(searchParams.get("day") || "");
  }, [searchParams]);

  useEffect(() => {
    const nextParams = new URLSearchParams();
    if (repoSearch) nextParams.set("repo", repoSearch);
    if (sourceFilter !== "all") nextParams.set("source", sourceFilter);
    if (rangeFilter !== "all") nextParams.set("range", rangeFilter);
    if (dayFilter) nextParams.set("day", dayFilter);
    setSearchParams(nextParams, { replace: true });
  }, [dayFilter, repoSearch, rangeFilter, setSearchParams, sourceFilter]);

  useEffect(() => {
    if (parsed.turns.length === 0) {
      setOpenTurnId(null);
      return;
    }

    const preferredTurn =
      parsed.turns.find((turn) => turn.kind !== "setup") ?? parsed.turns[0];
    setOpenTurnId(preferredTurn.id);
  }, [selected?.id, parsed.turns]);

  const filteredRuns = useMemo(
    () =>
      runs.filter((r) => {
        if (sourceFilter !== "all" && (r.source || "unknown") !== sourceFilter)
          return false;
        if (repoSearch && !r.repo.toLowerCase().includes(repoSearch.toLowerCase()))
          return false;
        if (dayFilter && new Date(r.timestamp).toISOString().slice(0, 10) !== dayFilter)
          return false;
        const rangeStart = getRangeStart(rangeFilter);
        if (rangeStart && new Date(r.timestamp).getTime() < rangeStart)
          return false;
        return true;
      }),
    [runs, dayFilter, repoSearch, rangeFilter, sourceFilter],
  );

  const inputStyle: React.CSSProperties = {
    fontSize: "12px",
    borderRadius: "6px",
    padding: "5px 8px",
    backgroundColor: "#18181b",
    color: "#ededed",
    border: "1px solid #27272a",
    outline: "none",
    width: "100%",
    boxSizing: "border-box",
  };
  const btnStyle: React.CSSProperties = {
    fontSize: "12px",
    padding: "6px",
    borderRadius: "6px",
    border: "1px solid #27272a",
    backgroundColor: "transparent",
    color: "#ededed",
    cursor: "pointer",
    width: "100%",
  };
  const selectStyle: React.CSSProperties = {
    ...inputStyle,
    padding: "6px 8px",
  };

  const clearFilters = () => {
    setRepoSearch("");
    setSourceFilter("all");
    setRangeFilter("all");
    setDayFilter("");
  };

  return (
    <div style={{ display: "flex", gap: "16px", flex: 1, minHeight: 0 }}>
      {/* LEFT: sidebar */}
      <div
        style={{
          width: "320px",
          flexShrink: 0,
          display: "flex",
          flexDirection: "column",
          gap: "8px",
          minHeight: 0,
        }}
      >
        <button onClick={load} style={btnStyle}>
          Refresh
        </button>

        {loading ? (
          <p style={muted}>{"Loading\u2026"}</p>
        ) : runs.length === 0 ? (
          <p style={muted}>No sessions yet.</p>
        ) : (
          <>
            <div
              style={{
                ...card,
                padding: "8px 10px",
                display: "flex",
                flexDirection: "column",
                gap: "6px",
              }}
            >
              <input
                placeholder={"Filter by repo\u2026"}
                value={repoSearch}
                onChange={(e) => setRepoSearch(e.target.value)}
                list="repo-options"
                style={inputStyle}
              />
              <select
                value={sourceFilter}
                onChange={(e) => setSourceFilter(e.target.value)}
                style={selectStyle}
              >
                <option value="all">All sources</option>
                {sourceOptions.map((source) => (
                  <option key={source} value={source}>
                    {formatSourceLabel(source)}
                  </option>
                ))}
              </select>
              <select
                value={rangeFilter}
                onChange={(e) => {
                  setRangeFilter(
                    e.target.value as "24h" | "7d" | "30d" | "all",
                  );
                  setDayFilter("");
                }}
                style={selectStyle}
              >
                <option value="all">All time</option>
                <option value="24h">Last 24h</option>
                <option value="7d">Last 7d</option>
                <option value="30d">Last 30d</option>
              </select>
              <datalist id="repo-options">
                {repoOptions.map((r) => (
                  <option key={r} value={r} />
                ))}
              </datalist>
              {(repoSearch ||
                sourceFilter !== "all" ||
                rangeFilter !== "all" ||
                dayFilter) && (
                <>
                  <p style={{ ...muted, textAlign: "center" }}>
                    {dayFilter ? `Day ${dayFilter} · ` : ""}
                    {filteredRuns.length} / {runs.length} sessions
                  </p>
                  <button onClick={clearFilters} style={btnStyle}>
                    Clear filters
                  </button>
                </>
              )}
            </div>

            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "4px",
                overflowY: "auto",
                flex: 1,
                minHeight: 0,
              }}
            >
              {filteredRuns.length === 0 ? (
                <p style={muted}>No sessions match the filter.</p>
              ) : (
                filteredRuns.map((run) => (
                  <button
                    key={run.id}
                    onClick={() => void loadDetail(run)}
                    style={{
                      textAlign: "left",
                      borderRadius: "6px",
                      border: `1px solid ${
                        selected?.id === run.id ? "#52525b" : "#27272a"
                      }`,
                      backgroundColor:
                        selected?.id === run.id ? "#27272a" : "#111113",
                      padding: "8px 10px",
                      cursor: "pointer",
                      width: "100%",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        gap: "8px",
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "6px",
                          minWidth: 0,
                        }}
                      >
                        <SourceBadge source={run.source} />
                        {run.status === "error" && (
                          <span
                            title={run.error_message || "Error"}
                            style={{
                              display: "inline-block",
                              width: 8,
                              height: 8,
                              borderRadius: "50%",
                              backgroundColor: "#ef4444",
                              flexShrink: 0,
                            }}
                          />
                        )}
                      </div>
                      <span
                        style={{
                          fontSize: "10px",
                          color: "#71717a",
                          fontFamily: "ui-monospace, monospace",
                          flexShrink: 0,
                        }}
                        title={run.id}
                      >
                        {shortId(run.id)}
                      </span>
                    </div>
                    <span
                      style={{
                        fontSize: "11px",
                        fontFamily: "ui-monospace, monospace",
                        color: "#ededed",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                        display: "block",
                        marginTop: "8px",
                      }}
                    >
                      {run.repo || "No repo captured"}
                    </span>
                    <p
                      style={{
                        ...muted,
                        marginTop: "3px",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {new Date(run.timestamp).toLocaleString()} {"\u00b7"}{" "}
                      {run.tool_call_count} calls {"\u00b7"}{" "}
                      {formatNumber(run.token_usage.total)} tokens
                    </p>
                    {run.user_prompt_preview && (
                      <p
                        style={{
                          ...muted,
                          marginTop: "2px",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {run.user_prompt_preview}
                      </p>
                    )}
                  </button>
                ))
              )}
            </div>
          </>
        )}
      </div>

      {/* RIGHT: detail */}
      <div style={{ flex: 1, minWidth: 0, minHeight: 0, overflowY: "auto" }}>
        {selected ? (
          <div
            style={{
              display: "flex",
              overflow: "hidden",
              flexDirection: "column",
              gap: "10px",
              minHeight: 0,
            }}
          >
            {/* header */}
            <div
              style={{
                ...card,
                padding: "16px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: "16px",
                  flexWrap: "wrap",
                  alignItems: "flex-start",
                }}
              >
                <div style={{ flex: "1 1 360px", minWidth: 0 }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      flexWrap: "wrap",
                      marginBottom: "10px",
                    }}
                  >
                    <SourceBadge source={selected.source} />
                    <MetaPill label="session" value={shortId(selected.id)} />
                  </div>
                  <p
                    style={{
                      fontSize: "18px",
                      fontWeight: 700,
                      color: "#ededed",
                      flexShrink: 0,
                      whiteSpace: "nowrap",
                      margin: 0,
                      lineHeight: 1.2,
                    }}
                  >
                    {selected.repo || "Session without repo label"}
                  </p>
                  <p style={{ ...muted, marginTop: "6px", fontSize: "12px" }}>
                    Started {new Date(selected.timestamp).toLocaleString()}
                  </p>
                  <div
                    style={{
                      display: "flex",
                      gap: "8px",
                      flexWrap: "wrap",
                      marginTop: "12px",
                    }}
                  >
                    <MetaPill
                      label="model"
                      value={selected.model || "unknown"}
                    />
                    <MetaPill
                      label="duration"
                      value={formatDuration(selected.duration_ms)}
                    />
                    <MetaPill
                      label="turns"
                      value={String(parsed.turns.length)}
                    />
                    <MetaPill
                      label="calls"
                      value={String(selected.tool_call_count)}
                    />
                  </div>
                </div>
                <div
                  style={{
                    flex: "0 1 420px",
                    minWidth: 0,
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
                    gap: "10px",
                    width: "100%",
                    maxWidth: "420px",
                  }}
                >
                  <StatTile
                    label="Total Tokens"
                    value={formatNumber(selectedUsage?.total || 0)}
                  />
                  <StatTile
                    label="Input Base"
                    value={formatNumber(selectedUsage?.input || 0)}
                  />
                  <StatTile
                    label="Cache Read"
                    value={formatNumber(selectedUsage?.cache_read || 0)}
                  />
                  <StatTile
                    label="Cache Write"
                    value={formatNumber(selectedUsage?.cache_write || 0)}
                  />
                  <StatTile
                    label="Output"
                    value={formatNumber(selectedUsage?.output || 0)}
                  />
                  {selected.cost_usd != null && (
                    <StatTile
                      label="Cost"
                      value={`$${selected.cost_usd.toFixed(4)}`}
                    />
                  )}
                </div>
              </div>
              {(diagnostics.counts.oversized > 0 ||
                diagnostics.counts.fallback > 0 ||
                diagnostics.counts.repeat > 0 ||
                diagnostics.counts.empty > 0) && (
                <div
                  style={{
                    display: "flex",
                    gap: "8px",
                    flexWrap: "wrap",
                    marginTop: "8px",
                  }}
                >
                  {(
                    [
                      {
                        label: "flagged",
                        value: diagnostics.steps.filter(
                          (s) => s.flags.length > 0,
                        ).length,
                        color: "#fca5a5",
                      },
                      {
                        label: "oversized",
                        value: diagnostics.counts.oversized,
                        color: "#fdba74",
                      },
                      {
                        label: "fallbacks",
                        value: diagnostics.counts.fallback,
                        color: "#f9a8d4",
                      },
                      {
                        label: "repeats",
                        value: diagnostics.counts.repeat,
                        color: "#fcd34d",
                      },
                    ] as const
                  )
                    .filter((item) => item.value > 0)
                    .map((item) => (
                      <span
                        key={item.label}
                        style={{
                          fontSize: "11px",
                          color: item.color,
                          backgroundColor: "#18181b",
                          border: "1px solid #27272a",
                          borderRadius: "6px",
                          padding: "2px 8px",
                        }}
                      >
                        {item.value} {item.label}
                      </span>
                    ))}
                </div>
              )}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  flexWrap: "wrap",
                  marginTop: "10px",
                  paddingTop: "10px",
                  borderTop: "1px solid #1f1f22",
                }}
              >
                {annotations
                  .filter((a) => a.target === "session")
                  .map((a) => (
                    <AnnotationBadge
                      key={a.ts}
                      marker={a.marker}
                      note={a.note}
                    />
                  ))}
                {showSessionAnnotationForm ? (
                  <AnnotationForm
                    onSubmit={(marker, note) => {
                      void handleAnnotate(marker, note);
                      setShowSessionAnnotationForm(false);
                    }}
                    onCancel={() => setShowSessionAnnotationForm(false)}
                  />
                ) : (
                  <button
                    onClick={() => setShowSessionAnnotationForm(true)}
                    style={{
                      fontSize: "11px",
                      padding: "3px 10px",
                      borderRadius: "9999px",
                      border: "1px solid #3f3f46",
                      backgroundColor: "transparent",
                      color: "#71717a",
                      cursor: "pointer",
                    }}
                  >
                    + annotate session
                  </button>
                )}
              </div>
            </div>

            <Section
              title="Trace timeline"
              badge={`${parsed.turns.length} turns / ${parsed.events.length} events`}
              defaultOpen={true}
            >
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "8px",
                  padding: "10px",
                  maxHeight: "min(62vh, 52rem)",
                  overflowY: "auto",
                  minHeight: 0,
                }}
              >
                {parsed.turns.length === 0 ? (
                  <p style={{ ...muted, padding: "10px 4px" }}>
                    No trace events available.
                  </p>
                ) : (
                  parsed.turns.map((turn) => (
                    <TurnCard
                      key={turn.id}
                      turn={turn}
                      stepMetas={selected?.step_meta?.filter(
                        (m) => m.turn === turn.index,
                      )}
                      provenanceEntries={selected?.search_provenance}
                      annotations={annotations}
                      onAnnotate={handleAnnotate}
                      isOpen={openTurnId === turn.id}
                      onToggle={handleTurnToggle}
                    />
                  ))
                )}
              </div>
            </Section>

            <Section
              title="Session highlights"
              badge="summary"
              defaultOpen={false}
            >
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "10px",
                  minHeight: 0,
                }}
              >
                <div>
                  <p style={{ ...labelStyle, margin: "10px 14px 0 14px" }}>
                    Prompt
                  </p>
                  <CopyableBlock
                    value={prompt}
                    maxHeight={chunkSummaryMaxHeight}
                  />
                </div>
                <div>
                  <p style={{ ...labelStyle, margin: "10px 14px 0 14px" }}>
                    Final answer
                  </p>
                  <CopyableBlock
                    value={answer}
                    maxHeight={chunkSummaryMaxHeight}
                  />
                </div>
              </div>
            </Section>

            <Section
              title="Tool analysis"
              badge={String(freq.length)}
              defaultOpen={false}
            >
              <div style={{ padding: "12px 14px" }}>
                <p style={{ ...labelStyle, marginBottom: "8px" }}>
                  Tool frequency
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                  {freq.length === 0 ? (
                    <p style={muted}>No tool activity in this session.</p>
                  ) : (
                    freq.map(({ toolName, count }) => (
                      <span
                        key={toolName}
                        title={`${count} call${count === 1 ? "" : "s"}`}
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "6px",
                          fontSize: "11px",
                          padding: "3px 10px",
                          borderRadius: "9999px",
                          border: "1px solid #27272a",
                          backgroundColor: "#18181b",
                          color: "#ededed",
                        }}
                      >
                        {toolName}
                        <span
                          style={{
                            fontSize: "10px",
                            fontWeight: 700,
                            backgroundColor: "#3f3f46",
                            borderRadius: "9999px",
                            padding: "1px 6px",
                            color: "#ededed",
                          }}
                        >
                          {count}
                        </span>
                      </span>
                    ))
                  )}
                </div>
              </div>
            </Section>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "10px",
              }}
            >
              <Section
                title="Tool calls"
                badge={String(parsed.calls.length)}
                defaultOpen={false}
              >
                <div
                  style={{
                    maxHeight: "28rem",
                    overflowY: "auto",
                    minHeight: 0,
                  }}
                >
                  {parsed.calls.length === 0 ? (
                    <p style={{ ...muted, padding: "10px 14px" }}>
                      No tool calls in trace.
                    </p>
                  ) : (
                    parsed.calls.map((c) => (
                      <EntryRow
                        key={c.id}
                        kind="call"
                        index={c.index}
                        toolName={c.toolName}
                        payload={c.input}
                        flags={flagsById.get(c.id)}
                      />
                    ))
                  )}
                </div>
              </Section>
              <Section
                title="Tool results"
                badge={String(parsed.results.length)}
                defaultOpen={false}
              >
                <div
                  style={{
                    maxHeight: "28rem",
                    overflowY: "auto",
                    minHeight: 0,
                  }}
                >
                  {parsed.results.length === 0 ? (
                    <p style={{ ...muted, padding: "10px 14px" }}>
                      No tool results in trace.
                    </p>
                  ) : (
                    parsed.results.map((r) => (
                      <EntryRow
                        key={r.id}
                        kind="result"
                        index={r.index}
                        toolName={r.toolName}
                        payload={r.output}
                        flags={flagsById.get(r.id)}
                      />
                    ))
                  )}
                </div>
              </Section>
            </div>

            {/* raw trace */}
            {selected.trace != null && (
              <Section title="Raw trace" badge="debug" defaultOpen={false}>
                <CopyableBlock value={selected.trace} maxHeight="28rem" />
              </Section>
            )}
          </div>
        ) : (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: "200px",
            }}
          >
            <p style={muted}>Select a session to view details</p>
          </div>
        )}
      </div>
    </div>
  );
}
