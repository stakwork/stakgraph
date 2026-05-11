import { previewStr } from "../utils";
import type {
  CallEntry,
  ResultEntry,
  TraceEvent,
  TraceEventKind,
  TraceTurn,
  ParsedTrace,
} from "./types";

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
      .filter(
        (event) =>
          event.kind === "tool-call" || event.kind === "tool-result",
      )
      .map((event) => event.toolName),
  );
  const firstUser = events.find((event) => event.kind === "user")?.text ?? "";
  const assistantTexts = events
    .filter((event) => event.kind === "assistant-text")
    .map((event) => event.text)
    .filter(Boolean);
  const lastAssistant = assistantTexts[assistantTexts.length - 1] ?? "";
  const firstSystem =
    events.find((event) => event.kind === "system")?.text ?? "";
  const kind =
    firstSystem ? "setup" : toolNames.length > 0 ? "tool" : "direct";
  const title =
    kind === "setup"
      ? "System instructions"
      : firstUser || lastAssistant || toolNames.join(", ") || `Turn ${index}`;
  const preview =
    kind === "setup"
      ? firstSystem
      : firstUser || assistantTexts[0] || toolNames.join(", ") || "No prompt";
  const outputPreview =
    lastAssistant || previewStr(findLastToolResult(events)?.payload ?? "");

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

export function parseTrace(trace: unknown): ParsedTrace {
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
      const text =
        getContent(content) || (typeof content === "string" ? content : "");
      ei += 1;
      events.push(createTraceEvent(ei, role, "user", content, text));
    }
    if (role === "assistant" && typeof content === "string") {
      ei += 1;
      events.push(
        createTraceEvent(ei, role, "assistant-text", content, content),
      );
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
