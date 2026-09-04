import { useState, useCallback, useEffect, useRef } from "preact/hooks";
import * as api from "../api";
import * as storage from "../storage";
import { formatJson } from "../helpers";
import { CloseIcon, HistoryIcon, CopyIcon, CheckIcon } from "../icons";
import { ToolResultView } from "./ToolResultView";
import { FlyoutResizer } from "./FlyoutResizer";
import { Markdown } from "./Markdown";

// ── Chat Flyout (AI workflow builder) ──────────────────────────────────────

/** A tool RESULT paired to its call. `isError` = the tool threw. */
type ToolResult = { output: unknown; isError: boolean };
type ToolCall = api.ToolCallInfo & { result?: ToolResult };
type ToolGroup = { name: string; calls: ToolCall[] };

type ChatEntry =
  | { kind: "user"; content: string }
  | { kind: "notice"; content: string }
  | { kind: "text"; content: string }
  | { kind: "tool"; groups: ToolGroup[] };

// Persist the active chat id so closing/reopening the flyout (or the whole
// browser) reattaches to the same detached session.
const CHAT_ID_KEY = "activeChatId";

// The active chat is also mirrored into the URL (?chat=<id>) so a link
// deep-links straight into it; the param wins over the persisted id.
const CHAT_URL_PARAM = "chat";

function setChatUrlParam(id: string | null) {
  const url = new URL(location.href);
  if (id) url.searchParams.set(CHAT_URL_PARAM, id);
  else url.searchParams.delete(CHAT_URL_PARAM);
  history.replaceState(null, "", url);
}

// Server-initiated wake-up messages (a detached run finished) are stored as
// user-role messages with this prefix; render them as a notice, not a bubble.
const NOTIFICATION_PREFIX = "[run-notification]";

// While the flyout is open and idle, poll for server-initiated turns (a
// detached run finishing starts a turn no client action triggered).
const TURN_POLL_MS = 4000;

// Coalesce consecutive same-name tool calls into groups.
function groupCalls(calls: ToolCall[]): ToolGroup[] {
  const groups: ToolGroup[] = [];
  for (const tc of calls) {
    const last = groups[groups.length - 1];
    if (last && last.name === tc.name) {
      last.calls.push(tc);
    } else {
      groups.push({ name: tc.name, calls: [tc] });
    }
  }
  return groups;
}

/** Attach a result to the call with the same id, searching tool entries
 *  from the end (the call is almost always in the last one). Returns the
 *  matched call (for callers that need its input) or null. Pure: returns a
 *  new entries array when something changed. */
function attachResult(
  entries: ChatEntry[],
  toolCallId: string | undefined,
  result: ToolResult,
): { entries: ChatEntry[]; call: ToolCall | null } {
  if (!toolCallId) return { entries, call: null };
  for (let i = entries.length - 1; i >= 0; i--) {
    const e = entries[i]!;
    if (e.kind !== "tool") continue;
    for (let j = 0; j < e.groups.length; j++) {
      const g = e.groups[j]!;
      const k = g.calls.findIndex((c) => c.toolCallId === toolCallId);
      if (k < 0) continue;
      const call: ToolCall = { ...g.calls[k]!, result };
      const groups = e.groups.slice();
      groups[j] = { ...g, calls: g.calls.map((c, idx) => (idx === k ? call : c)) };
      const next = entries.slice();
      next[i] = { kind: "tool", groups };
      return { entries: next, call };
    }
  }
  return { entries, call: null };
}

/** Unwrap a stored AI SDK tool-result `output` ({ type, value }) into the
 *  plain value + error flag the UI renders. */
function storedOutputToResult(output: any): ToolResult {
  if (output && typeof output === "object" && typeof output.type === "string" && "value" in output) {
    return { output: output.value, isError: String(output.type).startsWith("error") };
  }
  return { output, isError: false };
}

/** Serialize the conversation as markdown — for pasting into an issue or
 *  another chat when debugging. Tool inputs and results are included as JSON
 *  blocks. */
function transcriptText(entries: ChatEntry[]): string {
  const blocks: string[] = [];
  for (const e of entries) {
    if (e.kind === "user") {
      blocks.push(`### User\n${e.content}`);
    } else if (e.kind === "notice") {
      blocks.push(`### Notice\n${e.content}`);
    } else if (e.kind === "text") {
      blocks.push(`### Assistant\n${e.content}`);
    } else {
      for (const g of e.groups) {
        for (const tc of g.calls) {
          blocks.push(`### Tool call: ${g.name}\n\`\`\`json\n${formatJson(tc.input)}\n\`\`\``);
          if (tc.result) {
            const label = tc.result.isError ? "Tool error" : "Tool result";
            blocks.push(`### ${label}: ${g.name}\n\`\`\`json\n${formatJson(tc.result.output)}\n\`\`\``);
          }
        }
      }
    }
  }
  return blocks.join("\n\n");
}

type ToolStatus = "pending" | "ok" | "error" | "unknown";

const STATUS_TITLE: Record<ToolStatus, string> = {
  pending: "Running",
  ok: "Completed",
  error: "Failed",
  unknown: "No result recorded",
};

/** Roll a group's calls up to one status for the chip's dot. A call with no
 *  result is "pending" only while this is the live tail of the chat;
 *  otherwise the result was never recorded (e.g. the turn died). */
function groupStatus(g: ToolGroup, live: boolean): ToolStatus {
  let status: ToolStatus = "ok";
  for (const c of g.calls) {
    if (!c.result) return live ? "pending" : "unknown";
    if (c.result.isError) status = "error";
  }
  return status;
}

/** Compact relative timestamp (e.g. "3m", "2h", "5d") with absolute fallback. */
function relativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const secs = Math.round((Date.now() - then) / 1000);
  if (secs < 60) return "just now";
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(iso).toLocaleDateString();
}

/** Pull plain text out of an AI SDK message content (string or parts array). */
function extractText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((p: any) => p?.type === "text" && typeof p.text === "string")
      .map((p: any) => p.text)
      .join("");
  }
  return "";
}

/** Render stored ModelMessages (the transcript) into display entries. Tool
 *  RESULT messages (role "tool") aren't bubbles — each result is attached to
 *  its call (by toolCallId) so the call's chip can disclose it. */
function transcriptToEntries(messages: { role: string; content: unknown }[]): ChatEntry[] {
  let entries: ChatEntry[] = [];
  for (const m of messages) {
    if (m.role === "tool" && Array.isArray(m.content)) {
      for (const part of m.content as any[]) {
        if (part?.type !== "tool-result") continue;
        entries = attachResult(entries, part.toolCallId, storedOutputToResult(part.output)).entries;
      }
      continue;
    }
    if (m.role === "user") {
      const text = extractText(m.content);
      if (text) {
        entries.push(
          text.startsWith(NOTIFICATION_PREFIX)
            ? { kind: "notice", content: text }
            : { kind: "user", content: text },
        );
      }
    } else if (m.role === "assistant") {
      if (typeof m.content === "string") {
        if (m.content) entries.push({ kind: "text", content: m.content });
      } else if (Array.isArray(m.content)) {
        const calls: ToolCall[] = [];
        for (const part of m.content as any[]) {
          if (part?.type === "text" && part.text) {
            entries.push({ kind: "text", content: part.text });
          } else if (part?.type === "tool-call") {
            calls.push({ name: part.toolName, input: part.input, toolCallId: part.toolCallId });
          }
        }
        if (calls.length) entries.push({ kind: "tool", groups: groupCalls(calls) });
      }
    }
  }
  return entries;
}

export function ChatFlyout(props: {
  onClose: () => void;
  onWorkflowCreated: (name: string) => void;
  onWorkflowRan: (name: string, runId: string) => void;
}) {
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [chatId, setChatId] = useState<string | null>(() =>
    new URLSearchParams(location.search).get(CHAT_URL_PARAM) ??
    storage.load<string | null>(CHAT_ID_KEY, null),
  );
  const [showHistory, setShowHistory] = useState(false);
  const [chats, setChats] = useState<api.ChatMeta[]>([]);
  const [copied, setCopied] = useState(false);

  const copyTranscript = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(transcriptText(entries));
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // Clipboard API unavailable (e.g. insecure context) — no-op.
    }
  }, [entries]);

  const toggleExpanded = (key: string) =>
    setExpanded((prev) => ({ ...prev, [key]: !prev[key] }));
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  // Highest turn this client has rendered/streamed — the poll below compares
  // against it to notice server-initiated turns (run notifications).
  const seenTurn = useRef(-1);
  // The stream this client is attached to (at most one). Switching chats or
  // starting a new one aborts it — the turn keeps running server-side and
  // `loadChat` reattaches when you come back. Several chats can be live at
  // once; `loading` only means "the chat I'm LOOKING AT has a live turn".
  const streamRef = useRef<AbortController | null>(null);
  const detach = useCallback(() => {
    streamRef.current?.abort();
    streamRef.current = null;
  }, []);
  useEffect(() => detach, [detach]);

  useEffect(() => { inputRef.current?.focus(); }, []);

  // Auto-grow the input with its content (CSS max-height caps it); shrinks
  // back when cleared on send.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [input]);

  // Mirror the active chat into the URL; clear the param when the flyout
  // closes so a reload doesn't unexpectedly reopen the panel.
  useEffect(() => { setChatUrlParam(chatId); }, [chatId]);
  useEffect(() => () => setChatUrlParam(null), []);

  // Auto-scroll on new content
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  // Build the incremental stream callbacks. Each `step.finish` starts a fresh
  // bubble; tool calls/results also drive the canvas (workflow created/ran).
  const streamCallbacks = useCallback((signal: AbortSignal): api.ChatCallbacks => {
    let textBuf = "";
    let toolBuf: ToolCall[] = [];
    // A trailing text/tool entry may only be replaced in place by the step
    // that created it — a new step must append, not clobber the previous
    // step's entry (which may be the same kind with no bubble in between).
    let stepHasTextEntry = false;
    let stepHasToolEntry = false;
    return {
      onTextDelta: (delta) => {
        if (signal.aborted) return;
        textBuf += delta;
        const content = textBuf;
        const replace = stepHasTextEntry;
        stepHasTextEntry = true;
        setEntries((prev) => {
          const last = prev[prev.length - 1];
          if (replace && last && last.kind === "text") {
            const next = [...prev];
            next[next.length - 1] = { kind: "text", content };
            return next;
          }
          return [...prev, { kind: "text", content }];
        });
      },
      onToolCall: (tc) => {
        if (signal.aborted) return;
        toolBuf.push(tc);
        if (tc.name === "create_workflow" && tc.input?.name) {
          props.onWorkflowCreated(tc.input.name);
        }
        const groups = groupCalls(toolBuf);
        const replace = stepHasToolEntry;
        stepHasToolEntry = true;
        setEntries((prev) => {
          const last = prev[prev.length - 1];
          if (replace && last && last.kind === "tool") {
            const next = [...prev];
            next[next.length - 1] = { kind: "tool", groups };
            return next;
          }
          return [...prev, { kind: "tool", groups }];
        });
      },
      onToolResult: (tr) => {
        if (signal.aborted) return;
        const result: ToolResult = { output: tr.output, isError: !!tr.isError };
        // Keep the step buffer in sync so a later same-step re-render
        // (another tool call) doesn't drop the result again.
        toolBuf = toolBuf.map((c) => (c.toolCallId === tr.toolCallId ? { ...c, result } : c));
        setEntries((prev) => attachResult(prev, tr.toolCallId, result).entries);
        // The tool-output event carries no input; recover it from the call.
        const input = tr.input ?? toolBuf.find((c) => c.toolCallId === tr.toolCallId)?.input;
        if (tr.name === "run_workflow" && !tr.isError && input?.name && tr.output?.runId) {
          props.onWorkflowRan(input.name, tr.output.runId);
        }
      },
      onStepFinish: () => {
        textBuf = "";
        toolBuf = [];
        stepHasTextEntry = false;
        stepHasToolEntry = false;
      },
      onFinish: () => {
        if (signal.aborted) return;
        setLoading(false);
      },
    };
  }, [props.onWorkflowCreated, props.onWorkflowRan]);

  /** Attach to a turn's stream (replacing any current attachment) and
   *  follow it to its end. */
  const attach = useCallback(async (id: string, turn: number) => {
    detach();
    const ac = new AbortController();
    streamRef.current = ac;
    setLoading(true);
    try {
      await api.streamChat(id, turn, streamCallbacks(ac.signal), ac.signal);
    } finally {
      if (streamRef.current === ac) streamRef.current = null;
    }
  }, [detach, streamCallbacks]);

  // Load a chat's transcript and — if a turn is still live server-side (we may
  // have closed the tab) — reattach to its stream. Shared by mount-restore and
  // clicking an entry in the history list.
  const loadChat = useCallback(async (id: string) => {
    detach();
    setLoading(false);
    setShowHistory(false);
    setExpanded({});
    setChatId(id);
    storage.save(CHAT_ID_KEY, id);
    try {
      const { meta, messages } = await api.getChat(id);
      setEntries(transcriptToEntries(messages));
      seenTurn.current = meta.currentTurn;
      if (meta.status === "live" && meta.currentTurn >= 0) {
        await attach(id, meta.currentTurn);
      }
    } catch {
      // Stale id (e.g. workspace wiped) — drop it and start fresh.
      storage.remove(CHAT_ID_KEY);
      setChatId(null);
      setEntries([]);
    }
  }, [detach, attach]);

  // On mount: restore the persisted chat (if any).
  useEffect(() => {
    if (chatId) loadChat(chatId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // While idle, watch for SERVER-INITIATED turns: when a detached run
  // finishes, the server appends a [run-notification] message and launches a
  // turn on its own — no client action to key off, so poll. On a new turn:
  // re-render the transcript (it now holds the notification) and, if the
  // turn is live, attach to its stream.
  useEffect(() => {
    if (!chatId || loading || showHistory) return;
    const t = setInterval(async () => {
      try {
        const { meta, messages } = await api.getChat(chatId);
        if (meta.currentTurn > seenTurn.current) {
          seenTurn.current = meta.currentTurn;
          setEntries(transcriptToEntries(messages));
          if (meta.status === "live") {
            await attach(chatId, meta.currentTurn);
          }
        }
      } catch {
        // Server briefly unreachable — keep polling.
      }
    }, TURN_POLL_MS);
    return () => clearInterval(t);
  }, [chatId, loading, showHistory, attach]);

  // Always available — a live turn in the current chat keeps running
  // detached; it's just no longer the one on screen.
  const newChat = useCallback(() => {
    detach();
    setLoading(false);
    storage.remove(CHAT_ID_KEY);
    setChatId(null);
    setEntries([]);
    setExpanded({});
    setShowHistory(false);
    seenTurn.current = -1;
  }, [detach]);

  // Toggle the history list, fetching the latest sessions when opening.
  const toggleHistory = useCallback(async () => {
    if (showHistory) {
      setShowHistory(false);
      return;
    }
    setShowHistory(true);
    try {
      setChats(await api.listChats());
    } catch {
      setChats([]);
    }
  }, [showHistory]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    setEntries((prev) => [...prev, { kind: "user", content: text }]);
    setInput("");
    setLoading(true);

    try {
      const { chatId: id, turn } = await api.sendChat(text, chatId ?? undefined);
      if (!chatId) {
        setChatId(id);
        storage.save(CHAT_ID_KEY, id);
      }
      seenTurn.current = turn;
      await attach(id, turn);
    } catch (err) {
      if ((err as { status?: number }).status === 409 && chatId) {
        // The chat is still working on an earlier message (we were
        // detached, e.g. after switching away) — reattach, keep the draft.
        setInput(text);
        await loadChat(chatId);
        return;
      }
      setEntries((prev) => [...prev, { kind: "text", content: "Error connecting to AI." }]);
      setLoading(false);
    }
  }, [input, loading, chatId, attach, loadChat]);

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div class="flyout chat-flyout">
      <FlyoutResizer />
      <div class="flyout-header">
        <div>
          <div class="flyout-eyebrow">AI Builder</div>
          <div class="flyout-title">Create Workflow</div>
        </div>
        <div class="chat-header-actions">
          {entries.length > 0 && (
            <button
              class="flyout-close chat-history-btn"
              onClick={copyTranscript}
              aria-label="Copy conversation"
              title={copied ? "Copied!" : "Copy conversation"}
            >
              {copied ? <CheckIcon size={15} /> : <CopyIcon size={15} />}
            </button>
          )}
          <button
            class={`flyout-close chat-history-btn${showHistory ? " is-active" : ""}`}
            onClick={toggleHistory}
            aria-label="Chat history"
            aria-expanded={showHistory}
          >
            <HistoryIcon />
          </button>
          <button class="btn" onClick={newChat}>New chat</button>
          <button class="flyout-close" onClick={props.onClose} aria-label="Close"><CloseIcon /></button>
        </div>
      </div>
      {showHistory ? (
        <div class="chat-history">
          {chats.length === 0 ? (
            <div class="chat-empty">No chat sessions yet.</div>
          ) : (
            chats.map((ch) => (
              <button
                key={ch.id}
                type="button"
                class={`chat-history-item${ch.id === chatId ? " is-current" : ""}`}
                onClick={() => loadChat(ch.id)}
              >
                {ch.status === "live" && (
                  <span class="chat-history-live" title="Working" aria-label="Working" />
                )}
                <span class="chat-history-title">{ch.title || "Untitled chat"}</span>
                <span class="chat-history-time">{relativeTime(ch.updatedAt)}</span>
              </button>
            ))
          )}
        </div>
      ) : (
      <div class="chat-messages" ref={scrollRef}>
        {entries.length === 0 && (
          <div class="chat-empty">Describe the workflow you want to build.</div>
        )}
        {entries.map((entry, i) => {
          if (entry.kind === "user") {
            return (
              <div key={i} class="chat-msg chat-msg-user">
                <div class="chat-msg-text">{entry.content}</div>
              </div>
            );
          }
          if (entry.kind === "notice") {
            return (
              <div key={i} class="chat-msg chat-msg-notice">
                <div class="chat-msg-text">{entry.content}</div>
              </div>
            );
          }
          if (entry.kind === "tool") {
            return (
              <div key={i} class="chat-tool-calls">
                {entry.groups.map((g, j) => {
                  const key = `${i}:${j}`;
                  const isOpen = !!expanded[key];
                  const count = g.calls.length;
                  const status = groupStatus(g, loading && i === entries.length - 1);
                  return (
                    <div key={j} class={`chat-tool-call${isOpen ? " is-open" : ""}`}>
                      <button
                        type="button"
                        class="chat-tool-head"
                        onClick={() => toggleExpanded(key)}
                        aria-expanded={isOpen}
                      >
                        <span class={`chat-tool-dot is-${status}`} title={STATUS_TITLE[status]} />
                        <span class="chat-tool-name">{g.name}</span>
                        <span class="chat-tool-meta">
                          {status === "error" && <span class="chat-tool-err">error</span>}
                          {count > 1 && <span class="chat-tool-count">×{count}</span>}
                        </span>
                      </button>
                      {isOpen && (
                        <div class="chat-tool-body">
                          {g.calls.map((tc, k) => {
                            const rkey = `${key}:${k}:result`;
                            return (
                              <div key={k} class="chat-tool-item">
                                <pre class="chat-tool-input">{formatJson(tc.input)}</pre>
                                <ToolResultView
                                  result={tc.result}
                                  open={!!expanded[rkey]}
                                  onToggle={() => toggleExpanded(rkey)}
                                />
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            );
          }
          // kind === "text" — plain text while tokens are still streaming into
          // this bubble (the last entry of a live turn); markdown once done.
          const streaming = loading && i === entries.length - 1;
          return (
            <div key={i} class="chat-msg chat-msg-assistant">
              {streaming
                ? <div class="chat-msg-text">{entry.content}</div>
                : <Markdown class="chat-msg-text" source={entry.content} />}
            </div>
          );
        })}
        {loading && (entries.length === 0 || entries[entries.length - 1]?.kind === "user") && (
          <div class="chat-msg chat-msg-assistant">
            <div class="chat-msg-text chat-thinking">Thinking...</div>
          </div>
        )}
      </div>
      )}
      <div class="chat-input-row">
        <textarea
          ref={inputRef}
          rows={1}
          value={input}
          onInput={(e) => setInput((e.target as HTMLTextAreaElement).value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe your workflow..."
          disabled={loading}
        />
        <button class="btn btn-primary" onClick={send} disabled={loading}>Send</button>
      </div>
    </div>
  );
}