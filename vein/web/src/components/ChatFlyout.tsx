import { useState, useCallback, useEffect, useRef } from "preact/hooks";
import * as api from "../api";
import * as storage from "../storage";
import { formatJson } from "../helpers";
import { CloseIcon, HistoryIcon } from "../icons";

// ── Chat Flyout (AI workflow builder) ──────────────────────────────────────

type ToolGroup = { name: string; calls: api.ToolCallInfo[] };

type ChatEntry =
  | { kind: "user"; content: string }
  | { kind: "text"; content: string }
  | { kind: "tool"; groups: ToolGroup[] };

// Persist the active chat id so closing/reopening the flyout (or the whole
// browser) reattaches to the same detached session.
const CHAT_ID_KEY = "activeChatId";

// Coalesce consecutive same-name tool calls into groups.
function groupCalls(calls: api.ToolCallInfo[]): ToolGroup[] {
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
 *  RESULT messages (role "tool") aren't shown as bubbles — only the calls. */
function transcriptToEntries(messages: { role: string; content: unknown }[]): ChatEntry[] {
  const entries: ChatEntry[] = [];
  for (const m of messages) {
    if (m.role === "user") {
      const text = extractText(m.content);
      if (text) entries.push({ kind: "user", content: text });
    } else if (m.role === "assistant") {
      if (typeof m.content === "string") {
        if (m.content) entries.push({ kind: "text", content: m.content });
      } else if (Array.isArray(m.content)) {
        const calls: api.ToolCallInfo[] = [];
        for (const part of m.content as any[]) {
          if (part?.type === "text" && part.text) {
            entries.push({ kind: "text", content: part.text });
          } else if (part?.type === "tool-call") {
            calls.push({ name: part.toolName, input: part.input });
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
    storage.load<string | null>(CHAT_ID_KEY, null),
  );
  const [showHistory, setShowHistory] = useState(false);
  const [chats, setChats] = useState<api.ChatMeta[]>([]);

  const toggleExpanded = (key: string) =>
    setExpanded((prev) => ({ ...prev, [key]: !prev[key] }));
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  // Auto-scroll on new content
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  // Build the incremental stream callbacks. Each `step.finish` starts a fresh
  // bubble; tool calls/results also drive the canvas (workflow created/ran).
  const streamCallbacks = useCallback((): api.ChatCallbacks => {
    let textBuf = "";
    let toolBuf: api.ToolCallInfo[] = [];
    return {
      onTextDelta: (delta) => {
        textBuf += delta;
        const content = textBuf;
        setEntries((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.kind === "text") {
            const next = [...prev];
            next[next.length - 1] = { kind: "text", content };
            return next;
          }
          return [...prev, { kind: "text", content }];
        });
      },
      onToolCall: (tc) => {
        toolBuf.push(tc);
        if (tc.name === "create_workflow" && tc.input?.name) {
          props.onWorkflowCreated(tc.input.name);
        }
        const groups = groupCalls(toolBuf);
        setEntries((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.kind === "tool") {
            const next = [...prev];
            next[next.length - 1] = { kind: "tool", groups };
            return next;
          }
          return [...prev, { kind: "tool", groups }];
        });
      },
      onToolResult: (tr) => {
        if (tr.name === "run_workflow" && tr.input?.name && tr.output?.runId) {
          props.onWorkflowRan(tr.input.name, tr.output.runId);
        }
      },
      onStepFinish: () => {
        textBuf = "";
        toolBuf = [];
      },
      onFinish: () => {
        setLoading(false);
      },
    };
  }, [props.onWorkflowCreated, props.onWorkflowRan]);

  // Load a chat's transcript and — if a turn is still live server-side (we may
  // have closed the tab) — reattach to its stream. Shared by mount-restore and
  // clicking an entry in the history list.
  const loadChat = useCallback(async (id: string) => {
    setShowHistory(false);
    setExpanded({});
    setChatId(id);
    storage.save(CHAT_ID_KEY, id);
    try {
      const { meta, messages } = await api.getChat(id);
      setEntries(transcriptToEntries(messages));
      if (meta.status === "live" && meta.currentTurn >= 0) {
        setLoading(true);
        await api.streamChat(id, meta.currentTurn, streamCallbacks());
      }
    } catch {
      // Stale id (e.g. workspace wiped) — drop it and start fresh.
      storage.remove(CHAT_ID_KEY);
      setChatId(null);
      setEntries([]);
    }
  }, [streamCallbacks]);

  // On mount: restore the persisted chat (if any).
  useEffect(() => {
    if (chatId) loadChat(chatId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const newChat = useCallback(() => {
    storage.remove(CHAT_ID_KEY);
    setChatId(null);
    setEntries([]);
    setExpanded({});
    setShowHistory(false);
  }, []);

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
      await api.streamChat(id, turn, streamCallbacks());
    } catch {
      setEntries((prev) => [...prev, { kind: "text", content: "Error connecting to AI." }]);
      setLoading(false);
    }
  }, [input, loading, chatId, streamCallbacks]);

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div class="flyout chat-flyout">
      <div class="flyout-header">
        <div>
          <div class="flyout-eyebrow">AI Builder</div>
          <div class="flyout-title">Create Workflow</div>
        </div>
        <div class="chat-header-actions">
          <button
            class={`flyout-close chat-history-btn${showHistory ? " is-active" : ""}`}
            onClick={toggleHistory}
            aria-label="Chat history"
            aria-expanded={showHistory}
          >
            <HistoryIcon />
          </button>
          <button class="btn" onClick={newChat} disabled={loading}>New chat</button>
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
          if (entry.kind === "tool") {
            return (
              <div key={i} class="chat-tool-calls">
                {entry.groups.map((g, j) => {
                  const key = `${i}:${j}`;
                  const isOpen = !!expanded[key];
                  const count = g.calls.length;
                  return (
                    <div key={j} class={`chat-tool-call${isOpen ? " is-open" : ""}`}>
                      <button
                        type="button"
                        class="chat-tool-head"
                        onClick={() => toggleExpanded(key)}
                        aria-expanded={isOpen}
                      >
                        <span class="chat-tool-name">{g.name}</span>
                        {count > 1 && <span class="chat-tool-count">×{count}</span>}
                      </button>
                      {isOpen && (
                        <div class="chat-tool-body">
                          {g.calls.map((tc, k) => (
                            <pre key={k} class="chat-tool-input">{formatJson(tc.input)}</pre>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            );
          }
          // kind === "text"
          return (
            <div key={i} class="chat-msg chat-msg-assistant">
              <div class="chat-msg-text">{entry.content}</div>
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
        <input
          ref={inputRef}
          type="text"
          value={input}
          onInput={(e) => setInput((e.target as HTMLInputElement).value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe your workflow..."
          disabled={loading}
        />
        <button class="btn btn-primary" onClick={send} disabled={loading}>Send</button>
      </div>
    </div>
  );
}