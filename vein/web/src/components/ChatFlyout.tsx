import { useState, useCallback, useEffect, useRef } from "preact/hooks";
import * as api from "../api";
import * as storage from "../storage";
import { formatJson } from "../helpers";
import { CloseIcon, HistoryIcon, CopyIcon, CheckIcon } from "../icons";
import { FlyoutResizer } from "./FlyoutResizer";

// ── Chat Flyout (AI workflow builder) ──────────────────────────────────────

type ToolGroup = { name: string; calls: api.ToolCallInfo[] };

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

/** Serialize the conversation as markdown — for pasting into an issue or
 *  another chat when debugging. Tool inputs are included as JSON blocks. */
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
        }
      }
    }
  }
  return blocks.join("\n\n");
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
  const inputRef = useRef<HTMLInputElement>(null);
  // Highest turn this client has rendered/streamed — the poll below compares
  // against it to notice server-initiated turns (run notifications).
  const seenTurn = useRef(-1);

  useEffect(() => { inputRef.current?.focus(); }, []);

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
  const streamCallbacks = useCallback((): api.ChatCallbacks => {
    let textBuf = "";
    let toolBuf: api.ToolCallInfo[] = [];
    // A trailing text/tool entry may only be replaced in place by the step
    // that created it — a new step must append, not clobber the previous
    // step's entry (which may be the same kind with no bubble in between).
    let stepHasTextEntry = false;
    let stepHasToolEntry = false;
    return {
      onTextDelta: (delta) => {
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
        if (tr.name === "run_workflow" && tr.input?.name && tr.output?.runId) {
          props.onWorkflowRan(tr.input.name, tr.output.runId);
        }
      },
      onStepFinish: () => {
        textBuf = "";
        toolBuf = [];
        stepHasTextEntry = false;
        stepHasToolEntry = false;
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
      seenTurn.current = meta.currentTurn;
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
            setLoading(true);
            await api.streamChat(chatId, meta.currentTurn, streamCallbacks());
          }
        }
      } catch {
        // Server briefly unreachable — keep polling.
      }
    }, TURN_POLL_MS);
    return () => clearInterval(t);
  }, [chatId, loading, showHistory, streamCallbacks]);

  const newChat = useCallback(() => {
    storage.remove(CHAT_ID_KEY);
    setChatId(null);
    setEntries([]);
    setExpanded({});
    setShowHistory(false);
    seenTurn.current = -1;
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
      seenTurn.current = turn;
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