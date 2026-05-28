import { useState, useCallback, useEffect, useRef } from "preact/hooks";
import * as api from "../api";
import { formatJson } from "../helpers";
import { CloseIcon } from "../icons";

// ── Chat Flyout (AI workflow builder) ──────────────────────────────────────

type ToolGroup = { name: string; calls: api.ToolCallInfo[] };

type ChatEntry =
  | { kind: "user"; content: string }
  | { kind: "text"; content: string }
  | { kind: "tool"; groups: ToolGroup[] };

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

export function ChatFlyout(props: {
  onClose: () => void;
  onWorkflowCreated: (name: string) => void;
  onWorkflowRan: (name: string, runId: string) => void;
}) {
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

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

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    setEntries((prev) => [...prev, { kind: "user", content: text }]);
    setInput("");
    setLoading(true);

    // Build API messages: flatten entries into role/content pairs
    const allEntries = [...entries, { kind: "user" as const, content: text }];
    const apiMessages: api.ChatMessage[] = [];
    for (const e of allEntries) {
      if (e.kind === "user") {
        apiMessages.push({ role: "user", content: e.content });
      } else if (e.kind === "text" && e.content) {
        apiMessages.push({ role: "assistant", content: e.content });
      }
    }

    let textBuf = "";
    let toolBuf: api.ToolCallInfo[] = [];

    try {
      await api.chat(apiMessages, {
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
          // Reset buffers so next step starts a fresh bubble
          textBuf = "";
          toolBuf = [];
        },
        onFinish: () => {
          setLoading(false);
        },
      });
    } catch {
      setEntries((prev) => [...prev, { kind: "text", content: "Error connecting to AI." }]);
      setLoading(false);
    }
  }, [input, loading, entries, props.onWorkflowCreated]);

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
        <button class="flyout-close" onClick={props.onClose} aria-label="Close"><CloseIcon /></button>
      </div>
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