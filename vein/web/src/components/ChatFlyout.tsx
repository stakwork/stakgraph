import { useState, useCallback, useEffect, useRef } from "preact/hooks";
import * as api from "../api";
import { formatJson } from "../helpers";
import { CloseIcon } from "../icons";

// ── Chat Flyout (AI workflow builder) ──────────────────────────────────────

type ChatEntry =
  | { kind: "user"; content: string }
  | { kind: "text"; content: string }
  | { kind: "tool"; calls: api.ToolCallInfo[] };

export function ChatFlyout(props: {
  onClose: () => void;
  onWorkflowCreated: (name: string) => void;
}) {
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
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
          const calls = [...toolBuf];
          setEntries((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.kind === "tool") {
              const next = [...prev];
              next[next.length - 1] = { kind: "tool", calls };
              return next;
            }
            return [...prev, { kind: "tool", calls }];
          });
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
                {entry.calls.map((tc, j) => (
                  <div key={j} class="chat-tool-call">
                    <span class="chat-tool-name">{tc.name}</span>
                    <pre class="chat-tool-input">{formatJson(tc.input)}</pre>
                  </div>
                ))}
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