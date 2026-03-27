import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import { useChat } from "@/stores/useChat";
import { useAgentChat } from "@/hooks/useAgentChat";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { ToolCallFlow } from "./ToolCallFlow";

export function Chat() {
  const { messages, toolCalls, streamingText, status } = useChat();
  const { send, clearChat } = useAgentChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const isActive = status === "pending" || status === "streaming";
  const hasMessages = messages.length > 0 || isActive;

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, toolCalls, streamingText]);

  return (
    <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-end z-10">
      <div className="pointer-events-none flex flex-col justify-end max-h-[85vh] w-full max-w-2xl px-4">
      {/* Message history - scrolls up from bottom */}
      {hasMessages && (
        <div className="flex-1 min-h-0 overflow-y-auto pb-2 pointer-events-none">
          <div className="space-y-2 flex flex-col items-center">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}

            {/* Tool call flow while agent is working */}
            {isActive && toolCalls.length > 0 && (
              <ToolCallFlow toolCalls={toolCalls} isActive={isActive} />
            )}

            {/* Streaming text preview */}
            {isActive && streamingText && (
              <ChatMessage
                message={{
                  id: "streaming",
                  role: "assistant",
                  content: streamingText,
                  timestamp: Date.now(),
                }}
                isStreaming
              />
            )}

            {/* Pending indicator */}
            {status === "pending" && toolCalls.length === 0 && (
              <div className="pointer-events-auto flex justify-center">
                <div className="rounded-2xl px-4 py-3 bg-muted/10 backdrop-blur-sm">
                  <span className="text-sm text-muted-foreground">Thinking</span>
                  <PulsingDots />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>
      )}

      {/* Clear button */}
      {messages.length > 0 && (
        <div className="pointer-events-auto flex justify-center pb-1">
          <button
            type="button"
            onClick={clearChat}
            className="rounded-full border border-border/50 bg-muted/30 px-3 py-1.5 text-xs text-muted-foreground hover:border-border hover:bg-muted/60 hover:text-foreground transition-all flex items-center gap-1.5"
          >
            <X className="w-3.5 h-3.5" />
            Clear
          </button>
        </div>
      )}

      {/* Input - anchored to bottom */}
      <div className="pointer-events-auto shrink-0">
        <ChatInput onSend={send} disabled={isActive} />
      </div>
      </div>
    </div>
  );
}

function PulsingDots() {
  return (
    <span className="inline-flex gap-0.5 ml-1">
      {[0, 0.2, 0.4].map((delay, i) => (
        <span
          key={i}
          className="text-muted-foreground animate-pulse"
          style={{ animationDelay: `${delay}s` }}
        >
          .
        </span>
      ))}
    </span>
  );
}
