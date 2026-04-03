import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import type { ChatMessage as ChatMessageType } from "@/stores/useChat";

interface ChatMessageProps {
  message: ChatMessageType;
  isStreaming?: boolean;
}

export function ChatMessage({
  message,
  isStreaming = false,
}: ChatMessageProps) {
  const isUser = message.role === "user";
  const [streamingContent, setStreamingContent] = useState(message.content);
  const lastUpdateRef = useRef(0);
  const timeoutRef = useRef<number | null>(null);

  useEffect(() => {
    if (!isStreaming) {
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      setStreamingContent(message.content);
      lastUpdateRef.current = Date.now();
      return;
    }

    const THROTTLE_MS = 80;
    const now = Date.now();
    const elapsed = now - lastUpdateRef.current;

    const flush = () => {
      setStreamingContent(message.content);
      lastUpdateRef.current = Date.now();
      timeoutRef.current = null;
    };

    if (elapsed >= THROTTLE_MS) {
      flush();
      return;
    }

    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = window.setTimeout(flush, THROTTLE_MS - elapsed);

    return () => {
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [isStreaming, message.content]);

  if (!message.content.trim()) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="pointer-events-auto flex justify-center w-full"
    >
      <div
        className={`max-w-[75vw] sm:max-w-[520px] md:max-w-[620px] lg:max-w-[720px] ${
          isUser ? "" : "w-full"
        }`}
      >
        <div
          className={`rounded-2xl px-4 py-3 shadow-sm backdrop-blur-sm ${
            isUser ? "bg-white/10 text-white inline-block" : "bg-muted/10"
          }`}
        >
          <MarkdownRenderer className="text-sm">
            {isStreaming ? streamingContent : message.content}
          </MarkdownRenderer>
        </div>
      </div>
    </motion.div>
  );
}
