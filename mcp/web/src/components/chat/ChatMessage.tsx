import { motion } from "framer-motion";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import type { ChatMessage as ChatMessageType } from "@/stores/useChat";

interface ChatMessageProps {
  message: ChatMessageType;
  isStreaming?: boolean;
}

export function ChatMessage({ message, isStreaming = false }: ChatMessageProps) {
  const isUser = message.role === "user";

  if (!message.content.trim()) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="pointer-events-auto flex justify-center w-full"
    >
      <div
        className={`max-w-[70vw] sm:max-w-[450px] md:max-w-[500px] lg:max-w-[600px] ${
          isUser ? "" : "w-full"
        }`}
      >
        <div
          className={`rounded-2xl px-4 py-3 shadow-sm backdrop-blur-sm ${
            isUser
              ? "bg-white/10 text-white inline-block"
              : "bg-muted/10"
          }`}
        >
          {isStreaming ? (
            <div className="text-sm whitespace-pre-wrap text-foreground/90">
              {message.content}
            </div>
          ) : (
            <MarkdownRenderer className="text-sm">
              {message.content}
            </MarkdownRenderer>
          )}
        </div>
      </div>
    </motion.div>
  );
}
