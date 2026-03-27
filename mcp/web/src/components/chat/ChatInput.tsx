import { useState, useRef, useEffect } from "react";
import { Send } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Ask about your codebase...",
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const [rows, setRows] = useState(1);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!input) {
      setRows(1);
      return;
    }
    const lineCount = (input.match(/\n/g) || []).length + 1;
    setRows(Math.min(Math.max(1, lineCount + 1), 8));
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || disabled) return;
    onSend(input.trim());
    setInput("");
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-2 w-full py-4">
      <div className="relative flex-1 min-w-0">
        <textarea
          ref={inputRef}
          placeholder={placeholder}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={rows}
          className={`w-full px-4 py-3 pr-12 rounded-3xl bg-background/90 border border-border/50 text-sm text-foreground/95 placeholder:text-muted-foreground/40 focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all resize-none ${
            disabled ? "opacity-50 cursor-not-allowed" : ""
          }`}
        />
        <Button
          type="submit"
          size="icon"
          disabled={!input.trim() || disabled}
          className="absolute right-1.5 bottom-3 h-8 w-8 rounded-full"
        >
          <Send className="w-4 h-4" />
        </Button>
      </div>
    </form>
  );
}
