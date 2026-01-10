import { ModelMessage } from "ai";

export function sanitizePrompt(prompt: string | undefined | null): string {
  if (!prompt || typeof prompt !== "string") {
    return "";
  }

  return prompt
    .replace(/\$/g, "\\$")
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "")
    .replace(/\s{10,}/g, " ".repeat(5));
}

export function sanitizeMessages(messages: ModelMessage[]): ModelMessage[] {
  return messages.map((message) => {
    if (typeof message.content === "string") {
      return {
        ...message,
        content: sanitizePrompt(message.content),
      } as ModelMessage;
    }

    if (Array.isArray(message.content)) {
      const sanitizedContent = message.content.map((part: any) => {
        if (part.type === "text" && typeof part.text === "string") {
          return {
            ...part,
            text: sanitizePrompt(part.text),
          };
        }
        return part;
      });
      return {
        ...message,
        content: sanitizedContent,
      } as ModelMessage;
    }

    return message;
  });
}
