import { create } from "zustand";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

export interface ToolCallEvent {
  id: string;
  toolName: string;
  input?: unknown;
  timestamp: string;
}

export type AgentStatus = "idle" | "pending" | "streaming" | "done" | "error";

interface ChatState {
  messages: ChatMessage[];
  /** Tool calls received via SSE for the current request */
  toolCalls: ToolCallEvent[];
  /** Accumulated text chunks from SSE for the current response */
  streamingText: string;
  status: AgentStatus;
  /** Current request_id (for SSE subscription) */
  requestId: string | null;
  /** JWT token for SSE auth */
  eventsToken: string | null;
  /** Session ID for multi-turn conversation */
  sessionId: string | null;
  errorMessage: string | null;

  addUserMessage: (content: string) => void;
  setPending: (requestId: string, eventsToken: string | null) => void;
  setStreaming: () => void;
  addToolCall: (tc: ToolCallEvent) => void;
  appendText: (text: string) => void;
  finishResponse: (finalAnswer?: string, sessionId?: string) => void;
  setError: (message: string) => void;
  clearChat: () => void;
}

let msgCounter = 0;

export const useChat = create<ChatState>((set) => ({
  messages: [],
  toolCalls: [],
  streamingText: "",
  status: "idle",
  requestId: null,
  eventsToken: null,
  sessionId: null,
  errorMessage: null,

  addUserMessage: (content) =>
    set((s) => ({
      messages: [
        ...s.messages,
        {
          id: `msg-${++msgCounter}`,
          role: "user",
          content,
          timestamp: Date.now(),
        },
      ],
    })),

  setPending: (requestId, eventsToken) =>
    set({
      status: "pending",
      requestId,
      eventsToken,
      toolCalls: [],
      streamingText: "",
      errorMessage: null,
    }),

  setStreaming: () => set({ status: "streaming" }),

  addToolCall: (tc) =>
    set((s) => ({
      status: "streaming",
      toolCalls: [...s.toolCalls, tc],
    })),

  appendText: (text) =>
    set((s) => ({
      status: "streaming",
      streamingText: s.streamingText + text,
    })),

  finishResponse: (finalAnswer, sessionId) =>
    set((s) => {
      const content = finalAnswer || s.streamingText || "(no response)";
      return {
        status: "done",
        messages: [
          ...s.messages,
          {
            id: `msg-${++msgCounter}`,
            role: "assistant",
            content,
            timestamp: Date.now(),
          },
        ],
        toolCalls: [],
        streamingText: "",
        requestId: null,
        eventsToken: null,
        ...(sessionId ? { sessionId } : {}),
      };
    }),

  setError: (message) =>
    set((s) => ({
      status: "error",
      errorMessage: message,
      messages: [
        ...s.messages,
        {
          id: `msg-${++msgCounter}`,
          role: "assistant",
          content: `Error: ${message}`,
          timestamp: Date.now(),
        },
      ],
      toolCalls: [],
      streamingText: "",
      requestId: null,
      eventsToken: null,
    })),

  clearChat: () =>
    set({
      messages: [],
      toolCalls: [],
      streamingText: "",
      status: "idle",
      requestId: null,
      eventsToken: null,
      sessionId: null,
      errorMessage: null,
    }),
}));
