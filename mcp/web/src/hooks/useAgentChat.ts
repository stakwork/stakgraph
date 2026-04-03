import { useCallback, useEffect, useRef, useMemo } from "react";
import { useChat, type ToolCallEvent } from "@/stores/useChat";
import { useIngestion } from "@/stores/useIngestion";
import { useGraphData } from "@/stores/useGraphData";
import { useSettings } from "@/stores/useSettings";
import { resolveRepoUrl } from "@/lib/utils";

const API_BASE = import.meta.env.VITE_API_BASE || "";

/**
 * Hook that calls POST /repo/agent with stream=true and
 * consumes the AI SDK UI message stream for real-time text + tool calls.
 */
export function useAgentChat() {
  const {
    status,
    sessionId,
    addUserMessage,
    setPending,
    addToolCall,
    appendText,
    finishResponse,
    setError,
    clearChat,
  } = useChat();

  const { repoUrl: storedRepoUrl, username, pat } = useIngestion();
  const data = useGraphData((s) => s.data);
  const { model, apiKey } = useSettings();

  const repoUrl = useMemo(
    () => resolveRepoUrl(data, storedRepoUrl),
    [data, storedRepoUrl],
  );

  const abortRef = useRef<AbortController | null>(null);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  /** Parse a single AI SDK UIMessageChunk */
  const processChunk = useCallback(
    (chunk: Record<string, unknown>) => {
      switch (chunk.type) {
        case "text-delta":
          appendText((chunk.delta || chunk.textDelta || "") as string);
          break;
        case "tool-input-available":
          addToolCall({
            id: (chunk.toolCallId || chunk.toolName + "-" + Date.now()) as string,
            toolName: chunk.toolName as string,
            input: chunk.input,
            timestamp: new Date().toISOString(),
          } satisfies ToolCallEvent);
          break;
        // ignore tool-output-available, start-step, finish-step, etc.
      }
    },
    [appendText, addToolCall],
  );

  const send = useCallback(
    async (prompt: string) => {
      if (!repoUrl) {
        setError("No repository configured. Ingest a repo first.");
        return;
      }

      addUserMessage(prompt);
      setPending("streaming", null);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const body: Record<string, unknown> = {
          repo_url: repoUrl,
          prompt,
          stream: true,
          toolsConfig: { learn_concepts: true },
        };
        if (username) body.username = username;
        if (pat) body.pat = pat;
        if (sessionId) body.sessionId = sessionId;
        if (model) body.model = model;
        if (apiKey) body.apiKey = apiKey;

        const res = await fetch(`${API_BASE}/repo/agent`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.error || `HTTP ${res.status}`);
        }

        const reader = res.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // SSE format: "data: {json}\n\n"
          const events = buffer.split("\n\n");
          buffer = events.pop() || ""; // keep incomplete event in buffer

          for (const event of events) {
            for (const line of event.split("\n")) {
              if (!line.startsWith("data: ")) continue;
              const payload = line.slice(6); // strip "data: "

              try {
                const chunk = JSON.parse(payload);
                processChunk(chunk);
              } catch {
                // not valid JSON, skip
              }
            }
          }
        }

        // Stream finished normally
        finishResponse();
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        abortRef.current = null;
      }
    },
    [repoUrl, username, pat, sessionId, addUserMessage, setPending, setError, finishResponse, processChunk],
  );

  return { send, clearChat, status, repoUrl };
}
