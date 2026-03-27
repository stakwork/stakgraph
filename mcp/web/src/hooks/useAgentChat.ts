import { useCallback, useEffect, useRef, useMemo } from "react";
import { useChat, type ToolCallEvent } from "@/stores/useChat";
import { useIngestion } from "@/stores/useIngestion";
import { useGraphData } from "@/stores/useGraphData";

const API_BASE = import.meta.env.VITE_API_BASE || "";

/**
 * Hook that wires together:
 *  1. POST /repo/agent  (kick off the agent)
 *  2. GET /events/:request_id  (SSE for tool calls + text + done)
 *  3. GET /progress?request_id=...  (poll fallback if SSE misses "done")
 */
export function useAgentChat() {
  const {
    status,
    requestId,
    eventsToken,
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

  // Derive repo URL from graph Repository nodes, fall back to ingestion store
  const repoUrl = useMemo(() => {
    if (data?.nodes) {
      const repoNodes = data.nodes.filter((n) => n.node_type === "Repository");
      if (repoNodes.length > 0) {
        return repoNodes
          .map((n) => {
            const sourceLink = n.properties.source_link as string | undefined;
            if (sourceLink) return sourceLink;
            return `https://github.com/${n.properties.name}`;
          })
          .join(",");
      }
    }
    return storedRepoUrl;
  }, [data, storedRepoUrl]);

  const esRef = useRef<EventSource | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Clean up SSE + polling on unmount
  useEffect(() => {
    return () => {
      esRef.current?.close();
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // Subscribe to SSE whenever requestId changes
  useEffect(() => {
    // Close previous
    esRef.current?.close();
    esRef.current = null;
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }

    if (!requestId) return;

    const params = new URLSearchParams();
    if (eventsToken) params.set("token", eventsToken);
    const url = `${API_BASE}/events/${requestId}${params.toString() ? "?" + params.toString() : ""}`;

    const es = new EventSource(url);
    esRef.current = es;

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        switch (data.type) {
          case "tool_call":
            addToolCall({
              id: data.toolName + "-" + Date.now(),
              toolName: data.toolName,
              input: data.input,
              timestamp: data.timestamp,
            } satisfies ToolCallEvent);
            break;
          case "text":
            if (data.text) appendText(data.text);
            break;
          case "done":
            es.close();
            esRef.current = null;
            finishResponse(
              data.result?.final_answer,
              data.result?.sessionId,
            );
            break;
          case "error":
            es.close();
            esRef.current = null;
            setError(data.error || "Agent error");
            break;
        }
      } catch {
        // ignore malformed events
      }
    };

    es.onerror = () => {
      es.close();
      esRef.current = null;
      // Fall back to polling
      startPolling(requestId);
    };

    return () => {
      es.close();
      esRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestId]);

  function startPolling(rid: string) {
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/progress?request_id=${rid}`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.status === "complete" || data.result) {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          finishResponse(
            data.result?.final_answer,
            data.result?.sessionId,
          );
        } else if (data.status === "error" || data.error) {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          setError(data.error || "Agent error");
        }
      } catch {
        // keep polling
      }
    }, 3000);
  }

  const send = useCallback(
    async (prompt: string) => {
      if (!repoUrl) {
        setError("No repository configured. Ingest a repo first.");
        return;
      }

      addUserMessage(prompt);

      try {
        const body: Record<string, unknown> = {
          repo_url: repoUrl,
          prompt,
          toolsConfig: { learn_concepts: true },
        };
        if (username) body.username = username;
        if (pat) body.pat = pat;
        if (sessionId) body.sessionId = sessionId;

        const res = await fetch(`${API_BASE}/repo/agent`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.error || `HTTP ${res.status}`);
        }

        const data = await res.json();
        setPending(data.request_id, data.events_token || null);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [repoUrl, username, pat, sessionId, addUserMessage, setPending, setError],
  );

  return { send, clearChat, status, repoUrl };
}
