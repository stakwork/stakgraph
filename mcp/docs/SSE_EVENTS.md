# Real-time Agent Events (SSE)

## Flow

1. `POST /repo/agent` returns `{ request_id, status: "pending", events_token }`
2. Open an SSE connection to `GET /events/:request_id?token=<events_token>`
3. Receive events as the agent works. Each event is a JSON object on a `data:` line.
4. The stream ends with a `done` or `error` event, then the connection closes.

`events_token` is a 1-hour JWT scoped to the `request_id`. It's only present when `API_TOKEN` is set on the server. You can also auth with the `x-api-token` header instead.

## Event types

### `tool_call`

The LLM is invoking a tool. Fires once per tool call with the full input args.

```json
{
  "type": "tool_call",
  "toolName": "file_summary",
  "input": { "path": "src/index.ts" },
  "timestamp": "..."
}
```

### `text`

The LLM produced text output for this step. This is the complete text for the step, not token-by-token chunks.

```json
{ "type": "text", "text": "Based on my analysis...", "timestamp": "..." }
```

### `done`

Agent finished successfully. The full result is still available via `GET /progress?request_id=...`.

```json
{
  "type": "done",
  "result": {
    "final_answer": "...",
    "usage": { "inputTokens": 0, "outputTokens": 0, "totalTokens": 0 }
  },
  "timestamp": "..."
}
```

### `error`

Agent failed.

```json
{ "type": "error", "error": "something went wrong", "timestamp": "..." }
```

Tool results are deliberately excluded from the stream (they can be very large).

## Example (React hook)

```tsx
function useAgentEvents(requestId: string | null, token: string | null) {
  const [events, setEvents] = useState<StepEvent[]>([]);
  const [status, setStatus] = useState<"idle" | "streaming" | "done" | "error">(
    "idle",
  );

  useEffect(() => {
    if (!requestId || !token) return;
    setStatus("streaming");

    const es = new EventSource(`/events/${requestId}?token=${token}`);

    es.onmessage = (e) => {
      const event = JSON.parse(e.data);
      setEvents((prev) => [...prev, event]);
      if (event.type === "done") {
        setStatus("done");
        es.close();
      } else if (event.type === "error") {
        setStatus("error");
        es.close();
      }
    };

    es.onerror = () => {
      setStatus("error");
      es.close();
    };

    return () => es.close();
  }, [requestId, token]);

  return { events, status };
}
```

## Notes

- Events are step-level, not token-level. Each `text` event contains the full text the LLM produced in one tool-loop step.
- The stream auto-closes after `done` or `error`. The bus is cleaned up server-side after 5 seconds.
- If you connect after the agent has already started, you'll only see events from that point forward (no replay).
- The full final result is always available via polling `GET /progress?request_id=...` regardless of whether you used SSE.
