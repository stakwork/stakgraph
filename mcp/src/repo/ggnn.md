Makes sense. For the custom tool you just need three things exposed as config:

- GGNN_URL — the base URL (https://ggnn.sphinx.chat)
- GGNN_API_KEY — the bearer token
- which endpoints to enable (probably just /check to start, /predict before task begins)
  The tool call for /check would be something like:

```ts
async function checkExecution(taskDescription: string, traceSoFar: Message[]) {
  const res = await fetch(`${GGNN_URL}/check`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${GGNN_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      task_description: taskDescription,
      languages: ["typescript"],
      trace_so_far: traceSoFar,
    }),
  });
  return res.json();
}
```

### 1. Before a task: `/predict`

Call this when you know what the agent needs to do but hasn't started yet.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{"task_description": "Add Stripe webhook for subscriptions", "languages": ["typescript"]}'
```

Returns:

- **strategy** — which cluster of past traces this task is most similar to, average efficiency, avg steps
- **guardrails** — concrete limits like "stop after 20 steps" or "don't make 3+ consecutive searches"
- **anti_patterns** — what agents did wrong on similar tasks
- **similar_past_traces** — the K most similar past tasks and how they went

If no model is trained yet, returns `{"cold_start": true}`.

### 2. Mid-execution: `/check`

Call this periodically (e.g. every 5-10 tool calls) with the trace so far. No LLM call — just the GGNN forward pass (~15ms).

```bash
curl -X POST http://localhost:8000/check \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Add Stripe webhook for subscriptions",
    "languages": ["typescript"],
    "trace_so_far": [
      {"role": "user", "content": [{"type": "text", "text": "..."}]},
      {"role": "assistant", "content": [{"type": "tool-call", "toolName": "shell", "input": {"command": "grep webhook src/"}}]},
      {"role": "tool", "content": [{"type": "tool-result", "toolName": "shell", "content": "..."}]}
    ]
  }'
```

Returns:

- **status** — `on_track`, `warning`, or `off_track`
- **efficiency_prediction** — 0.0 to 1.0, how efficient this execution looks so far
- **node_assessments** — per-step labels (`necessary`, `redundant`, `dead_end`, `wrong_direction`) with confidence
- **remaining_budget** — estimated steps remaining before you should reconsider
- **suggested_action** — what high-efficiency agents did in similar situations

### 3. Score a plan: `/score-plan`

Call this after the agent makes a plan but before executing it.

```bash
curl -X POST http://localhost:8000/score-plan \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Add Stripe webhook for subscriptions",
    "languages": ["typescript"],
    "plan": "1. Find existing webhook handlers\n2. Read the pattern\n3. Create new endpoint\n4. Add signature verification\n5. Write tests"
  }'
```

---

### Using GGNN via the agent endpoint

Pass a `ggnn` object in the request body to `POST /repo/agent`. The agent gets GGNN tools registered automatically. The `/check` tool receives the live message trace via a `messagesRef` updated each step.

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/stakwork/hive",
    "prompt": "Add Stripe webhook for subscriptions. Call ggnn_check every 5 tool calls to validate your approach.",
    "ggnn": {
      "url": "https://ggnn.sphinx.chat",
      "apiKey": "sk-",
      "languages": ["typescript"],
      "tools": [
        {
          "name": "ggnn_check",
          "endpoint": "/check",
          "description": "Call this every 5-10 tool calls to check execution. Returns status (on_track/warning/off_track), efficiency prediction, and suggested actions.",
          "bodyType": "check"
        }
      ]
    }
  }' \
  "http://localhost:3355/repo/agent"
```
