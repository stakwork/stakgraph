# Session ingest API

Record an agent run that happens **outside stakgraph** (e.g. hive's ai-sdk agents) as a
live Turn chain in the graph:

```
(AgentSession)-[:HAS_TURN]->(Turn)-[:NEXT]->(Turn)-[:NEXT]->…
                              └─[:READ_CONCEPT]->(Concept)
```

Post turns as the run happens; the session is watchable in the sessions UI while it is
still running. The chains are identical in shape to in-process ones (same `turn_id`,
`node_key`, and edges), so nothing downstream can tell them apart.

**Base URL** the stakgraph server (default `:3355`).
**Auth** `x-api-token: $API_TOKEN` on every POST (or `Authorization: Bearer <jwt>`).
When the server runs without `API_TOKEN` set, auth is skipped.
All bodies are JSON. All writes are idempotent unless noted.

---

## 1. Start the session

```
POST /api/sessions
{
  "session_id": "hive-run-8f21",   // required; your id, becomes the node key
  "source": "hive",                 // shows up as the session's source facet
  "repo": "stakwork/hive",          // optional
  "agent_name": "planner",          // optional; free-form label for grouping runs
  "parent_session_id": "...",       // optional; creates (parent)-[:SPAWNED]->(this)
  "spawn_tool_call_id": "...",      // optional; pins a sub-agent to the parent's Turn
  "start_time": 1755400000000       // optional epoch ms, defaults to now
}
→ 201 { "session_id": "hive-run-8f21", "status": "running", "start_time": … }
```

Call this **first** — turns anchor on the session node, and posting turns for an unknown
session returns `404`. Re-posting the same `session_id` is safe (keeps the original node).

## 2. Stream turns

```
POST /api/sessions/:id/turns
{
  "agent": "hive",        // label in turn_ids ("hive-<session>-turn-0"); pass it consistently
  "turns": [
    { "turn_type": "user_input", "content": "fix the flaky test" },
    { "turn_type": "reasoning",  "content": "checking the retry helper" },
    { "turn_type": "tool_call",   "content": "{\"query\":\"retry\"}", "tool": "search", "tool_call_id": "c1" },
    { "turn_type": "tool_result", "content": "…raw tool output…",      "tool": "search", "tool_call_id": "c1",
      "concepts": [{ "ref_id": "…" }] }
  ]
}
→ 201 { "session_id": …, "written": 4, "next_order": 4,
        "turns": [{ "order": 0, "turn_id": "hive-…-turn-0", "node_key": "turn-…", "turn_type": "user_input" }, …] }
```

- `turn_type` ∈ `user_input` | `reasoning` | `tool_call` | `tool_result` | `response`.
- **Ordering is automatic**: each batch continues from the session's highest existing
  turn. The graph is the cursor, so a crashed process resumes by just posting again.
  Pass `start_order` to pin it yourself (re-posting an order overwrites that turn).
- One writer per session at a time. If you must post concurrently, pin `start_order`.
- `timestamp` (epoch ms) is optional per turn; defaults to server receipt time.
- `concepts` is optional per turn: `[{ "ref_id": … } | { "id": … }]` — gitree Concept
  graph ref_id (preferred) or gitree id. Unmatched entries are skipped, not errors.
- Max 500 turns per batch. Post one batch per agent step for a live feed.

### Mapping ai-sdk steps to turns

| ai-sdk part | `turn_type` | `content` | also send |
|---|---|---|---|
| user message text | `user_input` | the text | — |
| assistant text | `reasoning` | the text | — |
| `tool-call` | `tool_call` | `JSON.stringify(input)` | `tool`, `tool_call_id` |
| `tool-result` | `tool_result` | the raw output (string or object) | `tool`, `tool_call_id` |

Within a step, send assistant parts first and tool results after. Skip system messages
and empty text parts — the in-process emitter does, and turns with no content are noise.

Emit the run's final answer as `reasoning` like any other assistant text; `/end` retypes
it to `response`.

`tool_result` content is stored as `{"type":"text","value":<content>}` (or `"json"` for
objects) truncated to 100 chars — parity with in-process sessions, and it keeps large
payloads out of the graph. Other content is capped at 100k chars.

## 3. End the session

```
POST /api/sessions/:id/end
{
  "status": "success",              // success | error | aborted (default success)
  "error_message": "",              // optional
  "model": "claude-opus-5",         // optional; provider is derived when omitted
  "provider": "anthropic",          // optional
  "end_time": 1755400123000,        // optional epoch ms, defaults to now
  "duration_ms": 123000,            // optional, defaults to end_time - start_time
  "usage": { "input_tokens": 0, "output_tokens": 0,
             "cache_read_tokens": 0, "cache_write_tokens": 0, "total_tokens": 0 },
  "finalize_response": true         // default true: retype the last `reasoning` turn to `response`
}
→ 200 { "session_id": …, "status": "success", "end_time": …, "finalized_turn": "turn-…" }
```

**Call this exactly once per run** — token counts accumulate on the session node, so a
second call double-counts them. (Resuming a session later and ending it again is the one
legitimate case: the totals are meant to sum across runs.)

## 4. Concept rollup (optional)

Session-level `READ_CONCEPT` edges with the agent's own ranking — the counterpart of the
per-turn `concepts` above, which record *that* a concept was read; these record *how
load-bearing it was*.

```
POST /api/sessions/:id/concepts
{ "concepts": [
    { "ref_id": "…", "read_order": 0, "rank": 1,
      "evidence": "why it mattered", "contradicts": "what it got wrong" }
] }
→ 200 { "session_id": …, "linked": 1, "submitted": 1 }
```

Full-state mirror, not a merge: every call overwrites the edge properties, and a `null`
`rank` clears it. Send the complete list each time.

## Reading it back

```
GET /api/sessions/:id/turns?after=<order>&limit=1000   # poll for a live feed
GET /api/sessions/:id                                  # session detail + totals
GET /api/sessions?source=hive                          # list runs
```

`after` is exclusive; start at `-1`. Read endpoints need no auth.

## Errors

| code | meaning |
|---|---|
| 400 | malformed body (bad `turn_type`, empty batch, batch > 500, `start_order` < 0, bad session id) |
| 401 | missing/invalid API token |
| 404 | session node doesn't exist — `POST /api/sessions` first |
| 503 | graph unavailable |

## Minimal flow

```bash
H='-H content-type:application/json -H x-api-token:'"$API_TOKEN"
curl -sX POST $STAKGRAPH/api/sessions $H \
  -d '{"session_id":"hive-run-1","source":"hive","repo":"stakwork/hive"}'

curl -sX POST $STAKGRAPH/api/sessions/hive-run-1/turns $H \
  -d '{"agent":"hive","turns":[{"turn_type":"user_input","content":"fix the flaky test"}]}'

curl -sX POST $STAKGRAPH/api/sessions/hive-run-1/end $H \
  -d '{"status":"success","model":"claude-opus-5","usage":{"input_tokens":12000,"output_tokens":800}}'
```
