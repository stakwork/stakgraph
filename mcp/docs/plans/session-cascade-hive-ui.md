# Session Cascade in Hive — Legal Benchmark Runs

Build the **session cascade visualization** (mockup: `mcp/docs/mockups/session-cascade.html`
in the stakgraph repo — open it in a browser; it is self-contained and interactive)
into the hive UI's legal benchmark Runs section, showing the **entire agent stack of
one benchmark run**: every top-level agent in sequence, each agent's sub-agent tree,
and every Concept action (read/created), live while the run executes and identically
after it finishes.

All server-side data comes from the stakgraph mcp service (`:3355` on the swarm) via
the Turn-chain work on stakgraph PR #1568. No new stakgraph endpoints are needed —
this plan is hive-side only.

---

## 1. The data (stakgraph `:3355`, header `x-api-token: <swarm key>`)

### 1a. All sessions of a run

```
GET :3355/api/sessions?agent_name_contains=<runIdentifier>&limit=200
```

Each row (only relevant fields shown):

```json
{
  "id": "b0e35724-4f39-49b5-9394-5037a4931bf5",
  "parent_session_id": "",
  "agent_name": "repair-agent-147813394",
  "source": "repo_agent",
  "status": "running",            // 'running' | 'success' | 'error' | 'aborted'
  "turn_count": 61,               // VERSION COUNTER — unchanged = nothing new
  "last_turn_at": 1755450000000,  // epoch ms | null
  "timestamp": "2026-08-17T18:00:00.000Z",
  "model": "claude-sonnet-5",
  "repo": "stakwork/hive",
  "token_usage": { "input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "total": 412000 },
  "child_count": 2
}
```

- `agent_name` is set by the benchmark workflow when it POSTs `/repo/agent` with
  `agentName`; the run identifier is embedded in the name. **Top-level sessions of
  the run** = rows returned by this filter (children have `agent_name: ""`,
  `source: "graph_sub_agent"` and are NOT matched by the filter). Order them by
  `timestamp` ascending — that is the agent chain.
- **Open question for the builder to confirm with one real run**: whether the
  workflow embeds the Stakwork **projectId** (`StakworkRun.projectId`, the number in
  names like `repair-agent-147813394`) or the hive cuid. The proxy route should look
  up the run row server-side and try `projectId` first.

### 1b. Sub-agent tree of one session

```
GET :3355/api/sessions/:id            → detail; `children: [...]` = direct children
GET :3355/api/sessions/:id?recursive=true  → adds `descendants: [...]` (flat, any depth,
                                             each with parent_session_id — rebuild the tree client-side)
```

Children are also discoverable purely by id shape: `<parent>-sub-<8hex>` nests
arbitrarily (`X-sub-a1b2c3d4-sub-e5f6a7b8`).

### 1c. The turn chain of one session (the polling workhorse)

```
GET :3355/api/sessions/:id/turns?after=<order>&limit=1000
```

```json
{
  "session_id": "…",
  "status": "running",
  "turn_count": 61,
  "last_turn_at": 1755450000000,
  "turns": [
    { "order": 0, "turn_id": "repair-agent-147813394-<sid>-turn-0",
      "turn_type": "user_input", "tool": null, "tool_call_id": null,
      "content": "Inspect the current running environment…", "timestamp": 1755449000000,
      "concepts": [] },
    { "order": 5, "turn_type": "tool_call", "tool": "graph_get", "tool_call_id": "c9",
      "content": "{\"ref_id\":\"abc-123\"}", "timestamp": 1755449020000, "concepts": [] },
    { "order": 6, "turn_type": "tool_result", "tool": "graph_get", "tool_call_id": "c9",
      "content": "{\"type\":\"json\",\"value\":{…", "timestamp": 1755449021000,
      "concepts": [ { "ref_id": "abc-123", "id": "g-1", "name": "wfa-ontology" } ] }
  ]
}
```

- `turn_type` ∈ `user_input | reasoning | tool_call | tool_result | response`.
- `content`: full text for user_input/reasoning/response; full input JSON for
  tool_call; tool_result truncated to 100 chars (display as-is; it is a preview).
- `concepts` non-empty ⇒ this turn READ those Concepts (the graph has a
  `(Turn)-[:READ_CONCEPT]->(Concept)` edge).
- `timestamp` is null on backfilled (pre-feature) sessions — render the time gutter
  blank, never fake it.
- Protocol: first call `after=-1` (or omit) → history; then poll with
  `after=<max order seen>`; **stop polling when `status !== 'running'`**. Dedupe by
  `order` — overlap is harmless by design.

### 1d. Live protocol for the whole run (flat cost regardless of run size)

1. Poll **1a** (the run's session list) on the legal section's standard cadence.
2. Diff `turn_count` per session against what's rendered. Unchanged ⇒ skip.
3. For each changed session (in a sequential pipeline that is ~1–2 at a time),
   fetch **1c** with its cursor.
4. New sessions appearing in the list ⇒ new agents starting (render immediately —
   the AgentSession node exists from run start with `status: 'running'`).
5. Whole run is finished when no session has `status === 'running'` (plus the
   StakworkRun's own status from hive's existing hooks).

---

## 2. Mapping the mockup to real data

The mockup renders "story rows", not raw turns. The fold from a turn array to rows
(pure function, unit-testable — put it in `src/lib/legal-cascade/derive.ts`):

- **user row** ← each `user_input` turn.
- **concept row** ← each turn with `concepts.length > 0` (one row per concept).
  Verb `READ`. These NEVER collapse into pills.
- **concept CREATED row (display-parse, v1)** ← each `tool_call` turn whose `tool` ∈
  `create_triplet | create_batch_triplet | create_node | edit_node`. Parse
  `content` (full input JSON) for a human label (e.g. `node_data.name`,
  `node_type`); verb `+ CREATED` (or `EDITED` for edit_node). This is display-only
  provenance — the graph does not yet carry `WROTE` edges (flagged as follow-up in
  stakgraph). If parsing fails, fold the turn into the surrounding pill instead.
- **response row** ← each `response` turn.
- **pill row** ← every maximal contiguous run of remaining
  `reasoning | tool_call | tool_result` turns. Pill data: span `[o0, o1]`
  (min/max `order`), `calls` = count of tool_call, `texts` = reasoning contents,
  `mix` = per-tool tally string (`"graph_search ×14 · graph_get ×5"`), duration
  from first/last `timestamp` when present.
  - Expanding a pill = rendering the turns already in memory (the fold keeps them);
    no extra fetch needed since 1c returns full pages of 1000.
- **sub-agent fork row** ← a `tool_call` turn with `tool === "graph_sub_agent"`.
  Join to the child session: the tool_call `content` JSON has `prompt`, and the
  child's turn 0 (`user_input`) has the same text; fallback join by start-time
  order among unmatched children. The matching `tool_result` (same `tool_call_id`)
  is the **merge point** where the child's lane curves back.
  - The child session's own rows render on the next lane (depth = number of
    `-sub-` segments in its id), between fork and merge, exactly as the mockup
    draws lanes 1 and 2. Clicking the agent header reveals its turn-0 prompt
    (already in memory from the child's own 1c fetch).
- **live head** ← last session with `status === 'running'`: pulsing dot below its
  last row; the bottom pill's count ticks up in place as new turns arrive
  (append to the open pill rather than re-folding the whole list — fold is
  incremental-friendly since new turns only ever extend the tail).

### The run-level view (what the mockup doesn't show)

The mockup is one session. The run page stacks the cascades: one **top-level
section per agent** (ordered by `timestamp`), each headed by its `agent_name`,
status badge, token/turn counts, with its cascade below (children nested inside).
Between consecutive agents draw the lane-0 spine continuing with the dashed
"hand-off" style the mockup uses between same-lane segments. A run-level summary
strip on top: N agents, M sub-agents, K concepts touched (union of chips),
total tokens. Optionally a right-hand run-wide concept rail aggregating every
chip with counts (a concept read by 4 agents shows ×4) — v2 if time-boxed.

---

## 3. Hive integration (follow these conventions exactly)

### 3a. Placement — new sibling route under the run

`/w/[slug]/legal/benchmarks/runs/[runId]/cascade` → new file
`src/app/w/[slug]/legal/benchmarks/runs/[runId]/cascade/page.tsx`.

- Copy the authz + data-load preamble **verbatim** from
  `src/app/w/[slug]/legal/benchmarks/runs/[runId]/report/page.tsx:41-107`
  (`getServerSession` → `resolveWorkspaceAccess` → `requireMemberAccess` →
  `canReadRunReport(member.role)` → IDOR-guarded
  `db.stakworkRun.findFirst({ where: { id, workspaceId, type: LEGAL_BENCHMARK_RUNNER } })`).
- Link it from `ReportCell` in `src/components/legal/BenchmarkRunsHistory.tsx:506-535`
  next to "View Report" (e.g. "Trace" / "Cascade"), and optionally from the report
  page header.

### 3b. Proxy routes (the browser NEVER calls `:3355` directly)

New route dir `src/app/api/workspaces/[slug]/legal/benchmarks/cascade/`, following
the rubrics template `src/app/api/workspaces/[slug]/legal/benchmarks/rubrics/route.ts`
step-for-step (`runtime = "nodejs"`, `fetchCache = "force-no-store"`, openlaw slug
gate, `checkRateLimit`, `getWorkspaceSwarmAccess`, `USE_MOCKS` branch). Base URL:
`transformSwarmUrlToRepo2Graph(swarmUrl)` (→ `:3355`), header
`x-api-token: swarmApiKey`. Two handlers:

1. `GET …/cascade/sessions?runId=<cuid>` — server looks up the StakworkRun row,
   derives the run identifier (`projectId` first — see §1a open question), calls
   stakgraph `GET /api/sessions?agent_name_contains=…`, returns an explicit field
   allowlist (mirror `RunResponseRow` discipline in
   `src/app/api/stakwork/runs/route.ts:23-39`).
2. `GET …/cascade/turns?sessionId=…&after=…` — proxies §1c. Also accept
   `recursive` session-detail proxying here or as a third handler
   (`…/cascade/session?sessionId=…&recursive=true`) for the child tree.

Validate that any `sessionId` requested belongs to the run (prefix/agent-name
check against the sessions list, cached per request) so the proxy can't be used
to read arbitrary swarm sessions.

Mock mode: add `src/app/api/mock/…` fixtures with a small 2-agent, 1-child,
2-concept run so the page works under `USE_MOCKS=true`.

### 3c. Components & files

```
src/lib/legal-cascade/derive.ts        # fold: turns[] -> Row[]; pill spans; fork/merge joins; run assembly
src/lib/legal-cascade/types.ts         # Turn, SessionSummary, Row union, RunCascadeModel
src/hooks/useRunCascade.ts             # polling per §1d; legal-section conventions:
                                       #   plain fetch + setInterval (15s like useLegalBenchmarkRunList),
                                       #   poll only while any session is 'running',
                                       #   Pusher STAKWORK_RUN_UPDATE as a refetch nudge
src/components/legal/RunCascade.tsx    # the trace: lanes, spines, fork/merge beziers (inline SVG à la HillClimbChart)
src/components/legal/CascadeRow.tsx    # row renderers: user / pill / concept chip / agent header / response
src/components/legal/CascadeHeader.tsx # run summary strip
```

- **Rendering**: absolutely-positioned rows over one `<svg>` for spines/curves,
  exactly like the mockup's DOM structure — it translates 1:1 to React. Use the
  mockup's geometry (34px rows, lane x-offsets, bezier fork/merge) as the spec;
  keep `min-w-[980px]` + `overflow-x-auto` like `Gantt.tsx`.
- **Style**: Tailwind v4 utility classes with explicit `dark:` variants (the
  section's rule — see `BenchmarkRunsHistory.tsx:604`). Reuse
  `src/components/run-report/chrome.tsx` primitives (`Section`, `Panel`, `Chip`,
  `StatusBadge`, `Kicker`) so it matches the report's editorial/monospace house
  style. Map the mockup's palette to theme-appropriate Tailwind colors (lanes:
  cyan/violet/rose families; concept amber) rather than hardcoding its hex values.
- **Interactions from the mockup to keep**: pill click → unroll/fold that span;
  agent header click → reveal its turn-0 prompt; hover tooltips showing
  `turn <order> · <turn_type> · <tool>` and content preview (shadcn `Tooltip`);
  "expand all"; reduced-motion respected (`motion-safe:` variants). The cascade
  entry animation is nice-to-have; skip if it fights React re-renders — the live
  view's "append without reflow" matters more.
- `data-testid` on rows/pills/chips (`cascade-agent-${sid}`, `cascade-pill-${o0}`,
  `cascade-concept-${ref_id}`) — the section relies on them for tests.

### 3d. Tests

- Unit-test `derive.ts` hard (this is where all the correctness lives): folding,
  pill spans, fork/merge joining by prompt and by fallback order, CREATED parse,
  incremental append (feeding turns in two batches must equal one batch).
  Location: `src/__tests__/unit/lib/legal-cascade/derive.test.ts`.
- Hook test mirroring `useLegalBenchmarkRun*.test.ts`: polls while running, stops
  on completion, dedupes by order.
- Component smoke test with the mock fixture.

---

## 4. Build order

1. `types.ts` + `derive.ts` + unit tests (pure, no UI, no network).
2. Proxy routes + mock fixtures (verify with curl against a real swarm run).
3. `useRunCascade` hook (history load → live polling → stop).
4. `RunCascade` render, static first (mock fixture), then live.
5. Page route + authz preamble + link from `ReportCell`.
6. Polish: tooltips, expand-all, summary strip, empty/error states
   (`EmptyPanel`, `SectionErrorBoundary`).

**Acceptance**: open a finished benchmark run → full stack renders (agents in
order, children nested, concept chips on the rail, pills collapse/expand); open a
run mid-flight → rows appear as the agent works, new agents appear when the
workflow advances, polling stops when the run finishes; a pre-feature (backfilled)
run renders identically minus timestamps; `USE_MOCKS=true` works offline.

## 5. Known limits / follow-ups (do not block on these)

- **CREATED chips are display-parsed** from tool_call inputs in v1; real
  `(Turn)-[:WROTE]->(node)` edges are a stakgraph follow-up.
- Backfilled sessions have **no per-turn timestamps** (time gutter blank) and
  legacy tool_results show `tool: "unknown"` in old orphan chains only — sessions
  served by the turns endpoint always have real tool names.
- stakgraph's `/api/sessions` router is currently mounted pre-auth on the swarm;
  hive must still send `x-api-token` (harmless now, required if/when stakgraph
  gates that router — flagged on stakgraph PR #1568).
- If a child session's chain is empty (crashed before its node existed), render
  the agent header with an "no trace" badge rather than omitting the fork.
