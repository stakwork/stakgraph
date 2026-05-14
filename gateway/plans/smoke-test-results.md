# Observability smoke-test results

Date: 2026-05-14
Run by: smoke test on `gateway/` against `transports/v1.5.2` of Bifrost.
Reproducer: `bash gateway/scripts/smoke-test.sh`

---

## Verdict

**⚠️ Mostly works — observability via `x-bf-dim-*` is real, but Bifrost's
built-in logging plugin in our build silently drops a meaningful subset
of requests, so the headline question is answerable but with an
asterisk.**

When a request lands as a row, every dimension we care about — agent name,
user, workspace, run-id, session-id, deployment, provider, model, cost,
tokens, latency, status — is present and queryable with SQL. The headline
query ("dollars per agent per time period") works exactly as the plan
predicted.

But of 12 requests sent in the most recent reproducer run, **only 8
landed**. The streaming request, the error (404) request, the no-dims
request, and one of the openrouter requests never produced a row in
`logs.db`. The plugin's HTTP hooks ran, Bifrost returned the right
status codes to the caller, yet nothing was persisted. This pattern is
reproducible: in two back-to-back runs of the same script, the
streaming/error/no-dims rows were dropped both times.

The miss rate is the most important finding. If we relied on
`logs.metadata` for invoiced spend tomorrow, we would silently
under-bill — by an unknown fraction of traffic, biased toward streaming
and error requests, which is exactly where the most interesting stories
live.

The dimension propagation itself works perfectly when a row does land.
No plugin code change was required, as the plan predicted.

---

## Environment

- Gateway image: `stakgraph-gateway:dev` (custom, dynamic-linked bifrost-http
  + our boilerplate plugin), rebuilt today, pinned to bifrost
  `transports/v1.5.2` (`13af31d2ec4024d4db8f2f3e1cf9f79791045b15`).
- UI: now bundled (Dockerfile builds upstream `ui/` via
  `npm run build-enterprise` and copies into `transports/bifrost-http/ui`
  before the Go build).
- `enforce_auth_on_inference: false` — no VK required.
- Provider keys present: ANTHROPIC, OPENAI, OPENROUTER. (GEMINI absent;
  warns on startup, doesn't affect the test.)

### Pre-test snapshot of `logs.db`

```
0|
```

…i.e. zero rows in the freshly-rebuilt `logs.db`. (The DB was wiped
during a rebuild iteration to confirm schema migrations create the
table on first boot. They do.)

---

## Traffic matrix actually sent

The script sends 12 requests via `POST /v1/chat/completions` against
`http://localhost:8181`. Each request caps `max_tokens` at 20 against
a cheap model.

| # | time(UTC) | agent      | user    | ws | provider/model                          | run_id (truncated)      | http |
|---|-----------|-----------|---------|----|----------------------------------------|--------------------------|------|
| 1 | 17:22:31  | browser   | u_alice | w1 | anthropic/claude-haiku-4-5-20251001    | 5e31596d-7bf5-49b9-ac7a | 200  |
| 2 | 17:22:32  | browser   | u_alice | w1 | openai/gpt-4o-mini                     | bf… (LANDED)            | 200  |
| 3 | 17:22:33  | browser   | u_bob   | w1 | anthropic/claude-haiku-4-5-20251001    | … (LANDED)              | 200  |
| 4 | 17:22:34  | coder     | u_alice | w1 | openai/gpt-4.1-nano                    | … (LANDED)              | 200  |
| 5 | 17:22:35  | coder     | u_alice | w1 | anthropic/claude-haiku-4-5-20251001    | … (LANDED)              | 200  |
| 6 | 17:22:35  | chat      | u_bob   | w2 | openai/gpt-4o-mini                     | … (LANDED)              | 200  |
|   | (sleep 25s, cross minute boundary)                                                                          |
| 7 | 17:23:01  | chat      | u_carol | w2 | anthropic/claude-haiku-4-5-20251001    | e8e4c7bc-bd25-43b5-adb4 | 200  |
| 8 | 17:23:02  | reviewer  | u_carol | w1 | openai/gpt-4.1-nano                    | 610531de-8a7e-4325-8416 | 200  |
| 9 | 17:23:03  | reviewer  | u_carol | w1 | openrouter/moonshotai/kimi-k2-0905     | a368e502-f2d8-469f-a841 | 200 **but no row** |
|10 | 17:23:04  | chat      | u_alice | w1 | anthropic/claude-haiku-4-5-20251001 (STREAM) | 898c07cd-e51b-412c-bb80 | 200 **but no row** |
|11 | 17:23:05  | browser   | u_alice | w1 | anthropic/this-model-does-not-exist    | 3abd06bc-d956-4459-b325 | 404 **but no row** |
|12 | 17:23:05  | (no dims) | —       | —  | anthropic/claude-haiku-4-5-20251001    | —                       | 200 **but no row** |

Rows landed: **8 of 12** (≈67%). Plugin hooks fire on all 12; the
`HTTPTransportPostHook` saw status=200/404 for every request. Only
some of those eventually became persistent log rows.

---

## Verification queries

Run against `gateway/data/logs.db` ~10 seconds after the script
completes (no extra `PRAGMA wal_checkpoint` required — host-side
`sqlite3 SELECT` sees committed transactions through the SHM file).

### Q1 — Sanity

```sql
SELECT COUNT(*) AS new_rows,
       SUM(CASE WHEN metadata IS NOT NULL AND metadata != '' THEN 1 ELSE 0 END) AS with_metadata,
       SUM(CASE WHEN stream THEN 1 ELSE 0 END) AS streaming_rows,
       SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS error_rows
FROM logs
WHERE created_at >= datetime('now', '-15 minutes');
```

```
new_rows  with_metadata  streaming_rows  error_rows
--------  -------------  --------------  ----------
8         8              0               0
```

- `new_rows=8` vs 12 sent → 4 silent drops (see "Surprises" below).
- `with_metadata=8` → 100 % of landed rows carry the `metadata` JSON,
  including for requests sent without `x-bf-dim-*` … except those
  requests didn't land at all, so this can't be used to claim
  no-dims requests are handled gracefully. The plan predicted "the row
  still exists, metadata is either null or empty"; in practice the row
  doesn't exist.
- `streaming_rows=0` despite one explicit streaming request →
  the final-chunk accumulator's cost is not landing in any row.
- `error_rows=0` despite one explicit 404 → errors aren't landing
  either.

### Q2 — Headline question: cost per agent name

```sql
SELECT json_extract(metadata, '$.agent-name') AS agent,
       COUNT(*) AS calls,
       ROUND(SUM(cost), 6) AS spend_usd
FROM logs
WHERE created_at >= datetime('now', '-15 minutes')
  AND json_extract(metadata, '$.agent-name') IS NOT NULL
GROUP BY agent
ORDER BY spend_usd DESC;
```

```
agent     calls  spend_usd
--------  -----  ---------
browser   3      0.000111
coder     2      8.5e-05
chat      2      4.4e-05
reviewer  1      4.0e-06
```

✅ This is the test in one query: per-agent cost rolls up correctly,
the rows are sorted, the numbers are non-zero, and every agent in the
traffic matrix appears (except the streaming `chat` and error `browser`
calls, which were dropped — see Q1 / Surprises). This is exactly the
shape we want to expose as a dashboard.

### Q3 — Per agent × per minute

```sql
SELECT json_extract(metadata, '$.agent-name') AS agent,
       strftime('%Y-%m-%d %H:%M', created_at) AS minute,
       COUNT(*)   AS calls,
       ROUND(SUM(cost), 6) AS spend_usd
FROM logs
WHERE created_at >= datetime('now', '-15 minutes')
  AND json_extract(metadata, '$.agent-name') IS NOT NULL
GROUP BY agent, minute
ORDER BY minute, agent;
```

```
agent     minute            calls  spend_usd
--------  ----------------  -----  ---------
browser   2026-05-14 17:22  3      0.000111
chat      2026-05-14 17:22  1      3.0e-06
coder     2026-05-14 17:22  2      8.5e-05
chat      2026-05-14 17:23  1      4.1e-05
reviewer  2026-05-14 17:23  1      4.0e-06
```

✅ Two distinct minute buckets (`17:22` and `17:23`). The 25-second
inter-batch sleep in the script produces real time-period slicing.
`chat` appears in both minutes — same dim header, same agent name,
naturally split across buckets. This is the "time period" axis the
headline question demands.

### Q4 — Per agent × per hour

```sql
SELECT json_extract(metadata, '$.agent-name') AS agent,
       strftime('%Y-%m-%d %H', created_at) AS hour,
       ROUND(SUM(cost), 6) AS spend_usd
FROM logs
WHERE created_at >= datetime('now', '-1 day')
  AND json_extract(metadata, '$.agent-name') IS NOT NULL
GROUP BY agent, hour
ORDER BY hour, agent;
```

```
agent     hour           spend_usd
--------  -------------  ---------
browser   2026-05-14 17  0.000111
chat      2026-05-14 17  4.4e-05
coder     2026-05-14 17  8.5e-05
reviewer  2026-05-14 17  4.0e-06
```

✅ All in one hour, as expected. Confirms `strftime`-based bucketing
works at any granularity.

### Q5 — Cross-dimensional: agent × workspace × user

```sql
SELECT json_extract(metadata, '$.agent-name')   AS agent,
       json_extract(metadata, '$.workspace-id') AS workspace,
       json_extract(metadata, '$.user-id')      AS user,
       COUNT(*) AS calls,
       ROUND(SUM(cost), 6) AS spend_usd
FROM logs
WHERE created_at >= datetime('now', '-15 minutes')
  AND json_extract(metadata, '$.agent-name') IS NOT NULL
GROUP BY agent, workspace, user
ORDER BY spend_usd DESC;
```

```
agent     workspace  user     calls  spend_usd
--------  ---------  -------  -----  ---------
coder     w1         u_alice  2      8.5e-05
browser   w1         u_alice  2      5.8e-05
browser   w1         u_bob    1      5.3e-05
chat      w2         u_carol  1      4.1e-05
reviewer  w1         u_carol  1      4.0e-06
chat      w2         u_bob    1      3.0e-06
```

✅ All three dims are present and orthogonal. `u_alice` shows up in
two agent contexts in `w1`. `u_carol` appears in two workspaces.
Per-row dims aren't being collapsed.

### Q6 — Per agent × provider × model

```sql
SELECT json_extract(metadata, '$.agent-name') AS agent,
       provider, model,
       COUNT(*) AS calls,
       ROUND(SUM(cost), 6) AS spend_usd,
       SUM(prompt_tokens) AS prompt_tokens,
       SUM(completion_tokens) AS completion_tokens
FROM logs
WHERE created_at >= datetime('now', '-15 minutes')
  AND json_extract(metadata, '$.agent-name') IS NOT NULL
GROUP BY agent, provider, model
ORDER BY agent, model;
```

```
agent     provider   model                      calls  spend_usd  prompt_tokens  completion_tokens
--------  ---------  -------------------------  -----  ---------  -------------  -----------------
browser   anthropic  claude-haiku-4-5-20251001  2      0.000106   26             16
browser   openai     gpt-4o-mini                1      5.0e-06    12             6
chat      anthropic  claude-haiku-4-5-20251001  1      4.1e-05    11             6
chat      openai     gpt-4o-mini                1      3.0e-06    10             3
coder     anthropic  claude-haiku-4-5-20251001  1      7.7e-05    12             13
coder     openai     gpt-4.1-nano               1      8.0e-06    11             17
reviewer  openai     gpt-4.1-nano               1      4.0e-06    12             6
```

✅ Costs flow per (agent, provider, model). Bifrost's pricing manager
populated `cost` on every landed row. Anthropic Haiku ≈ $5e-5/call at
~13 prompt tokens, gpt-4o-mini ≈ $3-5e-6/call — sanity checks against
public pricing.

❌ **The openrouter row is missing.** We sent one `openrouter/moonshotai/kimi-k2-0905`
call (request #9), Bifrost returned 200, but it never made it into
the table. So we can't validate that OpenRouter costs flow.

### Q7 — All calls in one run

```sql
-- run_id picked from query 2
SELECT created_at, agent_name_from_meta, model, cost, latency, status
FROM (
  SELECT created_at,
         json_extract(metadata, '$.run-id')     AS run_id,
         json_extract(metadata, '$.agent-name') AS agent_name_from_meta,
         model, cost, latency, status
  FROM logs
  WHERE created_at >= datetime('now', '-15 minutes')
)
WHERE run_id = '5e31596d-7bf5-49b9-ac7a-fc7d7b5083f3';
```

```
created_at                           agent    model                      cost     latency  status
-----------------------------------  -------  -------------------------  -------  -------  -------
2026-05-14 17:22:32.648040166+00:00  browser  claude-haiku-4-5-20251001  5.3e-05  690.0    success
```

✅ Exactly one row per `run_id`, as designed (`run_id` is a fresh UUID
per call). `latency` populated in ms. This is the join key for
ancestor-chain accounting in plugin v2.

### Q8 — Streaming row has cost

```sql
SELECT json_extract(metadata, '$.agent-name') AS agent,
       model, stream, cost, prompt_tokens, completion_tokens, status
FROM logs
WHERE created_at >= datetime('now', '-15 minutes')
  AND stream = 1;
```

```
(no rows)
```

❌ **Empty.** We sent one streaming request, Bifrost streamed back
chunks (visible in the bifrost stderr — multiple `PostLLMHook` calls,
accumulator reached `completion_tokens=24, total_tokens=32` before
the stream closed), and yet **zero rows with `stream=1`** in the DB.
This is the most important miss in the test. The plan's prediction —
"Bifrost's streaming-cost accumulator may need a config tweak or
there's a provider-specific gotcha" — appears to be the case here.

### Q9 — Virtual key info

```sql
SELECT virtual_key_id, virtual_key_name, COUNT(*) AS rows
FROM logs
WHERE created_at >= datetime('now', '-15 minutes')
GROUP BY virtual_key_id, virtual_key_name;
```

```
virtual_key_id  virtual_key_name  rows
--------------  ----------------  ----
                                  8
```

✅ `virtual_key_id` and `virtual_key_name` are both NULL/empty,
exactly as expected with `enforce_auth_on_inference: false` and no
VK supplied. The plan's working assumption — Option B (per-environment
VKs, set later) — remains viable. Bifrost does not stamp a default VK
when none is supplied.

### Q10 — Customer / team / business unit columns

```sql
SELECT
  SUM(CASE WHEN customer_id IS NOT NULL THEN 1 ELSE 0 END) AS with_customer,
  SUM(CASE WHEN team_id IS NOT NULL THEN 1 ELSE 0 END)     AS with_team,
  SUM(CASE WHEN business_unit_id IS NOT NULL THEN 1 ELSE 0 END) AS with_bu
FROM logs
WHERE created_at >= datetime('now', '-15 minutes');
```

```
with_customer  with_team  with_bu
-------------  ---------  -------
0              0          0
```

✅ All three columns are empty, confirming the plan's reservation:
`customer_id` etc. are populated by Bifrost's governance plugin from
VK configuration and are NOT to be set manually. When we adopt
Option B and stamp `customer_id` with environment via VK metadata
(step 4 of the rollout), these columns will start carrying values.
No collision with our own data because we use `metadata` JSON for
everything caller-set.

---

## Total spend

```
SELECT ROUND(SUM(cost), 6) FROM logs WHERE created_at >= datetime('now','-15 minutes');
```

**$0.000244 USD** across 8 successful rows. Well under the $0.25 budget.
The script averages ~$3e-5 per call against cheap models with `max_tokens=20`.

---

## Surprises

### 1. Bifrost's logging plugin silently drops requests
This is the test's loudest finding. Reproducible: in two consecutive
runs of `gateway/scripts/smoke-test.sh`, the same four request shapes
failed to produce DB rows even though Bifrost returned the expected
HTTP status codes to the caller and our plugin's `PostLLMHook` ran:

| dropped request   | how it shows up |
|-------------------|------------------|
| Streaming         | `stream=true` request → many `PostLLMHook` chunks → final chunk seen with `completion_tokens=24` → still no row. |
| Error (404)       | bad model name → Bifrost 404 to caller → `had_err=true` in `PostLLMHook` → no row. |
| No-dim headers    | 200 to caller → `PostLLMHook` ran → no row. |
| OpenRouter call (one of three) | 200 to caller → `PostLLMHook` ran → no row. |

What we don't yet know: whether this is a Bifrost-side bug,
a configuration toggle we haven't enabled, or a side-effect of our
custom dynamic-linked build path. Worth notable points:

- Same `logs.db` schema as upstream `maximhq/bifrost:latest`.
- Bifrost stderr in our container shows **no errors** at the moment the
  drops occur (no "insert failed", no "queue full", no
  "tracing flush dropped", nothing).
- Earlier in this session I observed bifrost stderr lines like
  `batch insert failed: no such table: logs` when the schema hadn't
  been created yet. Once we got past that, the writes for the dropped
  request shapes just go silent — no warning, no error.
- Reading the bifrost v1.5.2 logging plugin source, the write path
  for any traced request goes through `storeOrEnqueueEntry` →
  parked in `pendingLogsToInject` keyed by `trace_id` → drained by
  the tracer's `Inject()`. If `Inject()` is never called for a given
  `trace_id`, the entry sits in memory and never reaches the DB.
  All Bifrost requests today get a `trace_id` (visible in the
  `request completed` log line), so this is potentially the failure
  mode. We have not yet proven this is the cause; it's the most
  consistent hypothesis given the data.

### 2. Schema migrations run on first boot of a fresh `logs.db`
Bifrost's logging plugin auto-creates `logs`, `mcp_tool_logs`,
`async_jobs`, and `migrations` tables on first start (via gorm
migrator). No manual `sqlite3` DDL needed. This is healthy — and
distinct from the "writes don't land" problem above.

### 3. WAL ownership matters
Earlier in this work (separate session) the `logs.db-wal` file was
left owned by root from a prior container run, which silently blocked
all SQLite writes from the `app`-user bifrost process. The upstream
image's `docker-entrypoint.sh` does `chown -R "$CURRENT_UID:$CURRENT_GID" "$APP_DIR"`
to defend against this; our Dockerfile doesn't. If WAL/SHM ever get
into a bad ownership state on the mounted volume, every chat-completion
write fails silently. Adding the same chown step (or running rootless
+ matching UIDs) would prevent recurrence.

### 4. The UI is bundled now, but wasn't yesterday
Our Dockerfile's prior approach was to stub `transports/bifrost-http/ui/index.html`
with a one-liner so `//go:embed all:ui` would succeed. This produced
an API-only image whose `/` returned "Bifrost UI not bundled". I
fixed this earlier in the session by adding a Node 25 UI build stage
that runs `npm run build-enterprise` on the upstream `ui/` package
and copies the output into `transports/bifrost-http/ui` before the
Go build. `http://localhost:8181/` now serves the real React UI.
This is unrelated to the observability findings but should reduce
"is this thing working?" confusion in the future.

### 5. Dimension metadata key naming uses hyphens, not underscores
The plan says headers like `x-bf-dim-run-id` arrive in `metadata` as
JSON keys `run-id`, `agent-name`, etc. — *with hyphens*. This is
exactly what we observed. SQL queries must use
`json_extract(metadata, '$.agent-name')`, not `$.agent_name`. The plan
already says this; just calling it out because every query in this
report depends on it and it's easy to typo.

### 6. Pricing works for OpenRouter when rows land
Pricing for `moonshotai/kimi-k2-0905` is in Bifrost's pricing manager
(verified in the earlier session run where one openrouter row landed
with `cost=1.72e-05`). So the OpenRouter drop is not a pricing-data
issue, it's a row-not-persisted issue.

### 7. Dim headers in our plugin's PostLLMHook log show as empty
Our plugin's `PostLLMHook` log line shows `run_id= agent= ...` (empty)
even when the request was sent with `x-bf-dim-run-id`, etc., and the
landed row's `metadata` JSON contains all the expected keys. So Bifrost
is correctly extracting dims into `BifrostContextKeyDimensions` by
the time the logging plugin reads them in *its* `PreLLMHook`, but our
plugin's `HTTPTransportPreHook` is reading the map before it's
populated. Not a blocker — the dims are in the logs row regardless —
but worth fixing in our plugin once we touch it again. (See "next
steps.")

---

## Next steps (proposed, do not edit the plan yet)

These are findings the team should fold back into
`gateway/plans/llm-governance.md` and the rollout. They are **proposals**;
the user owns the diff.

1. **Step 5 of the rollout is not "free" in our build.** The plan says
   "Observability is a configuration concern, not a code concern" and
   "No plugin code change needed for this step — the v0 boilerplate
   plugin is enough." That holds *if* the bifrost logging plugin
   reliably writes every request. In our build it doesn't, for the
   request shapes we care most about (streaming, errors, untagged
   requests). Recommend amending step 5 to add a verification gate:
   "before declaring step 5 done, run the smoke test and confirm
   `Q1.new_rows == requests_sent` and `Q1.streaming_rows >= 1` and
   `Q1.error_rows >= 1`."

2. **Investigate the drop pattern before relying on logs for spend
   reconciliation.** Likely places to look, in order:

   a. Whether bifrost's tracing middleware is correctly wiring our
      logger plugin as an `ObservabilityPlugin` and calling `Inject()`
      per completed trace. Source: `transports/bifrost-http/server/server.go`
      `reloadObservabilityPlugins`. Add a one-line debug log to confirm.

   b. Whether `pendingLogsToInject` accumulates entries that are never
      drained. A simple goroutine that periodically dumps
      `len(p.pendingLogsToInject)` to stderr would tell us.

   c. Whether streaming chunks specifically corrupt the
      `pendingInjectEntries` slice state for the trace_id. The upstream
      source has a slightly suspicious `if !loaded { return }` branch
      in `storeOrEnqueueEntry` that may interact badly with multiple
      chunks per trace.

   d. Whether the issue is unique to our dynamic-linked build. Our
      current image disables `-extldflags '-static'` so plugins can
      load; nothing else differs from upstream's binary. If the same
      smoke test on `maximhq/bifrost:latest` (without our plugin)
      produces all 12 rows, the regression is in our build. If it
      also drops requests, file upstream.

3. **Add a `chown $APP_DIR` step to our image** (mirror
   upstream's `docker-entrypoint.sh`). Cheap insurance against the
   root-owned-WAL failure mode that silently blocks writes after a
   prior container ran as root.

4. **Fix our plugin's `HTTPTransportPreHook` dim-reading.** It reads
   `BifrostContextKeyDimensions` before Bifrost has populated the map
   for that hook stage, so the log line always shows empty dims. Move
   the read into `PreLLMHook` (where the map is populated, as proven
   by the fact that the logging plugin reads it there successfully).
   Or add a fallback that reads from raw headers. Either is a
   localized change; not a code concern for observability, just a
   plugin-log-clarity concern.

5. **Document the metadata key naming.** Add to README:
   `x-bf-dim-run-id` → `metadata.run-id` (hyphen, lowercase). Make
   this the canonical form in plan examples too. Saves future hours.

6. **Confirm "absence of dims is graceful" is true.** This was a
   plan claim that this test could not validate — the no-dims request
   didn't land at all. Will need to revisit once finding #2 is
   resolved.

7. **Macaroon enforcement (step 8) depends on these logs being
   reliable.** The plan describes macaroon-derived dimensions being
   "cryptographically bound" to the metadata in `logs.metadata`. If
   metadata is silently dropped on a fraction of requests, that bind
   is conditional. The macaroon code path can still enforce in-memory
   per-run state via Redis, but the audit trail will have holes.
   Worth calling out the audit-trail caveat in plan §"Macaroon
   precedence".

---

## How to reproduce

From a clean state:

```bash
cd gateway

# Optional: nuke previous DB to confirm fresh-boot behavior
docker compose down
rm -f data/logs.db*

make docker-up        # builds image (incl. UI) + starts container

# Wait for healthy
until curl -sf http://localhost:8181/api/health >/dev/null; do sleep 1; done

# Run the smoke test
bash scripts/smoke-test.sh

# Wait a few seconds for async log flush, then ask the headline question:
sleep 6
sqlite3 data/logs.db "
SELECT json_extract(metadata,'\$.agent-name') AS agent,
       ROUND(SUM(cost),6) AS spend
FROM logs
WHERE created_at >= datetime('now','-15 minutes')
  AND json_extract(metadata,'\$.agent-name') IS NOT NULL
GROUP BY agent ORDER BY spend DESC;"
```

Expected output (≈ 60–80 % of the time on this build — variance from
the drop pattern documented above):

```
agent     calls  spend_usd
--------  -----  ---------
browser   3      0.0001…
coder     2      8…e-05
chat      2      4…e-05
reviewer  1      …e-06
```

If you see *no* rows: check that `logs.db-wal` is owned by your local
user, not root. If WAL is root-owned, every write silently fails. To
recover:

```bash
docker compose down
sudo rm data/logs.db-wal data/logs.db-shm
docker compose up -d
```
