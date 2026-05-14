# Bifrost log-drop investigation — post-mortem

**Status:** Resolved. Root cause is not bifrost; it's the
macOS Docker Desktop virtiofs bind mount interacting with SQLite's
WAL journaling mode. Fix is a one-line `docker-compose.yml` change
from bind mount to docker-managed volume.

## Symptom recap

Smoke test fires 12 LLM calls. A reproducible fraction never appears
as a row in `logs.db`, even though the requests succeed end-to-end and
no errors are logged. The dropped fraction skewed toward streaming,
error (404), and no-dim-header requests — which made the bug look like
it had a code-level cause. It did not.

## What we tested

| # | Setup | Result |
|---|-------|--------|
| 1 | Our build (`stakgraph-gateway:dev`, bind mount `./data → /app/data`) | 8 of 12 land |
| 2 | Upstream `maximhq/bifrost:latest`, **bind mount** `/tmp/bifrost-upstream-test/data → /app/data` | 9 of 12 land |
| 3 | Upstream `maximhq/bifrost:latest`, **single isolated request**, bind mount | **0 of 1 lands** |
| 4 | Upstream `maximhq/bifrost:latest`, **docker named volume** `bifrost-upstream-data → /app/data` | **12 of 12 land** |

Run 3 was the key disambiguator: an isolated, well-formed,
fully-dim'd, non-streaming chat-completion request — the exact shape
that always lands in larger batches — fails to land on its own. That
killed the "streaming/error/no-dim is special" interpretation.

Run 4 was the confirmation: same image, same smoke test, same provider
keys, only volume driver changed. All rows land.

## What we ruled out and why

### H1 — Tracer never calls `Inject()` for some traces

Looked plausible from a code read of `pendingLogsToInject` /
`cleanupStalePendingLogs` (silent deletion path, no `droppedRequests`
bump). **Falsified** by adding `LOG_LEVEL=debug` and observing
`Inject: enqueuing log entry <id>` for every dropped request. The
tracer participated; entries reached the write queue.

### H2 — `if !loaded { return }` race in `storeOrEnqueueEntry`

Re-read with fresh eyes:

```go
existing, loaded := p.pendingLogsToInject.LoadOrStore(
    traceID,
    &pendingInjectEntries{entries: []*logstore.Log{entry}, createdAt: time.Now()},
)
if !loaded {
    return
}
```

`LoadOrStore` returns `loaded=false` when the value was just stored.
The freshly-stored value already contains the entry in its `entries`
slice. The early return skips a redundant append — it is not losing
data. **Falsified by reading the code more carefully.**

### H3 — Our dynamic-linked build differs from upstream

Falsified by run 2: upstream's official static build drops requests
under the same volume conditions. **Our build is not the cause.**

### A few hypotheses we generated mid-investigation

- **`BatchCreateIfNotExists` silently swallowing rows via `ON CONFLICT
  (id) DO NOTHING`.** Plausible because the function returns `nil`
  with the row absent and no fallback log line. Falsified by checking
  request IDs: all 12 were unique, no conflicts possible.
- **NOT NULL constraint violation on minimal error-entry path.** Path
  only runs for the no-pending case; logs showed all 3 dropped
  requests had `found=true`, so they did not take that path.
- **Batch writer starvation / never flushing.** Falsified by waiting
  3+ minutes after the smoke test — the row count never changed.

## Root cause

macOS Docker Desktop uses **virtiofs** to bind-mount host directories
into Linux containers. virtiofs has documented inconsistencies with
SQLite's WAL mode because:

1. SQLite's WAL flush path relies on `mmap`'d access to the `-shm`
   file and on `fdatasync` flushing the `-wal` file's tail bytes.
2. virtiofs caches `mmap` pages and may delay `fsync` propagation to
   the host filesystem.
3. As a result, a successful `INSERT` (from gorm's perspective, no
   error returned) can leave the row visible only inside the WAL
   pages held in the container's mmap'd cache, never reaching the
   host-visible WAL or main DB file. A query from outside the
   container (host `sqlite3` against the same path) sees only what
   actually flushed.

Evidence in our setup:

- WAL file (`logs.db-wal`) was **0 bytes** on the host after multiple
  requests, despite the SHM file being live and the bifrost process
  holding all three file descriptors open.
- A direct `INSERT … ON CONFLICT (id) DO NOTHING` from host
  `sqlite3` against the same path worked instantly.
- Switching the same image to a **docker-managed named volume** (which
  uses the Linux VM's native overlay/ext4 filesystem, no virtiofs)
  produced 12-of-12 landings, deterministically.
- A bifrost startup line confirms the I/O backend:
  > `WRN failed to cleanup old processing LLM logs: disk I/O error;
  >  cannot rollback - no transaction is active`
  Only fired during the startup cleanup pass; the routine inserts
  return `nil` to gorm but never make it to disk.

## Why the "drop pattern" looked stream/error/no-dim biased

virtiofs flush behavior is timing-sensitive. Writes that happen
**inside an active batch with more writes immediately after** are more
likely to ride along on a subsequent flush. Writes at the **tail of a
batch** (the last requests in the smoke run) are more likely to sit in
the unflushed mmap'd WAL when the batch writer's 2-second timer
expires and the goroutine goes idle. In our smoke test, streaming /
error / no-dim happened to be the last three requests — they were
disproportionately tail-of-batch, not disproportionately
"problematic".

This is also why earlier runs showed different drop sets (the
OpenRouter call dropped once and landed twice). The drops were
non-deterministic because virtiofs flush timing is.

## The fix

`gateway/docker-compose.yml`: replace the bind mount with a
docker-managed named volume.

The config seed (`data/config.json`) moves into the image via a
`COPY` step in the Dockerfile, so first-boot still gets a seeded
`config.db`. After that, the named volume owns all state:
`config.db`, `logs.db`, `logs.db-shm`, `logs.db-wal`.

Trade-offs:

| | Bind mount (before) | Named volume (after) |
|---|---|---|
| `config.json` editable in-place? | yes (just save the file) | no — change → `make docker-build` |
| `logs.db` inspectable from host? | yes, directly | yes, via `docker run --rm -v bifrost-data:/data alpine sqlite3 /data/logs.db` |
| SQLite WAL writes reliable on macOS? | **no** | yes |
| Linux/cloud production behavior? | fine (native fs, no virtiofs) | fine |
| State survives `docker compose down`? | yes (host dir) | yes (named volume) |
| State survives `docker compose down -v`? | yes (host dir) | **no** (volume removed) |

The macOS-only reliability issue is the dominant concern. Config edits
during development are a 10-second rebuild; the WAL data loss was
silent and undermined trust in every observability claim downstream.

## Notes for the next person who hits this

- **The dropped_requests counter is misleading.** It only increments
  on Go-side queue-full or channel-closed paths. SQLite-side silent
  drops (virtiofs flush failure, ON CONFLICT swallow, GORM error
  paths) do not bump it. Don't trust `/api/logs/dropped == 0` as
  evidence that no rows were lost.
- **A 0-byte `logs.db-wal` while the process is actively writing is a
  red flag.** It means writes are either going straight to the main
  DB (rare with WAL mode) or sitting in mmap pages that never reach
  disk (the failure mode here).
- **First disambiguator should be the volume-driver swap, not code
  inspection.** It takes 90 seconds and falsifies every code-level
  hypothesis in one shot.
- **This is macOS-specific.** Production deployments on Linux hosts
  with native filesystems don't hit it. Don't let the local-dev
  finding contaminate cloud-deploy assumptions; both can be true.

## Optional upstream follow-up

Worth filing an upstream issue to:

1. Bump `droppedRequests` (or add a sibling counter) when
   `cleanupStalePendingLogs` deletes a non-empty entry from
   `pendingLogsToInject`. The current silent delete made this
   look like a bifrost bug for longer than it should have.
2. Emit a structured log line ("logging plugin dropped N pending
   entries via TTL cleanup") at the same time, so operators see it.

Neither would have *fixed* our virtiofs issue, but both would have
made the diagnostic path 30 minutes shorter.
