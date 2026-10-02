package adminapi

import (
	"context"
	"errors"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// Window cache for the rollup handlers.
//
// Every rollup (/spend/*, /histogram/*, /users/:id, /agents/:name/runs)
// used to call searchAll on each request: walk Bifrost's /api/logs a
// thousand rows at a time over the whole window and sum in Go. Each
// page costs Bifrost a full-window COUNT plus the page SELECT, so a
// 24h window on a busy swarm (tens of thousands of rows) took longer
// than the dashboard's fetch timeout — and the dashboard asks for the
// same window again every 30s from every open page, so Bifrost never
// got ahead. 2026-10-02: every /_plugin/spend/* call from Hive's
// iframe aborted at exactly 5s and nothing rendered.
//
// logWindowCache keeps one entry per (window length, dim filters).
// The first request for an entry walks the whole window once. After
// that:
//
//   - an entry younger than windowRefreshAfter is served as-is;
//   - an older entry is served as-is too, and ONE background refresh
//     starts: it re-reads only [coveredEnd − windowOverlap, now] and
//     splices that over the cached rows, so a 30s poll costs Bifrost
//     a page or two instead of the whole window;
//   - every windowRewalkAfter the refresh is a full walk instead,
//     which bounds how long a row that Bifrost updated late (a call
//     that ran longer than windowOverlap — Bifrost inserts the row at
//     request start and writes cost/tokens at the end) can stay stale;
//   - callers only ever wait for the FIRST fill. Everything after is
//     stale-while-revalidate, so a handler's latency once warm is its
//     own aggregation, not Bifrost. A refresh that keeps failing is
//     surfaced once the rows are older than windowStaleLimit.
//
// Fills run on a detached context. A browser that gives up, or a
// proxy that cuts the request, must not discard a half-finished walk
// that the next poll is about to ask for again — that is exactly the
// loop the dashboard was stuck in.
//
// Rows are shared between callers and are read-only. get hands out a
// fresh slice, but the Metadata maps inside are the cached ones.
const (
	// windowRefreshAfter is how old an entry may be before a request
	// kicks a background refresh. Shorter than the dashboard's 30s
	// poll so every poll sees the previous poll's refresh.
	windowRefreshAfter = 10 * time.Second

	// windowOverlap is how far behind coveredEnd an incremental
	// refresh re-reads. Covers cost/tokens landing on rows whose
	// timestamp (request start) is already inside the cached span.
	windowOverlap = 10 * time.Minute

	// windowRewalkAfter is the full re-walk cadence; the staleness
	// bound for rows updated later than windowOverlap after they
	// were created.
	windowRewalkAfter = 10 * time.Minute

	// windowIdleEvict drops entries nobody has asked for in a while
	// (a user page that was closed, a window the operator stopped
	// looking at) so memory tracks what is actually on screen.
	windowIdleEvict = 15 * time.Minute

	// windowFillTimeout bounds one detached walk. Generous: it is the
	// budget for the FIRST paint of a large window, and nothing is
	// waiting on it but the one request that triggered it.
	windowFillTimeout = 3 * time.Minute

	// windowStaleLimit is how long failed refreshes may keep serving
	// old rows before the error is returned instead.
	windowStaleLimit = 5 * time.Minute

	// windowPageSize / windowMaxRows are the searchAll arguments every
	// rollup used before the cache; kept as the single definition.
	windowPageSize = 1000
	windowMaxRows  = 200_000

	// windowSweepEvery rate-limits the idle-eviction pass.
	windowSweepEvery = time.Minute
)

// windowWalker is the function a cache fill calls: searchAll over the
// given opts (StartTime / EndTime / Metadata set).
type windowWalker func(ctx context.Context, o searchOpts) ([]logstoreLog, error)

type logWindowCache struct {
	walk windowWalker
	now  func() time.Time // injectable for tests

	mu        sync.Mutex
	entries   map[string]*logWindowEntry
	lastSweep time.Time
}

// windowRow is a cached log plus its parsed timestamp, so slicing and
// trimming never re-parse. A row whose timestamp Bifrost emitted in a
// shape we cannot parse keeps a zero ts: it was in Bifrost's answer
// for the window, so it is always served and never trimmed (the next
// full walk replaces it like every other row).
type windowRow struct {
	ts  time.Time
	log logstoreLog
}

type logWindowEntry struct {
	span    time.Duration
	filters map[string]string

	rows         []windowRow // newest first, as Bifrost returns them
	coveredStart time.Time
	coveredEnd   time.Time
	walkedAt     time.Time // last successful full walk
	refreshedAt  time.Time // last successful fill of any kind; zero ⇒ cold
	lastUsed     time.Time
	lastErr      error

	filling chan struct{} // non-nil while a fill is running; closed when it ends
}

func newLogWindowCache(walk windowWalker) *logWindowCache {
	return &logWindowCache{
		walk:    walk,
		now:     time.Now,
		entries: map[string]*logWindowEntry{},
	}
}

// windowKey canonicalises (span, filters) — map iteration order must
// not produce two entries for one window.
func windowKey(span time.Duration, filters map[string]string) string {
	keys := make([]string, 0, len(filters))
	for k := range filters {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	b.WriteString(span.String())
	for _, k := range keys {
		b.WriteByte(0)
		b.WriteString(k)
		b.WriteByte('=')
		b.WriteString(filters[k])
	}
	return b.String()
}

// get returns the rows with timestamps in [start, end] that match
// filters, from the cache when it has them. Only the first request
// for a (window length, filters) pair blocks on Bifrost; see the file
// comment for everything after.
func (c *logWindowCache) get(ctx context.Context, start, end time.Time, filters map[string]string) ([]logstoreLog, error) {
	span := end.Sub(start)
	if span <= 0 {
		return []logstoreLog{}, nil
	}
	now := c.now()

	c.mu.Lock()
	c.sweepLocked(now)
	key := windowKey(span, filters)
	e, ok := c.entries[key]
	if !ok {
		e = &logWindowEntry{span: span, filters: cloneFilters(filters)}
		c.entries[key] = e
	}
	e.lastUsed = now

	if e.refreshedAt.IsZero() {
		// Cold: join the running first fill or start it, then wait.
		wait := e.filling
		if wait == nil {
			wait = c.startFillLocked(e, true, now)
		}
		c.mu.Unlock()
		select {
		case <-wait:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		c.mu.Lock()
		defer c.mu.Unlock()
		if e.refreshedAt.IsZero() {
			if e.lastErr != nil {
				return nil, e.lastErr
			}
			return nil, errors.New("log window: fill produced no result")
		}
		return e.slice(start, end), nil
	}
	defer c.mu.Unlock()

	// Warm: serve what we have; kick ONE refresh if due.
	if e.filling == nil && now.Sub(e.refreshedAt) >= windowRefreshAfter {
		full := now.Sub(e.walkedAt) >= windowRewalkAfter
		c.startFillLocked(e, full, now)
	}
	if e.lastErr != nil && now.Sub(e.refreshedAt) > windowStaleLimit {
		return nil, e.lastErr
	}
	return e.slice(start, end), nil
}

// startFillLocked launches a detached walk for e and returns the
// channel that closes when it ends. Caller holds c.mu. `full` walks
// the whole window; otherwise only the overlap tail plus whatever is
// newer than coveredEnd.
func (c *logWindowCache) startFillLocked(e *logWindowEntry, full bool, now time.Time) chan struct{} {
	done := make(chan struct{})
	e.filling = done

	windowStart := now.Add(-e.span)
	from := windowStart
	if !full {
		if f := e.coveredEnd.Add(-windowOverlap); f.After(from) {
			from = f
		}
	}
	to := now
	opts := searchOpts{StartTime: &from, EndTime: &to, Metadata: cloneFilters(e.filters)}
	kind := "refresh"
	if full {
		kind = "walk"
	}

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), windowFillTimeout)
		defer cancel()
		logs, err := c.walk(ctx, opts)

		c.mu.Lock()
		defer c.mu.Unlock()
		defer close(done)
		e.filling = nil
		if err != nil {
			e.lastErr = err
			pluginlog.Warnf("adminapi: log window %s %s: %v", e.span, kind, err)
			return
		}
		e.lastErr = nil
		fresh := parseRows(logs)
		if full {
			e.rows = fresh
			e.walkedAt = now
		} else {
			e.rows = mergeRows(e.rows, fresh, from)
		}
		e.coveredStart = windowStart
		e.coveredEnd = to
		e.refreshedAt = now
		e.rows = trimRows(e.rows, windowStart, windowMaxRows)
	}()
	return done
}

// sweepLocked evicts idle entries, at most once per windowSweepEvery.
func (c *logWindowCache) sweepLocked(now time.Time) {
	if now.Sub(c.lastSweep) < windowSweepEvery {
		return
	}
	c.lastSweep = now
	for k, e := range c.entries {
		if e.filling == nil && now.Sub(e.lastUsed) > windowIdleEvict {
			delete(c.entries, k)
		}
	}
}

// slice copies out the rows inside [start, end]. Unparseable-ts rows
// are always included (see windowRow).
func (e *logWindowEntry) slice(start, end time.Time) []logstoreLog {
	out := make([]logstoreLog, 0, len(e.rows))
	for _, r := range e.rows {
		if r.ts.IsZero() || (!r.ts.Before(start) && !r.ts.After(end)) {
			out = append(out, r.log)
		}
	}
	return out
}

func parseRows(logs []logstoreLog) []windowRow {
	rows := make([]windowRow, 0, len(logs))
	for _, l := range logs {
		ts, err := time.Parse(time.RFC3339Nano, l.Timestamp)
		if err != nil {
			ts = time.Time{}
		}
		rows = append(rows, windowRow{ts: ts, log: l})
	}
	return rows
}

// mergeRows splices a fresh read of [from, …] over the cached rows:
// everything cached at or after `from` is replaced by the fresh set
// (updated cost lands, deleted rows vanish), everything older is kept.
// Both inputs are newest-first and the fresh rows are all ≥ from, so
// concatenation preserves the order.
func mergeRows(old, fresh []windowRow, from time.Time) []windowRow {
	out := make([]windowRow, 0, len(fresh)+len(old))
	out = append(out, fresh...)
	for _, r := range old {
		if r.ts.IsZero() || r.ts.Before(from) {
			out = append(out, r)
		}
	}
	return out
}

// trimRows drops rows that slid out of the window and enforces the
// row cap (oldest rows go first — they are at the tail).
func trimRows(rows []windowRow, windowStart time.Time, maxRows int) []windowRow {
	kept := rows[:0]
	for _, r := range rows {
		if r.ts.IsZero() || !r.ts.Before(windowStart) {
			kept = append(kept, r)
		}
	}
	if len(kept) > maxRows {
		kept = kept[:maxRows]
	}
	// Release the tail the trim exposed so the backing array doesn't
	// pin rows nobody can reach.
	for i := len(kept); i < len(rows); i++ {
		rows[i] = windowRow{}
	}
	return kept
}

func cloneFilters(m map[string]string) map[string]string {
	if m == nil {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}
