package adminapi

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

// Unit tests for the window cache (logwindow.go). The walker and the
// clock are both fakes: these pin the fill/refresh/rewalk policy and
// the merge semantics, not Bifrost's wire format (that's
// logstore_client_test.go).

type fakeWalker struct {
	mu    sync.Mutex
	calls []searchOpts
	rows  func(o searchOpts) []logstoreLog
	err   error
	gate  chan struct{} // when non-nil, walk blocks until it is closed
}

func (f *fakeWalker) walk(_ context.Context, o searchOpts) ([]logstoreLog, error) {
	f.mu.Lock()
	f.calls = append(f.calls, o)
	gate, err, rows := f.gate, f.err, f.rows
	f.mu.Unlock()
	if gate != nil {
		<-gate
	}
	if err != nil {
		return nil, err
	}
	if rows == nil {
		return nil, nil
	}
	return rows(o), nil
}

func (f *fakeWalker) nCalls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.calls)
}

func (f *fakeWalker) call(i int) searchOpts {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls[i]
}

func (f *fakeWalker) setErr(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.err = err
}

func (f *fakeWalker) setRows(rows func(o searchOpts) []logstoreLog) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.rows = rows
}

func wrow(id string, ts time.Time, cost float64) logstoreLog {
	return logstoreLog{ID: id, Timestamp: ts.Format(time.RFC3339Nano), Cost: cost,
		Metadata: metadataMap{"agent-name": "coder"}}
}

// inWindow filters a fixed row set the way Bifrost would for the walk's
// bounds — newest first.
func inWindow(all []logstoreLog) func(o searchOpts) []logstoreLog {
	return func(o searchOpts) []logstoreLog {
		var out []logstoreLog
		for i := len(all) - 1; i >= 0; i-- {
			ts, _ := time.Parse(time.RFC3339Nano, all[i].Timestamp)
			if ts.Before(*o.StartTime) || ts.After(*o.EndTime) {
				continue
			}
			out = append(out, all[i])
		}
		return out
	}
}

type testClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *testClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *testClock) advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

func newTestWindowCache(t *testing.T, w *fakeWalker) (*logWindowCache, *testClock) {
	t.Helper()
	clock := &testClock{t: time.Date(2026, 10, 2, 12, 0, 0, 0, time.UTC)}
	c := newLogWindowCache(w.walk)
	c.now = clock.now
	return c, clock
}

// waitIdle blocks until no entry has a fill in flight.
func waitIdle(t *testing.T, c *logWindowCache) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		c.mu.Lock()
		idle := true
		for _, e := range c.entries {
			if e.filling != nil {
				idle = false
			}
		}
		c.mu.Unlock()
		if idle {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatal("background fill did not finish")
}

func ids(rows []logstoreLog) []string {
	out := make([]string, 0, len(rows))
	for _, r := range rows {
		out = append(out, r.ID)
	}
	return out
}

func TestLogWindow_ColdFillWalksOnceAndCoalesces(t *testing.T) {
	w := &fakeWalker{gate: make(chan struct{})}
	c, clock := newTestWindowCache(t, w)
	now := clock.now()
	w.setRows(inWindow([]logstoreLog{wrow("a", now.Add(-time.Hour), 1)}))

	const n = 6
	results := make([][]logstoreLog, n)
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = c.get(context.Background(), now.Add(-24*time.Hour), now, nil)
		}(i)
	}
	// Every caller is parked on the one fill.
	deadline := time.Now().Add(time.Second)
	for w.nCalls() == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	close(w.gate)
	wg.Wait()

	if got := w.nCalls(); got != 1 {
		t.Fatalf("walks = %d, want 1 (callers must coalesce on the cold fill)", got)
	}
	for i := 0; i < n; i++ {
		if errs[i] != nil || len(results[i]) != 1 || results[i][0].ID != "a" {
			t.Errorf("caller %d: rows=%v err=%v", i, ids(results[i]), errs[i])
		}
	}
	o := w.call(0)
	if !o.StartTime.Equal(now.Add(-24*time.Hour)) || !o.EndTime.Equal(now) {
		t.Errorf("cold walk bounds = [%v, %v], want the whole window", o.StartTime, o.EndTime)
	}
}

func TestLogWindow_WarmEntryServesWithoutWalking(t *testing.T) {
	w := &fakeWalker{}
	c, clock := newTestWindowCache(t, w)
	now := clock.now()
	w.setRows(inWindow([]logstoreLog{wrow("a", now.Add(-time.Hour), 1)}))

	if _, err := c.get(context.Background(), now.Add(-24*time.Hour), now, nil); err != nil {
		t.Fatal(err)
	}
	clock.advance(windowRefreshAfter / 2)
	now = clock.now()
	rows, err := c.get(context.Background(), now.Add(-24*time.Hour), now, nil)
	if err != nil || len(rows) != 1 {
		t.Fatalf("rows=%v err=%v", ids(rows), err)
	}
	if got := w.nCalls(); got != 1 {
		t.Fatalf("walks = %d, want 1 (entry younger than refreshAfter must not refetch)", got)
	}
}

func TestLogWindow_IncrementalRefreshSplicesTail(t *testing.T) {
	w := &fakeWalker{}
	c, clock := newTestWindowCache(t, w)
	t0 := clock.now()
	span := 24 * time.Hour
	// Rows at first walk: `old` will slide out of the window, `kept`
	// sits before the overlap, `updated` inside it, `gone` inside it
	// but deleted by the time of the refresh.
	old := wrow("old", t0.Add(-span).Add(10*time.Second), 1)
	kept := wrow("kept", t0.Add(-time.Hour), 1)
	updated := wrow("updated", t0.Add(-2*time.Minute), 0) // cost lands later
	gone := wrow("gone", t0.Add(-time.Minute), 1)
	w.setRows(inWindow([]logstoreLog{old, kept, updated, gone}))
	if _, err := c.get(context.Background(), t0.Add(-span), t0, nil); err != nil {
		t.Fatal(err)
	}

	// 30s later: the poll is served from cache immediately and kicks
	// a refresh that reads only the overlap tail.
	clock.advance(30 * time.Second)
	t1 := clock.now()
	updated.Cost = 5
	fresh := wrow("fresh", t1.Add(-5*time.Second), 2)
	w.setRows(inWindow([]logstoreLog{old, kept, updated, fresh}))

	rows, err := c.get(context.Background(), t1.Add(-span), t1, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Served before the refresh landed: the stale view, minus the row
	// that already slid out of [t1-span, t1].
	if got := ids(rows); fmt.Sprint(got) != "[gone updated kept]" {
		t.Errorf("stale-while-revalidate rows = %v", got)
	}
	waitIdle(t, c)
	if got := w.nCalls(); got != 2 {
		t.Fatalf("walks = %d, want 2", got)
	}
	o := w.call(1)
	if wantFrom := t0.Add(-windowOverlap); !o.StartTime.Equal(wantFrom) || !o.EndTime.Equal(t1) {
		t.Errorf("refresh bounds = [%v, %v], want [%v, %v]", o.StartTime, o.EndTime, wantFrom, t1)
	}

	rows, err = c.get(context.Background(), t1.Add(-span), t1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(rows); fmt.Sprint(got) != "[fresh updated kept]" {
		t.Errorf("merged rows = %v (fresh added, gone dropped, old slid out, kept retained)", got)
	}
	for _, r := range rows {
		if r.ID == "updated" && r.Cost != 5 {
			t.Errorf("updated row cost = %v, want 5 (overlap re-read must replace)", r.Cost)
		}
	}
	if got := w.nCalls(); got != 2 {
		t.Errorf("walks = %d, want 2 (second get within refreshAfter of the refresh)", got)
	}
}

func TestLogWindow_FullRewalkAfterInterval(t *testing.T) {
	w := &fakeWalker{}
	c, clock := newTestWindowCache(t, w)
	t0 := clock.now()
	span := 6 * time.Hour
	w.setRows(inWindow([]logstoreLog{wrow("a", t0.Add(-time.Hour), 1)}))
	if _, err := c.get(context.Background(), t0.Add(-span), t0, nil); err != nil {
		t.Fatal(err)
	}

	clock.advance(windowRewalkAfter)
	t1 := clock.now()
	if _, err := c.get(context.Background(), t1.Add(-span), t1, nil); err != nil {
		t.Fatal(err)
	}
	waitIdle(t, c)
	if got := w.nCalls(); got != 2 {
		t.Fatalf("walks = %d, want 2", got)
	}
	o := w.call(1)
	if !o.StartTime.Equal(t1.Add(-span)) || !o.EndTime.Equal(t1) {
		t.Errorf("rewalk bounds = [%v, %v], want the whole window [%v, %v]", o.StartTime, o.EndTime, t1.Add(-span), t1)
	}
}

func TestLogWindow_CallerCancelDoesNotCancelFill(t *testing.T) {
	w := &fakeWalker{gate: make(chan struct{})}
	c, clock := newTestWindowCache(t, w)
	now := clock.now()
	w.setRows(inWindow([]logstoreLog{wrow("a", now.Add(-time.Hour), 1)}))

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := c.get(ctx, now.Add(-24*time.Hour), now, nil)
		done <- err
	}()
	deadline := time.Now().Add(time.Second)
	for w.nCalls() == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	cancel() // the browser gave up
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled caller err = %v, want context.Canceled", err)
	}

	close(w.gate) // the walk finishes anyway
	waitIdle(t, c)
	rows, err := c.get(context.Background(), now.Add(-24*time.Hour), now, nil)
	if err != nil || len(rows) != 1 {
		t.Fatalf("rows=%v err=%v", ids(rows), err)
	}
	if got := w.nCalls(); got != 1 {
		t.Errorf("walks = %d, want 1 (the abandoned fill must be reused)", got)
	}
}

func TestLogWindow_ColdFillErrorIsReturnedAndRetried(t *testing.T) {
	w := &fakeWalker{}
	c, clock := newTestWindowCache(t, w)
	now := clock.now()
	boom := errors.New("bifrost down")
	w.setErr(boom)

	if _, err := c.get(context.Background(), now.Add(-time.Hour), now, nil); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want %v", err, boom)
	}
	w.setErr(nil)
	w.setRows(inWindow([]logstoreLog{wrow("a", now.Add(-time.Minute), 1)}))
	rows, err := c.get(context.Background(), now.Add(-time.Hour), now, nil)
	if err != nil || len(rows) != 1 {
		t.Fatalf("after retry: rows=%v err=%v", ids(rows), err)
	}
	if got := w.nCalls(); got != 2 {
		t.Errorf("walks = %d, want 2", got)
	}
}

func TestLogWindow_RefreshErrorsServeStaleUntilLimit(t *testing.T) {
	w := &fakeWalker{}
	c, clock := newTestWindowCache(t, w)
	t0 := clock.now()
	w.setRows(inWindow([]logstoreLog{wrow("a", t0.Add(-time.Minute), 1)}))
	if _, err := c.get(context.Background(), t0.Add(-time.Hour), t0, nil); err != nil {
		t.Fatal(err)
	}

	boom := errors.New("bifrost down")
	w.setErr(boom)
	clock.advance(windowRefreshAfter)
	t1 := clock.now()
	rows, err := c.get(context.Background(), t1.Add(-time.Hour), t1, nil)
	if err != nil || len(rows) != 1 {
		t.Fatalf("within stale limit: rows=%v err=%v (must serve the cached rows)", ids(rows), err)
	}
	waitIdle(t, c)

	clock.advance(windowStaleLimit)
	t2 := clock.now()
	if _, err := c.get(context.Background(), t2.Add(-time.Hour), t2, nil); !errors.Is(err, boom) {
		t.Fatalf("past stale limit: err = %v, want %v", err, boom)
	}
}

func TestLogWindow_FiltersAndSpanAreSeparateEntries(t *testing.T) {
	w := &fakeWalker{}
	c, clock := newTestWindowCache(t, w)
	now := clock.now()
	w.setRows(inWindow([]logstoreLog{wrow("a", now.Add(-time.Minute), 1)}))

	for _, call := range []struct {
		span    time.Duration
		filters map[string]string
	}{
		{time.Hour, nil},
		{time.Hour, map[string]string{"user-id": "alice"}},
		{time.Hour, map[string]string{"user-id": "alice"}}, // same as previous
		{6 * time.Hour, nil},
	} {
		if _, err := c.get(context.Background(), now.Add(-call.span), now, call.filters); err != nil {
			t.Fatal(err)
		}
	}
	if got := w.nCalls(); got != 3 {
		t.Errorf("walks = %d, want 3 (one per distinct window×filters)", got)
	}
	if got := w.call(1).Metadata["user-id"]; got != "alice" {
		t.Errorf("filtered walk metadata = %v", w.call(1).Metadata)
	}
}

func TestLogWindow_IdleEntriesAreEvicted(t *testing.T) {
	w := &fakeWalker{}
	c, clock := newTestWindowCache(t, w)
	now := clock.now()
	w.setRows(inWindow(nil))
	if _, err := c.get(context.Background(), now.Add(-time.Hour), now, nil); err != nil {
		t.Fatal(err)
	}
	clock.advance(windowIdleEvict + windowSweepEvery + time.Second)
	now = clock.now()
	// A request for another window runs the sweep.
	if _, err := c.get(context.Background(), now.Add(-6*time.Hour), now, nil); err != nil {
		t.Fatal(err)
	}
	c.mu.Lock()
	n := len(c.entries)
	c.mu.Unlock()
	if n != 1 {
		t.Errorf("entries after sweep = %d, want 1 (the idle 1h entry is gone)", n)
	}
}

func TestLogWindow_KeyIsOrderIndependent(t *testing.T) {
	a := windowKey(time.Hour, map[string]string{"x": "1", "y": "2"})
	b := windowKey(time.Hour, map[string]string{"y": "2", "x": "1"})
	if a != b {
		t.Errorf("keys differ: %q vs %q", a, b)
	}
	if windowKey(time.Hour, nil) == windowKey(2*time.Hour, nil) {
		t.Error("span must be part of the key")
	}
}
