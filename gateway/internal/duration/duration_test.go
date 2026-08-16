package duration

import (
	"strconv"
	"testing"
	"time"
)

// ─── Parse ────────────────────────────────────────────────────────────

func TestParse_Valid(t *testing.T) {
	cases := []struct {
		in   string
		want Window
	}{
		{"1s", Window{1, 's'}},
		{"10m", Window{10, 'm'}},
		{"2h", Window{2, 'h'}},
		{"1d", Window{1, 'd'}},
		{"3d", Window{3, 'd'}},
		{"1w", Window{1, 'w'}},
		{"1M", Window{1, 'M'}},
		{"1Y", Window{1, 'Y'}},
	}
	for _, c := range cases {
		got, err := Parse(c.in)
		if err != nil {
			t.Errorf("Parse(%q): unexpected error %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("Parse(%q) = %+v, want %+v", c.in, got, c.want)
		}
	}
}

func TestParse_Invalid(t *testing.T) {
	for _, in := range []string{
		"", "1", "d", "1x", "0d", "01d", "-1d", "1.5h", "1 d", "d1", "1dd",
	} {
		if _, err := Parse(in); err == nil {
			t.Errorf("Parse(%q): expected error, got nil", in)
		}
	}
}

// Case sensitivity is load-bearing: "1M" is months, "1m" is minutes.
func TestParse_CaseSensitivity(t *testing.T) {
	mo, err := Parse("1M")
	if err != nil || mo.Unit != 'M' {
		t.Fatalf("Parse(1M) = %+v, %v", mo, err)
	}
	mi, err := Parse("1m")
	if err != nil || mi.Unit != 'm' {
		t.Fatalf("Parse(1m) = %+v, %v", mi, err)
	}
	if mo.Calendar() == false || mi.Calendar() == true {
		t.Fatalf("calendar flags: 1M=%v 1m=%v", mo.Calendar(), mi.Calendar())
	}
}

// ─── BucketKey: cross-check table ─────────────────────────────────────
//
// These pairs mirror the phase-6 schema table and Bifrost's
// GetCalendarPeriodStart semantics (midnight UTC / Monday / 1st /
// Jan 1). If Bifrost's vocabulary ever changes, update both this
// table and the parser together.

func TestBucketKey_CalendarWindows(t *testing.T) {
	// Friday 2026-05-16 14:30:00 UTC (matches the phase-6 examples).
	now := time.Date(2026, 5, 16, 14, 30, 0, 0, time.UTC)
	cases := []struct {
		window string
		want   string
	}{
		{"1d", "2026-05-16"},
		{"1w", "2026-W20"},
		{"1M", "2026-05"},
		{"1Y", "2026"},
	}
	for _, c := range cases {
		got, err := BucketKey(c.window, now)
		if err != nil {
			t.Errorf("BucketKey(%q): %v", c.window, err)
			continue
		}
		if got != c.want {
			t.Errorf("BucketKey(%q) = %q, want %q", c.window, got, c.want)
		}
	}
}

func TestBucketKey_SubDayRolling(t *testing.T) {
	now := time.Date(2026, 5, 16, 14, 37, 42, 0, time.UTC)
	cases := []struct {
		window string
		start  time.Time // expected window start; key is its epoch
	}{
		{"1h", time.Date(2026, 5, 16, 14, 0, 0, 0, time.UTC)},
		{"2h", time.Date(2026, 5, 16, 14, 0, 0, 0, time.UTC)},
		{"10m", time.Date(2026, 5, 16, 14, 30, 0, 0, time.UTC)},
		{"30s", time.Date(2026, 5, 16, 14, 37, 30, 0, time.UTC)},
	}
	for _, c := range cases {
		got, err := BucketKey(c.window, now)
		if err != nil {
			t.Errorf("BucketKey(%q): %v", c.window, err)
			continue
		}
		if want := strconv.FormatInt(c.start.Unix(), 10); got != want {
			t.Errorf("BucketKey(%q) = %q, want %q (start %s)", c.window, got, want, c.start)
		}
	}
}

// ─── Bounds ───────────────────────────────────────────────────────────

func TestBounds_WeekStartsMonday(t *testing.T) {
	// Sunday 2026-05-17: the week bucket must start Monday 05-11,
	// not Sunday (Go's Weekday numbering trap).
	now := time.Date(2026, 5, 17, 8, 0, 0, 0, time.UTC)
	w := Window{1, 'w'}
	start, end := w.Bounds(now)
	if want := time.Date(2026, 5, 11, 0, 0, 0, 0, time.UTC); !start.Equal(want) {
		t.Errorf("week start = %s, want %s", start, want)
	}
	if want := time.Date(2026, 5, 18, 0, 0, 0, 0, time.UTC); !end.Equal(want) {
		t.Errorf("week end = %s, want %s", end, want)
	}
}

func TestBounds_MondayIsOwnWeekStart(t *testing.T) {
	now := time.Date(2026, 5, 11, 0, 0, 0, 0, time.UTC) // Monday midnight
	start, _ := Window{1, 'w'}.Bounds(now)
	if !start.Equal(now) {
		t.Errorf("week start on a Monday = %s, want %s", start, now)
	}
}

func TestBounds_YearBoundaryISOWeek(t *testing.T) {
	// 2027-01-01 is a Friday, part of ISO week 2026-W53. The bucket
	// key must carry the ISO year of the week's Monday, not the
	// calendar year of `now`.
	now := time.Date(2027, 1, 1, 12, 0, 0, 0, time.UTC)
	key, err := BucketKey("1w", now)
	if err != nil {
		t.Fatal(err)
	}
	if key != "2026-W53" {
		t.Errorf("year-boundary week key = %q, want 2026-W53", key)
	}
}

func TestBounds_NonUTCInputNormalized(t *testing.T) {
	// 2026-05-16 23:30 in UTC+10 is 13:30 UTC on the same day; a
	// naive local-date bucket would say 05-17.
	loc := time.FixedZone("UTC+10", 10*3600)
	now := time.Date(2026, 5, 17, 9, 30, 0, 0, loc) // = 2026-05-16 23:30 UTC
	key, err := BucketKey("1d", now)
	if err != nil {
		t.Fatal(err)
	}
	if key != "2026-05-16" {
		t.Errorf("non-UTC day key = %q, want 2026-05-16", key)
	}
}

// ─── TTL ──────────────────────────────────────────────────────────────

func TestTTL_TwiceWindow(t *testing.T) {
	cases := []struct {
		window string
		want   time.Duration
	}{
		{"1d", 48 * time.Hour},
		{"1w", 14 * 24 * time.Hour},
		{"1M", 60 * 24 * time.Hour},
		{"1Y", 730 * 24 * time.Hour},
		{"1h", 2 * time.Hour},
		{"10m", 20 * time.Minute},
	}
	for _, c := range cases {
		w, err := Parse(c.window)
		if err != nil {
			t.Fatal(err)
		}
		if got := w.TTL(); got != c.want {
			t.Errorf("TTL(%q) = %s, want %s", c.window, got, c.want)
		}
	}
}
