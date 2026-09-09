// Package duration mirrors Bifrost's duration vocabulary for windowed
// budgets (phase-6 §"Duration vocabulary") without taking a dep on
// bifrost/framework. The reference implementation is
// bifrost/framework/configstore/tables/utils.go (ParseDuration and
// GetCalendarPeriodStart); tests cross-check a fixed input set against
// the outputs recorded from that implementation so drift is caught in
// CI rather than in an operator's budget math.
//
// Vocabulary (case-sensitive — "1M" is months, "1m" is minutes):
//
//	Ns, Nm, Nh   rolling windows, no calendar boundary
//	Nd           calendar day, midnight UTC
//	Nw           calendar week, midnight UTC of the most recent Monday
//	NM           calendar month, midnight UTC of the 1st
//	NY           calendar year, midnight UTC of Jan 1
//
// Bucket-key formats follow the phase-6 schema table: calendar
// windows use human-readable date keys ("2026-05-16", "2026-W20",
// "2026-05", "2026"), sub-day rolling windows use the unix epoch of
// the window start. All boundaries are UTC.
package duration

import (
	"fmt"
	"strconv"
	"time"
)

// Window is a parsed duration string: a positive count and a unit
// suffix. Phase 6 ships only N=1 calendar windows; larger N parses
// (the vocabulary allows it) and buckets align to the N=1 boundary of
// the unit, with the window length spanning N units from that
// boundary.
type Window struct {
	N    int
	Unit byte // one of 's' 'm' 'h' 'd' 'w' 'M' 'Y'
}

// Parse parses a Bifrost duration string ("10m", "2h", "1d", "1w",
// "1M", "1Y"). The numeric prefix must be a positive base-10 integer
// with no leading zeroes; the suffix must be exactly one of the seven
// vocabulary units.
func Parse(s string) (Window, error) {
	if len(s) < 2 {
		return Window{}, fmt.Errorf("duration %q: too short", s)
	}
	unit := s[len(s)-1]
	switch unit {
	case 's', 'm', 'h', 'd', 'w', 'M', 'Y':
	default:
		return Window{}, fmt.Errorf("duration %q: unknown unit %q", s, string(unit))
	}
	num := s[:len(s)-1]
	if num[0] == '0' {
		return Window{}, fmt.Errorf("duration %q: leading zero", s)
	}
	n, err := strconv.Atoi(num)
	if err != nil || n <= 0 {
		return Window{}, fmt.Errorf("duration %q: bad count", s)
	}
	return Window{N: n, Unit: unit}, nil
}

// Calendar reports whether the window aligns to a calendar boundary
// (d/w/M/Y) as opposed to rolling sub-day semantics (s/m/h).
func (w Window) Calendar() bool {
	switch w.Unit {
	case 'd', 'w', 'M', 'Y':
		return true
	}
	return false
}

// Bounds returns the [start, end) interval of the bucket containing
// `now`. Calendar windows anchor start at the unit's standard
// boundary (midnight UTC / Monday / 1st / Jan 1) and span N units;
// sub-day windows truncate `now` to the N×unit interval.
//
// `now` is converted to UTC before any boundary math — every bucket
// boundary in the schema is UTC by contract.
func (w Window) Bounds(now time.Time) (start, end time.Time) {
	now = now.UTC()
	switch w.Unit {
	case 's', 'm', 'h':
		interval := w.rollingInterval()
		start = now.Truncate(interval)
		return start, start.Add(interval)
	case 'd':
		start = time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC)
		return start, start.AddDate(0, 0, w.N)
	case 'w':
		// Go's Weekday: Sunday=0..Saturday=6; ISO weeks start Monday.
		wd := int(now.Weekday())
		if wd == 0 {
			wd = 7
		}
		start = time.Date(now.Year(), now.Month(), now.Day()-(wd-1), 0, 0, 0, 0, time.UTC)
		return start, start.AddDate(0, 0, 7*w.N)
	case 'M':
		start = time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, time.UTC)
		return start, start.AddDate(0, w.N, 0)
	case 'Y':
		start = time.Date(now.Year(), 1, 1, 0, 0, 0, 0, time.UTC)
		return start, start.AddDate(w.N, 0, 0)
	}
	return time.Time{}, time.Time{} // unreachable for parsed windows
}

// BucketKey returns the Redis bucket-key segment for the window
// containing `now`, per the phase-6 schema table. The caller
// assembles the full key (`bifrost:cost:agent:<name>:<key>`).
func (w Window) BucketKey(now time.Time) string {
	start, _ := w.Bounds(now)
	switch w.Unit {
	case 's', 'm', 'h':
		return strconv.FormatInt(start.Unix(), 10)
	case 'd':
		return start.Format("2006-01-02")
	case 'w':
		y, wk := start.ISOWeek()
		return fmt.Sprintf("%04d-W%02d", y, wk)
	case 'M':
		return start.Format("2006-01")
	case 'Y':
		return start.Format("2006")
	}
	return ""
}

// TTL is the Redis expiry for a bucket key: 2× the window duration
// (phase-6 §cost:agent — "a 1d bucket lives 48h, a 1w bucket lives
// 14d, a 1M bucket lives ~60d"). Calendar months and years use the
// 30d/365d approximations; the imprecision only stretches or trims
// how long a dead bucket lingers, never which bucket is written.
func (w Window) TTL() time.Duration {
	return 2 * w.length()
}

func (w Window) length() time.Duration {
	n := time.Duration(w.N)
	switch w.Unit {
	case 's':
		return n * time.Second
	case 'm':
		return n * time.Minute
	case 'h':
		return n * time.Hour
	case 'd':
		return n * 24 * time.Hour
	case 'w':
		return n * 7 * 24 * time.Hour
	case 'M':
		return n * 30 * 24 * time.Hour
	case 'Y':
		return n * 365 * 24 * time.Hour
	}
	return 0
}

func (w Window) rollingInterval() time.Duration {
	switch w.Unit {
	case 's':
		return time.Duration(w.N) * time.Second
	case 'm':
		return time.Duration(w.N) * time.Minute
	case 'h':
		return time.Duration(w.N) * time.Hour
	}
	return 0
}

// BucketKey is the package-level convenience for one-shot callers:
// parse + key in one call. Returns an error on a malformed window
// string so hot-path callers can decide their own degrade behavior.
func BucketKey(window string, now time.Time) (string, error) {
	w, err := Parse(window)
	if err != nil {
		return "", err
	}
	return w.BucketKey(now), nil
}
