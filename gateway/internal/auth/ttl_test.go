package auth

import (
	"testing"
	"time"
)

func TestRunKeyTTL_Clamps(t *testing.T) {
	now := time.Date(2026, 5, 14, 10, 0, 0, 0, time.UTC)
	cases := []struct {
		name string
		exp  time.Time
		want time.Duration
	}{
		{"already_expired_uses_floor", now.Add(-1 * time.Hour), 1 * time.Hour},
		{"nominal_30m_plus_grace", now.Add(30 * time.Minute), 90 * time.Minute},
		{"nominal_2h", now.Add(2 * time.Hour), 3 * time.Hour},
		{"long_lived_session_8h", now.Add(8 * time.Hour), 9 * time.Hour},
		{"way_over_uses_ceiling", now.Add(30 * 24 * time.Hour), 7 * 24 * time.Hour},
		{"exactly_at_ceiling", now.Add(7*24*time.Hour - 1*time.Hour), 7 * 24 * time.Hour},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := runKeyTTL(c.exp, now)
			if got != c.want {
				t.Fatalf("got %s want %s", got, c.want)
			}
		})
	}
}

func TestParseRFC3339_BadInput(t *testing.T) {
	if got := parseRFC3339("nonsense"); !got.IsZero() {
		t.Fatalf("expected zero time on parse error, got %v", got)
	}
	if got := parseRFC3339("2026-05-14T10:00:00Z"); got.IsZero() {
		t.Fatal("valid RFC3339 returned zero")
	}
}
