package env

import "testing"

func TestEnforceMacaroonsValue(t *testing.T) {
	cases := []struct {
		raw       string
		wantVal   bool
		wantSet   bool
		wantError bool
	}{
		{"", false, false, false},
		{"   ", false, false, false},
		{"true", true, true, false},
		{"TRUE", true, true, false},
		{"1", true, true, false},
		{"yes", true, true, false},
		{"on", true, true, false},
		{" true ", true, true, false},
		{"false", false, true, false},
		{"0", false, true, false},
		{"no", false, true, false},
		{"OFF", false, true, false},
		// Typos must surface as errors, never as a silent default.
		{"ture", false, true, true},
		{"enabled", false, true, true},
		{"2", false, true, true},
	}
	for _, c := range cases {
		t.Setenv(EnforceMacaroons, c.raw)
		val, set, err := EnforceMacaroonsValue()
		if (err != nil) != c.wantError {
			t.Fatalf("%q: err=%v, wantError=%v", c.raw, err, c.wantError)
		}
		if set != c.wantSet {
			t.Fatalf("%q: set=%v, want %v", c.raw, set, c.wantSet)
		}
		if err == nil && val != c.wantVal {
			t.Fatalf("%q: value=%v, want %v", c.raw, val, c.wantVal)
		}
	}
}
