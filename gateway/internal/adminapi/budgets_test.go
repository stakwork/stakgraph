package adminapi

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

// newBudgetTestServer stands up an adminapi mux with a miniredis-backed
// store and a synthetic auth.Config. Phase-7 observability isn't needed
// for these tests, so logstore is nil.
func newBudgetTestServer(t *testing.T, budgets map[string]auth.AgentBudget) (*httptest.Server, *miniredis.Miniredis) {
	t.Helper()
	mr := miniredis.RunT(t)
	rc := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() {
		_ = rc.Close()
		redisclient.SetClientForTest(nil)
		auth.SetConfigForTest(auth.Config{})
	})
	redisclient.SetClientForTest(rc)
	auth.SetConfigForTest(auth.Config{AgentBudgets: budgets})

	mux := http.NewServeMux()
	registerRoutes(mux, routeDeps{
		adminUser:         "admin",
		adminPass:         "hunter2",
		provisioningToken: testToken,
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, mr
}

func TestBudget_WithCap_TodayBucket(t *testing.T) {
	srv, mr := newBudgetTestServer(t, map[string]auth.AgentBudget{
		"coder": {CapUSD: 5.00, Window: "1d"},
	})

	// Seed the day-bucket with $1.25 of spend.
	today := time.Now().UTC().Format("2006-01-02")
	mr.HSet("bifrost:cost:agent:coder:"+today, "total", "1.25")

	req, _ := http.NewRequest(http.MethodGet, srv.URL+"/_plugin/agents/coder/budget", nil)
	req.Header.Set("Authorization", "Bearer "+testToken)
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: %d", resp.StatusCode)
	}
	var got AgentBudgetResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}

	if got.AgentName != "coder" {
		t.Errorf("agent_name: %q", got.AgentName)
	}
	if got.CapUSD == nil || *got.CapUSD != 5.00 {
		t.Errorf("cap: %+v", got.CapUSD)
	}
	if got.Window != "1d" {
		t.Errorf("window: %q", got.Window)
	}
	if got.SpentUSD != 1.25 {
		t.Errorf("spent: %v", got.SpentUSD)
	}
	if got.RemainingUSD == nil || *got.RemainingUSD != 3.75 {
		t.Errorf("remaining: %+v", got.RemainingUSD)
	}
	if got.Ratio == nil || *got.Ratio != 0.25 {
		t.Errorf("ratio: %+v", got.Ratio)
	}
	if got.PeriodStart == "" || got.PeriodEnd == "" {
		t.Errorf("period bounds missing: %+v", got)
	}
}

func TestBudget_WithoutCap_SurfacesDaySpend(t *testing.T) {
	srv, mr := newBudgetTestServer(t, nil)

	// No cap configured, but there's still spend today.
	today := time.Now().UTC().Format("2006-01-02")
	mr.HSet("bifrost:cost:agent:nobudget:"+today, "total", "0.42")

	req, _ := http.NewRequest(http.MethodGet, srv.URL+"/_plugin/agents/nobudget/budget", nil)
	req.Header.Set("Authorization", "Bearer "+testToken)
	resp, _ := srv.Client().Do(req)
	defer resp.Body.Close()
	var got AgentBudgetResponse
	_ = json.NewDecoder(resp.Body).Decode(&got)

	if got.CapUSD != nil {
		t.Errorf("expected nil cap, got %v", got.CapUSD)
	}
	if got.RemainingUSD != nil || got.Ratio != nil {
		t.Errorf("expected nil remaining/ratio when cap is nil: %+v", got)
	}
	if got.SpentUSD != 0.42 {
		t.Errorf("spent: %v", got.SpentUSD)
	}
}

func TestBudget_HourlyWindow(t *testing.T) {
	srv, mr := newBudgetTestServer(t, map[string]auth.AgentBudget{
		"web-search": {CapUSD: 1.00, Window: "1h"},
	})

	now := time.Now().UTC()
	hourStart := now.Truncate(time.Hour).Unix()
	mr.HSet("bifrost:cost:agent:web-search:"+strconv.FormatInt(hourStart, 10),
		"total", "0.10")

	req, _ := http.NewRequest(http.MethodGet, srv.URL+"/_plugin/agents/web-search/budget", nil)
	req.Header.Set("Authorization", "Bearer "+testToken)
	resp, _ := srv.Client().Do(req)
	defer resp.Body.Close()
	var got AgentBudgetResponse
	_ = json.NewDecoder(resp.Body).Decode(&got)

	if got.SpentUSD != 0.10 {
		t.Errorf("spent: %v", got.SpentUSD)
	}
	if got.Ratio == nil || *got.Ratio != 0.10 {
		t.Errorf("ratio: %+v", got.Ratio)
	}
}

func TestBudget_404OnDeepPath(t *testing.T) {
	srv, _ := newBudgetTestServer(t, nil)
	// phase-9 will own /:name/state and /:name/kill; phase-8 only
	// owns /:name/budget. Anything else under /_plugin/agents/
	// should 404.
	for _, p := range []string{
		"/_plugin/agents/coder",
		"/_plugin/agents/coder/state",
		"/_plugin/agents/coder/budget/extra",
		"/_plugin/agents//budget",
	} {
		req, _ := http.NewRequest(http.MethodGet, srv.URL+p, nil)
		req.Header.Set("Authorization", "Bearer "+testToken)
		resp, _ := srv.Client().Do(req)
		resp.Body.Close()
		if resp.StatusCode != http.StatusNotFound {
			t.Errorf("%s: want 404, got %d", p, resp.StatusCode)
		}
	}
}

func TestBudget_RequiresAuth(t *testing.T) {
	srv, _ := newBudgetTestServer(t, nil)
	resp, err := http.Get(srv.URL + "/_plugin/agents/coder/budget")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("want 401, got %d", resp.StatusCode)
	}
}
