package adminapi

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// logstoreBaseURL is the loopback address bifrost-http binds to.
// The plugin never reaches Bifrost over the public listener — that
// would route back through the wrapper, doubling hops for no reason.
const logstoreBaseURL = "http://127.0.0.1:8080"

// logstoreTimeout caps a single call to Bifrost's /api/logs. Generous
// because a fresh-cache rankings query against a large SQLite can
// take a few hundred ms; tight enough that an unresponsive Bifrost
// surfaces fast to the dashboard.
const logstoreTimeout = 5 * time.Second

// logstoreClient is the plugin's HTTP client for Bifrost's logging
// API. Phase 8 ships four read-only handlers; this client owns the
// query-string composition, the Basic-auth header, and the
// SearchFilters/MetadataFilters wire shape so callers stay declarative.
//
// Why an HTTP client (not a direct SQLite handle)
// -----------------------------------------------
// See phase-7-observability.md §"No SQLite handle in the plugin":
//   - Bifrost owns logs.db; schema changes don't break us.
//   - A future Postgres swap is invisible to the plugin.
//   - Bifrost's matview optimisations apply for free.
//   - The extra loopback hop costs ~sub-ms, dwarfed by the query.
//
// Basic auth
// ----------
// Bifrost's /api/logs is gated by the same admin user/password the
// dashboard logs in with — they're configured in data/config.json
// (env.BIFROST_ADMIN_USER / env.BIFROST_ADMIN_PASS). The plugin
// presents those credentials as `Authorization: Basic` on every
// call. They never leave loopback.
type logstoreClient struct {
	base       string
	httpClient *http.Client
	authHeader string // pre-encoded "Basic xxx"
}

// newLogstoreClient builds a client wired to the loopback Bifrost
// with the admin user/pass the plugin already has. Returns nil when
// either credential is missing — the caller (server.go) treats that
// as "skip phase-7 observability routes" rather than serving 500s.
func newLogstoreClient(adminUser, adminPass string) *logstoreClient {
	if adminUser == "" || adminPass == "" {
		return nil
	}
	return &logstoreClient{
		base: logstoreBaseURL,
		httpClient: &http.Client{
			Timeout: logstoreTimeout,
		},
		authHeader: basicAuth(adminUser, adminPass),
	}
}

// basicAuth pre-encodes "user:pass" as a complete Authorization
// header value. Captured once at client construction so the hot path
// doesn't allocate.
func basicAuth(user, pass string) string {
	return "Basic " + base64.StdEncoding.EncodeToString([]byte(user+":"+pass))
}

// ─── search ──────────────────────────────────────────────────────────

// logstoreLog is the subset of Bifrost's Log columns phase 8 reads.
// Defined inline — tygo doesn't emit this; the SPA only sees the
// shapes the plugin's own handlers return.
type logstoreLog struct {
	ID         string  `json:"id"`
	Timestamp  string  `json:"timestamp"`
	Provider   string  `json:"provider"`
	Model      string  `json:"model"`
	Status     string  `json:"status"`
	Cost       float64 `json:"cost"`
	Latency    float64 `json:"latency"`
	CustomerID string  `json:"customer_id"`

	// Bifrost emits metadata as a JSON string on the wire (the gorm
	// model marks it `json:"-"` then a separate hook re-attaches as
	// "metadata"). Decoding as map[string]string covers every dim
	// header the plugin canonicalises.
	Metadata map[string]string `json:"metadata"`
}

// logstoreSearchResult is the trimmed shape of /api/logs.
type logstoreSearchResult struct {
	Logs       []logstoreLog `json:"logs"`
	Pagination struct {
		Limit      int   `json:"limit"`
		Offset     int   `json:"offset"`
		TotalCount int64 `json:"total_count"`
	} `json:"pagination"`
	Stats struct {
		TotalRequests int64   `json:"total_requests"`
		TotalCost     float64 `json:"total_cost"`
		TotalTokens   int64   `json:"total_tokens"`
	} `json:"stats"`
	HasLogs bool `json:"has_logs"`
}

// searchOpts is the small struct phase-8 callers use to compose
// /api/logs queries. Mirrors only the SearchFilters fields phase 8
// uses; future handlers extend it.
type searchOpts struct {
	StartTime *time.Time
	EndTime   *time.Time

	// Metadata filters are sent as `metadata_<key>=<value>` query
	// params, which Bifrost's handler parses into
	// SearchFilters.MetadataFilters. ALL conditions must match
	// (AND), and a single key may only have one value — that's
	// fine for phase 8's filters (run-id, agent-name, etc.).
	Metadata map[string]string

	Limit  int
	Offset int
	SortBy string // "timestamp", "latency", "tokens", "cost"
	Order  string // "asc", "desc"
}

// search hits GET /api/logs with the supplied filters and returns
// the decoded result. The single call paginates up to `Limit` rows;
// callers that need a full window over many pages use searchAll.
func (c *logstoreClient) search(ctx context.Context, o searchOpts) (*logstoreSearchResult, error) {
	q := url.Values{}
	addTime(q, "start_time", o.StartTime)
	addTime(q, "end_time", o.EndTime)
	for k, v := range o.Metadata {
		q.Set("metadata_"+k, v)
	}
	if o.Limit > 0 {
		q.Set("limit", strconv.Itoa(o.Limit))
	}
	if o.Offset > 0 {
		q.Set("offset", strconv.Itoa(o.Offset))
	}
	if o.SortBy != "" {
		q.Set("sort_by", o.SortBy)
	}
	if o.Order != "" {
		q.Set("order", o.Order)
	}

	var out logstoreSearchResult
	if err := c.getJSON(ctx, "/api/logs?"+q.Encode(), &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// searchAll pages through /api/logs until it has exhausted the
// window (or hit `maxRows` as a safety cap). Used by the agent-name
// aggregation, where Bifrost's GetDimensionCostHistogram doesn't
// support metadata columns natively and we have to compute
// per-agent sums in Go.
//
// `pageSize` is bounded by Bifrost's own server-side cap (1000).
// `maxRows` is the global safety cap so a runaway query can't OOM
// the plugin — phase 8's busiest expected window (30d × thousands
// of calls/day) is comfortably under 200k rows.
func (c *logstoreClient) searchAll(
	ctx context.Context,
	o searchOpts,
	pageSize, maxRows int,
) ([]logstoreLog, error) {
	if pageSize <= 0 || pageSize > 1000 {
		pageSize = 1000
	}
	if maxRows <= 0 {
		maxRows = 200_000
	}
	o.Limit = pageSize
	o.Offset = 0
	var all []logstoreLog
	for {
		res, err := c.search(ctx, o)
		if err != nil {
			return nil, err
		}
		all = append(all, res.Logs...)
		if len(res.Logs) < pageSize {
			break // last page
		}
		if len(all) >= maxRows {
			break // safety cap; document in the caller
		}
		o.Offset += pageSize
	}
	return all, nil
}

// ─── user rankings (bifrost native) ──────────────────────────────────

// logstoreUserRanking mirrors UserRankingEntry from
// framework/logstore.tables.go. UserName isn't in the Bifrost
// struct (only UserID) but Bifrost's /api/logs/rankings endpoint
// hydrates the name from another table — phase 8 currently treats
// user-id and name as interchangeable per the v2 invariant that
// customer_id = user_id.
type logstoreUserRanking struct {
	UserID        string  `json:"user_id"`
	TotalRequests int64   `json:"total_requests"`
	TotalTokens   int64   `json:"total_tokens"`
	TotalCost     float64 `json:"total_cost"`
}

// userRankings is currently UNUSED — phase 8 derives by-user from
// SearchStats per-customer rather than Bifrost's rankings endpoint
// because the latter applies a "trends" calculation that we don't
// want surfaced through phase 8's clean shape. Kept here, with the
// matching JSON shape, so phase 9 can swap in if it wants the
// trend deltas.

// ─── helpers ─────────────────────────────────────────────────────────

// getJSON GETs `path` (must include leading slash) against Bifrost's
// loopback API, decodes the JSON body into `out`. Any non-2xx is
// returned as an upstreamError so handlers can map it to a 502 with
// the right error code.
func (c *logstoreClient) getJSON(ctx context.Context, path string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+path, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", c.authHeader)
	req.Header.Set("Accept", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return &upstreamError{cause: err}
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// Bifrost emits BifrostError on failure — read up to 2 KB
		// for a useful log line, never propagated to the browser
		// because /api/logs's error envelope differs from ours.
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return &upstreamError{
			status: resp.StatusCode,
			body:   strings.TrimSpace(string(body)),
		}
	}
	dec := json.NewDecoder(resp.Body)
	if err := dec.Decode(out); err != nil {
		return fmt.Errorf("logstore: decode %s: %w", path, err)
	}
	return nil
}

// upstreamError marks a failure that maps to phase-7's
// `upstream_unavailable` error code. Handler-level glue calls
// errors.As to switch on it.
type upstreamError struct {
	status int
	body   string
	cause  error
}

func (e *upstreamError) Error() string {
	if e.cause != nil {
		return fmt.Sprintf("logstore: upstream: %v", e.cause)
	}
	return fmt.Sprintf("logstore: upstream status=%d body=%s", e.status, e.body)
}

// addTime is a small helper that adds an RFC3339 timestamp to a
// url.Values only when the pointer is non-nil. Keeps the call sites
// in search() one line each.
func addTime(v url.Values, key string, t *time.Time) {
	if t == nil {
		return
	}
	v.Set(key, t.UTC().Format(time.RFC3339Nano))
}
