// Package graphclient is the gateway's thin HTTP client for neo4j's
// native transactional Cypher endpoint (`/db/<db>/tx/commit`).
//
// Why HTTP and not bolt
// ---------------------
// The gateway is a Bifrost .so plugin built from pure stdlib HTTP. We
// don't want to pull the neo4j Go bolt driver (cgo-free but a large
// dependency) into the plugin just to write a handful of catalog
// upserts. neo4j ships a first-class JSON-over-HTTP Cypher API on its
// 7474 port; Basic auth with the neo4j password (already in the swarm
// env) is all it needs. This mirrors how internal/adminapi's
// logstoreClient wraps Bifrost's loopback API — same shape, different
// upstream.
//
// Only the gateway holds the neo4j credential and owns the upsert;
// catalog sources POST manifests to the gateway, never to neo4j
// directly (see gateway/plans/agent-catalog.md).
package graphclient

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/stakwork/stakgraph/gateway/internal/env"
)

// requestTimeout caps a single tx/commit round-trip. Catalog writes
// are small (a fleet manifest is kilobytes) and reads are single-agent
// lookups, so a tight-ish timeout keeps an unresponsive neo4j from
// hanging the dashboard.
const requestTimeout = 8 * time.Second

// Client talks to one neo4j database over its HTTP Cypher API. nil is
// a valid "not configured" value — callers (the catalog handlers)
// nil-check and return 503, the same contract redisclient.Client()
// uses for observability mode.
type Client struct {
	endpoint   string // fully-composed .../db/<db>/tx/commit URL
	httpClient *http.Client
	authHeader string // pre-encoded "Basic xxx"
}

// New builds a Client from env config, or returns nil when neo4j is
// not configured (NEO4J_PASSWORD unset). Wiring this at server start
// means the per-request path never re-reads env.
func New() *Client {
	cfg, ok := env.Neo4jHTTPConfigValue()
	if !ok {
		return nil
	}
	return NewFromConfig(cfg)
}

// NewFromConfig is New with an explicit config — handy for tests that
// point at a local neo4j without touching the process environment.
func NewFromConfig(cfg env.Neo4jConfig) *Client {
	endpoint := strings.TrimRight(cfg.BaseURL, "/") + "/db/" + cfg.Database + "/tx/commit"
	return &Client{
		endpoint:   endpoint,
		httpClient: &http.Client{Timeout: requestTimeout},
		authHeader: "Basic " + base64.StdEncoding.EncodeToString([]byte(cfg.User+":"+cfg.Password)),
	}
}

// Statement is one parameterised Cypher statement in a tx/commit
// batch. Parameters is sent as-is; neo4j accepts scalars, lists and
// maps (used for UNWIND-driven batch creates).
type Statement struct {
	Statement  string         `json:"statement"`
	Parameters map[string]any `json:"parameters,omitempty"`
}

// Result is one statement's result block: ordered columns and rows.
// Each row's values are kept as raw JSON so callers decode positionally
// into whatever shape the RETURN clause produced (a node's properties
// object, a scalar, a list, …).
type Result struct {
	Columns []string `json:"columns"`
	Data    []struct {
		Row []json.RawMessage `json:"row"`
	} `json:"data"`
}

// wire shapes for the tx/commit envelope.
type txRequest struct {
	Statements []Statement `json:"statements"`
}

type txResponse struct {
	Results []Result `json:"results"`
	Errors  []neoErr `json:"errors"`
}

type neoErr struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// Run posts one or more statements as a single atomic transaction
// (tx/commit commits or rolls back the whole batch). It returns one
// Result per statement, in order. Any neo4j-reported error fails the
// whole call — the transaction is already rolled back server-side.
func (c *Client) Run(ctx context.Context, statements ...Statement) ([]Result, error) {
	body, err := json.Marshal(txRequest{Statements: statements})
	if err != nil {
		return nil, fmt.Errorf("graphclient: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("graphclient: build request: %w", err)
	}
	req.Header.Set("Authorization", c.authHeader)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("graphclient: neo4j unreachable: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// 401 (bad password) and 404 (missing db) land here; read a
		// bounded slice for a useful log line.
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("graphclient: neo4j http %d: %s",
			resp.StatusCode, strings.TrimSpace(string(snippet)))
	}

	var out txResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("graphclient: decode response: %w", err)
	}
	if len(out.Errors) > 0 {
		// neo4j returns 200 with a populated errors[] for Cypher-level
		// failures (syntax, constraint violation). Surface the first.
		return nil, fmt.Errorf("graphclient: cypher error %s: %s",
			out.Errors[0].Code, out.Errors[0].Message)
	}
	return out.Results, nil
}

// Query is the single-statement convenience over Run, returning that
// statement's Result (or an empty Result if the statement produced no
// result block).
func (c *Client) Query(ctx context.Context, cypher string, params map[string]any) (Result, error) {
	results, err := c.Run(ctx, Statement{Statement: cypher, Parameters: params})
	if err != nil {
		return Result{}, err
	}
	if len(results) == 0 {
		return Result{}, nil
	}
	return results[0], nil
}
