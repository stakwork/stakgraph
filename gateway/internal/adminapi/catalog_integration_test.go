package adminapi

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/stakwork/stakgraph/gateway/internal/env"
	"github.com/stakwork/stakgraph/gateway/internal/graphclient"
)

// Integration test against a live neo4j. Skipped unless RUN_NEO4J_TESTS
// is set, so the default `go test` run stays hermetic. Run with:
//
//	RUN_NEO4J_TESTS=1 NEO4J_HTTP_URL=http://localhost:7474 \
//	  NEO4J_USER=neo4j NEO4J_PASSWORD=testtest \
//	  go test ./internal/adminapi/ -run TestCatalogIntegration -v
func TestCatalogIntegration(t *testing.T) {
	if os.Getenv("RUN_NEO4J_TESTS") == "" {
		t.Skip("set RUN_NEO4J_TESTS=1 (and NEO4J_* env) to run the neo4j integration test")
	}
	cfg, ok := env.Neo4jHTTPConfigValue()
	if !ok {
		t.Fatal("NEO4J_PASSWORD must be set for the integration test")
	}
	graph := graphclient.NewFromConfig(cfg)
	cat := newCatalogHandlers(graph)
	ctx := context.Background()

	const agent = "test-catalog-agent"
	cleanup := func() {
		_, err := graph.Query(ctx,
			fmt.Sprintf("MATCH (a:%s {name:$n}) OPTIONAL MATCH (a)-->(c) DETACH DELETE a, c", graphclient.LabelAgent),
			map[string]any{"n": agent})
		if err != nil {
			t.Fatalf("cleanup: %v", err)
		}
	}
	cleanup()
	t.Cleanup(cleanup)

	// ── push from "hive": tools (+ one prompt, one skill) ──
	hivePush := map[string]any{
		"source": "hive",
		"agents": []map[string]any{{
			"name":          agent,
			"display_name":  "Test Catalog Agent",
			"description":   "Used by the integration test.",
			"default_model": "sonnet",
			"version":       "git:abc123",
			"prompts": []map[string]any{
				{"name": "system", "role": "system", "body": "You are a test."},
			},
			"tools": []map[string]any{
				{"name": "read_file", "description": "Read a file",
					"schema": map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}}},
				{"name": "write_file", "description": "Write a file"},
			},
			"skills": []map[string]any{
				{"name": "security-review", "description": "OWASP review"},
			},
		}},
	}
	wrote := doPush(t, cat, hivePush)
	if wrote.Written.Agents != 1 || wrote.Written.Prompts != 1 || wrote.Written.Tools != 2 || wrote.Written.Skills != 1 {
		t.Fatalf("unexpected write counts: %+v", wrote.Written)
	}

	got := doRead(t, cat, agent, http.StatusOK)
	if got.DisplayName != "Test Catalog Agent" {
		t.Errorf("display_name = %q", got.DisplayName)
	}
	if got.DefaultModel != "sonnet" {
		t.Errorf("default_model = %q, want sonnet", got.DefaultModel)
	}
	if len(got.Tools) != 2 {
		t.Fatalf("tools = %d, want 2", len(got.Tools))
	}
	// schema round-trips as raw JSON.
	var schemaTool *CatalogTool
	for i := range got.Tools {
		if got.Tools[i].Name == "read_file" {
			schemaTool = &got.Tools[i]
		}
	}
	if schemaTool == nil || !strings.Contains(string(schemaTool.Schema), `"path"`) {
		t.Errorf("read_file schema didn't round-trip: %+v", schemaTool)
	}

	// ── idempotency: re-push the same hive manifest ──
	doPush(t, cat, hivePush)
	got = doRead(t, cat, agent, http.StatusOK)
	if len(got.Tools) != 2 || len(got.Prompts) != 1 || len(got.Skills) != 1 {
		t.Fatalf("re-push duplicated children: tools=%d prompts=%d skills=%d",
			len(got.Tools), len(got.Prompts), len(got.Skills))
	}

	// ── second source (prompt-manager) contributes a prompt ──
	doPush(t, cat, map[string]any{
		"source": "prompt-manager",
		"agents": []map[string]any{{
			"name":    agent,
			"version": "rev-42",
			"prompts": []map[string]any{{"name": "guardrails", "role": "system", "body": "Be careful."}},
		}},
	})
	got = doRead(t, cat, agent, http.StatusOK)
	if len(got.Sources) != 2 {
		t.Errorf("sources = %v, want hive + prompt-manager", got.Sources)
	}
	if len(got.Prompts) != 2 { // hive's system + prompt-manager's guardrails
		t.Errorf("prompts = %d, want 2 (hive's preserved across other source's push)", len(got.Prompts))
	}
	// hive's tools must survive prompt-manager's push (replace-by-source).
	if len(got.Tools) != 2 {
		t.Errorf("tools = %d, want 2 (hive's tools clobbered by other source)", len(got.Tools))
	}

	// ── 404 for unknown agent ──
	doRead(t, cat, "no-such-agent-xyz", http.StatusNotFound)

	// ── list includes our agent with correct counts ──
	req := httptest.NewRequest(http.MethodGet, "/_plugin/agents/catalog", nil)
	rec := httptest.NewRecorder()
	cat.list(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("list status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var list CatalogListResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
		t.Fatalf("list decode: %v", err)
	}
	var row *CatalogAgentSummary
	for i := range list.Agents {
		if list.Agents[i].Name == agent {
			row = &list.Agents[i]
		}
	}
	if row == nil {
		t.Fatalf("list missing %q (got %d agents)", agent, len(list.Agents))
	}
	if row.DefaultModel != "sonnet" {
		t.Errorf("list default_model = %q", row.DefaultModel)
	}
	if row.Prompts != 2 || row.Tools != 2 || row.Skills != 1 {
		t.Errorf("list counts = p%d t%d s%d, want p2 t2 s1",
			row.Prompts, row.Tools, row.Skills)
	}
	if len(row.Sources) != 2 {
		t.Errorf("list sources = %v, want hive + prompt-manager", row.Sources)
	}
}

func doPush(t *testing.T, cat *catalogHandlers, body any) catalogWriteResponse {
	t.Helper()
	raw, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/_plugin/agents", strings.NewReader(string(raw)))
	rec := httptest.NewRecorder()
	cat.push(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("push status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var out catalogWriteResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("push decode: %v", err)
	}
	return out
}

func doRead(t *testing.T, cat *catalogHandlers, name string, wantStatus int) AgentCatalogResponse {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/_plugin/agents/"+name+"/catalog", nil)
	rec := httptest.NewRecorder()
	cat.read(rec, req)
	if rec.Code != wantStatus {
		t.Fatalf("read %s status = %d (want %d), body = %s", name, rec.Code, wantStatus, rec.Body.String())
	}
	var out AgentCatalogResponse
	if wantStatus == http.StatusOK {
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatalf("read decode: %v", err)
		}
	}
	return out
}
