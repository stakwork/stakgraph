package adminapi

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/stakwork/stakgraph/gateway/internal/graphclient"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// ─── wire shapes: write (POST /_plugin/agents) ───────────────────────

// catalogPushRequest is the whole-fleet manifest one source pushes.
// "Whole fleet object" (not one-agent-per-call) because a source
// typically knows its entire fleet at once and one call per deploy is
// simpler to operate. A source that only knows one agent sends a
// one-element Agents slice. See gateway/plans/agent-catalog.md.
type catalogPushRequest struct {
	Source string             `json:"source"` // REQUIRED contributing system
	Agents []catalogPushAgent `json:"agents"`
}

type catalogPushAgent struct {
	Name         string              `json:"name"` // REQUIRED; matches x-bf-dim-agent-name
	DisplayName  string              `json:"display_name"`
	Description  string              `json:"description"`
	DefaultModel string              `json:"default_model"` // default LLM for this agent
	Version      string              `json:"version"`       // source's own stamp (git sha, rev id…)
	Prompts      []catalogPushPrompt `json:"prompts"`
	Tools        []catalogPushTool   `json:"tools"`
	Skills       []catalogPushSkill  `json:"skills"`
}

type catalogPushPrompt struct {
	Name   string `json:"name"`
	Role   string `json:"role"`
	Body   string `json:"body"`
	Source string `json:"source"` // optional per-item source override
}

type catalogPushTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Schema      json.RawMessage `json:"schema"` // JSON parameter schema, stored as a string
	Source      string          `json:"source"`
}

type catalogPushSkill struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Source      string `json:"source"`
}

// catalogWriteResponse reports how many nodes the push wrote.
type catalogWriteResponse struct {
	Written struct {
		Agents  int `json:"agents"`
		Prompts int `json:"prompts"`
		Tools   int `json:"tools"`
		Skills  int `json:"skills"`
	} `json:"written"`
}

// ─── wire shapes: read (GET /_plugin/agents/:name/catalog) ───────────

// AgentCatalogResponse is the merged view across all sources for one
// agent — what the agent is _made of_ (prompts/tools/skills), as
// opposed to the budget view's what it's _allowed to spend_.
type AgentCatalogResponse struct {
	Name         string          `json:"name"`
	DisplayName  string          `json:"display_name,omitempty"`
	Description  string          `json:"description,omitempty"`
	DefaultModel string          `json:"default_model,omitempty"`
	Sources      []string        `json:"sources"`
	Prompts      []CatalogPrompt `json:"prompts"`
	Tools        []CatalogTool   `json:"tools"`
	Skills       []CatalogSkill  `json:"skills"`
}

type CatalogPrompt struct {
	Name      string `json:"name"`
	Role      string `json:"role"`
	Body      string `json:"body"`
	Source    string `json:"source"`
	Version   string `json:"version,omitempty"`
	UpdatedAt string `json:"updated_at"`
}

type CatalogTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Schema      json.RawMessage `json:"schema,omitempty"`
	Source      string          `json:"source"`
	Version     string          `json:"version,omitempty"`
	UpdatedAt   string          `json:"updated_at"`
}

type CatalogSkill struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Source      string `json:"source"`
	Version     string `json:"version,omitempty"`
	UpdatedAt   string `json:"updated_at"`
}

// ─── handler ─────────────────────────────────────────────────────────

// catalogHandlers owns the neo4j graph client for the agent catalog.
// graph is nil when neo4j isn't configured (NEO4J_PASSWORD unset) — in
// that mode every endpoint returns 503, matching the Redis-style
// graceful degradation the rest of the plugin uses.
type catalogHandlers struct {
	graph *graphclient.Client

	schemaMu    sync.Mutex
	schemaReady bool // indexes/constraints created? ("on first write")
}

func newCatalogHandlers(graph *graphclient.Client) *catalogHandlers {
	return &catalogHandlers{graph: graph}
}

// maxCatalogBody caps the push payload. A fleet manifest is kilobytes;
// 4 MiB is generous headroom (large system prompts) while bounding
// memory.
const maxCatalogBody = 4 << 20

// push handles POST /_plugin/agents — bearer-only, the server-to-server
// write front door. Replace-by-source semantics: the push is the
// complete picture of what THIS source knows about each agent, so the
// gateway deletes that source's existing children for the agent and
// re-creates them in one transaction. Other sources' contributions are
// untouched.
func (h *catalogHandlers) push(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	if h.graph == nil {
		writeError(w, http.StatusServiceUnavailable, "catalog_unavailable",
			"agent catalog not configured (neo4j unset on this swarm)")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxCatalogBody)
	var req catalogPushRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}
	if strings.TrimSpace(req.Source) == "" {
		writeError(w, http.StatusBadRequest, "missing_source", "source is required")
		return
	}
	for i := range req.Agents {
		if strings.TrimSpace(req.Agents[i].Name) == "" {
			writeError(w, http.StatusBadRequest, "missing_agent_name",
				fmt.Sprintf("agents[%d].name is required", i))
			return
		}
	}

	ctx := r.Context()
	if err := h.ensureSchema(ctx); err != nil {
		// Non-fatal for correctness (MERGE on name still works), but
		// log it — a persistently missing constraint is worth seeing.
		pluginlog.Warnf("adminapi: catalog ensureSchema: %v", err)
	}

	stmts, counts := buildCatalogWrite(req)
	if _, err := h.graph.Run(ctx, stmts...); err != nil {
		pluginlog.Errf("adminapi: catalog push: %v", err)
		writeError(w, http.StatusBadGateway, "catalog_write_failed", "neo4j write failed")
		return
	}

	var resp catalogWriteResponse
	resp.Written.Agents = counts.Agents
	resp.Written.Prompts = counts.Prompts
	resp.Written.Tools = counts.Tools
	resp.Written.Skills = counts.Skills
	writeJSON(w, http.StatusOK, resp)
}

// read handles GET /_plugin/agents/:name/catalog — cookie-or-bearer,
// the read-only dashboard view. Returns the merged view across all
// sources, or 404 when the agent has no catalog node yet.
func (h *catalogHandlers) read(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	if h.graph == nil {
		writeError(w, http.StatusServiceUnavailable, "catalog_unavailable",
			"agent catalog not configured (neo4j unset on this swarm)")
		return
	}

	const prefix = "/_plugin/agents/"
	rest := strings.TrimPrefix(r.URL.Path, prefix)
	parts := strings.Split(rest, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] != "catalog" {
		http.NotFound(w, r)
		return
	}
	name := parts[0]

	out, found, err := h.fetchCatalog(r.Context(), name)
	if err != nil {
		pluginlog.Errf("adminapi: catalog read agent=%s: %v", name, err)
		writeError(w, http.StatusBadGateway, "catalog_read_failed", "neo4j read failed")
		return
	}
	if !found {
		writeError(w, http.StatusNotFound, "not_found", "no catalog for agent")
		return
	}
	writeJSON(w, http.StatusOK, out)
}

// ─── schema (indexes on first write) ─────────────────────────────────

// ensureSchema creates the catalog's constraint + indexes once. Each
// DDL statement runs as its own transaction (neo4j forbids mixing
// schema and data writes, and older versions forbid multiple schema
// updates per tx). On failure schemaReady stays false so the next
// write retries.
func (h *catalogHandlers) ensureSchema(ctx context.Context) error {
	h.schemaMu.Lock()
	defer h.schemaMu.Unlock()
	if h.schemaReady {
		return nil
	}
	for _, s := range graphclient.SchemaStatements() {
		if _, err := h.graph.Run(ctx, s); err != nil {
			return err
		}
	}
	h.schemaReady = true
	return nil
}

// ─── write builder ───────────────────────────────────────────────────

// buildCatalogWrite turns a push request into an ordered batch of
// statements (one atomic tx) plus the counts for the response.
//
// Per agent the statements are:
//  1. MERGE agent + source + DEFINED_BY; delete this push's sources'
//     existing children (replace-by-source).
//  2. UNWIND-create prompts.
//  3. UNWIND-create tools.
//  4. UNWIND-create skills.
//
// Label and relationship names are interpolated from graphclient
// constants (never user input), so there's no injection surface; the
// per-node data all flows through bound $parameters.
func buildCatalogWrite(req catalogPushRequest) ([]graphclient.Statement, struct {
	Agents, Prompts, Tools, Skills int
}) {
	now := time.Now().UTC().Format(time.RFC3339)
	var stmts []graphclient.Statement
	var counts struct{ Agents, Prompts, Tools, Skills int }

	srcOr := func(override string) string {
		if strings.TrimSpace(override) != "" {
			return override
		}
		return req.Source
	}

	for _, ag := range req.Agents {
		// Collect the distinct set of sources this push touches for
		// the agent so cleanup removes exactly what we're about to
		// re-create (plus the push source itself, so an emptied kind
		// is cleared). Children owned by *other* sources survive.
		sourceSet := map[string]struct{}{req.Source: {}}

		prompts := make([]map[string]any, 0, len(ag.Prompts))
		for _, p := range ag.Prompts {
			src := srcOr(p.Source)
			sourceSet[src] = struct{}{}
			prompts = append(prompts, map[string]any{
				"node_key": nodeKey(ag.Name, src, "prompt", p.Name),
				"name":     p.Name,
				"role":     p.Role,
				"body":     p.Body,
				"source":   src,
				"version":  ag.Version,
			})
		}
		tools := make([]map[string]any, 0, len(ag.Tools))
		for _, t := range ag.Tools {
			src := srcOr(t.Source)
			sourceSet[src] = struct{}{}
			tools = append(tools, map[string]any{
				"node_key":    nodeKey(ag.Name, src, "tool", t.Name),
				"name":        t.Name,
				"description": t.Description,
				"schema":      rawToString(t.Schema),
				"source":      src,
				"version":     ag.Version,
			})
		}
		skills := make([]map[string]any, 0, len(ag.Skills))
		for _, s := range ag.Skills {
			src := srcOr(s.Source)
			sourceSet[src] = struct{}{}
			skills = append(skills, map[string]any{
				"node_key":    nodeKey(ag.Name, src, "skill", s.Name),
				"name":        s.Name,
				"description": s.Description,
				"source":      src,
				"version":     ag.Version,
			})
		}

		sources := make([]string, 0, len(sourceSet))
		for s := range sourceSet {
			sources = append(sources, s)
		}

		counts.Agents++
		counts.Prompts += len(prompts)
		counts.Tools += len(tools)
		counts.Skills += len(skills)

		stmts = append(stmts,
			graphclient.Statement{
				Statement: fmt.Sprintf(
					"MERGE (a:%[1]s {name: $name}) "+
						"SET a.updated_at = $now, "+
						"a.display_name = CASE WHEN $display_name <> '' THEN $display_name ELSE a.display_name END, "+
						"a.description = CASE WHEN $description <> '' THEN $description ELSE a.description END, "+
						"a.default_model = CASE WHEN $default_model <> '' THEN $default_model ELSE a.default_model END "+
						"MERGE (s:%[2]s {name: $source}) SET s.updated_at = $now "+
						"MERGE (a)-[:%[3]s]->(s) "+
						"WITH a "+
						"OPTIONAL MATCH (a)-[:%[4]s|%[5]s|%[6]s]->(c) WHERE c.source IN $sources "+
						"DETACH DELETE c",
					graphclient.LabelAgent, graphclient.LabelSource, graphclient.RelDefinedBy,
					graphclient.RelHasPrompt, graphclient.RelHasTool, graphclient.RelHasSkill),
				Parameters: map[string]any{
					"name":          ag.Name,
					"display_name":  ag.DisplayName,
					"description":   ag.Description,
					"default_model": ag.DefaultModel,
					"source":        req.Source,
					"sources":       sources,
					"now":           now,
				},
			},
			unwindCreate(graphclient.RelHasPrompt, graphclient.LabelPrompt, ag.Name, now, prompts,
				"node_key: row.node_key, name: row.name, role: row.role, body: row.body, source: row.source, version: row.version"),
			unwindCreate(graphclient.RelHasTool, graphclient.LabelTool, ag.Name, now, tools,
				"node_key: row.node_key, name: row.name, description: row.description, schema: row.schema, source: row.source, version: row.version"),
			unwindCreate(graphclient.RelHasSkill, graphclient.LabelSkill, ag.Name, now, skills,
				"node_key: row.node_key, name: row.name, description: row.description, source: row.source, version: row.version"),
		)
	}
	return stmts, counts
}

// unwindCreate builds a single UNWIND-CREATE statement for one child
// kind. Over an empty `rows` slice it creates nothing (UNWIND of [] is
// a no-op) — so an agent that dropped all its tools still runs the
// statement harmlessly after cleanup cleared the old ones.
func unwindCreate(rel, label, agentName, now string, rows []map[string]any, props string) graphclient.Statement {
	return graphclient.Statement{
		Statement: fmt.Sprintf(
			"MATCH (a:%[1]s {name: $name}) "+
				"UNWIND $rows AS row "+
				"CREATE (a)-[:%[2]s]->(:%[3]s {%[4]s, updated_at: $now})",
			graphclient.LabelAgent, rel, label, props),
		Parameters: map[string]any{
			"name": agentName,
			"rows": rows,
			"now":  now,
		},
	}
}

// ─── read builder ────────────────────────────────────────────────────

// fetchCatalog runs one tx/commit batch of five statements (existence
// + scalars, sources, prompts, tools, skills) and assembles the merged
// view. Separate statements rather than one big OPTIONAL MATCH avoid
// the cartesian-product blow-up of prompts × tools × skills.
func (h *catalogHandlers) fetchCatalog(ctx context.Context, name string) (*AgentCatalogResponse, bool, error) {
	p := map[string]any{"name": name}
	results, err := h.graph.Run(ctx,
		graphclient.Statement{
			Statement:  fmt.Sprintf("MATCH (a:%s {name: $name}) RETURN a.name, a.display_name, a.description, a.default_model", graphclient.LabelAgent),
			Parameters: p,
		},
		graphclient.Statement{
			Statement: fmt.Sprintf("MATCH (a:%s {name: $name})-[:%s]->(s:%s) RETURN s.name ORDER BY s.name",
				graphclient.LabelAgent, graphclient.RelDefinedBy, graphclient.LabelSource),
			Parameters: p,
		},
		graphclient.Statement{
			Statement: fmt.Sprintf("MATCH (a:%s {name: $name})-[:%s]->(p:%s) "+
				"RETURN p.name, p.role, p.body, p.source, p.version, p.updated_at ORDER BY p.source, p.name",
				graphclient.LabelAgent, graphclient.RelHasPrompt, graphclient.LabelPrompt),
			Parameters: p,
		},
		graphclient.Statement{
			Statement: fmt.Sprintf("MATCH (a:%s {name: $name})-[:%s]->(t:%s) "+
				"RETURN t.name, t.description, t.schema, t.source, t.version, t.updated_at ORDER BY t.source, t.name",
				graphclient.LabelAgent, graphclient.RelHasTool, graphclient.LabelTool),
			Parameters: p,
		},
		graphclient.Statement{
			Statement: fmt.Sprintf("MATCH (a:%s {name: $name})-[:%s]->(k:%s) "+
				"RETURN k.name, k.description, k.source, k.version, k.updated_at ORDER BY k.source, k.name",
				graphclient.LabelAgent, graphclient.RelHasSkill, graphclient.LabelSkill),
			Parameters: p,
		},
	)
	if err != nil {
		return nil, false, err
	}
	if len(results) < 5 || len(results[0].Data) == 0 {
		return nil, false, nil // agent has no catalog node
	}

	agentRow := results[0].Data[0].Row
	out := &AgentCatalogResponse{
		Name:         rowString(at(agentRow, 0)),
		DisplayName:  rowString(at(agentRow, 1)),
		Description:  rowString(at(agentRow, 2)),
		DefaultModel: rowString(at(agentRow, 3)),
		Sources:      []string{},
		Prompts:      []CatalogPrompt{},
		Tools:        []CatalogTool{},
		Skills:       []CatalogSkill{},
	}

	for _, row := range results[1].Data {
		out.Sources = append(out.Sources, rowString(at(row.Row, 0)))
	}
	for _, row := range results[2].Data {
		out.Prompts = append(out.Prompts, CatalogPrompt{
			Name:      rowString(at(row.Row, 0)),
			Role:      rowString(at(row.Row, 1)),
			Body:      rowString(at(row.Row, 2)),
			Source:    rowString(at(row.Row, 3)),
			Version:   rowString(at(row.Row, 4)),
			UpdatedAt: rowString(at(row.Row, 5)),
		})
	}
	for _, row := range results[3].Data {
		out.Tools = append(out.Tools, CatalogTool{
			Name:        rowString(at(row.Row, 0)),
			Description: rowString(at(row.Row, 1)),
			Schema:      stringToRaw(rowString(at(row.Row, 2))),
			Source:      rowString(at(row.Row, 3)),
			Version:     rowString(at(row.Row, 4)),
			UpdatedAt:   rowString(at(row.Row, 5)),
		})
	}
	for _, row := range results[4].Data {
		out.Skills = append(out.Skills, CatalogSkill{
			Name:        rowString(at(row.Row, 0)),
			Description: rowString(at(row.Row, 1)),
			Source:      rowString(at(row.Row, 2)),
			Version:     rowString(at(row.Row, 3)),
			UpdatedAt:   rowString(at(row.Row, 4)),
		})
	}
	return out, true, nil
}

// ─── helpers ─────────────────────────────────────────────────────────

// nodeKey is the deterministic identity of a prompt/tool/skill node:
// a hash of (agent, source, kind, name). Re-pushing the same node
// yields the same key, so the catalog is idempotent and a child
// belongs to exactly one (agent, source, kind, name) tuple.
func nodeKey(agent, source, kind, name string) string {
	h := sha256.Sum256([]byte(agent + "\x00" + source + "\x00" + kind + "\x00" + name))
	return hex.EncodeToString(h[:])
}

// rawToString renders a tool's JSON schema for storage. neo4j has no
// nested-object property type, so the schema lives as a JSON string
// and is re-emitted verbatim on read. Empty/null ⇒ "".
func rawToString(raw json.RawMessage) string {
	s := strings.TrimSpace(string(raw))
	if s == "" || s == "null" {
		return ""
	}
	return s
}

// stringToRaw is rawToString's inverse for the read path: a stored
// schema string becomes raw JSON in the response (nil ⇒ omitted).
func stringToRaw(s string) json.RawMessage {
	if s == "" {
		return nil
	}
	return json.RawMessage(s)
}

// rowString decodes a single returned cell into a string. neo4j null
// (unset property) unmarshals into "" without error.
func rowString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return ""
	}
	return s
}

// at returns the i-th cell of a row, or empty when the RETURN clause
// produced fewer columns than expected (defensive; shouldn't happen).
func at(row []json.RawMessage, i int) json.RawMessage {
	if i < 0 || i >= len(row) {
		return nil
	}
	return row[i]
}
