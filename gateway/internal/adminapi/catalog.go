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

// catalogPushPrompt references an existing `:Prompt` node by name. The
// gateway does NOT store a prompt body — the body lives on the shared
// `:Prompt` node (authored by the Stakwork prompt workflow). A push
// carries only the name so the gateway can wire HiveAgent-[:HAS_PROMPT]
// ->Prompt; a name with no matching node is skipped silently.
type catalogPushPrompt struct {
	Name   string `json:"name"`
	Source string `json:"source"` // optional per-item source override
	// Role is the slot this prompt fills for the agent: "SYSTEM" (the
	// system prompt) vs "USER" (the main/task prompt). The shared
	// `:Prompt` node is authored elsewhere, so this is stored on the
	// HAS_PROMPT relationship the catalog owns. Empty ⇒ unknown.
	Role string `json:"role"`
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

// CatalogAgentSummary is one row of the catalog list — enough to merge
// the registry into the spend-derived /agents table without pulling
// every agent's full prompt/tool/skill bodies. Counts let the list
// render "📄 2 · 🔧 5 · ✦ 1" badges cheaply.
type CatalogAgentSummary struct {
	Name         string   `json:"name"`
	DisplayName  string   `json:"display_name,omitempty"`
	Description  string   `json:"description,omitempty"`
	DefaultModel string   `json:"default_model,omitempty"`
	Sources      []string `json:"sources"`
	Prompts      int      `json:"prompts"`
	Tools        int      `json:"tools"`
	Skills       int      `json:"skills"`
	UpdatedAt    string   `json:"updated_at"`
}

// CatalogListResponse is the whole registry — every HiveAgent node,
// traffic or not. The UI unions this with spend-by-agent so seeded
// agents that have never been invoked still appear.
type CatalogListResponse struct {
	Agents []CatalogAgentSummary `json:"agents"`
}

// CatalogPrompt is one prompt linked to an agent. name/body come from
// the shared `:Prompt` node the agent links to; source/updated_at come
// from the HAS_PROMPT relationship (which system wired it, and when).
type CatalogPrompt struct {
	Name string `json:"name"`
	Body string `json:"body"`
	// Role is the prompt's slot for this agent — "SYSTEM" or "USER"
	// (the main/task prompt). Stored on the HAS_PROMPT relationship;
	// empty when the wiring source didn't classify it.
	Role      string `json:"role,omitempty"`
	Source    string `json:"source"`
	UpdatedAt string `json:"updated_at"`
}

type CatalogTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Schema      json.RawMessage `json:"schema,omitempty"`
	Source      string          `json:"source"`
	Version     string          `json:"version,omitempty"`
	// Enabled is the per-swarm operator toggle, mirroring skills:
	// seeded enabled, flipped in the dashboard, preserved across
	// Hive re-seeds. Legacy nodes with no value read back as true.
	Enabled   bool   `json:"enabled"`
	UpdatedAt string `json:"updated_at"`
}

type CatalogSkill struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Source      string `json:"source"`
	Version     string `json:"version,omitempty"`
	// Enabled is the per-swarm operator toggle. Skills seed enabled by
	// default; an operator flips this in the dashboard and the gateway
	// preserves it across Hive re-seeds (a push only refreshes the
	// palette + metadata, never the toggle). Legacy nodes with no
	// stored value read back as true (coalesce on the read query).
	Enabled   bool   `json:"enabled"`
	UpdatedAt string `json:"updated_at"`
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

// ─── tool / skill enable/disable toggle ──────────────────────────────

// childToggleRequest is the PATCH /_plugin/agents/:name/{tools,skills}
// body. A child is addressed by (agent, source, name) — the same tuple
// the node_key hashes — so the UI sends back the source/name it read
// from the catalog view plus the desired state.
type childToggleRequest struct {
	Name    string `json:"name"`
	Source  string `json:"source"`
	Enabled bool   `json:"enabled"`
}

// childToggleResponse echoes the applied state.
type childToggleResponse struct {
	Name    string `json:"name"`
	Source  string `json:"source"`
	Enabled bool   `json:"enabled"`
}

// toggleTool handles PATCH /_plugin/agents/:name/tools.
func (h *catalogHandlers) toggleTool(w http.ResponseWriter, r *http.Request) {
	h.toggleChild(w, r, "tools", "tool", graphclient.RelHasTool, graphclient.LabelTool)
}

// toggleSkill handles PATCH /_plugin/agents/:name/skills.
func (h *catalogHandlers) toggleSkill(w http.ResponseWriter, r *http.Request) {
	h.toggleChild(w, r, "skills", "skill", graphclient.RelHasSkill, graphclient.LabelSkill)
}

// toggleChild flips one tool/skill's `enabled` flag — cookie-or-bearer
// (an operator action in the dashboard, unlike the bearer-only catalog
// push). This is the one piece of catalog state the gateway owns rather
// than a source: Hive seeds the palette (enabled by default) and
// re-seeds preserve the toggle, so an operator's on/off choice survives
// deploys. `view` is the URL segment ("tools"/"skills"), `kind` the
// node_key discriminator, and rel/label locate the child in the graph.
func (h *catalogHandlers) toggleChild(w http.ResponseWriter, r *http.Request, view, kind, rel, label string) {
	if r.Method != http.MethodPatch {
		methodNotAllowed(w, http.MethodPatch)
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
	if len(parts) != 2 || parts[0] == "" || parts[1] != view {
		http.NotFound(w, r)
		return
	}
	agent := parts[0]

	r.Body = http.MaxBytesReader(w, r.Body, 1<<16)
	var req childToggleRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}
	if strings.TrimSpace(req.Name) == "" || strings.TrimSpace(req.Source) == "" {
		writeError(w, http.StatusBadRequest, "missing_field", "name and source are required")
		return
	}

	key := nodeKey(agent, req.Source, kind, req.Name)
	now := time.Now().UTC().Format(time.RFC3339)
	results, err := h.graph.Run(r.Context(), graphclient.Statement{
		Statement: fmt.Sprintf(
			"MATCH (a:%[1]s {name: $name})-[:%[2]s]->(c:%[3]s {node_key: $key}) "+
				"SET c.enabled = $enabled, c.updated_at = $now "+
				"RETURN c.node_key",
			graphclient.LabelAgent, rel, label),
		Parameters: map[string]any{
			"name":    agent,
			"key":     key,
			"enabled": req.Enabled,
			"now":     now,
		},
	})
	if err != nil {
		pluginlog.Errf("adminapi: %s toggle agent=%s name=%s: %v", kind, agent, req.Name, err)
		writeError(w, http.StatusBadGateway, "catalog_write_failed", "neo4j write failed")
		return
	}
	if len(results) == 0 || len(results[0].Data) == 0 {
		writeError(w, http.StatusNotFound, "not_found", "no such "+kind+" for agent")
		return
	}

	writeJSON(w, http.StatusOK, childToggleResponse{
		Name:    req.Name,
		Source:  req.Source,
		Enabled: req.Enabled,
	})
}

// list handles GET /_plugin/agents/catalog — cookie-or-bearer. Returns
// every agent in the registry with child counts, so the /agents list
// can union the catalog with spend-derived agents (seeded agents that
// have never produced traffic still show up). Empty registry returns
// an empty array, not 404 — "no agents seeded yet" is a normal state.
func (h *catalogHandlers) list(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	if h.graph == nil {
		writeError(w, http.StatusServiceUnavailable, "catalog_unavailable",
			"agent catalog not configured (neo4j unset on this swarm)")
		return
	}

	out, err := h.fetchCatalogList(r.Context())
	if err != nil {
		pluginlog.Errf("adminapi: catalog list: %v", err)
		writeError(w, http.StatusBadGateway, "catalog_read_failed", "neo4j read failed")
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

		// Prompts are links to existing `:Prompt` nodes, not owned
		// children — we carry only the name + the wiring source.
		prompts := make([]map[string]any, 0, len(ag.Prompts))
		for _, p := range ag.Prompts {
			src := srcOr(p.Source)
			sourceSet[src] = struct{}{}
			prompts = append(prompts, map[string]any{
				"name":   p.Name,
				"source": src,
				"role":   p.Role,
			})
		}
		tools := make([]map[string]any, 0, len(ag.Tools))
		toolKeys := make([]string, 0, len(ag.Tools))
		for _, t := range ag.Tools {
			src := srcOr(t.Source)
			sourceSet[src] = struct{}{}
			key := nodeKey(ag.Name, src, "tool", t.Name)
			toolKeys = append(toolKeys, key)
			tools = append(tools, map[string]any{
				"node_key":    key,
				"name":        t.Name,
				"description": t.Description,
				"schema":      rawToString(t.Schema),
				"source":      src,
				"version":     ag.Version,
			})
		}
		skills := make([]map[string]any, 0, len(ag.Skills))
		skillKeys := make([]string, 0, len(ag.Skills))
		for _, s := range ag.Skills {
			src := srcOr(s.Source)
			sourceSet[src] = struct{}{}
			key := nodeKey(ag.Name, src, "skill", s.Name)
			skillKeys = append(skillKeys, key)
			skills = append(skills, map[string]any{
				"node_key":    key,
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
		// Prompt count is the number of links *requested*; a name with
		// no matching `:Prompt` node is skipped, so the live link count
		// may be lower. Reads report the true linked set.
		counts.Prompts += len(prompts)
		counts.Tools += len(tools)
		counts.Skills += len(skills)

		stmts = append(stmts,
			// 1. Upsert the agent + source + DEFINED_BY. No child cleanup
			//    here anymore: prompts are replaced in statement 2, and
			//    tools/skills self-prune in their own merge-preserve pairs
			//    (statements 4-5 / 6-7) so their per-swarm `enabled`
			//    toggle survives re-seeds.
			graphclient.Statement{
				Statement: fmt.Sprintf(
					"MERGE (a:%[1]s {name: $name}) "+
						"SET a.updated_at = $now, "+
						"a.display_name = CASE WHEN $display_name <> '' THEN $display_name ELSE a.display_name END, "+
						"a.description = CASE WHEN $description <> '' THEN $description ELSE a.description END, "+
						"a.default_model = CASE WHEN $default_model <> '' THEN $default_model ELSE a.default_model END "+
						"MERGE (s:%[2]s {name: $source}) SET s.updated_at = $now "+
						"MERGE (a)-[:%[3]s]->(s)",
					graphclient.LabelAgent, graphclient.LabelSource, graphclient.RelDefinedBy),
				Parameters: map[string]any{
					"name":          ag.Name,
					"display_name":  ag.DisplayName,
					"description":   ag.Description,
					"default_model": ag.DefaultModel,
					"source":        req.Source,
					"now":           now,
				},
			},
			// 2. Replace this push's sources' prompt links: drop the old
			//    HAS_PROMPT relationships (never the Prompt nodes), then
			//    re-link. A prompt name with no matching `:Prompt` node is
			//    skipped silently (the inner MATCH yields no row).
			graphclient.Statement{
				Statement: fmt.Sprintf(
					"MATCH (a:%[1]s {name: $name})-[r:%[2]s]->(:%[3]s) "+
						"WHERE r.source IN $sources DELETE r",
					graphclient.LabelAgent, graphclient.RelHasPrompt, graphclient.LabelPrompt),
				Parameters: map[string]any{
					"name":    ag.Name,
					"sources": sources,
				},
			},
			graphclient.Statement{
				// MERGE keys on `source` too, so a prompt linked by two
				// sources gets one relationship each (mirroring how
				// tool/skill child nodes are per-source) rather than one
				// source clobbering the other's link.
				Statement: fmt.Sprintf(
					"MATCH (a:%[1]s {name: $name}) "+
						"UNWIND $rows AS row "+
						"MATCH (p:%[3]s {name: row.name}) "+
						"MERGE (a)-[r:%[2]s {source: row.source}]->(p) "+
						"SET r.updated_at = $now, r.role = row.role",
					graphclient.LabelAgent, graphclient.RelHasPrompt, graphclient.LabelPrompt),
				Parameters: map[string]any{
					"name": ag.Name,
					"rows": prompts,
					"now":  now,
				},
			},
		)
		// 4-5. Tools and 6-7. skills: both are Hive-owned child nodes
		//      whose per-swarm `enabled` toggle must survive re-seeds, so
		//      each gets a prune-removed + merge-preserve pair rather than
		//      a destructive delete+recreate. Tools additionally carry a
		//      `schema` column.
		stmts = append(stmts, childMergePreserve(childWrite{
			rel: graphclient.RelHasTool, label: graphclient.LabelTool,
			agentName: ag.Name, now: now, rows: tools, keys: toolKeys, sources: sources,
			setProps: "c.name = row.name, c.description = row.description, " +
				"c.schema = row.schema, c.source = row.source, c.version = row.version",
		})...)
		stmts = append(stmts, childMergePreserve(childWrite{
			rel: graphclient.RelHasSkill, label: graphclient.LabelSkill,
			agentName: ag.Name, now: now, rows: skills, keys: skillKeys, sources: sources,
			setProps: "c.name = row.name, c.description = row.description, " +
				"c.source = row.source, c.version = row.version",
		})...)
	}
	return stmts, counts
}

// childWrite is the input to childMergePreserve — everything needed to
// emit one owned child kind's prune + merge pair.
type childWrite struct {
	rel       string           // HAS_TOOL / HAS_SKILL
	label     string           // HiveTool / HiveSkill
	agentName string           // owning agent
	now       string           // RFC3339 stamp
	rows      []map[string]any // one map per child (must carry node_key + the setProps columns)
	keys      []string         // node_keys in this push (the survivor set)
	sources   []string         // sources this push touches
	setProps  string           // comma-joined `c.<col> = row.<col>` (NEVER `enabled`)
}

// childMergePreserve builds the two-statement pair for a Hive-owned
// child kind (tool/skill) whose per-swarm `enabled` toggle must survive
// re-seeds:
//
//  1. Prune — delete this push's sources' children whose node_key is no
//     longer in the manifest. Survivors (and other sources') keep their
//     toggle. Over an empty push this clears the source's kind entirely
//     (`NOT c.node_key IN []` is true for every row).
//  2. Merge — MERGE by node_key (not delete+create) so an existing
//     child keeps its operator-set `enabled`; only a brand-new node
//     defaults to enabled=true. `setProps` refresh every push; UNWIND
//     of [] is a harmless no-op.
func childMergePreserve(c childWrite) []graphclient.Statement {
	return []graphclient.Statement{
		{
			Statement: fmt.Sprintf(
				"MATCH (a:%[1]s {name: $name})-[:%[2]s]->(c:%[3]s) "+
					"WHERE c.source IN $sources AND NOT c.node_key IN $keys "+
					"DETACH DELETE c",
				graphclient.LabelAgent, c.rel, c.label),
			Parameters: map[string]any{
				"name":    c.agentName,
				"sources": c.sources,
				"keys":    c.keys,
			},
		},
		{
			Statement: fmt.Sprintf(
				"MATCH (a:%[1]s {name: $name}) "+
					"UNWIND $rows AS row "+
					"MERGE (c:%[3]s {node_key: row.node_key}) "+
					"ON CREATE SET c.enabled = true "+
					"SET %[4]s, c.updated_at = $now "+
					"MERGE (a)-[:%[2]s]->(c)",
				graphclient.LabelAgent, c.rel, c.label, c.setProps),
			Parameters: map[string]any{
				"name": c.agentName,
				"rows": c.rows,
				"now":  c.now,
			},
		},
	}
}

// ─── read builder ────────────────────────────────────────────────────

// fetchCatalogList returns every agent with child counts in one query.
// Counts are done with COUNT{} subqueries per kind so the row count
// stays one-per-agent (no cartesian fan-out), and source names are
// collected the same way.
func (h *catalogHandlers) fetchCatalogList(ctx context.Context) (*CatalogListResponse, error) {
	cypher := fmt.Sprintf(
		"MATCH (a:%[1]s) "+
			"RETURN a.name, a.display_name, a.description, a.default_model, "+
			"COUNT { (a)-[:%[2]s]->(:%[3]s) } AS prompts, "+
			"COUNT { (a)-[:%[4]s]->(:%[5]s) } AS tools, "+
			"COUNT { (a)-[:%[6]s]->(:%[7]s) } AS skills, "+
			"[ (a)-[:%[8]s]->(s:%[9]s) | s.name ] AS sources, "+
			"a.updated_at "+
			"ORDER BY coalesce(a.display_name, a.name)",
		graphclient.LabelAgent,
		graphclient.RelHasPrompt, graphclient.LabelPrompt,
		graphclient.RelHasTool, graphclient.LabelTool,
		graphclient.RelHasSkill, graphclient.LabelSkill,
		graphclient.RelDefinedBy, graphclient.LabelSource,
	)
	res, err := h.graph.Query(ctx, cypher, nil)
	if err != nil {
		return nil, err
	}
	out := &CatalogListResponse{Agents: []CatalogAgentSummary{}}
	for _, row := range res.Data {
		out.Agents = append(out.Agents, CatalogAgentSummary{
			Name:         rowString(at(row.Row, 0)),
			DisplayName:  rowString(at(row.Row, 1)),
			Description:  rowString(at(row.Row, 2)),
			DefaultModel: rowString(at(row.Row, 3)),
			Prompts:      rowInt(at(row.Row, 4)),
			Tools:        rowInt(at(row.Row, 5)),
			Skills:       rowInt(at(row.Row, 6)),
			Sources:      rowStrings(at(row.Row, 7)),
			UpdatedAt:    rowString(at(row.Row, 8)),
		})
	}
	return out, nil
}

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
			Statement: fmt.Sprintf("MATCH (a:%s {name: $name})-[r:%s]->(p:%s) "+
				"RETURN p.name, p.body, r.source, r.updated_at, r.role ORDER BY r.source, p.name",
				graphclient.LabelAgent, graphclient.RelHasPrompt, graphclient.LabelPrompt),
			Parameters: p,
		},
		graphclient.Statement{
			Statement: fmt.Sprintf("MATCH (a:%s {name: $name})-[:%s]->(t:%s) "+
				"RETURN t.name, t.description, t.schema, t.source, t.version, coalesce(t.enabled, true), t.updated_at "+
				"ORDER BY t.source, t.name",
				graphclient.LabelAgent, graphclient.RelHasTool, graphclient.LabelTool),
			Parameters: p,
		},
		graphclient.Statement{
			Statement: fmt.Sprintf("MATCH (a:%s {name: $name})-[:%s]->(k:%s) "+
				"RETURN k.name, k.description, k.source, k.version, coalesce(k.enabled, true), k.updated_at "+
				"ORDER BY k.source, k.name",
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
			Body:      rowString(at(row.Row, 1)),
			Source:    rowString(at(row.Row, 2)),
			UpdatedAt: rowString(at(row.Row, 3)),
			Role:      rowString(at(row.Row, 4)),
		})
	}
	for _, row := range results[3].Data {
		out.Tools = append(out.Tools, CatalogTool{
			Name:        rowString(at(row.Row, 0)),
			Description: rowString(at(row.Row, 1)),
			Schema:      stringToRaw(rowString(at(row.Row, 2))),
			Source:      rowString(at(row.Row, 3)),
			Version:     rowString(at(row.Row, 4)),
			Enabled:     rowBool(at(row.Row, 5), true),
			UpdatedAt:   rowString(at(row.Row, 6)),
		})
	}
	for _, row := range results[4].Data {
		out.Skills = append(out.Skills, CatalogSkill{
			Name:        rowString(at(row.Row, 0)),
			Description: rowString(at(row.Row, 1)),
			Source:      rowString(at(row.Row, 2)),
			Version:     rowString(at(row.Row, 3)),
			Enabled:     rowBool(at(row.Row, 4), true),
			UpdatedAt:   rowString(at(row.Row, 5)),
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

// rowInt decodes a single returned cell into an int. neo4j integers
// arrive as JSON numbers; null / garbage ⇒ 0.
func rowInt(raw json.RawMessage) int {
	if len(raw) == 0 {
		return 0
	}
	var n int
	if err := json.Unmarshal(raw, &n); err != nil {
		return 0
	}
	return n
}

// rowBool decodes a single returned cell into a bool. A neo4j null
// (unset property — e.g. a legacy skill node written before `enabled`
// existed) or a decode failure falls back to `def`.
func rowBool(raw json.RawMessage, def bool) bool {
	if len(raw) == 0 {
		return def
	}
	var b bool
	if err := json.Unmarshal(raw, &b); err != nil {
		return def
	}
	return b
}

// rowStrings decodes a returned cell that is a JSON array of strings
// (e.g. a Cypher list comprehension of source names). null / garbage
// ⇒ empty slice.
func rowStrings(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return []string{}
	}
	var ss []string
	if err := json.Unmarshal(raw, &ss); err != nil {
		return []string{}
	}
	return ss
}

// at returns the i-th cell of a row, or empty when the RETURN clause
// produced fewer columns than expected (defensive; shouldn't happen).
func at(row []json.RawMessage, i int) json.RawMessage {
	if i < 0 || i >= len(row) {
		return nil
	}
	return row[i]
}
