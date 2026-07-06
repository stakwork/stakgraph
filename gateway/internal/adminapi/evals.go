package adminapi

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/stakwork/stakgraph/gateway/internal/graphclient"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// evals.go serves the agent-detail "Evals" tab.
//
// Split of responsibility (see hivecallback.go for the why):
//   - READS come straight from neo4j. The Eval* subgraph
//     (EvalSet-[:HAS_REQUIREMENT]->EvalRequirement-[:HAS_TRIGGER]->
//     EvalTrigger-[:HAS_OUTPUT]->EvalTriggerOutput) lives in the same
//     graph as the agent catalog, so the dashboard renders without a
//     Hive round-trip. Eval sets surface under an agent via the
//     gateway-owned HiveAgent-[:HAS_EVAL_SET]->EvalSet edge.
//   - The HAS_EVAL_SET edge is written directly (edge-only, both
//     endpoints already exist — no Jarvis node schema involved).
//   - NODE mutations (create/update/delete a set or requirement) and
//     RUNs are delegated to Hive, which owns the Jarvis write path and
//     the Stakwork run orchestration.

// ─── read wire shapes ────────────────────────────────────────────────

// EvalSetSummary is one row of an agent's eval-set list.
type EvalSetSummary struct {
	RefID        string `json:"ref_id"`
	Name         string `json:"name,omitempty"`
	Description  string `json:"description,omitempty"`
	Requirements int    `json:"requirements"`
}

// AgentEvalsResponse is the agent-detail Evals tab payload — the sets
// linked to this agent via HAS_EVAL_SET.
type AgentEvalsResponse struct {
	Agent string           `json:"agent"`
	Sets  []EvalSetSummary `json:"sets"`
}

// EvalTriggerSummary is one captured trigger under a requirement, plus
// the outcome of its most-recent run (nil-ish fields when never run).
type EvalTriggerSummary struct {
	RefID       string  `json:"ref_id"`
	Agent       string  `json:"agent,omitempty"`
	Source      string  `json:"source,omitempty"`
	Environment string  `json:"environment,omitempty"`
	ChangeType  string  `json:"change_type,omitempty"`
	LastResult  string  `json:"last_result,omitempty"` // "pass" / "fail" / ""
	LastScore   float64 `json:"last_score,omitempty"`
	LastNotes   string  `json:"last_notes,omitempty"`
	LastAttempt int     `json:"last_attempt,omitempty"`
}

// EvalRequirementDetail is a requirement with its triggers.
type EvalRequirementDetail struct {
	RefID         string               `json:"ref_id"`
	Name          string               `json:"name,omitempty"`
	Description   string               `json:"description,omitempty"`
	PromptSnippet string               `json:"prompt_snippet,omitempty"`
	Order         int                  `json:"order"`
	Triggers      []EvalTriggerSummary `json:"triggers"`
}

// EvalSetDetailResponse is the expanded view of one set.
type EvalSetDetailResponse struct {
	RefID        string                  `json:"ref_id"`
	Name         string                  `json:"name,omitempty"`
	Description  string                  `json:"description,omitempty"`
	Requirements []EvalRequirementDetail `json:"requirements"`
}

// ─── write wire shapes ───────────────────────────────────────────────

// evalSetWriteRequest is the create/link/update body. On the agent
// route: a non-empty SetID links an existing set; otherwise Name
// creates a new one (and links it). On the set route: Name/Description
// update.
type evalSetWriteRequest struct {
	SetID       string `json:"set_id,omitempty"`
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
}

type evalRequirementWriteRequest struct {
	Name             string   `json:"name,omitempty"`
	Description      string   `json:"description,omitempty"`
	PromptSnippet    string   `json:"prompt_snippet,omitempty"`
	DesirableCases   []string `json:"desirable_cases,omitempty"`
	UndesirableCases []string `json:"undesirable_cases,omitempty"`
}

type evalRunRequest struct {
	Agent string `json:"agent,omitempty"`
}

// evalRefResponse is the create/link acknowledgement.
type evalRefResponse struct {
	RefID  string `json:"ref_id"`
	Linked bool   `json:"linked,omitempty"`
}

type evalRunResponse struct {
	// A requirement fans out to N triggers, so Hive dispatches one
	// Stakwork project per trigger and returns them all.
	ProjectIDs []any `json:"project_ids,omitempty"`
}

// ─── handler ─────────────────────────────────────────────────────────

// evalHandlers owns the graph client (reads + the HAS_EVAL_SET edge)
// and the Hive callback (node mutations + runs). graph is nil when
// neo4j is unconfigured → reads 503; hive backs the delegated writes.
type evalHandlers struct {
	graph *graphclient.Client
	hive  *hiveCallbackHandlers
}

func newEvalHandlers(graph *graphclient.Client, hive *hiveCallbackHandlers) *evalHandlers {
	return &evalHandlers{graph: graph, hive: hive}
}

func (h *evalHandlers) graphReady(w http.ResponseWriter) bool {
	if h.graph == nil {
		writeError(w, http.StatusServiceUnavailable, "catalog_unavailable",
			"eval graph not configured (neo4j unset on this swarm)")
		return false
	}
	return true
}

// ─── agent-scoped: /_plugin/agents/:name/evals[/:setId] ──────────────

// agentEvals dispatches the agent-scoped eval routes:
//
//	GET    /_plugin/agents/:name/evals          → list sets for agent
//	POST   /_plugin/agents/:name/evals          → create+link OR link a set
//	DELETE /_plugin/agents/:name/evals/:setId   → unlink a set (keep it)
func (h *evalHandlers) agentEvals(w http.ResponseWriter, r *http.Request, agent string, rest []string) {
	switch {
	case len(rest) == 0 && r.Method == http.MethodGet:
		h.listForAgent(w, r, agent)
	case len(rest) == 0 && r.Method == http.MethodPost:
		h.createOrLinkForAgent(w, r, agent)
	case len(rest) == 1 && rest[0] != "" && r.Method == http.MethodDelete:
		h.unlinkFromAgent(w, r, agent, rest[0])
	default:
		http.NotFound(w, r)
	}
}

func (h *evalHandlers) listForAgent(w http.ResponseWriter, r *http.Request, agent string) {
	if !h.graphReady(w) {
		return
	}
	res, err := h.graph.Query(r.Context(), fmt.Sprintf(
		"MATCH (a:%[1]s {name: $name})-[:%[2]s]->(es:%[3]s) "+
			"RETURN es.ref_id, es.name, es.description, "+
			"COUNT { (es)-[:%[4]s]->(:%[5]s) } AS reqs "+
			"ORDER BY coalesce(es.name, es.ref_id)",
		graphclient.LabelAgent, graphclient.RelHasEvalSet, graphclient.LabelEvalSet,
		graphclient.RelHasRequirement, graphclient.LabelEvalRequirement,
	), map[string]any{"name": agent})
	if err != nil {
		pluginlog.Errf("adminapi: evals listForAgent agent=%s: %v", agent, err)
		writeError(w, http.StatusBadGateway, "catalog_read_failed", "neo4j read failed")
		return
	}
	out := AgentEvalsResponse{Agent: agent, Sets: []EvalSetSummary{}}
	for _, row := range res.Data {
		out.Sets = append(out.Sets, EvalSetSummary{
			RefID:        rowString(at(row.Row, 0)),
			Name:         rowString(at(row.Row, 1)),
			Description:  rowString(at(row.Row, 2)),
			Requirements: rowInt(at(row.Row, 3)),
		})
	}
	writeJSON(w, http.StatusOK, out)
}

func (h *evalHandlers) createOrLinkForAgent(w http.ResponseWriter, r *http.Request, agent string) {
	if !h.graphReady(w) {
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 1<<16)
	var req evalSetWriteRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}
	ctx := r.Context()

	// Link an existing set: edge-only, no Hive round-trip.
	if strings.TrimSpace(req.SetID) != "" {
		if err := h.linkEdge(ctx, agent, req.SetID); err != nil {
			pluginlog.Errf("adminapi: evals link agent=%s set=%s: %v", agent, req.SetID, err)
			writeError(w, http.StatusBadGateway, "catalog_write_failed", "neo4j write failed")
			return
		}
		writeJSON(w, http.StatusOK, evalRefResponse{RefID: req.SetID, Linked: true})
		return
	}

	// Create a new set via Hive, then link it.
	if strings.TrimSpace(req.Name) == "" {
		writeError(w, http.StatusBadRequest, "missing_field", "name (or set_id) is required")
		return
	}
	var created evalRefResponse
	if err := h.hive.call(ctx, http.MethodPost, "/api/gateway/evals",
		map[string]any{"name": req.Name, "description": req.Description}, &created); err != nil {
		relayHiveError(w, err)
		return
	}
	if created.RefID == "" {
		writeError(w, http.StatusBadGateway, "hive_error", "Hive did not return a ref_id")
		return
	}
	if err := h.linkEdge(ctx, agent, created.RefID); err != nil {
		// The set exists in Jarvis; only the agent link failed. Surface
		// it so the operator can retry the link rather than silently
		// orphaning the set.
		pluginlog.Errf("adminapi: evals link-after-create agent=%s set=%s: %v", agent, created.RefID, err)
		writeError(w, http.StatusBadGateway, "catalog_write_failed",
			"set created but linking to agent failed")
		return
	}
	writeJSON(w, http.StatusOK, evalRefResponse{RefID: created.RefID})
}

// linkEdge MERGEs HiveAgent-[:HAS_EVAL_SET]->EvalSet. No-op (0 rows,
// no error) if either endpoint is missing.
func (h *evalHandlers) linkEdge(ctx context.Context, agent, setID string) error {
	_, err := h.graph.Query(ctx, fmt.Sprintf(
		"MATCH (a:%[1]s {name: $name}) MATCH (es:%[2]s {ref_id: $ref}) "+
			"MERGE (a)-[:%[3]s]->(es)",
		graphclient.LabelAgent, graphclient.LabelEvalSet, graphclient.RelHasEvalSet,
	), map[string]any{"name": agent, "ref": setID})
	return err
}

func (h *evalHandlers) unlinkFromAgent(w http.ResponseWriter, r *http.Request, agent, setID string) {
	if !h.graphReady(w) {
		return
	}
	_, err := h.graph.Query(r.Context(), fmt.Sprintf(
		"MATCH (a:%[1]s {name: $name})-[e:%[2]s]->(:%[3]s {ref_id: $ref}) DELETE e",
		graphclient.LabelAgent, graphclient.RelHasEvalSet, graphclient.LabelEvalSet,
	), map[string]any{"name": agent, "ref": setID})
	if err != nil {
		pluginlog.Errf("adminapi: evals unlink agent=%s set=%s: %v", agent, setID, err)
		writeError(w, http.StatusBadGateway, "catalog_write_failed", "neo4j write failed")
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// ─── set-scoped: /_plugin/evals[/:setId[/requirements[/:reqId[/run]]]] ─

// dispatch is the /_plugin/evals/ subtree router.
func (h *evalHandlers) dispatch(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, "/_plugin/evals")
	rest = strings.TrimPrefix(rest, "/")
	parts := splitNonEmpty(rest)

	switch {
	// POST /_plugin/evals — create a set (not yet linked to an agent).
	case len(parts) == 0 && r.Method == http.MethodPost:
		h.createSet(w, r)

	// /_plugin/evals/:setId — GET detail, PATCH update, DELETE.
	case len(parts) == 1:
		switch r.Method {
		case http.MethodGet:
			h.setDetail(w, r, parts[0])
		case http.MethodPatch:
			h.updateSet(w, r, parts[0])
		case http.MethodDelete:
			h.deleteSet(w, r, parts[0])
		default:
			methodNotAllowed(w, http.MethodGet, http.MethodPatch, http.MethodDelete)
		}

	// /_plugin/evals/:setId/requirements — POST create.
	case len(parts) == 2 && parts[1] == "requirements" && r.Method == http.MethodPost:
		h.createRequirement(w, r, parts[0])

	// /_plugin/evals/:setId/requirements/:reqId — PATCH / DELETE.
	case len(parts) == 3 && parts[1] == "requirements":
		switch r.Method {
		case http.MethodPatch:
			h.updateRequirement(w, r, parts[0], parts[2])
		case http.MethodDelete:
			h.deleteRequirement(w, r, parts[0], parts[2])
		default:
			methodNotAllowed(w, http.MethodPatch, http.MethodDelete)
		}

	// /_plugin/evals/:setId/requirements/:reqId/run — POST.
	case len(parts) == 4 && parts[1] == "requirements" && parts[3] == "run" && r.Method == http.MethodPost:
		h.runRequirement(w, r, parts[0], parts[2])

	default:
		http.NotFound(w, r)
	}
}

func (h *evalHandlers) setDetail(w http.ResponseWriter, r *http.Request, setID string) {
	if !h.graphReady(w) {
		return
	}
	p := map[string]any{"ref": setID}
	results, err := h.graph.Run(r.Context(),
		graphclient.Statement{
			Statement: fmt.Sprintf(
				"MATCH (es:%s {ref_id: $ref}) RETURN es.ref_id, es.name, es.description",
				graphclient.LabelEvalSet),
			Parameters: p,
		},
		graphclient.Statement{
			Statement: fmt.Sprintf(
				"MATCH (es:%[1]s {ref_id: $ref})-[hr:%[2]s]->(r:%[3]s) "+
					"RETURN r.ref_id, r.name, r.description, r.prompt_snippet, "+
					"toInteger(coalesce(hr.order, 0)) AS ord "+
					"ORDER BY ord, r.name",
				graphclient.LabelEvalSet, graphclient.RelHasRequirement, graphclient.LabelEvalRequirement),
			Parameters: p,
		},
		graphclient.Statement{
			// Latest output per trigger: order outputs by attempt desc
			// and keep the head. requirements with no trigger simply
			// contribute no rows here.
			Statement: fmt.Sprintf(
				"MATCH (es:%[1]s {ref_id: $ref})-[:%[2]s]->(r:%[3]s)-[:%[4]s]->(t:%[5]s) "+
					"OPTIONAL MATCH (t)-[:%[6]s]->(o:%[7]s) "+
					"WITH r, t, o ORDER BY toInteger(coalesce(o.attempt_number, 0)) DESC "+
					"WITH r, t, head(collect(o)) AS last "+
					"RETURN r.ref_id, t.ref_id, t.agent, t.source, t.environment, t.change_type, "+
					"last.result, last.score, last.judge_notes, "+
					"toInteger(coalesce(last.attempt_number, 0)) "+
					"ORDER BY r.ref_id, t.ref_id",
				graphclient.LabelEvalSet, graphclient.RelHasRequirement, graphclient.LabelEvalRequirement,
				graphclient.RelHasTrigger, graphclient.LabelEvalTrigger,
				graphclient.RelHasOutput, graphclient.LabelEvalOutput),
			Parameters: p,
		},
	)
	if err != nil {
		pluginlog.Errf("adminapi: evals setDetail set=%s: %v", setID, err)
		writeError(w, http.StatusBadGateway, "catalog_read_failed", "neo4j read failed")
		return
	}
	if len(results) < 3 || len(results[0].Data) == 0 {
		writeError(w, http.StatusNotFound, "not_found", "no such eval set")
		return
	}

	setRow := results[0].Data[0].Row
	out := &EvalSetDetailResponse{
		RefID:        rowString(at(setRow, 0)),
		Name:         rowString(at(setRow, 1)),
		Description:  rowString(at(setRow, 2)),
		Requirements: []EvalRequirementDetail{},
	}

	// Requirements, preserving order; index by ref for trigger attach.
	byRef := map[string]*EvalRequirementDetail{}
	for _, row := range results[1].Data {
		req := EvalRequirementDetail{
			RefID:         rowString(at(row.Row, 0)),
			Name:          rowString(at(row.Row, 1)),
			Description:   rowString(at(row.Row, 2)),
			PromptSnippet: rowString(at(row.Row, 3)),
			Order:         rowInt(at(row.Row, 4)),
			Triggers:      []EvalTriggerSummary{},
		}
		out.Requirements = append(out.Requirements, req)
		byRef[req.RefID] = &out.Requirements[len(out.Requirements)-1]
	}
	for _, row := range results[2].Data {
		reqRef := rowString(at(row.Row, 0))
		req, ok := byRef[reqRef]
		if !ok {
			continue
		}
		req.Triggers = append(req.Triggers, EvalTriggerSummary{
			RefID:       rowString(at(row.Row, 1)),
			Agent:       rowString(at(row.Row, 2)),
			Source:      rowString(at(row.Row, 3)),
			Environment: rowString(at(row.Row, 4)),
			ChangeType:  rowString(at(row.Row, 5)),
			LastResult:  rowString(at(row.Row, 6)),
			LastScore:   rowFloat(at(row.Row, 7)),
			LastNotes:   rowString(at(row.Row, 8)),
			LastAttempt: rowInt(at(row.Row, 9)),
		})
	}
	writeJSON(w, http.StatusOK, out)
}

// createSet creates a standalone set (no agent link yet). Delegated to
// Hive. Agent-scoped creation goes through createOrLinkForAgent.
func (h *evalHandlers) createSet(w http.ResponseWriter, r *http.Request) {
	req, ok := decodeSetWrite(w, r)
	if !ok {
		return
	}
	if strings.TrimSpace(req.Name) == "" {
		writeError(w, http.StatusBadRequest, "missing_field", "name is required")
		return
	}
	var created evalRefResponse
	if err := h.hive.call(r.Context(), http.MethodPost, "/api/gateway/evals",
		map[string]any{"name": req.Name, "description": req.Description}, &created); err != nil {
		relayHiveError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, created)
}

// updateSet is reached by an inbound PATCH from the SPA (apiFetch only
// speaks GET/POST/PATCH/DELETE) but forwards to Hive as PUT, which is
// the verb the /api/gateway/evals contract uses.
func (h *evalHandlers) updateSet(w http.ResponseWriter, r *http.Request, setID string) {
	req, ok := decodeSetWrite(w, r)
	if !ok {
		return
	}
	if err := h.hive.call(r.Context(), http.MethodPut,
		"/api/gateway/evals/"+urlSeg(setID),
		map[string]any{"name": req.Name, "description": req.Description}, nil); err != nil {
		relayHiveError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *evalHandlers) deleteSet(w http.ResponseWriter, r *http.Request, setID string) {
	// Drop the gateway-owned agent links first (best-effort — Jarvis's
	// node delete is DETACH, but clearing here keeps the graph tidy even
	// if the Hive call fails midway).
	if h.graph != nil {
		if _, err := h.graph.Query(r.Context(), fmt.Sprintf(
			"MATCH (:%[1]s)-[e:%[2]s]->(:%[3]s {ref_id: $ref}) DELETE e",
			graphclient.LabelAgent, graphclient.RelHasEvalSet, graphclient.LabelEvalSet,
		), map[string]any{"ref": setID}); err != nil {
			pluginlog.Warnf("adminapi: evals deleteSet unlink set=%s: %v", setID, err)
		}
	}
	if err := h.hive.call(r.Context(), http.MethodDelete,
		"/api/gateway/evals/"+urlSeg(setID), nil, nil); err != nil {
		relayHiveError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *evalHandlers) createRequirement(w http.ResponseWriter, r *http.Request, setID string) {
	req, ok := decodeReqWrite(w, r)
	if !ok {
		return
	}
	if strings.TrimSpace(req.Name) == "" {
		writeError(w, http.StatusBadRequest, "missing_field", "name is required")
		return
	}
	var created evalRefResponse
	if err := h.hive.call(r.Context(), http.MethodPost,
		"/api/gateway/evals/"+urlSeg(setID)+"/requirements", req, &created); err != nil {
		relayHiveError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, created)
}

func (h *evalHandlers) updateRequirement(w http.ResponseWriter, r *http.Request, setID, reqID string) {
	req, ok := decodeReqWrite(w, r)
	if !ok {
		return
	}
	if err := h.hive.call(r.Context(), http.MethodPut,
		"/api/gateway/evals/"+urlSeg(setID)+"/requirements/"+urlSeg(reqID), req, nil); err != nil {
		relayHiveError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *evalHandlers) deleteRequirement(w http.ResponseWriter, r *http.Request, setID, reqID string) {
	if err := h.hive.call(r.Context(), http.MethodDelete,
		"/api/gateway/evals/"+urlSeg(setID)+"/requirements/"+urlSeg(reqID), nil, nil); err != nil {
		relayHiveError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// runRequirement dispatches an eval run for a requirement. Hive fires
// the Stakwork workflow (Bifrost creds + Stakwork key are Hive-side)
// and the results are written back into Jarvis as EvalTriggerOutput
// nodes, which the setDetail read then surfaces.
func (h *evalHandlers) runRequirement(w http.ResponseWriter, r *http.Request, setID, reqID string) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<16)
	var req evalRunRequest
	// Body is optional (agent override) — tolerate an empty body.
	if r.ContentLength != 0 {
		if err := decodeJSON(r, &req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_body", err.Error())
			return
		}
	}
	var out evalRunResponse
	if err := h.hive.call(r.Context(), http.MethodPost,
		"/api/gateway/evals/"+urlSeg(setID)+"/requirements/"+urlSeg(reqID)+"/run",
		map[string]any{"agent": req.Agent}, &out); err != nil {
		relayHiveError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, out)
}

// ─── helpers ─────────────────────────────────────────────────────────

func decodeSetWrite(w http.ResponseWriter, r *http.Request) (evalSetWriteRequest, bool) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<16)
	var req evalSetWriteRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return req, false
	}
	return req, true
}

func decodeReqWrite(w http.ResponseWriter, r *http.Request) (evalRequirementWriteRequest, bool) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<16)
	var req evalRequirementWriteRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return req, false
	}
	return req, true
}

// splitNonEmpty splits a cleaned path tail on "/" dropping empties, so
// a trailing slash doesn't produce a phantom segment.
func splitNonEmpty(s string) []string {
	if s == "" {
		return nil
	}
	raw := strings.Split(s, "/")
	out := make([]string, 0, len(raw))
	for _, p := range raw {
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// urlSeg escapes a single path segment for the Hive callback URL.
func urlSeg(s string) string {
	// ref_ids are uuid4 (safe already) but escape defensively.
	return strings.ReplaceAll(strings.ReplaceAll(s, "/", "%2F"), " ", "%20")
}

// rowFloat decodes a returned cell into a float64. neo4j numbers arrive
// as JSON numbers; null / garbage ⇒ 0.
func rowFloat(raw json.RawMessage) float64 {
	if len(raw) == 0 {
		return 0
	}
	var f float64
	if err := json.Unmarshal(raw, &f); err != nil {
		return 0
	}
	return f
}
