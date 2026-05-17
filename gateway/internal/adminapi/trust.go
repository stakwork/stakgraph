package adminapi

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/trust"
)

// Route paths — kept as constants so server.go and tests stay in sync.
//
// `/_plugin/trust` handles POST (upsert) — no trailing slash.
// `/_plugin/trust/status` is its own leaf.
// Everything else lives under `/_plugin/trust/` (prefix routing) where
// we dispatch by HTTP method + sub-path:
//
//	GET    /_plugin/trust/<org_id>          → read one
//	DELETE /_plugin/trust/<org_id>          → remove
//	POST   /_plugin/trust/<org_id>/rotate   → rotate key
//
// Net/http's ServeMux gives us prefix routing for "/foo/" so we use
// a single handler that dispatches by inspecting the trailing path
// segment(s). Cheaper than pulling in a router library for four routes.
const (
	trustStatusPath = "/_plugin/trust/status"
	trustRootPath   = "/_plugin/trust"  // exact match → POST upsert
	trustPrefixPath = "/_plugin/trust/" // prefix → /:org_id and /:org_id/rotate
)

// trustHandlers bundles all four trust handlers so server.go can wire
// them in one place. Constructed once per process lifetime.
type trustHandlers struct {
	reg *trust.Registry
}

func newTrustHandlers(r *trust.Registry) *trustHandlers { return &trustHandlers{reg: r} }

// status implements GET /_plugin/trust/status. The body shape comes
// straight from trust.StatusResponse so the wire surface is owned by
// the trust package, not adminapi.
func (h *trustHandlers) status(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	writeJSON(w, http.StatusOK, h.reg.Status())
}

// upsert implements POST /_plugin/trust. Idempotent on the
// user-supplied fields — see trust.Registry.Upsert.
func (h *trustHandlers) upsert(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	var o trust.Org
	if err := decodeJSON(r, &o); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := h.reg.Upsert(o, trust.SeedSourceAPI); err != nil {
		if errors.Is(err, trust.ErrInvalidOrg) {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		pluginlog.Errf("adminapi: trust upsert: %v", err)
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "org_id": o.OrgID})
}

// dispatchPrefix routes everything under /_plugin/trust/ except the
// status leaf. The trailing path is one of:
//
//	<org_id>           → GET or DELETE
//	<org_id>/rotate    → POST
//
// Anything else returns 404 (org_id with extra path segments) so we
// don't silently accept malformed URLs that look like they should
// have done something.
func (h *trustHandlers) dispatchPrefix(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, trustPrefixPath)
	if rest == "" || rest == "status" {
		// "status" is registered as its own leaf — if we got here
		// with rest=="status", the user hit /_plugin/trust/status/
		// (trailing slash) which we treat the same as the leaf.
		if rest == "status" {
			h.status(w, r)
			return
		}
		http.NotFound(w, r)
		return
	}

	parts := strings.Split(rest, "/")
	switch len(parts) {
	case 1:
		// /_plugin/trust/<org_id>
		h.byOrg(w, r, parts[0])
	case 2:
		// /_plugin/trust/<org_id>/<verb>
		if parts[1] != "rotate" {
			http.NotFound(w, r)
			return
		}
		h.rotate(w, r, parts[0])
	default:
		http.NotFound(w, r)
	}
}

// byOrg dispatches GET / DELETE on /_plugin/trust/<org_id>.
func (h *trustHandlers) byOrg(w http.ResponseWriter, r *http.Request, orgID string) {
	switch r.Method {
	case http.MethodGet:
		o, ok := h.reg.Get(orgID)
		if !ok {
			http.NotFound(w, r)
			return
		}
		writeJSON(w, http.StatusOK, o)
	case http.MethodDelete:
		if err := h.reg.Delete(orgID); err != nil {
			if errors.Is(err, trust.ErrNotFound) {
				http.NotFound(w, r)
				return
			}
			pluginlog.Errf("adminapi: trust delete %s: %v", orgID, err)
			http.Error(w, "internal error", http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "removed": orgID})
	default:
		methodNotAllowed(w, http.MethodGet, http.MethodDelete)
	}
}

// rotate handles POST /_plugin/trust/<org_id>/rotate.
func (h *trustHandlers) rotate(w http.ResponseWriter, r *http.Request, orgID string) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	var req trust.RotateRequest
	if err := decodeJSON(r, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	resp, err := h.reg.Rotate(orgID, req.NewPubkey, req.GraceSeconds)
	if err != nil {
		if errors.Is(err, trust.ErrNotFound) {
			http.NotFound(w, r)
			return
		}
		if errors.Is(err, trust.ErrInvalidOrg) {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		// Pubkey validation errors and grace_seconds<0 are bare
		// errors.New from registry.Rotate — surface as 400.
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

// ─── helpers ──────────────────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, status int, body any) {
	raw, err := json.Marshal(body)
	if err != nil {
		pluginlog.Errf("adminapi: marshal response: %v", err)
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(raw)
}

func decodeJSON(r *http.Request, into any) error {
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields() // catches typos like "policy_max" → 400 not silent
	if err := dec.Decode(into); err != nil {
		return err
	}
	return nil
}

func methodNotAllowed(w http.ResponseWriter, allowed ...string) {
	w.Header().Set("Allow", strings.Join(allowed, ", "))
	http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
}
