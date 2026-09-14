package adminapi

import (
	"net/http"
	"strings"
	"time"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// Revocation routes — the HTTP face of auth/admin.go's helpers.
// Bearer-only: revocation is issuer territory (Hive's
// /macaroons/revoke fans out here), never a dashboard click.
//
//	POST   /_plugin/revoke/nonce/:nonce   body {exp?}    → 200 RevokeNonceResponse
//	DELETE /_plugin/revoke/nonce/:nonce                  → 204
//	PUT    /_plugin/revoke/user/:user_id  body {before?} → 200 RevokeUserResponse
//	GET    /_plugin/revoke/user/:user_id                 → 200 RevokeUserResponse | 404
//	DELETE /_plugin/revoke/user/:user_id                 → 204
//
// `exp` is the RFC3339 expiry of the layer whose nonce is being
// revoked; the tombstone's TTL is derived from it (phase-6: "TTL =
// layer.exp"). Omitted ⇒ the 7d ceiling, which is always safe
// (macaroon layers never outlive it). `before` defaults to now.

const revokePrefixPath = "/_plugin/revoke/"

// RevokeNonceRequest is the body for POST /_plugin/revoke/nonce/:nonce.
type RevokeNonceRequest struct {
	Exp string `json:"exp,omitempty"` // RFC3339; layer expiry
}

// RevokeNonceResponse is the wire shape for POST /_plugin/revoke/nonce/:nonce.
type RevokeNonceResponse struct {
	Nonce     string `json:"nonce"`
	ExpiresAt string `json:"expires_at"` // RFC3339 UTC; when the tombstone lapses
}

// RevokeUserRequest is the body for PUT /_plugin/revoke/user/:user_id.
type RevokeUserRequest struct {
	Before string `json:"before,omitempty"` // RFC3339; defaults to now
}

// RevokeUserResponse is the wire shape for PUT/GET /_plugin/revoke/user/:user_id.
type RevokeUserResponse struct {
	UserID string `json:"user_id"`
	Before string `json:"before"` // RFC3339 UTC
}

type revokeHandlers struct{}

func newRevokeHandlers() *revokeHandlers { return &revokeHandlers{} }

// dispatch routes /_plugin/revoke/{nonce,user}/<id>.
func (h *revokeHandlers) dispatch(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, revokePrefixPath), "/")
	if len(parts) != 2 || parts[1] == "" {
		http.NotFound(w, r)
		return
	}
	switch parts[0] {
	case "nonce":
		h.nonce(w, r, parts[1])
	case "user":
		h.user(w, r, parts[1])
	default:
		http.NotFound(w, r)
	}
}

func (h *revokeHandlers) nonce(w http.ResponseWriter, r *http.Request, nonce string) {
	switch r.Method {
	case http.MethodPost:
		var body RevokeNonceRequest
		if r.ContentLength != 0 {
			if err := decodeJSON(r, &body); err != nil {
				http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
				return
			}
		}
		now := time.Now().UTC()
		// Default to the ceiling: runKeyTTL clamps anything past 7d.
		exp := now.Add(365 * 24 * time.Hour)
		if body.Exp != "" {
			t, err := time.Parse(time.RFC3339, body.Exp)
			if err != nil {
				http.Error(w, "exp must be RFC3339", http.StatusBadRequest)
				return
			}
			exp = t
		}
		ttl := auth.RevocationTTL(exp, now)
		if err := auth.RevokeNonce(r.Context(), nonce, ttl); err != nil {
			writeHotStateErr(w, err, "revoke.nonce")
			return
		}
		pluginlog.Logf("adminapi: nonce revoked ttl=%s", ttl)
		writeJSON(w, http.StatusOK, RevokeNonceResponse{
			Nonce:     nonce,
			ExpiresAt: now.Add(ttl).Format(time.RFC3339),
		})
	case http.MethodDelete:
		if err := auth.UnrevokeNonce(r.Context(), nonce); err != nil {
			writeHotStateErr(w, err, "revoke.unrevoke_nonce")
			return
		}
		w.WriteHeader(http.StatusNoContent)
	default:
		methodNotAllowed(w, http.MethodPost, http.MethodDelete)
	}
}

func (h *revokeHandlers) user(w http.ResponseWriter, r *http.Request, userID string) {
	switch r.Method {
	case http.MethodPut:
		var body RevokeUserRequest
		if r.ContentLength != 0 {
			if err := decodeJSON(r, &body); err != nil {
				http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
				return
			}
		}
		before := time.Now().UTC()
		if body.Before != "" {
			t, err := time.Parse(time.RFC3339, body.Before)
			if err != nil {
				http.Error(w, "before must be RFC3339", http.StatusBadRequest)
				return
			}
			before = t.UTC()
		}
		if err := auth.SetUserRevokeCutoff(r.Context(), userID, before); err != nil {
			writeHotStateErr(w, err, "revoke.user")
			return
		}
		pluginlog.Logf("adminapi: user revoke cutoff set user=%s before=%s", userID, before.Format(time.RFC3339))
		writeJSON(w, http.StatusOK, RevokeUserResponse{
			UserID: userID,
			Before: before.Format(time.RFC3339),
		})
	case http.MethodGet:
		cutoff, ok, err := auth.GetUserRevokeCutoff(r.Context(), userID)
		if err != nil {
			writeHotStateErr(w, err, "revoke.get_user")
			return
		}
		if !ok {
			http.NotFound(w, r)
			return
		}
		writeJSON(w, http.StatusOK, RevokeUserResponse{
			UserID: userID,
			Before: cutoff.UTC().Format(time.RFC3339),
		})
	case http.MethodDelete:
		if err := auth.ClearUserRevokeCutoff(r.Context(), userID); err != nil {
			writeHotStateErr(w, err, "revoke.clear_user")
			return
		}
		w.WriteHeader(http.StatusNoContent)
	default:
		methodNotAllowed(w, http.MethodPut, http.MethodGet, http.MethodDelete)
	}
}
