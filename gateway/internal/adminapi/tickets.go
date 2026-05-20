package adminapi

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
	"github.com/stakwork/stakgraph/gateway/internal/sessions"
)

// ticketTTL is how long a freshly-minted ticket is valid for. Short
// enough that a leak via referrer / browser history / proxy log is
// already-expired by the time anyone could replay it, long enough
// that a slow page load between Hive issuing the iframe and the SPA
// redeeming the ticket comfortably succeeds.
const ticketTTL = 30 * time.Second

// ticketOpTimeout caps the Redis round-trips in mint / redeem.
// Same shape as loginTimeout — bounded so an unresponsive Redis
// surfaces as a 503 rather than a hung browser tab.
const ticketOpTimeout = 5 * time.Second

// TicketResponse is the JSON body returned by POST /_plugin/auth/ticket.
// Hive embeds the ticket in the iframe src as `?ticket=<value>`.
type TicketResponse struct {
	Ticket    string `json:"ticket"`
	ExpiresIn int    `json:"expires_in"` // seconds; mirrors ticketTTL
}

// redeemRequest is the JSON body POSTed to /_plugin/auth/redeem.
// The SPA pulls `ticket` out of its own query string and forwards it
// in a request body (not a URL) so the redemption itself doesn't
// re-leak the value into access logs.
type redeemRequest struct {
	Ticket string `json:"ticket"`
}

// ticketHandlers wraps the dependencies the mint / redeem handlers
// share. Same shape as loginHandlers — keeps tests injectable.
type ticketHandlers struct {
	store     *sessions.Store
	adminUser string
}

func newTicketHandlers(store *sessions.Store, adminUser string) *ticketHandlers {
	return &ticketHandlers{store: store, adminUser: adminUser}
}

// mint handles POST /_plugin/auth/ticket. Bearer-only — Hive is the
// sole intended caller. Generates a 32-byte random ticket, parks it
// in Redis with a 30s TTL, returns the value.
//
// The Redis value carries the admin user so a future multi-user
// setup can mint per-user tickets without changing the wire shape.
func (h *ticketHandlers) mint(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	if h.store == nil {
		http.Error(w, "session store unavailable", http.StatusServiceUnavailable)
		return
	}
	client := redisclient.Client()
	if client == nil {
		http.Error(w, "session store unavailable", http.StatusServiceUnavailable)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), ticketOpTimeout)
	defer cancel()

	ticket, err := newTicketID()
	if err != nil {
		pluginlog.Errf("adminapi: ticket id: %v", err)
		writeError(w, http.StatusInternalServerError, "internal", "could not mint ticket")
		return
	}

	payload, _ := json.Marshal(map[string]string{"user": h.adminUser})
	if err := client.Set(ctx, ticketKey(ticket), payload, ticketTTL).Err(); err != nil {
		pluginlog.Errf("adminapi: ticket set: %v", err)
		writeError(w, http.StatusServiceUnavailable, "session_store_unavailable",
			"could not persist ticket")
		return
	}

	writeJSON(w, http.StatusOK, TicketResponse{
		Ticket:    ticket,
		ExpiresIn: int(ticketTTL.Seconds()),
	})
}

// redeem handles POST /_plugin/auth/redeem. Anonymous — the ticket
// is the proof of authorization, redeemed exactly once via
// GETDEL so concurrent redemptions can't both succeed.
//
// Sets the session cookie and returns the canonical LoginResponse so
// the SPA's existing post-login code path needs no special case.
func (h *ticketHandlers) redeem(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	if h.store == nil {
		http.Error(w, "session store unavailable", http.StatusServiceUnavailable)
		return
	}
	client := redisclient.Client()
	if client == nil {
		http.Error(w, "session store unavailable", http.StatusServiceUnavailable)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), ticketOpTimeout)
	defer cancel()

	var req redeemRequest
	if err := decodeJSON(r, &req); err != nil || req.Ticket == "" {
		writeError(w, http.StatusBadRequest, "bad_request", "missing ticket")
		return
	}

	// GETDEL is atomic single-use: the second concurrent caller
	// for the same ticket gets redis.Nil and is rejected.
	raw, err := client.GetDel(ctx, ticketKey(req.Ticket)).Result()
	if err == redis.Nil {
		writeError(w, http.StatusUnauthorized, "invalid_ticket",
			"ticket unknown or already redeemed")
		return
	}
	if err != nil {
		pluginlog.Errf("adminapi: ticket getdel: %v", err)
		writeError(w, http.StatusServiceUnavailable, "session_store_unavailable",
			"could not redeem ticket")
		return
	}

	var meta struct {
		User string `json:"user"`
	}
	if err := json.Unmarshal([]byte(raw), &meta); err != nil || meta.User == "" {
		// Malformed payload — fail closed.
		writeError(w, http.StatusUnauthorized, "invalid_ticket", "ticket malformed")
		return
	}

	id, _, err := h.store.Create(ctx, meta.User)
	if err != nil {
		pluginlog.Errf("adminapi: ticket redeem create: %v", err)
		writeError(w, http.StatusServiceUnavailable, "session_store_unavailable",
			"could not create session")
		return
	}

	setSessionCookie(w, r, id)
	writeJSON(w, http.StatusOK, LoginResponse{User: meta.User})
}

// ticketKey is the Redis namespace for outstanding tickets. Distinct
// from the session keyspace so a stray SCAN can tell them apart.
func ticketKey(ticket string) string { return redisclient.Key("auth:ticket:" + ticket) }

// newTicketID returns a 43-char base64url-encoded 32-byte random
// string. Same shape as session IDs — URL-safe, no padding.
func newTicketID() (string, error) {
	var b [32]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b[:]), nil
}
