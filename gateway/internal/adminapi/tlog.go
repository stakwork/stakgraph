package adminapi

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"

	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/tlog"
)

// Transparency-log routes (phase 12, Part 1):
//
//	GET /_plugin/tlog/sth?since=N → 200 tlog.Page
//	                               400 since is not a base-10 integer
//	                               409 since > tree_size (TlogAheadResponse)
//	                               503 log disabled / not initialized
//	GET /_plugin/tlog/status      → 200 TlogStatusResponse, always
//
// /sth is bearer-only, registered in server.go. `since` defaults to
// 0 when absent. The handler fsyncs the leaf file before signing
// (inside tlog.Page), so a head the witness countersigns is never
// ahead of the disk after a crash. See
// gateway/plans/phases/phase-12-transparency-log.md "Witness API".
//
// /status is cookie-or-bearer: the dashboard card's read. It carries
// only what the gateway can attest about itself — size, root, the
// per-boot log key, the newest leaf's ts, and whether the log is up.
// No leaves and no STH signature, so a cookie session can never pull
// material the witness org-signs. It answers 200 even when the log
// is down (healthy:false + error) because the card has to render the
// broken state, not a 503.

const (
	tlogSthPath    = "/_plugin/tlog/sth"
	tlogStatusPath = "/_plugin/tlog/status"
)

// TlogStatusResponse is the 200 body of GET /_plugin/tlog/status.
// Exported for tygo (the dashboard's TlogCard decodes it).
//
// "Witnessed at head N" is deliberately not here: witnessing is
// Hive's fact, and this endpoint only reports the gateway's own.
type TlogStatusResponse struct {
	// Healthy is false when the log was never initialized or is
	// disabled (refused to come up, or stopped itself after an
	// unrecoverable write). Error carries the reason; null otherwise.
	Healthy bool    `json:"healthy"`
	Error   *string `json:"error"`
	// TreeSize is the number of leaves; RootHash their Merkle root
	// as hex (the RFC 9162 empty root at size 0).
	TreeSize uint64 `json:"tree_size"`
	RootHash string `json:"root_hash"`
	// LogPubkey is this boot's log key, compressed secp256k1 hex.
	// It is regenerated on every restart and is not an identity key.
	LogPubkey string `json:"log_pubkey"`
	// Path is the leaf file (BIFROST_PLUGIN_TLOG_PATH).
	Path string `json:"path"`
	// LastLeafTS is the RFC 3339 ts of the newest leaf; null while
	// the log is empty.
	LastLeafTS *string `json:"last_leaf_ts"`
}

// TlogAheadResponse is the 409 body: the witness's stored head is
// past this tree, which after a power loss means the witness must
// stop and an operator must reset its head. Never silently restart.
type TlogAheadResponse struct {
	Error    string `json:"error"`
	Since    uint64 `json:"since"`
	TreeSize uint64 `json:"tree_size"`
}

// TlogUnavailableResponse is the 503 body.
type TlogUnavailableResponse struct {
	Error  string `json:"error"`
	Detail string `json:"detail"`
}

type tlogHandlers struct {
	// log is resolved per request so tests can swap the process-wide
	// instance and so a log that comes up disabled still reports why.
	log func() *tlog.Log
}

func newTlogHandlers() *tlogHandlers { return &tlogHandlers{log: tlog.Default} }

// status serves the dashboard card. One tlog.Status() call under the
// log's mutex; no signing, no fsync, no file read.
func (h *tlogHandlers) status(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", "GET")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	l := h.log()
	if l == nil {
		empty := tlog.EmptyRoot()
		msg := tlog.ErrNotInitialized.Error()
		writeJSON(w, http.StatusOK, TlogStatusResponse{
			Healthy:  false,
			Error:    &msg,
			RootHash: hex.EncodeToString(empty[:]),
		})
		return
	}
	st := l.Status()
	resp := TlogStatusResponse{
		Healthy:   st.Err == nil,
		TreeSize:  st.Size,
		RootHash:  hex.EncodeToString(st.Root[:]),
		LogPubkey: st.PubkeyHex,
		Path:      st.Path,
	}
	if st.Err != nil {
		msg := st.Err.Error()
		resp.Error = &msg
	}
	if st.LastTS != "" {
		ts := st.LastTS
		resp.LastLeafTS = &ts
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *tlogHandlers) sth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", "GET")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var since uint64
	if raw := r.URL.Query().Get("since"); raw != "" {
		n, err := strconv.ParseUint(raw, 10, 64)
		if err != nil {
			http.Error(w, "since must be a base-10 integer in [0, tree_size]", http.StatusBadRequest)
			return
		}
		since = n
	}

	l := h.log()
	if l == nil {
		writeJSON(w, http.StatusServiceUnavailable, TlogUnavailableResponse{
			Error: "tlog_unavailable", Detail: tlog.ErrNotInitialized.Error(),
		})
		return
	}
	page, err := l.Page(since)
	if err != nil {
		var ahead *tlog.AheadError
		switch {
		case errors.As(err, &ahead):
			writeJSON(w, http.StatusConflict, TlogAheadResponse{
				Error: "since_ahead_of_tree", Since: ahead.Since, TreeSize: ahead.TreeSize,
			})
		default:
			pluginlog.Errf("adminapi: tlog page since=%d: %v", since, err)
			writeJSON(w, http.StatusServiceUnavailable, TlogUnavailableResponse{
				Error: "tlog_unavailable", Detail: err.Error(),
			})
		}
		return
	}

	// Encode without HTML escaping: the leaves are the canonical bytes
	// that were hashed, and they should reach the witness verbatim
	// (json.Marshal would rewrite `<`, `>`, `&` as \u escapes).
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(page); err != nil {
		pluginlog.Warnf("adminapi: tlog page write: %v", err)
	}
}
