package adminapi

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"

	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/tlog"
)

// Transparency-log witness route (phase 12, Part 1):
//
//	GET /_plugin/tlog/sth?since=N → 200 tlog.Page
//	                               400 since is not a base-10 integer
//	                               409 since > tree_size (TlogAheadResponse)
//	                               503 log disabled / not initialized
//
// Bearer-only, registered in server.go. `since` defaults to 0 when
// absent. The handler fsyncs the leaf file before signing (inside
// tlog.Page), so a head the witness countersigns is never ahead of
// the disk after a crash. See
// gateway/plans/phases/phase-12-transparency-log.md "Witness API".

const tlogSthPath = "/_plugin/tlog/sth"

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
