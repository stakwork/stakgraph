package handlers

import (
	"net/http"
)

// HealthCheck handler
func HealthCheck(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

// NotFoundHandler custom 404
func NotFoundHandler(w http.ResponseWriter, r *http.Request) {
	http.Error(w, "Endpoint not found", http.StatusNotFound)
}
