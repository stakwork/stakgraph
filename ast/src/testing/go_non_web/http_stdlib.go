package main

import (
	"net/http"
)

func RegisterRoutes() {
	http.HandleFunc("/health", HealthHandler)
	http.HandleFunc("/anon", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	http.Handle("/ready", http.HandlerFunc(ReadyHandler))
}

func HealthHandler(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func ReadyHandler(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
}
