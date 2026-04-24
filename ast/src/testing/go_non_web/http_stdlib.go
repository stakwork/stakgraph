package main

// @ast node: Endpoint "/health"
// @ast edge: Handler -> Function "HealthHandler" "http_stdlib.go"
// @ast node: Endpoint "/ready"
// @ast edge: Handler -> Function "ReadyHandler" "http_stdlib.go"
// @ast node: Endpoint "/anon"
// @ast edge: Handler -> Function "HANDLEFUNC_anon_func_L19" "http_stdlib.go"
// @ast node: Function "RegisterRoutes"
// @ast node: Function "HealthHandler"
// @ast node: Function "ReadyHandler"
// @ast node: Function "HANDLEFUNC_anon_func_L19"

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
