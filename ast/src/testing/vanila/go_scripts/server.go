package main

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"go_scripts/handlers"
	"go_scripts/utils"
)

type Server struct {
	httpServer *http.Server
	logger     *utils.Logger
}

func NewServer(cfg *Config, logger *utils.Logger) *Server {
	mux := http.NewServeMux()

	// Register routes
	mux.HandleFunc("/health", handlers.HealthCheck)
	mux.HandleFunc("/users", handlers.HandleUsers)
	mux.HandleFunc("/users/", handlers.HandleUsers) // For /users/{id}
	mux.HandleFunc("/login", handlers.HandleLogin)

	// Wrap mux with logging middleware
	handler := loggingMiddleware(logger, mux)

	return &Server{
		httpServer: &http.Server{
			Addr:         fmt.Sprintf(":%d", cfg.Port),
			Handler:      handler,
			ReadTimeout:  cfg.RequestTimeout,
			WriteTimeout: cfg.RequestTimeout,
		},
		logger: logger,
	}
}

func (s *Server) Start() error {
	s.logger.Info("Starting server on %s", s.httpServer.Addr)
	return s.httpServer.ListenAndServe()
}

func (s *Server) Stop(ctx context.Context) error {
	s.logger.Info("Shutting down server...")
	return s.httpServer.Shutdown(ctx)
}

func loggingMiddleware(logger *utils.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		
		// Create a custom response writer to capture status code
		rw := &responseWriter{ResponseWriter: w, status: http.StatusOK}
		
		next.ServeHTTP(rw, r)
		
		logger.LogRequest(r.Method, r.URL.Path, time.Since(start), rw.status)
	})
}

type responseWriter struct {
	http.ResponseWriter
	status int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.status = code
	rw.ResponseWriter.WriteHeader(code)
}
