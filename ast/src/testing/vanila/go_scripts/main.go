package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"
	"time"

	"go_scripts/utils"
)

func main() {
	logger := utils.NewLogger()
	cfg := LoadConfig()

	server := NewServer(cfg, logger)

	// Start server in goroutine
	go func() {
		if err := server.Start(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("Failed to start server: %v", err)
		}
	}()

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	ctx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
	defer cancel()

	if err := server.Stop(ctx); err != nil {
		logger.Fatal("Server forced to shutdown: %v", err)
	}

	logger.Info("Server exited properly")
}
