package main

import (
	"os"
	"strconv"
	"time"
)

type Config struct {
	Port            int
	Environment     string
	DBPath          string
	RequestTimeout  time.Duration
	ShutdownTimeout time.Duration
	SecretKey       string
}

func LoadConfig() *Config {
	return &Config{
		Port:            getEnvAsInt("PORT", 8080),
		Environment:     getEnv("GO_ENV", "development"),
		DBPath:          getEnv("DB_PATH", "memory"),
		RequestTimeout:  time.Duration(getEnvAsInt("REQUEST_TIMEOUT_SEC", 10)) * time.Second,
		ShutdownTimeout: time.Duration(getEnvAsInt("SHUTDOWN_TIMEOUT_SEC", 5)) * time.Second,
		SecretKey:       getEnv("SECRET_KEY", "super-secret-default-key"),
	}
}

func getEnv(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}

func getEnvAsInt(key string, fallback int) int {
	strValue := getEnv(key, "")
	if value, err := strconv.Atoi(strValue); err == nil {
		return value
	}
	return fallback
}
