package main

import (
	"testing"
)

func TestBuildCommands(t *testing.T) {
	commands := BuildCommands()
	if len(commands) != 2 {
		t.Errorf("Expected 2 commands, got %d", len(commands))
	}
}

func TestPing(t *testing.T) {
	err := Ping()
	if err != nil {
		t.Errorf("Expected no error from Ping, got %v", err)
	}
}

func TestVersion(t *testing.T) {
	err := Version()
	if err != nil {
		t.Errorf("Expected no error from Version, got %v", err)
	}
}
