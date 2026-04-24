package main

// @ast node: UnitTest "TestBuildCommands"
// @ast edge: Calls -> Function "BuildCommands" "cli.go"
// @ast node: UnitTest "TestPing"
// @ast edge: Calls -> Function "Ping" "cli.go"
// @ast node: UnitTest "TestVersion"
// @ast edge: Calls -> Function "Version" "cli.go"

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
