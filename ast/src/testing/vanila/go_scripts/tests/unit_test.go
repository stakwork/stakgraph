package tests

import (
	"testing"

	"go_scripts/models"
	"go_scripts/utils"
)

func TestValidation(t *testing.T) {
	if utils.IsValidEmail("bad-email") {
		t.Error("Expected validation error for bad email")
	}
	if !utils.IsValidEmail("good@example.com") {
		t.Error("Expected valid email")
	}
}

func TestUserModel(t *testing.T) {
	user, err := models.NewUser("testuser", "test@example.com", "StrongP@ss1")
	if err != nil {
		t.Fatalf("Failed to create user: %v", err)
	}
	if user.Username != "testuser" {
		t.Errorf("Expected username testuser, got %s", user.Username)
	}
}
