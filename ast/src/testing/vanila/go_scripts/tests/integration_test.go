package tests

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"go_scripts/handlers"
	"go_scripts/models"
	"go_scripts/utils"
)

func TestCreateUserFlow(t *testing.T) {
	// Setup request
	payload := map[string]string{
		"username": "integration",
		"email":    "int@example.com",
		"password": "StrongPassword1!",
	}
	body, _ := json.Marshal(payload)
	req := httptest.NewRequest("POST", "/users", bytes.NewBuffer(body))
	w := httptest.NewRecorder()

	// Execute handler directly
	handlers.HandleUsers(w, req)

	// Assert response
	if w.Code != http.StatusCreated {
		t.Errorf("Expected status 201, got %d. Body: %s", w.Code, w.Body.String())
	}

	// Verify DB state
	var response utils.JSONResponse
	json.NewDecoder(w.Body).Decode(&response)
	
	// Assuming response.Data holds the user map
	dataMap, ok := response.Data.(map[string]interface{})
	if ok {
		id := dataMap["id"].(string)
		_, err := models.DB.GetUser(id)
		if err != nil {
			t.Errorf("User not found in DB after creation")
		}
	}
}
