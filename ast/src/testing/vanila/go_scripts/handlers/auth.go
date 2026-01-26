package handlers

import (
	"encoding/json"
	"net/http"

	"go_scripts/utils"
)

type LoginRequest struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

func HandleLogin(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		utils.WriteError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	var req LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		utils.WriteError(w, http.StatusBadRequest, "Invalid request")
		return
	}

	// Mock authentication
	if req.Email == "admin@example.com" && req.Password == "Password123!" {
		token, _ := utils.GenerateID()
		utils.WriteJSON(w, http.StatusOK, map[string]string{
			"token": token,
			"role":  "admin",
		})
		return
	}

	utils.WriteError(w, http.StatusUnauthorized, "Invalid credentials")
}
