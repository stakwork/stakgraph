package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"go_scripts/models"
	"go_scripts/utils"
)

type CreateUserRequest struct {
	Username string `json:"username"`
	Email    string `json:"email"`
	Password string `json:"password"`
}

func HandleUsers(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		createUser(w, r)
	case http.MethodGet:
		getUser(w, r)
	default:
		utils.WriteError(w, http.StatusMethodNotAllowed, "Method not allowed")
	}
}

func createUser(w http.ResponseWriter, r *http.Request) {
	var req CreateUserRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		utils.WriteError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	user, err := models.NewUser(req.Username, req.Email, req.Password)
	if err != nil {
		utils.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}

	if err := models.DB.CreateUser(user); err != nil {
		utils.WriteError(w, http.StatusConflict, err.Error())
		return
	}

	utils.WriteJSON(w, http.StatusCreated, user)
}

func getUser(w http.ResponseWriter, r *http.Request) {
	// Simple path extraction: /users/{id}
	pathParts := strings.Split(r.URL.Path, "/")
	if len(pathParts) < 3 || pathParts[2] == "" {
		utils.WriteError(w, http.StatusBadRequest, "Missing user ID")
		return
	}
	id := pathParts[2]

	user, err := models.DB.GetUser(id)
	if err != nil {
		utils.WriteError(w, http.StatusNotFound, "User not found")
		return
	}

	utils.WriteJSON(w, http.StatusOK, user)
}
