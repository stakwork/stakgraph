package models

import (
	"errors"
	"time"

	"go_scripts/utils"
)

type User struct {
	ID        string    `json:"id"`
	Username  string    `json:"username"`
	Email     string    `json:"email"`
	Password  string    `json:"-"` // Don't serialize password
	CreatedAt time.Time `json:"created_at"`
}

var (
	ErrInvalidEmail    = errors.New("invalid email format")
	ErrWeakPassword    = errors.New("password too weak")
	ErrInvalidUsername = errors.New("invalid username")
)

func NewUser(username, email, password string) (*User, error) {
	if !utils.IsValidSlug(username) {
		return nil, ErrInvalidUsername
	}
	if !utils.IsValidEmail(email) {
		return nil, ErrInvalidEmail
	}
	if !utils.IsStrongPassword(password) {
		return nil, ErrWeakPassword
	}

	id, err := utils.GenerateID()
	if err != nil {
		return nil, err
	}

	return &User{
		ID:        id,
		Username:  username,
		Email:     email,
		Password:  password, // In real app, hash this!
		CreatedAt: time.Now(),
	}, nil
}
