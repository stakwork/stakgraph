package models

import (
	"errors"
	"time"

	"go_scripts/utils"
)

type Post struct {
	ID        string    `json:"id"`
	AuthorID  string    `json:"author_id"`
	Title     string    `json:"title"`
	Content   string    `json:"content"`
	Published bool      `json:"published"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

var ErrEmptyContent = errors.New("post content cannot be empty")

func NewPost(authorID, title, content string) (*Post, error) {
	if content == "" {
		return nil, ErrEmptyContent
	}
	
	id, err := utils.GenerateID()
	if err != nil {
		return nil, err
	}

	now := time.Now()
	return &Post{
		ID:        id,
		AuthorID:  authorID,
		Title:     title,
		Content:   content,
		Published: false,
		CreatedAt: now,
		UpdatedAt: now,
	}, nil
}
