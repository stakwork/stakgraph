package models

import (
	"errors"
	"sync"
)

var (
	ErrNotFound      = errors.New("record not found")
	ErrDuplicateKey  = errors.New("record with this key already exists")
)

// Database simulates a thread-safe in-memory data store
type Database struct {
	usersMu sync.RWMutex
	users   map[string]*User

	postsMu sync.RWMutex
	posts   map[string]*Post
}

func NewDatabase() *Database {
	return &Database{
		users: make(map[string]*User),
		posts: make(map[string]*Post),
	}
}

// User methods
func (db *Database) CreateUser(u *User) error {
	db.usersMu.Lock()
	defer db.usersMu.Unlock()

	if _, exists := db.users[u.ID]; exists {
		return ErrDuplicateKey
	}
	
	// Check for unique email (naive O(N) check for simulation)
	for _, existing := range db.users {
		if existing.Email == u.Email {
			return errors.New("email already taken")
		}
	}

	db.users[u.ID] = u
	return nil
}

func (db *Database) GetUser(id string) (*User, error) {
	db.usersMu.RLock()
	defer db.usersMu.RUnlock()

	u, ok := db.users[id]
	if !ok {
		return nil, ErrNotFound
	}
	return u, nil
}

// Post methods
func (db *Database) CreatePost(p *Post) error {
	db.postsMu.Lock()
	defer db.postsMu.Unlock()

	db.posts[p.ID] = p
	return nil
}

func (db *Database) GetPost(id string) (*Post, error) {
	db.postsMu.RLock()
	defer db.postsMu.RUnlock()

	p, ok := db.posts[id]
	if !ok {
		return nil, ErrNotFound
	}
	return p, nil
}

// Global DB instance
var DB = NewDatabase()
