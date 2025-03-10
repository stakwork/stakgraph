package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/go-chi/chi"
	"github.com/go-chi/chi/middleware"
	"github.com/rs/cors"
)

// NewRouter creates a chi router
func NewRouter() *http.Server {
	r := initChi()

	r.Group(func(r chi.Router) {
		r.Get("/person/{id}", GetPerson)
		r.Post("/person", CreatePerson)
	})

	PORT := os.Getenv("PORT")
	if PORT == "" {
		PORT = "5002"
	}

	server := &http.Server{Addr: ":" + PORT, Handler: r}

	go func() {
		fmt.Printf("Listening on port %s\n", PORT)
		if err := server.ListenAndServe(); err != nil {
			fmt.Printf("server err: %s\n", err.Error())
		}
	}()

	return server
}

func GetPerson(w http.ResponseWriter, r *http.Request) {
	idstr := chi.URLParam(r, "id")
	id, _ := strconv.Atoi(idstr)
	p, err := DB.GetPerson(uint(id))
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(err)
		return
	}
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(p)
}

func CreatePerson(w http.ResponseWriter, r *http.Request) {
	p := Person{}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		w.WriteHeader(http.StatusNotAcceptable)
		return
	}
	r.Body.Close()
	err = json.Unmarshal(body, &p)
	if err != nil {
		w.WriteHeader(http.StatusNotAcceptable)
		return
	}
	err = DB.CreatePerson(p)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(err)
		return
	}
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(p)
}

func initChi() *chi.Mux {
	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	cors := cors.New(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-CSRF-Token", "X-User", "authorization", "x-jwt", "Referer", "User-Agent", "x-session-id"},
		AllowCredentials: true,
		MaxAge:           300,
	})
	r.Use(cors.Handler)
	r.Use(middleware.Timeout(60 * time.Second))
	return r
}
