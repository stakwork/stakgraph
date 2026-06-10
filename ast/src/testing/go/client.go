package main

// @ast node: Function "FetchPerson"
// @ast node: Request "http://localhost:5002/person/1" [verb=GET]
// @ast node: Function "SubmitPerson"
// @ast node: Request "http://localhost:5002/person" [verb=POST]
// @ast edge: Calls -> Endpoint "/person" "routes.go" [verb=POST]
// @ast node: Function "RemovePerson"
// @ast node: Request "http://localhost:5002/person/1" [verb=DELETE]

import (
	"bytes"
	"net/http"
)

func FetchPerson() (*http.Response, error) {
	return http.Get("http://localhost:5002/person/1")
}

func SubmitPerson(payload []byte) (*http.Response, error) {
	return http.Post("http://localhost:5002/person", "application/json", bytes.NewBuffer(payload))
}

func RemovePerson() (*http.Response, error) {
	req, err := http.NewRequest("DELETE", "http://localhost:5002/person/1", nil)
	if err != nil {
		return nil, err
	}
	return http.DefaultClient.Do(req)
}
