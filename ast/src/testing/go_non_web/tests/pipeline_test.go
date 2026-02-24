package main

import (
	"sync"
	"testing"
)

func TestWorker(t *testing.T) {
	jobs := make(chan Job, 1)
	results := make(chan Result, 1)
	var wg sync.WaitGroup

	wg.Add(1)
	go worker(jobs, results, &wg)

	testJob := Job{ID: 42}
	jobs <- testJob
	close(jobs)

	wg.Wait()
	close(results)

	result := <-results
	if result.ID != 42 {
		t.Errorf("Expected result ID 42, got %d", result.ID)
	}
}
