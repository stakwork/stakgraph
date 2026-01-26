package main

import (
	"fmt"
	"sync"
)

// Job represents a unit of work
type Job struct {
	ID    int
	Value int
}

// Result represents the outcome of a job
type Result struct {
	JobID int
	Value int
}

// Worker processes jobs from the jobs channel and sends results to the results channel
func Worker(id int, jobs <-chan Job, results chan<- Result, wg *sync.WaitGroup) {
	defer wg.Done()
	for j := range jobs {
		// Simulate work
		res := j.Value * 2
		results <- Result{JobID: j.ID, Value: res}
	}
}

// StartWorkerPool initializes the worker pool
func StartWorkerPool(numWorkers int, numJobs int) {
	jobs := make(chan Job, numJobs)
	results := make(chan Result, numJobs)
	var wg sync.WaitGroup

	// Start workers
	for w := 1; w <= numWorkers; w++ {
		wg.Add(1)
		go Worker(w, jobs, results, &wg)
	}

	// Send jobs
	for j := 1; j <= numJobs; j++ {
		jobs <- Job{ID: j, Value: j}
	}
	close(jobs)

	// Wait for workers to finish
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	for r := range results {
		fmt.Printf("Job %d result: %d\n", r.JobID, r.Value)
	}
}
