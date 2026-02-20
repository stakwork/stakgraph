package main

import "sync"

type Job struct {
	ID int
}

type Result struct {
	ID int
}

func StartPipeline() {
	jobs := make(chan Job, 2)
	results := make(chan Result, 2)

	var wg sync.WaitGroup
	wg.Add(1)
	go worker(jobs, results, &wg)

	jobs <- Job{ID: 1}
	close(jobs)

	wg.Wait()
	close(results)

	for range results {
	}
}

func worker(jobs <-chan Job, results chan<- Result, wg *sync.WaitGroup) {
	defer wg.Done()
	for job := range jobs {
		results <- Result{ID: job.ID}
	}
}
