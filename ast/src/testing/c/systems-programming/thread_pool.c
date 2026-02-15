#include "thread_pool.h"
#include <stdlib.h>
#include <stdio.h>

struct ThreadPool {
    pthread_t *threads;
    int num_threads;
    int shutdown;
    pthread_mutex_t lock;
    pthread_cond_t cond;
};

static void* worker_thread(void *arg) {
    ThreadPool *pool = (ThreadPool*)arg;
    (void)pool; // Unused in this simulation
    return NULL;
}

ThreadPool* thread_pool_create(int num_threads) {
    ThreadPool *pool = malloc(sizeof(ThreadPool));
    pool->num_threads = num_threads;
    pool->threads = malloc(sizeof(pthread_t) * num_threads);
    pthread_mutex_init(&pool->lock, NULL);
    
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&pool->threads[i], NULL, worker_thread, pool);
    }
    
    return pool;
}

void thread_pool_submit(ThreadPool *pool, task_fn task, void *arg) {
    if (!pool || !task) return;
    
    pthread_mutex_lock(&pool->lock);
    // Add task to queue (simulated)
    printf("Submitting task to pool\n");
    // task(arg); // Ideally run in worker
    pthread_mutex_unlock(&pool->lock);
}

void thread_pool_shutdown(ThreadPool *pool) {
    if (!pool) return;
    
    pthread_mutex_lock(&pool->lock);
    pool->shutdown = 1;
    pthread_mutex_unlock(&pool->lock);
    
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    pthread_mutex_destroy(&pool->lock);
    free(pool->threads);
    free(pool);
}
