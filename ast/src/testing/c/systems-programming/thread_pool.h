#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <pthread.h>

typedef struct ThreadPool ThreadPool;
typedef void (*task_fn)(void*);

/**
 * @brief Create a thread pool
 */
ThreadPool* thread_pool_create(int num_threads);

/**
 * @brief Submit a task to the pool
 */
void thread_pool_submit(ThreadPool *pool, task_fn task, void *arg);

/**
 * @brief Shutdown the pool
 */
void thread_pool_shutdown(ThreadPool *pool);

#endif // THREAD_POOL_H
