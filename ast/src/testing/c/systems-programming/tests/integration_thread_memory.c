#include "../thread_pool.h"
#include "../memory_pool.h"
#include <assert.h>
#include <stdio.h>
#include <pthread.h>

static MemoryPool *shared_pool;
static pthread_mutex_t test_lock = PTHREAD_MUTEX_INITIALIZER;
static int alloc_count = 0;

void task_allocate_from_pool(void *arg) {
    (void)arg;
    pthread_mutex_lock(&test_lock);
    void *block = pool_alloc(shared_pool);
    if (block) {
        alloc_count++;
    }
    pthread_mutex_unlock(&test_lock);
}

void integration_test_thread_pool_with_memory_pool(void) {
    shared_pool = pool_create(128, 10);
    ThreadPool *tpool = thread_pool_create(4);
    
    assert(shared_pool != NULL && "Memory pool creation should succeed");
    assert(tpool != NULL && "Thread pool creation should succeed");
    
    for (int i = 0; i < 5; i++) {
        thread_pool_submit(tpool, task_allocate_from_pool, NULL);
    }
    
    thread_pool_shutdown(tpool);
    
    pool_destroy(shared_pool);
    printf("PASS: integration_test_thread_pool_with_memory_pool\n");
}

void integration_test_pool_stress(void) {
    MemoryPool *pool = pool_create(256, 100);
    ThreadPool *tpool = thread_pool_create(8);
    
    assert(pool != NULL && "Large memory pool should be created");
    assert(tpool != NULL && "Thread pool with 8 threads should be created");
    
    thread_pool_shutdown(tpool);
    pool_destroy(pool);
    printf("PASS: integration_test_pool_stress\n");
}

int main(void) {
    printf("Running thread-memory integration tests...\n");
    integration_test_thread_pool_with_memory_pool();
    integration_test_pool_stress();
    printf("All integration tests passed!\n");
    return 0;
}
