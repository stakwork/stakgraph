#include "../memory_pool.h"
#include <assert.h>
#include <stdio.h>
#include <stdint.h>

void test_pool_create(void) {
    MemoryPool *pool = pool_create(64, 10);
    assert(pool != NULL && "Pool creation should succeed");
    pool_destroy(pool);
    printf("PASS: test_pool_create\n");
}

void test_pool_alloc_single(void) {
    MemoryPool *pool = pool_create(128, 5);
    void *block = pool_alloc(pool);
    assert(block != NULL && "First allocation should succeed");
    pool_destroy(pool);
    printf("PASS: test_pool_alloc_single\n");
}

void test_pool_alloc_multiple(void) {
    MemoryPool *pool = pool_create(32, 3);
    void *block1 = pool_alloc(pool);
    void *block2 = pool_alloc(pool);
    void *block3 = pool_alloc(pool);
    
    assert(block1 != NULL && "First allocation should succeed");
    assert(block2 != NULL && "Second allocation should succeed");
    assert(block3 != NULL && "Third allocation should succeed");
    assert(block1 != block2 && "Blocks should be different");
    assert(block2 != block3 && "Blocks should be different");
    
    pool_destroy(pool);
    printf("PASS: test_pool_alloc_multiple\n");
}

void test_pool_alloc_exhaustion(void) {
    MemoryPool *pool = pool_create(16, 2);
    void *block1 = pool_alloc(pool);
    void *block2 = pool_alloc(pool);
    void *block3 = pool_alloc(pool);
    
    assert(block1 != NULL && "First allocation should succeed");
    assert(block2 != NULL && "Second allocation should succeed");
    assert(block3 == NULL && "Third allocation should fail (pool exhausted)");
    
    pool_destroy(pool);
    printf("PASS: test_pool_alloc_exhaustion\n");
}

void test_pool_free_and_realloc(void) {
    MemoryPool *pool = pool_create(64, 3);
    void *block1 = pool_alloc(pool);
    void *block2 = pool_alloc(pool);
    
    pool_free(pool, block1);
    void *block3 = pool_alloc(pool);
    
    assert(block3 != NULL && "Allocation after free should succeed");
    assert(block3 == block1 && "Freed block should be reused");
    
    pool_destroy(pool);
    printf("PASS: test_pool_free_and_realloc\n");
}

int main(void) {
    printf("Running memory pool unit tests...\n");
    test_pool_create();
    test_pool_alloc_single();
    test_pool_alloc_multiple();
    test_pool_alloc_exhaustion();
    test_pool_free_and_realloc();
    printf("All memory pool unit tests passed!\n");
    return 0;
}
