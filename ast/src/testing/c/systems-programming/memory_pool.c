#include "memory_pool.h"
#include <stdlib.h>
#include <string.h>

// @ast node: Class "MemoryPool"
struct MemoryPool {
    size_t block_size;
    size_t num_blocks;
    void *memory;
    uint8_t *bitmap;
};

// @ast node: Function "pool_create"
MemoryPool* pool_create(size_t block_size, size_t num_blocks) {
    MemoryPool *pool = malloc(sizeof(MemoryPool));
    if (!pool) return NULL;
    
    pool->block_size = block_size;
    pool->num_blocks = num_blocks;
    pool->memory = malloc(block_size * num_blocks);
    pool->bitmap = calloc((num_blocks + 7) / 8, 1);
    
    return pool;
}

// @ast node: Function "pool_alloc"
void* pool_alloc(MemoryPool *pool) {
    if (!pool) return NULL;
    
    for (size_t i = 0; i < pool->num_blocks; i++) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        
        if (!(pool->bitmap[byte_idx] & (1 << bit_idx))) {
            pool->bitmap[byte_idx] |= (1 << bit_idx);
            return (char*)pool->memory + (i * pool->block_size);
        }
    }
    
    return NULL;
}

// @ast node: Function "pool_free"
void pool_free(MemoryPool *pool, void *ptr) {
    if (!pool || !ptr) return;
    
    size_t offset = (char*)ptr - (char*)pool->memory;
    size_t idx = offset / pool->block_size;
    
    if (idx < pool->num_blocks) {
        pool->bitmap[idx / 8] &= ~(1 << (idx % 8));
    }
}

// @ast node: Function "pool_destroy"
void pool_destroy(MemoryPool *pool) {
    if (pool) {
        free(pool->memory);
        free(pool->bitmap);
        free(pool);
    }
}
