#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <stddef.h>

typedef struct MemoryPool MemoryPool;

/**
 * @brief Create a new memory pool
 */
MemoryPool* pool_create(size_t block_size, size_t num_blocks);

/**
 * @brief Allocate a block from the pool
 */
void* pool_alloc(MemoryPool *pool);

/**
 * @brief Free a block back to the pool
 */
void pool_free(MemoryPool *pool, void *ptr);

/**
 * @brief Destroy the pool
 */
void pool_destroy(MemoryPool *pool);

#endif // MEMORY_POOL_H
