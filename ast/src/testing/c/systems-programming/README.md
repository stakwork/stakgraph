# Systems Programming Test Server

This simulates low-level systems programming in C to test capture of:

- Advanced memory management (`malloc`, pointers)
- Concurrency primitives (`pthread`, `mutex`, `cond`)
- Atomic operations (`_Atomic`, `atomic_compare_exchange`)
- Complex data structures linked via pointers
- Header file inclusion and interface definitions

## Structure

- `memory_pool.c`: Custom allocator implementation
- `thread_pool.c`: Worker thread pattern
- `lock_free_queue.c`: Atomics usage
