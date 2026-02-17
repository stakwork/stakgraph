cmake_minimum_required(VERSION 3.10)
project(systems_programming C)

add_executable(app 
    file_dummy.c # We need a main, just creating a dummy target or linking these as objects
)

# Or better, just a library
add_library(systems 
    memory_pool.c 
    thread_pool.c
    lock_free_queue.c
)

target_link_libraries(systems pthread)
