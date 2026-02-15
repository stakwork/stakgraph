cmake_minimum_required(VERSION 3.10)
project(StakGraphCTest)

set(CMAKE_C_STANDARD 11)

include_directories(lib)

add_subdirectory(web-http)
add_subdirectory(embedded-iot)
add_subdirectory(systems-programming)
add_subdirectory(data-structures)

add_library(utils lib/string_utils.c)
