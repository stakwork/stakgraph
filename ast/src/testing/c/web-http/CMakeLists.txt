cmake_minimum_required(VERSION 3.10)
project(web_http_server C)

find_package(onion REQUIRED)

add_executable(server 
    server.c 
    routes.c 
    models.c
)

target_link_libraries(server onion)
