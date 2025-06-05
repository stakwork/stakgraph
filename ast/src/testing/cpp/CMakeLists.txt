cmake_minimum_required(VERSION 3.10)
project(stakgraph_cpp_server)

set(CMAKE_CXX_STANDARD 17)

find_package(nlohmann_json REQUIRED)
find_package(SQLite3 REQUIRED)

include_directories(/opt/homebrew/include)
link_directories(/opt/homebrew/lib)

add_executable(server main.cpp model.cpp routes.cpp)

target_link_libraries(server
    nlohmann_json::nlohmann_json
    SQLite::SQLite3
    pthread
)