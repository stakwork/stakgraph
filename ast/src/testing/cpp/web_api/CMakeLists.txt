cmake_minimum_required(VERSION 3.10)
project(stakgraph_web_api_server)

set(CMAKE_CXX_STANDARD 17)

find_package(nlohmann_json REQUIRED)
find_package(SQLite3 REQUIRED)

include_directories(/opt/homebrew/include)
link_directories(/opt/homebrew/lib)

add_executable(web_api_server main.cpp model.cpp routes.cpp)

target_link_libraries(web_api_server
    nlohmann_json::nlohmann_json
    SQLite::SQLite3
    pthread
)
