#include "crow.h"

int main_anon()
{
    crow::SimpleApp app;

    CROW_ROUTE(app, "/anon-get")
    ([](){
        return "Hello world";
    });

    CROW_ROUTE(app, "/anon-post")
    .methods("POST"_method)
    ([](){
        return "Created";
    });

    app.port(18080).multithreaded().run();
}
