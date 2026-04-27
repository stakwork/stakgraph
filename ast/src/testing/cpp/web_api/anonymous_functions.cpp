// @ast node: Function "main_anon"
// @ast node: Endpoint "/anon-post" [verb=POST]
// @ast edge: Handler -> Function "POST_anon-post_lambda_L18" "anonymous_functions.cpp"
// @ast node: Function "POST_anon-post_lambda_L18"
// @ast edge: NestedIn -> Function "main_anon" "anonymous_functions.cpp"
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
