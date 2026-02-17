#include "crow.h"
#include "routes.h"
#include "model.h"

std::string app_name = "StakGraph";
int main() {
    Database db("people.db");
    crow::SimpleApp app;
    setup_routes(app, db);
    app.port(18080).multithreaded().run();
}