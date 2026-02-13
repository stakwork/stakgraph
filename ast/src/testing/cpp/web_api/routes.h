#pragma once
#include "crow.h"
#include "model.h"

void setup_routes(crow::SimpleApp& app, Database& db);

crow::response get_person_by_id(const crow::request& req, int id, Database& db);
crow::response new_person(const crow::request& req, Database& db);