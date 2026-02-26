#include "../routes.h"
#include "../model.h"
#include "crow.h"
#include <cassert>
#include <nlohmann/json.hpp>

#define assert_eq(actual, expected) assert((actual) == (expected))

void integration_test_create_and_fetch_person() {
    Database db(":memory:");
    crow::request req;
    nlohmann::json payload = {
        {"id", 12},
        {"name", "Lin"},
        {"email", "lin@example.com"}
    };
    req.body = payload.dump();

    auto create_res = new_person(req, db);
    assert_eq(create_res.code, 201);

    auto get_res = get_person_by_id(req, 12, db);
    assert_eq(get_res.code, 200);
    assert_eq(get_res.body, payload.dump());
}

void integration_test_invalid_json() {
    Database db(":memory:");
    crow::request req;
    req.body = "invalid";

    auto res = new_person(req, db);
    assert_eq(res.code, 400);
    assert_eq(res.body, std::string("Invalid JSON"));
}

void integration_test_not_found() {
    Database db(":memory:");
    crow::request req;

    auto res = get_person_by_id(req, 99, db);
    assert_eq(res.code, 404);
    assert_eq(res.body, std::string("Not found"));
}

int main() {
    integration_test_create_and_fetch_person();
    integration_test_invalid_json();
    integration_test_not_found();
    return 0;
}
