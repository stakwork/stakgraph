#include "../model.h"
#include <cassert>

#define assert_eq(actual, expected) assert((actual) == (expected))

void test_database_create_and_fetch() {
    Database db(":memory:");
    Person p{1, "Ada", "ada@example.com"};

    bool created = db.createPerson(p);
    assert_eq(created, true);

    auto fetched = db.getPerson(1);
    assert_eq(fetched.has_value(), true);
    assert_eq(fetched->id, 1);
    assert_eq(fetched->name, std::string("Ada"));
    assert_eq(fetched->email, std::string("ada@example.com"));
}

void test_database_missing_person() {
    Database db(":memory:");
    auto missing = db.getPerson(999);
    assert_eq(missing.has_value(), false);
}

int main() {
    test_database_create_and_fetch();
    test_database_missing_person();
    return 0;
}
