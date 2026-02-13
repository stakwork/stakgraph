#pragma once
#include <string>
#include <sqlite3.h>
#include <optional>

struct Person {
    int id;
    std::string name;
    std::string email;
};

class Database {
public:
    Database(const std::string& db_file);
    ~Database();
    bool createPerson(const Person& p);
    std::optional<Person> getPerson(int id);
private:
    sqlite3* db;
};