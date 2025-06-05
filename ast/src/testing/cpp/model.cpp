#include "model.h"
#include <iostream>

Database::Database(const std::string& db_file) {
    if (sqlite3_open(db_file.c_str(), &db)) {
        std::cerr << "Can't open DB: " << sqlite3_errmsg(db) << std::endl;
        db = nullptr;
    } else {
        const char* sql = "CREATE TABLE IF NOT EXISTS people (id INTEGER PRIMARY KEY, name TEXT, email TEXT);";
        char* errMsg = nullptr;
        sqlite3_exec(db, sql, nullptr, nullptr, &errMsg);
        if (errMsg) sqlite3_free(errMsg);
    }
}

Database::~Database() {
    if (db) sqlite3_close(db);
}

bool Database::createPerson(const Person& p) {
    const char* sql = "INSERT INTO people (id, name, email) VALUES (?, ?, ?);";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
    sqlite3_bind_int(stmt, 1, p.id);
    sqlite3_bind_text(stmt, 2, p.name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, p.email.c_str(), -1, SQLITE_STATIC);
    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    return success;
}

std::optional<Person> Database::getPerson(int id) {
    const char* sql = "SELECT id, name, email FROM people WHERE id = ?;";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;
    sqlite3_bind_int(stmt, 1, id);
    Person p;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        p.id = sqlite3_column_int(stmt, 0);
        p.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        p.email = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        sqlite3_finalize(stmt);
        return p;
    }
    sqlite3_finalize(stmt);
    return std::nullopt;
}