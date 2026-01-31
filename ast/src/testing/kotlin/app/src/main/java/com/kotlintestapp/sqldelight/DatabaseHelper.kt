package com.kotlintestapp.sqldelight

import android.content.Context
import app.cash.sqldelight.db.SqlDriver
import app.cash.sqldelight.driver.android.AndroidSqliteDriver
import com.kotlintestapp.db.Person
import com.kotlintestapp.db.PersonDatabase

class DatabaseHelper(context: Context) {
    private val driver: SqlDriver = AndroidSqliteDriver(PersonDatabase.Schema, context, "PersonDatabase.db")
    private val database = PersonDatabase(driver)

    private val queries = database.personQueries

    /**
     * Insert a person into the database.
     */
    fun insertPerson(
        id: Long,
        alias: String,
        image: String?,
        publicKey: String,
        routeHint: String?
    ) {
        queries.insertPerson(id, alias, image, publicKey, routeHint)
    }

    fun updatePerson(
        id: Long,
        alias: String
    ) {
        queries.updatePerson(alias, id)
    }

    fun getAllPersons(): List<Person> {
        return queries.selectAll().executeAsList()
    }

    fun clearDatabase() {
        queries.deleteAll()
    }
}
