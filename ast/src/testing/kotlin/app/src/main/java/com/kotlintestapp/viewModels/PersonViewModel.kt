package com.kotlintestapp.viewModels
// @ast node: Class "PersonViewModel"
// @ast edge: Operand -> Function "fetchAndStorePersons" "PersonViewModel.kt"
// @ast edge: Operand -> Function "fetchPersonsFromApi" "PersonViewModel.kt"
// @ast edge: Operand -> Function "insertPersonsIntoDatabase" "PersonViewModel.kt"
// @ast edge: Operand -> Function "updatePerson" "PersonViewModel.kt"
// @ast node: Function "fetchAndStorePersons"
// @ast edge: Calls -> Function "clearDatabase" "DatabaseHelper.kt"
// @ast edge: Calls -> Function "getAllPersons" "DatabaseHelper.kt"
// @ast node: Function "fetchPeople"
// @ast edge: Calls -> Request "https://people.sphinx.chat/people" "PersonViewModel.kt"
// @ast node: Function "fetchPersonsFromApi"
// @ast node: Function "insertPersonsIntoDatabase"
// @ast edge: Calls -> Function "insertPerson" "DatabaseHelper.kt"
// @ast node: Function "postUpdateProfile"
// @ast edge: Calls -> Request "https://people.sphinx.chat/person?token=testToken" "PersonViewModel.kt"
// @ast node: Function "updatePerson"
// @ast node: Request "https://people.sphinx.chat/people"
// @ast node: Request "https://people.sphinx.chat/person?token=testToken"
// @ast node: Import "import-imports-srctestingkotlinappsrcmainjavacomkotlintestappviewmodelspersonviewmodelkt-0"

import android.content.Context
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.kotlintestapp.db.Person
import com.kotlintestapp.sqldelight.DatabaseHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.Response
import java.io.IOException

class PersonViewModel(
    context: Context
) : ViewModel() {

    private val databaseHelper = DatabaseHelper(context);
    // To hold the list of Persons
    val allPersons = mutableStateOf<List<Person>>(emptyList())

    init {
        fetchAndStorePersons()
    }

    private fun fetchAndStorePersons() {
        viewModelScope.launch {
            databaseHelper.clearDatabase()

            val personsFromApi = fetchPersonsFromApi()
            insertPersonsIntoDatabase(personsFromApi)

            val updatedPersons = databaseHelper.getAllPersons()
            allPersons.value = updatedPersons
        }
    }

    private suspend fun fetchPersonsFromApi(): List<com.kotlintestapp.models.Person> {
        return withContext(Dispatchers.IO) {
            val response = ApiClient.fetchPeople()
            response
        }
    }

    private suspend fun insertPersonsIntoDatabase(persons: List<com.kotlintestapp.models.Person>) {
        withContext(Dispatchers.IO) {
            persons.forEach { person ->
                databaseHelper.insertPerson(
                    id = person.id.toLong(),
                    alias = person.owner_alias,
                    image = person.img,
                    publicKey = person.owner_pubkey,
                    routeHint = person.owner_route_hint
                )
            }
        }
    }

    fun updatePerson(person: com.kotlintestapp.models.Person) {
        ApiClient.postUpdateProfile(person) { success ->
            if (success) {
                viewModelScope.launch {
                    withContext(Dispatchers.IO) {
                        databaseHelper.updatePerson(person.id.toLong(), person.owner_alias)
                    }
                }
            }
        }
    }
}

object ApiClient {
    private val client = OkHttpClient()

    fun fetchPeople(): List<com.kotlintestapp.models.Person> {
        val request = Request.Builder()
            .url("https://people.sphinx.chat/people")
            .get()
            .build()

        return try {
            val response = client.newCall(request).execute()
            if (response.isSuccessful) {
                val json = response.body?.string()
                val listType = object : TypeToken<List<com.kotlintestapp.models.Person>>() {}.type
                Gson().fromJson(json, listType)
            } else {
                println("Error: ${response.code}")
                emptyList()
            }
        } catch (e: IOException) {
            e.printStackTrace()
            emptyList()
        }
    }

    fun postUpdateProfile(person: com.kotlintestapp.models.Person, callback: (Boolean) -> Unit) {
        val url = "https://people.sphinx.chat/person?token=testToken" // Fake API
        val json = Gson().toJson(person)
        val body = RequestBody.create("application/json".toMediaType(), json)

        val request = Request.Builder()
            .url("https://people.sphinx.chat/person?token=testToken")
            .post(body)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                e.printStackTrace()
                callback(false)
            }

            override fun onResponse(call: Call, response: Response) {
                callback(response.isSuccessful)
            }
        })
    }
}
