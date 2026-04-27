package com.kotlintestapp.data.api
// @ast node: Class "ApiService"
// @ast edge: Operand -> Function "createUser" "ApiService.kt"
// @ast edge: Operand -> Function "getUser" "ApiService.kt"
// @ast edge: Operand -> Function "getUsers" "ApiService.kt"
// @ast node: Function "createUser"
// @ast edge: Calls -> Request "/users" "ApiService.kt"
// @ast node: Function "getUser"
// @ast edge: Calls -> Request "/users/{id}" "ApiService.kt"
// @ast node: Function "getUsers"
// @ast edge: Calls -> Request "/users" "ApiService.kt"
// @ast node: Request "/users"
// @ast node: Request "/users"
// @ast node: Request "/users/{id}"
// @ast node: Import "import-imports-srctestingkotlinappsrcmainjavacomkotlintestappdataapiapiservicekt-0"

import com.kotlintestapp.data.models.User
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Body
import retrofit2.http.Path

interface ApiService {
    @GET("/users")
    suspend fun getUsers(): List<User>

    @GET("/users/{id}")
    suspend fun getUser(@Path("id") id: String): User

    @POST("/users")
    suspend fun createUser(@Body user: User): User
}
