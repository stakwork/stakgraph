package com.kotlintestapp.data.api

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
