package com.kotlintestapp.data.models

data class User(
    val id: String,
    val name: String,
    val email: String,
    val role: UserRole
)

enum class UserRole {
    ADMIN, USER, GUEST
}

sealed interface ApiResult<out T> {
    data class Success<out T>(val data: T) : ApiResult<T>
    data class Error(val exception: Exception) : ApiResult<Nothing>
    object Loading : ApiResult<Nothing>
}
