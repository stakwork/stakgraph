package com.kotlintestapp.data.models
// @ast node: Class "ApiResult"
// @ast node: Class "Error"
// @ast node: Class "Success"
// @ast node: Class "User"
// @ast node: Class "UserRole"
// @ast node: DataModel "ApiResult"
// @ast node: DataModel "Error"
// @ast node: DataModel "Success"
// @ast node: DataModel "User"
// @ast node: DataModel "UserRole"
// @ast node: Import "import-imports-srctestingkotlinappsrcmainjavacomkotlintestappdatamodelsusermodelskt-0"

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
