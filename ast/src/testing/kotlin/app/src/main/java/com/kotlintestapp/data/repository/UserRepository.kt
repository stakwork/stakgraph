package com.kotlintestapp.data.repository
// @ast node: Class "UserRepository"
// @ast edge: Operand -> Function "addUser" "UserRepository.kt"
// @ast edge: Operand -> Function "getUsers" "UserRepository.kt"
// @ast node: Function "addUser"
// @ast edge: Calls -> Function "createUser" "ApiService.kt"
// @ast node: Function "getUsers"
// @ast edge: Calls -> Function "getUsers" "ApiService.kt"
// @ast node: Import "import-imports-srctestingkotlinappsrcmainjavacomkotlintestappdatarepositoryuserrepositorykt-0"

import com.kotlintestapp.data.api.ApiService
import com.kotlintestapp.data.models.ApiResult
import com.kotlintestapp.data.models.User
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import javax.inject.Inject

class UserRepository @Inject constructor(
    private val apiService: ApiService
) {
    fun getUsers(): Flow<ApiResult<List<User>>> = flow {
        emit(ApiResult.Loading)
        try {
            val users = apiService.getUsers()
            emit(ApiResult.Success(users))
        } catch (e: Exception) {
            emit(ApiResult.Error(e))
        }
    }

    suspend fun addUser(user: User): ApiResult<User> {
        return try {
            val createdUser = apiService.createUser(user)
            ApiResult.Success(createdUser)
        } catch (e: Exception) {
            ApiResult.Error(e)
        }
    }
}
