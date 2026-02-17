package com.kotlintestapp.data.repository

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
